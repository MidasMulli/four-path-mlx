#!/usr/bin/env python3
"""
Custom ANE Model Conversion for Speculative Drafting
=====================================================

Converts Qwen3-1.7B to CoreML with:
- 1024 context length (fits full tool prompts)
- Explicit KV cache as model I/O (warm cache across calls)
- Conv-first layout for ANE fast datapath
- Optimized for single-token decode (the spec decode use case)

This is a purpose-built conversion, not a general-purpose tool.

Usage:
    ~/.mlx-env/bin/python3.11 ane_convert.py
"""

import os
import sys
import time
import math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ──────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-1.7B"
CONTEXT_LENGTH = 1024
OUTPUT_DIR = Path.home() / "models" / "qwen3-1.7b-ane-1024"

# ── ANE-Optimized Modules ──────────────────────────────────────


class Conv1x1(nn.Module):
    """Linear projection as 1x1 convolution for ANE fast datapath.
    Input:  (batch, channels, 1, seq)
    Output: (batch, out_channels, 1, seq)"""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.conv(x)


class RMSNorm(nn.Module):
    """RMSNorm in channel-first 4D format for ANE.
    Pure FP16 implementation for CoreML compatibility."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float16))
        self.eps_tensor = nn.Parameter(torch.tensor(eps, dtype=torch.float16), requires_grad=False)

    def forward(self, x):
        # x: (batch, dim, 1, seq) in fp16
        variance = x.pow(2).mean(dim=1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps_tensor)
        return x * self.weight.view(1, -1, 1, 1)


class RMSNormFlat(nn.Module):
    """RMSNorm for attention heads - works on (batch, heads, seq, dim).
    Pure FP16 for CoreML."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float16))
        self.eps_tensor = nn.Parameter(torch.tensor(eps, dtype=torch.float16), requires_grad=False)

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps_tensor)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputed rotary embeddings for ANE."""

    def __init__(self, head_dim, max_seq_len=1024, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # Register as buffers (not parameters)
        self.register_buffer("cos_cached", cos.half())
        self.register_buffer("sin_cached", sin.half())

    def forward(self, position_ids):
        # position_ids: (batch,) or (batch, seq)
        cos = self.cos_cached[position_ids]  # (batch, seq, head_dim//2)
        sin = self.sin_cached[position_ids]
        return cos, sin


def apply_rotary(x, cos, sin):
    """Apply rotary embeddings. x: (batch, heads, seq, head_dim)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class QwenAttention(nn.Module):
    """GQA attention with conv-first projections and explicit KV cache."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        # Conv-first projections
        self.q_proj = Conv1x1(hidden_size, num_heads * head_dim)
        self.k_proj = Conv1x1(hidden_size, num_kv_heads * head_dim)
        self.v_proj = Conv1x1(hidden_size, num_kv_heads * head_dim)
        self.o_proj = Conv1x1(num_heads * head_dim, hidden_size)

        # Q/K norms (Qwen3 has these) - use our FP16-compatible RMSNorm
        self.q_norm = RMSNormFlat(head_dim, eps=1e-6)
        self.k_norm = RMSNormFlat(head_dim, eps=1e-6)

    def forward(self, x, cos, sin, kv_cache_k, kv_cache_v, position, mask):
        """
        x: (batch, hidden, 1, seq)
        kv_cache_k/v: (batch, kv_heads, context_len, head_dim)
        position: current position index
        mask: causal mask
        Returns: output, new_k_cache, new_v_cache
        """
        batch = x.shape[0]
        seq_len = x.shape[3]

        # Project Q, K, V via conv
        q = self.q_proj(x)  # (batch, heads*dim, 1, seq)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to attention format: (batch, heads, seq, dim)
        q = q.squeeze(2).view(batch, self.num_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        k = k.squeeze(2).view(batch, self.num_kv_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        v = v.squeeze(2).view(batch, self.num_kv_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)

        # Q/K norms
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Rotary embeddings
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Update KV cache
        # Write new K, V at the current position
        new_k = kv_cache_k.clone()
        new_v = kv_cache_v.clone()
        new_k[:, :, position:position + seq_len, :] = k
        new_v[:, :, position:position + seq_len, :] = v

        # GQA: repeat KV heads for multi-head attention
        if self.num_kv_groups > 1:
            k_expanded = new_k.repeat_interleave(self.num_kv_groups, dim=1)
            v_expanded = new_v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_expanded = new_k
            v_expanded = new_v

        # Attention: Q * K^T / sqrt(d) + mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn.float(), dim=-1).half()
        out = torch.matmul(attn, v_expanded)

        # Reshape back to (batch, heads*dim, 1, seq) for conv
        out = out.permute(0, 1, 3, 2).contiguous().view(batch, -1, 1, seq_len)

        # Output projection
        out = self.o_proj(out)
        return out, new_k, new_v


class QwenMLP(nn.Module):
    """SwiGLU MLP with conv-first projections."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = Conv1x1(hidden_size, intermediate_size)
        self.up_proj = Conv1x1(hidden_size, intermediate_size)
        self.down_proj = Conv1x1(intermediate_size, hidden_size)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class QwenDecoderLayer(nn.Module):
    """Single transformer layer with conv-first layout."""

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config["hidden_size"])
        self.post_attention_layernorm = RMSNorm(config["hidden_size"])
        self.self_attn = QwenAttention(
            config["hidden_size"],
            config["num_attention_heads"],
            config["num_key_value_heads"],
            config["head_dim"],
        )
        self.mlp = QwenMLP(config["hidden_size"], config["intermediate_size"])

    def forward(self, x, cos, sin, kv_k, kv_v, position, mask):
        residual = x
        x = self.input_layernorm(x)
        attn_out, new_k, new_v = self.self_attn(x, cos, sin, kv_k, kv_v, position, mask)
        x = residual + attn_out
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x, new_k, new_v


class QwenForANEDraft(nn.Module):
    """Complete Qwen3-1.7B for ANE speculative drafting.

    Inputs:
        input_ids: (1, seq_len) token IDs
        position: scalar, current position in context
        kv_caches: list of (k, v) tensors per layer
        mask: causal mask

    Outputs:
        logits: (1, vocab_size) next token logits
        updated kv_caches: list of (k, v) with new entries written
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            QwenDecoderLayer(config) for _ in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.rotary = RotaryEmbedding(config["head_dim"], max_seq_len=config["context_length"])

    def forward(self, input_ids, position, *kv_flat, mask=None):
        """Forward pass with explicit KV cache I/O.

        kv_flat: flattened list of [k0, v0, k1, v1, ...] tensors
        """
        # Embed
        h = self.embed_tokens(input_ids)  # (batch, seq, hidden)

        # Reshape to channel-first: (batch, hidden, 1, seq)
        h = h.permute(0, 2, 1).unsqueeze(2)

        # Rotary - position is (1,) tensor
        pos_val = position[0]
        pos_ids = pos_val.unsqueeze(0)  # Single position for single-token decode
        cos, sin = self.rotary(pos_ids)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Process layers with KV cache
        new_kv_flat = []
        for i, layer in enumerate(self.layers):
            kv_k = kv_flat[i * 2]
            kv_v = kv_flat[i * 2 + 1]
            h, new_k, new_v = layer(h, cos, sin, kv_k, kv_v, position, mask)
            new_kv_flat.extend([new_k, new_v])

        # Final norm and LM head
        h = self.norm(h)
        # Reshape back: (batch, hidden, 1, seq) -> (batch, seq, hidden)
        h = h.squeeze(2).permute(0, 2, 1)
        logits = self.lm_head(h[:, -1:, :])  # Only last token

        return (logits,) + tuple(new_kv_flat)


def load_weights(model, model_id):
    """Load weights from HuggingFace model into our ANE-optimized model."""
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_id} from HuggingFace...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    hf_state = hf_model.state_dict()

    # Map weights
    state_dict = {}

    # Embeddings
    state_dict["embed_tokens.weight"] = hf_state["model.embed_tokens.weight"]

    # LM head
    if "lm_head.weight" in hf_state:
        state_dict["lm_head.weight"] = hf_state["lm_head.weight"]
    else:
        state_dict["lm_head.weight"] = hf_state["model.embed_tokens.weight"]

    # Layers
    for i in range(model.config["num_hidden_layers"]):
        prefix = f"model.layers.{i}"
        our_prefix = f"layers.{i}"

        # Norms
        state_dict[f"{our_prefix}.input_layernorm.weight"] = hf_state[f"{prefix}.input_layernorm.weight"]
        state_dict[f"{our_prefix}.post_attention_layernorm.weight"] = hf_state[f"{prefix}.post_attention_layernorm.weight"]

        # Attention projections: Linear -> Conv1x1
        # HF shape: (out, in) -> Conv shape: (out, in, 1, 1)
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w = hf_state[f"{prefix}.self_attn.{proj}.weight"]
            state_dict[f"{our_prefix}.self_attn.{proj}.conv.weight"] = w.unsqueeze(-1).unsqueeze(-1)

        # Q/K norms
        state_dict[f"{our_prefix}.self_attn.q_norm.weight"] = hf_state[f"{prefix}.self_attn.q_norm.weight"]
        state_dict[f"{our_prefix}.self_attn.k_norm.weight"] = hf_state[f"{prefix}.self_attn.k_norm.weight"]

        # MLP projections: Linear -> Conv1x1
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            w = hf_state[f"{prefix}.mlp.{proj}.weight"]
            state_dict[f"{our_prefix}.mlp.{proj}.conv.weight"] = w.unsqueeze(-1).unsqueeze(-1)

    # Final norm
    state_dict["norm.weight"] = hf_state["model.norm.weight"]

    # Load
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}...")
    print(f"  Loaded {len(state_dict)} weight tensors")

    del hf_model
    return model


def convert_to_coreml(model, config):
    """Convert the ANE-optimized model to CoreML."""
    import coremltools as ct

    print("Tracing model...")
    model.eval()

    # Example inputs for tracing
    batch = 1
    seq_len = 1  # Single token decode
    n_layers = config["num_hidden_layers"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    ctx = config["context_length"]

    input_ids = torch.randint(0, 1000, (batch, seq_len), dtype=torch.long)
    position = torch.tensor([10], dtype=torch.long)

    # KV caches as flat list
    kv_flat = []
    for _ in range(n_layers):
        kv_flat.append(torch.zeros(batch, n_kv_heads, ctx, head_dim, dtype=torch.float16))
        kv_flat.append(torch.zeros(batch, n_kv_heads, ctx, head_dim, dtype=torch.float16))

    with torch.no_grad():
        traced = torch.jit.trace(model, (input_ids, position, *kv_flat))

    print("Converting to CoreML...")
    # Define inputs
    inputs = [
        ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="position", shape=(1,), dtype=np.int32),
    ]
    for i in range(n_layers):
        inputs.append(ct.TensorType(
            name=f"kv_cache_k_{i}",
            shape=(1, n_kv_heads, ctx, head_dim),
            dtype=np.float16,
        ))
        inputs.append(ct.TensorType(
            name=f"kv_cache_v_{i}",
            shape=(1, n_kv_heads, ctx, head_dim),
            dtype=np.float16,
        ))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,  # Let CoreML decide ANE vs GPU
        minimum_deployment_target=ct.target.iOS18,
    )

    # Save
    output_path = OUTPUT_DIR / "qwen3_1.7b_draft.mlpackage"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(f"Saved to {output_path}")
    return output_path


def main():
    config = {
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 6144,
        "vocab_size": 151936,
        "context_length": CONTEXT_LENGTH,
    }

    print(f"Building ANE draft model: Qwen3-1.7B @ {CONTEXT_LENGTH} context")
    print(f"  {config['num_hidden_layers']} layers, {config['hidden_size']} hidden, "
          f"{config['num_attention_heads']} heads, GQA {config['num_key_value_heads']} KV heads")

    # Build model
    model = QwenForANEDraft(config)
    model = model.half()

    # Load weights
    model = load_weights(model, MODEL_ID)

    # Quick test
    print("\nTesting forward pass...")
    with torch.no_grad():
        input_ids = torch.tensor([[1]], dtype=torch.long)
        position = torch.tensor(0, dtype=torch.long)
        kv_flat = []
        for _ in range(config["num_hidden_layers"]):
            kv_flat.append(torch.zeros(1, config["num_key_value_heads"],
                                        CONTEXT_LENGTH, config["head_dim"], dtype=torch.float16))
            kv_flat.append(torch.zeros(1, config["num_key_value_heads"],
                                        CONTEXT_LENGTH, config["head_dim"], dtype=torch.float16))

        outputs = model(input_ids, position, *kv_flat)
        logits = outputs[0]
        print(f"  Logits shape: {logits.shape}")
        print(f"  Top-5 tokens: {torch.topk(logits[0, 0], 5).indices.tolist()}")

    # Convert to CoreML
    convert_to_coreml(model, config)
    print("\nDone!")


if __name__ == "__main__":
    main()
