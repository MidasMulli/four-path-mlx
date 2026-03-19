"""
MTP Patch — Bolt MTP (Multi-Token Prediction) onto a stock mlx-lm Qwen3.5 model.

Loads MTP weights from a separate safetensors file and patches the model with:
  - model.language_model.mtp (MTPModule)
  - model.language_model.mtp_forward() method
  - model.language_model.make_mtp_cache() method

Based on AirRunner's PR #990 to ml-explore/mlx-lm.
"""

import os
import json
import types
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3_5 import Attention, MLP, TextModelArgs
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache


class MTPDecoderLayer(nn.Module):
    """Single transformer layer for MTP head (full attention only, no GatedDeltaNet)."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x, mask, cache):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class MTPModule(nn.Module):
    """
    Multi-Token Prediction head.

    Fuses backbone hidden state h_t with embedding of token t+1 via linear projection,
    runs through one transformer layer, applies final norm.
    """

    def __init__(self, args: TextModelArgs, num_layers: int = 1):
        super().__init__()
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.layers = [MTPDecoderLayer(args) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
        embeds = embed_tokens(next_token_ids)
        e = self.pre_fc_norm_embedding(embeds)
        h = self.pre_fc_norm_hidden(hidden_states)
        fused = self.fc(mx.concatenate([e, h], axis=-1))

        mask = create_attention_mask(fused, cache[0]) if cache else None

        for layer, c in zip(self.layers, cache or [None] * len(self.layers)):
            fused = layer(fused, mask, c)

        return self.norm(fused)


def _make_mtp_cache(self):
    """Create KV cache for MTP layers."""
    return [KVCache() for _ in self.mtp.layers]


def _mtp_forward(self, hidden_states, next_token_ids, mtp_cache):
    """Run MTP head: fuse hidden + embedding, predict next-next token."""
    mtp_out = self.mtp(hidden_states, next_token_ids, self.model.embed_tokens, mtp_cache)
    if self.args.tie_word_embeddings:
        return self.model.embed_tokens.as_linear(mtp_out)
    return self.lm_head(mtp_out)


def _patched_call(self, inputs, cache=None, input_embeddings=None, return_hidden=False, n_confirmed=0):
    """
    Patched __call__ that supports return_hidden for MTP.

    When return_hidden=True, returns (logits, hidden_states) where hidden_states
    are pre-norm (before final RMSNorm + lm_head).
    """
    # If return_hidden not requested, use the original fast path
    if not return_hidden and not hasattr(self, 'mtp'):
        return type(self)._original_call_unpatched(self, inputs, cache=cache,
                                                     input_embeddings=input_embeddings)

    # Run through the backbone manually to get pre-norm hidden states
    if input_embeddings is not None:
        h = input_embeddings
    else:
        h = self.model.embed_tokens(inputs)

    backbone = self.model
    if cache is None:
        cache_list = [None] * len(backbone.layers)
    else:
        cache_list = cache

    fa_mask = create_attention_mask(h, cache_list[backbone.fa_idx])

    from mlx_lm.models.base import create_ssm_mask
    ssm_mask = create_ssm_mask(h, cache_list[backbone.ssm_idx])

    for layer, c in zip(backbone.layers, cache_list):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=c)

    # h is now pre-norm hidden states
    normed = backbone.norm(h)

    if self.args.tie_word_embeddings:
        logits = backbone.embed_tokens.as_linear(normed)
    else:
        logits = self.lm_head(normed)

    if return_hidden:
        return logits, h  # Return pre-norm hidden for MTP
    return logits


def patch_mtp(model, mtp_weights_path: str) -> bool:
    """
    Patch a loaded Qwen3.5 model with MTP capability.

    Args:
        model: The loaded mlx model (top-level Model wrapper)
        mtp_weights_path: Path to directory containing MTP model safetensors

    Returns:
        True if MTP was successfully patched, False otherwise
    """
    mtp_path = Path(mtp_weights_path)
    safetensors_file = mtp_path / "model.safetensors"

    if not safetensors_file.exists():
        print(f"MTP weights not found at {safetensors_file}")
        return False

    # Get the language_model (TextModel)
    lang_model = model.language_model if hasattr(model, "language_model") else model

    # Get model args
    config_file = mtp_path / "config.json"
    if config_file.exists():
        config = json.load(open(config_file))
        text_config = config.get("text_config", config)
        mtp_num_layers = text_config.get("mtp_num_hidden_layers", 1)
    else:
        mtp_num_layers = 1

    # Load quantization config
    quant_config = config.get("quantization", {})
    quant_bits = quant_config.get("bits", 4)
    quant_group = quant_config.get("group_size", 64)

    # Create MTP module
    mtp_module = MTPModule(lang_model.args, num_layers=mtp_num_layers)

    # Quantize linear layers to match the model's quantization
    if quant_config:
        def quant_predicate(path, module):
            # Quantize all linear layers except the fc fusion layer
            if isinstance(module, nn.Linear) and "fc" not in path:
                return True
            return False

        nn.quantize(mtp_module, group_size=quant_group, bits=quant_bits,
                     class_predicate=quant_predicate)

    # Load MTP weights from safetensors
    all_weights = mx.load(str(safetensors_file))

    # Extract MTP-specific weights and remap keys
    mtp_weights = {}
    prefix = "language_model.mtp."
    for key, value in all_weights.items():
        if key.startswith(prefix):
            # Remove the prefix to get the MTP module's local key
            local_key = key[len(prefix):]
            mtp_weights[local_key] = value

    if not mtp_weights:
        print("No MTP weights found in safetensors file")
        return False

    # Load weights into the MTP module (strict=False for quantized weights)
    mtp_module.load_weights(list(mtp_weights.items()), strict=False)
    mx.eval(mtp_module.parameters())

    # Attach MTP module to the language model
    lang_model.mtp = mtp_module

    # Patch methods on the instance
    lang_model.mtp_forward = types.MethodType(_mtp_forward, lang_model)
    lang_model.make_mtp_cache = types.MethodType(_make_mtp_cache, lang_model)

    # Patch __call__ at the CLASS level (nn.Module dispatches via class, not instance)
    cls = type(lang_model)
    if not hasattr(cls, '_original_call_unpatched'):
        cls._original_call_unpatched = cls.__call__
        cls.__call__ = _patched_call

    print(f"MTP patched: {len(mtp_weights)} weights loaded, {mtp_num_layers} MTP layer(s)")
    return True
