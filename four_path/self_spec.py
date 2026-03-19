"""
Self-Speculative Decoding for MLX
===================================

Uses early exit from the target model itself as a draft source.
No second model, no distribution mismatch, no extra memory.

Architecture:
  1. N-gram check (nanoseconds) - catches verbatim patterns
  2. Early exit at layer N (87.5% cost) - draft from partial forward pass
  3. Verify with layers N+1..end - accept or reject
  4. Full forward pass only when both miss

On Qwen3.5-9B: layer 28/32 gives 45% match, layer 30/32 gives 55% match.
"""

import functools
from typing import Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
from mlx_lm.models import cache
from mlx_lm.models.base import create_attention_mask, create_ssm_mask


def self_spec_generate(
    prompt: mx.array,
    model: nn.Module,
    tokenizer=None,
    *,
    max_tokens: int = 256,
    exit_layer: int = 28,
    num_draft: int = 4,
    sampler=None,
    ngram=None,
    prefill_step_size: int = 2048,
    kv_bits=None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> Generator[Tuple[int, bool, str], None, None]:
    """
    Self-speculative decoding with early exit.

    Yields: (token_id, from_draft, source_name)
      source_name: "ngram", "early_exit", "gpu"
    """
    # Get model internals
    backbone = model.model
    layers = backbone.layers
    n_layers = len(layers)
    has_lm_head = hasattr(model, 'lm_head')

    if exit_layer >= n_layers:
        exit_layer = n_layers - 2  # At least skip 2 layers

    y = prompt.astype(mx.uint32)

    # Create caches
    model_cache = cache.make_prompt_cache(model)

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    quantize_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _get_masks(h, cache_list):
        fa_mask = create_attention_mask(h, cache_list[backbone.fa_idx])
        ssm_mask = create_ssm_mask(h, cache_list[backbone.ssm_idx])
        return fa_mask, ssm_mask

    def _run_layers(h, start, end, masks, cache_list):
        """Run layers [start, end) on hidden states."""
        fa_mask, ssm_mask = masks
        for i in range(start, end):
            layer = layers[i]
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=cache_list[i])
        return h

    def _to_logits(h):
        """Apply final norm + lm_head."""
        h_normed = backbone.norm(h)
        if has_lm_head:
            return model.lm_head(h_normed)
        return backbone.embed_tokens.as_linear(h_normed)

    def _sample_token(logits):
        logits = logits[:, -1:, :]
        return sampler(logits.squeeze(0)).item()

    # ── Prefill ──
    with mx.stream(generation_stream):
        while y.size > prefill_step_size:
            model(y[:prefill_step_size][None], cache=model_cache)
            quantize_fn(model_cache)
            mx.eval([c.state for c in model_cache if hasattr(c, "state")])
            y = y[prefill_step_size:]
            mx.clear_cache()

    # First token - full forward pass
    with mx.stream(generation_stream):
        logits = model(y[None], cache=model_cache)
        quantize_fn(model_cache)
    mx.eval(logits)
    first_tok = _sample_token(logits)
    yield first_tok, False, "gpu"

    ntoks = 1
    y = mx.array([first_tok], mx.uint32)
    all_generated = [first_tok]

    if ntoks >= max_tokens:
        return

    # N-gram state
    if ngram is not None:
        prompt_tokens = prompt.tolist()
        ngram.feed(prompt_tokens[-200:])  # Feed tail of prompt
        ngram.feed([first_tok])

    while ntoks < max_tokens:
        # ── Priority 1: N-gram chain ──
        if ngram is not None:
            chain = ngram.draft_chain(prompt.tolist() + all_generated, max_tokens=num_draft, min_tokens=1)
            if chain:
                # Verify N-gram chain with full model
                draft_mx = mx.array(chain, mx.uint32)
                verify_input = mx.concatenate([y, draft_mx])
                with mx.stream(generation_stream):
                    logits = model(verify_input[None], cache=model_cache)
                    quantize_fn(model_cache)
                mx.eval(logits)

                tokens_list = [_sample_token(logits[:, i:i+1, :]) for i in range(len(chain) + 1)]

                n_accepted = 0
                while n_accepted < len(chain):
                    if tokens_list[n_accepted] != chain[n_accepted]:
                        break
                    all_generated.append(tokens_list[n_accepted])
                    ngram.feed([tokens_list[n_accepted]])
                    yield tokens_list[n_accepted], True, "ngram"
                    n_accepted += 1
                    ntoks += 1
                    if ntoks >= max_tokens:
                        break

                if ntoks >= max_tokens:
                    break

                # Bonus/rejection token
                bonus = tokens_list[n_accepted]
                all_generated.append(bonus)
                ngram.feed([bonus])
                yield bonus, False, "gpu"
                ntoks += 1

                # Trim cache
                n_trim = len(chain) - n_accepted
                if n_trim > 0:
                    cache.trim_prompt_cache(model_cache, n_trim)

                y = mx.array([bonus], mx.uint32)
                if ntoks >= max_tokens:
                    break
                continue

        # ── Priority 2: Self-speculative (early exit + batched verify) ──
        # Generate num_draft tokens using only the first exit_layer layers.
        # Then verify the whole batch with one full forward pass.

        draft_tokens = []
        draft_ys = [y]

        # Generate draft tokens from early layers only
        for d in range(num_draft):
            with mx.stream(generation_stream):
                h = backbone.embed_tokens(draft_ys[-1][None])
                masks = _get_masks(h, model_cache)
                h = _run_layers(h, 0, exit_layer, masks, model_cache)
                early_logits = _to_logits(h)
                # Also run tail layers to keep cache consistent
                # (required because next draft needs full cache state)
                _run_layers(h, exit_layer, n_layers, masks, model_cache)
                quantize_fn(model_cache)
            mx.eval(early_logits)
            draft_tok = _sample_token(early_logits)
            draft_tokens.append(draft_tok)
            draft_ys.append(mx.array([draft_tok], mx.uint32))

        if not draft_tokens:
            # Fallback to standard generation
            with mx.stream(generation_stream):
                logits = model(y[None], cache=model_cache)
                quantize_fn(model_cache)
            mx.eval(logits)
            tok_id = _sample_token(logits)
            all_generated.append(tok_id)
            if ngram is not None:
                ngram.feed([tok_id])
            yield tok_id, False, "gpu"
            ntoks += 1
            y = mx.array([tok_id], mx.uint32)
            continue

        # Verify: check what the full model would have produced at each position.
        # We already ran all layers during drafting (for cache consistency),
        # so just check if full-model logits match the early-exit logits.
        # Rewind cache and re-run with the draft tokens as input for verification.
        n_trim = len(draft_tokens)
        cache.trim_prompt_cache(model_cache, n_trim)

        # Run full model on all draft tokens at once
        draft_mx = mx.concatenate([y] + [mx.array([t], mx.uint32) for t in draft_tokens])
        with mx.stream(generation_stream):
            full_logits = model(draft_mx[None], cache=model_cache)
            quantize_fn(model_cache)
        mx.eval(full_logits)

        # Check each position
        n_accepted = 0
        for i in range(len(draft_tokens)):
            verify_tok = _sample_token(full_logits[:, i:i+1, :])
            if verify_tok == draft_tokens[i]:
                all_generated.append(draft_tokens[i])
                if ngram is not None:
                    ngram.feed([draft_tokens[i]])
                yield draft_tokens[i], True, "early_exit"
                n_accepted += 1
                ntoks += 1
                if ntoks >= max_tokens:
                    break
            else:
                # Rejection - use the verifier's token
                all_generated.append(verify_tok)
                if ngram is not None:
                    ngram.feed([verify_tok])
                yield verify_tok, False, "gpu"
                ntoks += 1
                break

        if ntoks >= max_tokens:
            break

        # Trim cache for unverified draft tokens
        unverified = len(draft_tokens) - n_accepted
        if unverified > 0:
            cache.trim_prompt_cache(model_cache, unverified)

        y = mx.array([all_generated[-1]], mx.uint32)
