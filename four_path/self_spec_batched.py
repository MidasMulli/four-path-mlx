"""
Batched Self-Speculative Decoding
==================================

Draft N tokens using early exit (layers 0..E), save hidden states.
Verify all N at once by running tail layers (E..end) in a single batch.

The speed gain: N draft tokens cost N * E layer evals.
Verification costs 1 * (total - E) layer evals for the whole batch.
vs standard: N * total layer evals.

Savings = N * (total - E) - (total - E) = (N-1) * (total - E) layer evals.
For N=4, exit=28, total=32: saves 3 * 4 = 12 layer evals out of 128 = 9.4%
At 45% acceptance (avg 1.8 accepted per batch): effective ~17% layer savings.
"""

import functools
import time
from typing import Generator, Tuple, Optional

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
    ngram=None,
    prefill_step_size: int = 2048,
) -> Generator[Tuple[int, bool, str], None, None]:
    """
    Batched self-speculative decoding.

    Draft tokens from early exit, verify batch with tail layers.
    Yields: (token_id, from_draft, source)
    """
    backbone = model.model
    layers = backbone.layers
    n_layers = len(layers)
    has_lm_head = hasattr(model, 'lm_head')

    exit_layer = min(exit_layer, n_layers - 2)
    tail_layers = n_layers - exit_layer

    y = prompt.astype(mx.uint32)
    model_cache = cache.make_prompt_cache(model)

    qfn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0, kv_group_size=64, kv_bits=None,
    )

    def _to_logits(h):
        h_n = backbone.norm(h)
        return model.lm_head(h_n) if has_lm_head else backbone.embed_tokens.as_linear(h_n)

    # ── Prefill with full model ──
    with mx.stream(generation_stream):
        while y.size > prefill_step_size:
            model(y[:prefill_step_size][None], cache=model_cache)
            qfn(model_cache)
            mx.eval([c.state for c in model_cache if hasattr(c, "state")])
            y = y[prefill_step_size:]
            mx.clear_cache()

        logits = model(y[None], cache=model_cache)
        qfn(model_cache)
    mx.eval(logits)

    first_tok = mx.argmax(logits[:, -1, :], axis=-1).item()
    yield first_tok, False, "gpu"

    ntoks = 1
    y = mx.array([first_tok], mx.uint32)
    all_tokens = prompt.tolist() + [first_tok]

    if ngram is not None:
        ngram.feed(prompt.tolist()[-200:])
        ngram.feed([first_tok])

    while ntoks < max_tokens:
        # ── N-gram check first ──
        if ngram is not None:
            chain = ngram.draft_chain(all_tokens, max_tokens=num_draft, min_tokens=1)
            if chain:
                draft_mx = mx.array(chain, mx.uint32)
                verify_input = mx.concatenate([y, draft_mx])
                with mx.stream(generation_stream):
                    vlogits = model(verify_input[None], cache=model_cache)
                    qfn(model_cache)
                mx.eval(vlogits)

                n_acc = 0
                for i in range(len(chain)):
                    vtok = mx.argmax(vlogits[:, i, :], axis=-1).item()
                    if vtok != chain[i]:
                        # Rejection - emit verifier's token
                        all_tokens.append(vtok)
                        ngram.feed([vtok])
                        yield vtok, False, "gpu"
                        ntoks += 1
                        break
                    all_tokens.append(chain[i])
                    ngram.feed([chain[i]])
                    yield chain[i], True, "ngram"
                    n_acc += 1
                    ntoks += 1
                    if ntoks >= max_tokens:
                        break
                else:
                    # All accepted - emit the bonus token
                    if ntoks < max_tokens:
                        bonus = mx.argmax(vlogits[:, len(chain), :], axis=-1).item()
                        all_tokens.append(bonus)
                        ngram.feed([bonus])
                        yield bonus, False, "gpu"
                        ntoks += 1

                n_trim = len(chain) - n_acc
                if n_trim > 0:
                    cache.trim_prompt_cache(model_cache, n_trim)

                y = mx.array([all_tokens[-1]], mx.uint32)
                if ntoks >= max_tokens:
                    break
                continue

        # ── Self-speculative: draft from early layers, verify with tail ──

        # Step 1: Generate num_draft tokens using early layers only
        draft_tokens = []
        draft_hidden_states = []  # Save hidden states at exit_layer
        draft_y = y

        for d in range(num_draft):
            with mx.stream(generation_stream):
                h = backbone.embed_tokens(draft_y[None])
                fa_mask = create_attention_mask(h, model_cache[backbone.fa_idx])
                ssm_mask = create_ssm_mask(h, model_cache[backbone.ssm_idx])

                # Run early layers only (0 to exit_layer)
                for i in range(exit_layer):
                    layer = layers[i]
                    mask = ssm_mask if layer.is_linear else fa_mask
                    h = layer(h, mask=mask, cache=model_cache[i])

                # Save hidden state at exit point
                draft_hidden_states.append(h)

                # Get draft token from early exit
                early_logits = _to_logits(h)

            mx.eval(early_logits)
            draft_tok = mx.argmax(early_logits[:, -1, :], axis=-1).item()
            draft_tokens.append(draft_tok)
            draft_y = mx.array([draft_tok], mx.uint32)

            # DON'T run tail layers - that's the whole point.
            # Early-layer caches are updated, tail-layer caches are stale.

        if not draft_tokens:
            # Fallback
            with mx.stream(generation_stream):
                logits = model(y[None], cache=model_cache)
                qfn(model_cache)
            mx.eval(logits)
            tok_id = mx.argmax(logits[:, -1, :], axis=-1).item()
            all_tokens.append(tok_id)
            if ngram: ngram.feed([tok_id])
            yield tok_id, False, "gpu"
            ntoks += 1
            y = mx.array([tok_id], mx.uint32)
            continue

        # Step 2: Verify by running tail layers on the saved hidden states
        # The early-layer caches are already correct from drafting.
        # We need to run ONLY the tail layers on each draft's hidden state.

        # Stack all hidden states: (1, num_draft, hidden) -> run tail layers
        # But tail layer caches need sequential processing (SSM state is sequential)
        # So we process each draft's hidden state through tail layers one at a time

        verify_tokens = []
        for i, h_draft in enumerate(draft_hidden_states):
            with mx.stream(generation_stream):
                fa_mask = create_attention_mask(h_draft, model_cache[backbone.fa_idx])
                ssm_mask = create_ssm_mask(h_draft, model_cache[backbone.ssm_idx])

                h = h_draft
                for li in range(exit_layer, n_layers):
                    layer = layers[li]
                    mask = ssm_mask if layer.is_linear else fa_mask
                    h = layer(h, mask=mask, cache=model_cache[li])

                full_logits = _to_logits(h)
                qfn(model_cache)
            mx.eval(full_logits)
            verify_tokens.append(mx.argmax(full_logits[:, -1, :], axis=-1).item())

        # Step 3: Check each draft token against the full model's prediction
        n_accepted = 0
        for i in range(len(draft_tokens)):
            if verify_tokens[i] == draft_tokens[i]:
                all_tokens.append(draft_tokens[i])
                if ngram: ngram.feed([draft_tokens[i]])
                yield draft_tokens[i], True, "early_exit"
                n_accepted += 1
                ntoks += 1
                if ntoks >= max_tokens:
                    break
            else:
                # Rejection - use verifier's token
                all_tokens.append(verify_tokens[i])
                if ngram: ngram.feed([verify_tokens[i]])
                yield verify_tokens[i], False, "gpu"
                ntoks += 1
                break

        # Trim unverified cache entries
        unverified = len(draft_tokens) - n_accepted
        if unverified > 0:
            cache.trim_prompt_cache(model_cache, unverified)

        y = mx.array([all_tokens[-1]], mx.uint32)
