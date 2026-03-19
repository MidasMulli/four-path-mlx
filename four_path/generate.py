"""
Four-Path Speculative Decoding on Apple Silicon
================================================

CPU  (N-gram hash)   → pattern-based draft chains       (nanoseconds)
ANE  (1.7B CoreML)   → neural lookahead draft tokens    (parallel, pre-generated)
MTP  (GPU head)      → hidden-state prediction          (free with forward pass)
GPU  (9B backbone)   → verification + generation        (the real model)

Adaptive routing per round:
  1. N-gram has chain?  → batch verify (fast, many tokens)
  2. ANE has tokens?    → batch verify ANE drafts
  3. Both miss?         → MTP single-token draft (always available)

MTP is the backstop — it catches tokens when neither pattern matching nor
neural lookahead fires. It uses the backbone's own hidden states, so it
has the highest per-token accuracy of any draft source.

The four sources catch fundamentally different patterns:
  N-gram:  verbatim repetition ("notwithstanding anything to the contrary")
  ANE:     semantic prediction from 1.7B model context
  MTP:     model's own next-token-plus-one from hidden states
  GPU:     novel tokens only a 9B model can generate
"""

import json
import os
import sys
import struct
import socket
import time
import threading
import functools
from typing import Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
from mlx_lm.models import cache

# Imports are package-relative
from four_path.ngram import NgramPredictor, EMPTY

# Import ANE lookahead from three_path
from four_path.three_path import ANELookahead, ane_generate_async


class FourPathDrafter:
    """
    Manages draft tokens from four sources: N-gram, ANE, MTP, and GPU.
    Adaptively routes to the best available source per round.
    """

    def __init__(self, ngram_n: int = 8, table_size: int = 4 * 1024 * 1024):
        self.ngram = NgramPredictor(n=ngram_n, table_size=table_size)
        self.ane_lookahead: Optional[ANELookahead] = None
        self.all_tokens: List[int] = []
        self.ane_position: int = 0

        # Stats per source
        self.stats = {
            "ngram": {"drafted": 0, "accepted": 0, "rounds": 0},
            "ane":   {"drafted": 0, "accepted": 0, "rounds": 0},
            "mtp":   {"drafted": 0, "accepted": 0, "rounds": 0},
            "gpu":   {"tokens": 0},
        }
        self.total_rounds = 0

    def feed_prompt(self, tokens: List[int]):
        self.all_tokens = list(tokens)
        self.ngram.feed(tokens)

    def set_ane_lookahead(self, lookahead: ANELookahead):
        self.ane_lookahead = lookahead

    def add_token(self, token: int):
        self.all_tokens.append(token)
        n = self.ngram.n
        if len(self.all_tokens) >= n + 1:
            self.ngram.feed(self.all_tokens[-(n + 1):])

    def get_ngram_chain(self, num_tokens: int) -> List[int]:
        return self.ngram.draft_chain(self.all_tokens, max_tokens=num_tokens, min_tokens=1)

    def get_ane_tokens(self, num_tokens: int, tokenizer=None) -> List[int]:
        if not self.ane_lookahead:
            return []
        if not self.ane_lookahead.is_ready():
            self.ane_lookahead.wait(timeout=0.01)
        if not self.ane_lookahead.is_ready() or self.ane_lookahead.error:
            return []
        ane_tokens = self.ane_lookahead.tokenize(tokenizer)
        result = []
        for i in range(num_tokens):
            pos = self.ane_position + i
            if pos < len(ane_tokens):
                result.append(ane_tokens[pos])
            else:
                break
        return result

    def record(self, source: str, drafted: int, accepted: int):
        if source in self.stats:
            self.stats[source]["drafted"] += drafted
            self.stats[source]["accepted"] += accepted
            self.stats[source]["rounds"] += 1

    def record_gpu_token(self):
        self.stats["gpu"]["tokens"] += 1

    def advance_ane(self, n: int):
        self.ane_position += n

    def summary(self) -> dict:
        total_drafted = sum(s.get("drafted", 0) for s in self.stats.values())
        total_accepted = sum(s.get("accepted", 0) for s in self.stats.values())
        return {
            "total_rounds": self.total_rounds,
            "total_drafted": total_drafted,
            "total_accepted": total_accepted,
            **{f"{k}_drafted": v.get("drafted", 0) for k, v in self.stats.items()},
            **{f"{k}_accepted": v.get("accepted", 0) for k, v in self.stats.items()},
            **{f"{k}_rounds": v.get("rounds", 0) for k, v in self.stats.items()},
            "gpu_only": self.stats["gpu"]["tokens"],
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n  Four-Path Drafter Summary")
        print(f"  ────────────────────────")
        print(f"  Rounds: {s['total_rounds']}")
        for src in ["ngram", "ane", "mtp"]:
            d = s.get(f"{src}_drafted", 0)
            a = s.get(f"{src}_accepted", 0)
            r = a / d if d else 0
            print(f"  {src:>6}: {d} drafted, {a} accepted ({r:.0%}), {s.get(f'{src}_rounds', 0)} rounds")
        print(f"     gpu: {s['gpu_only']} tokens (no draft available)")


def four_path_generate_step(
    prompt: mx.array,
    model: nn.Module,
    drafter: FourPathDrafter,
    tokenizer=None,
    *,
    num_draft_tokens: int = 32,
    max_tokens: int = 256,
    sampler=None,
    prompt_cache=None,
    prefill_step_size: int = 2048,
    kv_bits=None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> Generator[Tuple[int, mx.array, bool, str], None, None]:
    """
    Four-path speculative generation.

    Adaptive routing:
      1. N-gram chain available → batch verify
      2. ANE tokens available   → batch verify
      3. Neither available      → MTP single-token draft
      4. Nothing works          → standard GPU generation

    Yields: (token_id, logprobs, from_draft, source)
    """
    y = prompt.astype(mx.uint32)
    has_mtp = hasattr(model, 'mtp_forward')

    # Create caches
    if prompt_cache is None:
        model_cache = cache.make_prompt_cache(model)
        mtp_cache = model.make_mtp_cache() if has_mtp else None
    else:
        n_main = len(model.layers)
        model_cache = prompt_cache[:n_main]
        mtp_cache = prompt_cache[n_main:] if has_mtp else None

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _sample(logits):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tok = sampler(logprobs)
        return tok, logprobs

    # ── Standard forward (for N-gram/ANE batch verification) ──
    def _step_standard(y, n_predict=1):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=model_cache)
            logits = logits[:, -n_predict:, :]
            quantize_cache_fn(model_cache)
            return _sample(logits.squeeze(0))

    # ── MTP forward (returns hidden states for MTP head) ──
    def _step_mtp_backbone(y, n_predict=1, n_confirmed=0):
        with mx.stream(generation_stream):
            logits, hidden = model(
                y[None], cache=model_cache, return_hidden=True, n_confirmed=n_confirmed
            )
            logits = logits[:, -n_predict:, :]
            quantize_cache_fn(model_cache)
            tok, lp = _sample(logits.squeeze(0))
            return tok, lp, hidden

    def _step_mtp_head(hidden_last, main_tok):
        next_ids = main_tok.reshape(1, 1)
        with mx.stream(generation_stream):
            mtp_logits = model.mtp_forward(hidden_last, next_ids, mtp_cache)
            quantize_cache_fn(mtp_cache)
            mtp_logits = mtp_logits[:, -1, :].squeeze(0)
            return _sample(mtp_logits)

    # ── Prefill ──
    with mx.stream(generation_stream):
        while y.size > prefill_step_size:
            model(y[:prefill_step_size][None], cache=model_cache)
            quantize_cache_fn(model_cache)
            mx.eval([c.state for c in model_cache if hasattr(c, "state")])
            y = y[prefill_step_size:]
            mx.clear_cache()

    # Feed prompt to N-gram
    prompt_tokens = prompt.tolist()
    drafter.feed_prompt(prompt_tokens)

    # First token from GPU
    tokens, logprobs = _step_standard(y)
    mx.eval(tokens, logprobs)
    first_token = tokens.item() if tokens.ndim == 0 else tokens.tolist()[-1]
    drafter.add_token(first_token)

    ntoks = 1
    y = mx.array([first_token], mx.uint32)
    yield first_token, logprobs if logprobs.ndim == 1 else logprobs[-1], False, "gpu"
    drafter.record_gpu_token()

    if ntoks >= max_tokens:
        return

    # MTP state
    mtp_draft_tok = None
    mtp_draft_lp = None

    while True:
        drafter.total_rounds += 1
        num_draft = min(max_tokens - ntoks, num_draft_tokens)

        # ── Priority 1: N-gram chain ──
        ngram_chain = drafter.get_ngram_chain(num_draft)

        if ngram_chain:
            # Batch verify N-gram chain
            draft_mx = mx.array(ngram_chain, mx.uint32)
            verify_input = mx.concatenate([y, draft_mx])
            tokens, logprobs = _step_standard(verify_input, len(ngram_chain) + 1)
            mx.eval(tokens, logprobs)
            tokens_list = tokens.tolist()

            n_accepted = 0
            while n_accepted < len(ngram_chain):
                if tokens_list[n_accepted] != ngram_chain[n_accepted]:
                    break
                drafter.add_token(tokens_list[n_accepted])
                n_accepted += 1
                ntoks += 1
                lp = logprobs[n_accepted - 1] if logprobs.ndim > 1 else logprobs
                yield tokens_list[n_accepted - 1], lp, True, "ngram"
                if ntoks >= max_tokens:
                    break

            drafter.record("ngram", len(ngram_chain), n_accepted)

            if ntoks >= max_tokens:
                break

            # Emit rejection/bonus token
            reject_tok = tokens_list[n_accepted]
            reject_lp = logprobs[n_accepted] if logprobs.ndim > 1 else logprobs
            drafter.add_token(reject_tok)
            drafter.record_gpu_token()
            ntoks += 1
            yield reject_tok, reject_lp, False, "gpu"

            # Rewind cache
            n_to_trim = len(ngram_chain) - n_accepted
            if n_to_trim > 0:
                cache.trim_prompt_cache(model_cache, n_to_trim)

            y = mx.array([reject_tok], mx.uint32)
            mtp_draft_tok = None  # invalidate any MTP draft
            if ntoks >= max_tokens:
                break
            continue

        # ── Priority 2: ANE lookahead tokens ──
        ane_tokens = drafter.get_ane_tokens(num_draft, tokenizer=tokenizer)

        if ane_tokens:
            draft_mx = mx.array(ane_tokens, mx.uint32)
            verify_input = mx.concatenate([y, draft_mx])
            tokens, logprobs = _step_standard(verify_input, len(ane_tokens) + 1)
            mx.eval(tokens, logprobs)
            tokens_list = tokens.tolist()

            n_accepted = 0
            while n_accepted < len(ane_tokens):
                if tokens_list[n_accepted] != ane_tokens[n_accepted]:
                    break
                drafter.add_token(tokens_list[n_accepted])
                n_accepted += 1
                ntoks += 1
                lp = logprobs[n_accepted - 1] if logprobs.ndim > 1 else logprobs
                yield tokens_list[n_accepted - 1], lp, True, "ane"
                if ntoks >= max_tokens:
                    break

            drafter.record("ane", len(ane_tokens), n_accepted)
            drafter.advance_ane(n_accepted)

            if ntoks >= max_tokens:
                break

            reject_tok = tokens_list[n_accepted]
            reject_lp = logprobs[n_accepted] if logprobs.ndim > 1 else logprobs
            drafter.add_token(reject_tok)
            drafter.record_gpu_token()
            ntoks += 1
            yield reject_tok, reject_lp, False, "gpu"

            n_to_trim = len(ane_tokens) - n_accepted
            if n_to_trim > 0:
                cache.trim_prompt_cache(model_cache, n_to_trim)

            y = mx.array([reject_tok], mx.uint32)
            mtp_draft_tok = None
            if ntoks >= max_tokens:
                break
            continue

        # ── Priority 3: MTP single-token draft ──
        if has_mtp:
            if mtp_draft_tok is not None:
                # Verify pending MTP draft
                y_with_draft = mx.concatenate(
                    [y, mx.array([mtp_draft_tok.item()], mx.uint32)]
                )
                toks, lps, hidden = _step_mtp_backbone(
                    y_with_draft, n_predict=2, n_confirmed=1
                )
                mx.eval(toks, mtp_draft_tok)

                verify_pred = toks[0]
                bonus_tok = toks[1]

                if verify_pred.item() == mtp_draft_tok.item():
                    # Draft accepted — clear rollback
                    for c in model_cache:
                        if hasattr(c, "rollback_state") and c.rollback_state is not None:
                            c.rollback_state = None

                    drafter.add_token(mtp_draft_tok.item())
                    drafter.record("mtp", 1, 1)
                    ntoks += 1
                    yield mtp_draft_tok.item(), mtp_draft_lp, True, "mtp"
                    if ntoks >= max_tokens:
                        break

                    drafter.add_token(bonus_tok.item())
                    drafter.record_gpu_token()
                    ntoks += 1
                    yield bonus_tok.item(), lps[1], False, "gpu"
                    if ntoks >= max_tokens:
                        break

                    # Next MTP draft
                    mtp_draft_tok, mtp_draft_lp = _step_mtp_head(
                        hidden[:, 1:2, :], bonus_tok
                    )
                    mx.eval(mtp_draft_tok)
                    y = mx.array([bonus_tok.item()], mx.uint32)
                else:
                    # Rollback SSM state
                    for c in model_cache:
                        if hasattr(c, "rollback_state") and c.rollback_state is not None:
                            conv_snap, ssm_snap = c.rollback_state
                            c[0] = conv_snap
                            c[1] = ssm_snap
                            c.rollback_state = None
                        elif c.is_trimmable():
                            c.trim(1)
                    if mtp_cache:
                        cache.trim_prompt_cache(mtp_cache, 1)

                    drafter.add_token(verify_pred.item())
                    drafter.record("mtp", 1, 0)
                    drafter.record_gpu_token()
                    ntoks += 1
                    yield verify_pred.item(), lps[0], False, "gpu"
                    if ntoks >= max_tokens:
                        break

                    # New MTP draft from verified position
                    mtp_draft_tok, mtp_draft_lp = _step_mtp_head(
                        hidden[:, 0:1, :], verify_pred
                    )
                    mx.eval(mtp_draft_tok)
                    y = mx.array([verify_pred.item()], mx.uint32)
            else:
                # No MTP draft pending — generate one
                toks, lps, hidden = _step_mtp_backbone(y, n_predict=1)
                mx.eval(toks)
                main_tok = toks[0] if toks.ndim > 0 else toks

                drafter.add_token(main_tok.item())
                drafter.record_gpu_token()
                ntoks += 1
                yield main_tok.item(), lps[0] if lps.ndim > 1 else lps, False, "gpu"
                if ntoks >= max_tokens:
                    break

                # Generate MTP draft for next round
                mtp_draft_tok, mtp_draft_lp = _step_mtp_head(
                    hidden[:, -1:, :], main_tok
                )
                mx.eval(mtp_draft_tok)
                y = mx.array([main_tok.item()], mx.uint32)

            if ntoks >= max_tokens:
                break
            continue

        # ── Priority 4: Standard GPU (no draft sources available) ──
        tokens, logprobs = _step_standard(y)
        mx.eval(tokens, logprobs)
        tok = tokens.item() if tokens.ndim == 0 else tokens.tolist()[-1]
        lp = logprobs if logprobs.ndim == 1 else logprobs[-1]
        drafter.add_token(tok)
        drafter.record_gpu_token()
        ntoks += 1
        yield tok, lp, False, "gpu"
        y = mx.array([tok], mx.uint32)
        mtp_draft_tok = None
        if ntoks >= max_tokens:
            break
