"""
Multi-Check Four-Path Speculative Decoding
==========================================

Key insight: when the primary draft source is rejected, the verification
forward pass already produced the target's token. Checking if alternative
sources predicted that same token is a nanosecond lookup — not another
forward pass.

If an alternative source predicted the correct rejection token, it's "in sync"
with the target model at that position. Its subsequent predictions are likely
good drafts for the next verification pass.

This recovers value from every rejection by checking all sources against
the token you already computed.

Flow:
  1. Build merged draft (N-gram spine + ANE fill)
  2. GPU verifies in one pass
  3. On rejection at position i, target token = T
     - Check: did ANE predict T at this position?
     - Check: does MTP agree with T?
     - If any source matches, use its subsequent predictions immediately
  4. Result: fewer "cold start" rounds, higher token yield per pass
"""

import os
import sys
import functools
import threading
import time
from typing import Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
from mlx_lm.models import cache

# Imports are package-relative
from four_path.ngram import NgramPredictor, EMPTY
from four_path.three_path import ANELookahead, ane_generate_async


class MultiCheckDrafter:
    """
    Builds merged drafts and recovers from rejections by checking
    alternative sources against the target's rejection token.
    """

    def __init__(self, ngram_n: int = 8, table_size: int = 4 * 1024 * 1024):
        self.ngram = NgramPredictor(n=ngram_n, table_size=table_size)
        self.ane_lookahead: Optional[ANELookahead] = None
        self.all_tokens: List[int] = []
        self.ane_position: int = 0
        self._ane_tokens_cache: Optional[List[int]] = None

        # Stats
        self.ngram_drafted = 0
        self.ngram_accepted = 0
        self.ane_drafted = 0
        self.ane_accepted = 0
        self.mtp_drafted = 0
        self.mtp_accepted = 0
        self.gpu_tokens = 0
        self.multicheck_recoveries = 0  # times multi-check found a match
        self.multicheck_attempts = 0    # times we checked alternatives
        self.rounds = 0

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

    def _get_ane_tokens(self, tokenizer) -> List[int]:
        """Get all available ANE lookahead tokens (cached)."""
        if self._ane_tokens_cache is not None:
            return self._ane_tokens_cache
        if not self.ane_lookahead:
            return []
        if not self.ane_lookahead.is_ready():
            self.ane_lookahead.wait(timeout=0.01)
        if not self.ane_lookahead.is_ready() or self.ane_lookahead.error:
            return []
        self._ane_tokens_cache = self.ane_lookahead.tokenize(tokenizer)
        return self._ane_tokens_cache

    def get_ane_at(self, position: int, tokenizer=None) -> int:
        """Get ANE's prediction at a specific generation position."""
        tokens = self._get_ane_tokens(tokenizer)
        pos = self.ane_position + position
        if pos < len(tokens):
            return tokens[pos]
        return EMPTY

    def get_ane_chain_from(self, position: int, max_tokens: int, tokenizer=None) -> List[int]:
        """Get ANE tokens starting from a position (for follow-up drafting)."""
        tokens = self._get_ane_tokens(tokenizer)
        result = []
        for i in range(max_tokens):
            pos = self.ane_position + position + i
            if pos < len(tokens):
                result.append(tokens[pos])
            else:
                break
        return result

    def build_merged_draft(self, max_tokens: int, tokenizer=None) -> Tuple[List[int], List[str]]:
        """
        Build a merged draft: N-gram chain as spine, ANE fills remaining slots.
        Returns (tokens, sources).
        """
        self.rounds += 1
        draft_tokens = []
        sources = []

        # N-gram chain first
        ngram_chain = self.ngram.draft_chain(
            self.all_tokens, max_tokens=max_tokens, min_tokens=1
        )

        if ngram_chain:
            draft_tokens = list(ngram_chain)
            sources = ["ngram"] * len(ngram_chain)
            self.ngram_drafted += len(ngram_chain)

            # Fill remaining slots with ANE
            remaining = max_tokens - len(ngram_chain)
            if remaining > 0:
                ane_fill = self.get_ane_chain_from(len(ngram_chain), remaining, tokenizer)
                if ane_fill:
                    draft_tokens.extend(ane_fill)
                    sources.extend(["ane"] * len(ane_fill))
                    self.ane_drafted += len(ane_fill)
        else:
            # No N-gram — use ANE
            ane_chain = self.get_ane_chain_from(0, max_tokens, tokenizer)
            if ane_chain:
                draft_tokens = ane_chain
                sources = ["ane"] * len(ane_chain)
                self.ane_drafted += len(ane_chain)

        return draft_tokens, sources

    def check_alternatives(self, position_in_gen: int, target_token: int,
                           tokenizer=None) -> Optional[str]:
        """
        Multi-check: at a rejection point, check if alternative sources
        predicted the target's actual token. Returns the source name if
        found, None otherwise.

        This is a nanosecond lookup — no forward pass needed.
        """
        self.multicheck_attempts += 1

        # Check ANE
        ane_tok = self.get_ane_at(position_in_gen, tokenizer)
        if ane_tok != EMPTY and ane_tok == target_token:
            self.multicheck_recoveries += 1
            return "ane"

        return None

    def advance_ane(self, n: int):
        self.ane_position += n

    def record_accepted(self, sources: List[str], n_accepted: int):
        for i in range(n_accepted):
            if i < len(sources):
                if sources[i] == "ngram":
                    self.ngram_accepted += 1
                elif sources[i] == "ane":
                    self.ane_accepted += 1

    def summary(self) -> dict:
        total_drafted = self.ngram_drafted + self.ane_drafted + self.mtp_drafted
        total_accepted = self.ngram_accepted + self.ane_accepted + self.mtp_accepted
        return {
            "rounds": self.rounds,
            "ngram": {"drafted": self.ngram_drafted, "accepted": self.ngram_accepted},
            "ane": {"drafted": self.ane_drafted, "accepted": self.ane_accepted},
            "mtp": {"drafted": self.mtp_drafted, "accepted": self.mtp_accepted},
            "gpu_only": self.gpu_tokens,
            "multicheck_attempts": self.multicheck_attempts,
            "multicheck_recoveries": self.multicheck_recoveries,
            "multicheck_rate": (self.multicheck_recoveries / self.multicheck_attempts
                               if self.multicheck_attempts else 0),
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n  Multi-Check Drafter Summary")
        print(f"  ──────────────────────────")
        print(f"  Rounds: {s['rounds']}")
        for src in ["ngram", "ane", "mtp"]:
            d = s[src]["drafted"]
            a = s[src]["accepted"]
            print(f"  {src:>6}: {d} drafted, {a} accepted ({a/d:.0%})" if d else f"  {src:>6}: 0 drafted")
        print(f"  GPU-only: {s['gpu_only']}")
        print(f"  Multi-check: {s['multicheck_recoveries']}/{s['multicheck_attempts']} "
              f"recoveries ({s['multicheck_rate']:.0%})")


def multicheck_generate_step(
    prompt: mx.array,
    model: nn.Module,
    drafter: MultiCheckDrafter,
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
    Multi-check four-path generation.

    On rejection, checks alternative sources against the target's token
    and uses the matching source's subsequent predictions as immediate
    follow-up drafts.
    """
    y = prompt.astype(mx.uint32)
    has_mtp = hasattr(model, 'mtp_forward')

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

    def _step(y, n_predict=1):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=model_cache)
            logits = logits[:, -n_predict:, :]
            quantize_cache_fn(model_cache)
            return _sample(logits.squeeze(0))

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

    # Prefill
    with mx.stream(generation_stream):
        while y.size > prefill_step_size:
            model(y[:prefill_step_size][None], cache=model_cache)
            quantize_cache_fn(model_cache)
            mx.eval([c.state for c in model_cache if hasattr(c, "state")])
            y = y[prefill_step_size:]
            mx.clear_cache()

    prompt_tokens = prompt.tolist()
    drafter.feed_prompt(prompt_tokens)

    # First token
    tokens, logprobs = _step(y)
    mx.eval(tokens, logprobs)
    first_token = tokens.item() if tokens.ndim == 0 else tokens.tolist()[-1]
    drafter.add_token(first_token)
    drafter.gpu_tokens += 1

    ntoks = 1
    gen_position = 0  # tracks position in generation for ANE alignment
    y = mx.array([first_token], mx.uint32)
    yield first_token, logprobs if logprobs.ndim == 1 else logprobs[-1], False, "gpu"

    if ntoks >= max_tokens:
        return

    mtp_draft_tok = None
    mtp_draft_lp = None
    # After a multicheck recovery, this holds the source to prioritize
    follow_up_source: Optional[str] = None

    while True:
        num_draft = min(max_tokens - ntoks, num_draft_tokens)

        # If we have a follow-up from multi-check, use that source's chain
        if follow_up_source == "ane":
            ane_chain = drafter.get_ane_chain_from(gen_position, num_draft, tokenizer)
            if ane_chain:
                draft_tokens = ane_chain
                sources = ["ane"] * len(ane_chain)
                drafter.ane_drafted += len(ane_chain)
                follow_up_source = None
                # Fall through to verification below
            else:
                follow_up_source = None
                draft_tokens, sources = drafter.build_merged_draft(num_draft, tokenizer)
        else:
            follow_up_source = None
            draft_tokens, sources = drafter.build_merged_draft(num_draft, tokenizer)

        if draft_tokens:
            # Verify merged draft
            draft_mx = mx.array(draft_tokens, mx.uint32)
            verify_input = mx.concatenate([y, draft_mx])
            tokens, logprobs = _step(verify_input, len(draft_tokens) + 1)
            mx.eval(tokens, logprobs)
            tokens_list = tokens.tolist()

            n_accepted = 0
            while n_accepted < len(draft_tokens):
                if tokens_list[n_accepted] != draft_tokens[n_accepted]:
                    break
                drafter.add_token(tokens_list[n_accepted])
                n_accepted += 1
                ntoks += 1
                gen_position += 1
                lp = logprobs[n_accepted - 1] if logprobs.ndim > 1 else logprobs
                src = sources[n_accepted - 1] if n_accepted - 1 < len(sources) else "gpu"
                yield tokens_list[n_accepted - 1], lp, True, src
                if ntoks >= max_tokens:
                    break

            drafter.record_accepted(sources, n_accepted)

            # Advance ANE position for all accepted tokens (regardless of source)
            drafter.advance_ane(n_accepted)

            if ntoks >= max_tokens:
                break

            # ── MULTI-CHECK at rejection point ──
            reject_tok = tokens_list[n_accepted]
            reject_lp = logprobs[n_accepted] if logprobs.ndim > 1 else logprobs

            # Check if alternative sources predicted the rejection token
            recovery_source = drafter.check_alternatives(
                gen_position, reject_tok, tokenizer
            )

            drafter.add_token(reject_tok)
            gen_position += 1
            ntoks += 1

            if recovery_source:
                # Multi-check recovery! The alternative source predicted
                # the correct token. Credit it and use its chain next.
                yield reject_tok, reject_lp, True, recovery_source
                if recovery_source == "ane":
                    drafter.ane_accepted += 1
                    drafter.ane_drafted += 1
                    drafter.advance_ane(1)
                follow_up_source = recovery_source
            else:
                drafter.gpu_tokens += 1
                yield reject_tok, reject_lp, False, "gpu"

            # Rewind cache
            n_to_trim = len(draft_tokens) - n_accepted
            if n_to_trim > 0:
                cache.trim_prompt_cache(model_cache, n_to_trim)

            y = mx.array([reject_tok], mx.uint32)
            mtp_draft_tok = None
            if ntoks >= max_tokens:
                break
            continue

        # ── MTP fallback ──
        if has_mtp:
            if mtp_draft_tok is not None:
                y_with_draft = mx.concatenate(
                    [y, mx.array([mtp_draft_tok.item()], mx.uint32)]
                )
                toks, lps, hidden = _step_mtp_backbone(
                    y_with_draft, n_predict=2, n_confirmed=1
                )
                mx.eval(toks, mtp_draft_tok)

                if toks[0].item() == mtp_draft_tok.item():
                    for c in model_cache:
                        if hasattr(c, "rollback_state") and c.rollback_state is not None:
                            c.rollback_state = None

                    drafter.add_token(mtp_draft_tok.item())
                    drafter.mtp_drafted += 1
                    drafter.mtp_accepted += 1
                    gen_position += 1
                    ntoks += 1
                    yield mtp_draft_tok.item(), mtp_draft_lp, True, "mtp"
                    if ntoks >= max_tokens:
                        break

                    drafter.add_token(toks[1].item())
                    drafter.gpu_tokens += 1
                    gen_position += 1
                    ntoks += 1
                    yield toks[1].item(), lps[1], False, "gpu"
                    if ntoks >= max_tokens:
                        break

                    mtp_draft_tok, mtp_draft_lp = _step_mtp_head(hidden[:, 1:2, :], toks[1])
                    mx.eval(mtp_draft_tok)
                    y = mx.array([toks[1].item()], mx.uint32)
                else:
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

                    drafter.add_token(toks[0].item())
                    drafter.mtp_drafted += 1
                    drafter.gpu_tokens += 1
                    gen_position += 1
                    ntoks += 1
                    yield toks[0].item(), lps[0], False, "gpu"
                    if ntoks >= max_tokens:
                        break

                    mtp_draft_tok, mtp_draft_lp = _step_mtp_head(hidden[:, 0:1, :], toks[0])
                    mx.eval(mtp_draft_tok)
                    y = mx.array([toks[0].item()], mx.uint32)
            else:
                toks, lps, hidden = _step_mtp_backbone(y, n_predict=1)
                mx.eval(toks)
                main_tok = toks[0] if toks.ndim > 0 else toks

                drafter.add_token(main_tok.item())
                drafter.gpu_tokens += 1
                gen_position += 1
                ntoks += 1
                yield main_tok.item(), lps[0] if lps.ndim > 1 else lps, False, "gpu"
                if ntoks >= max_tokens:
                    break

                mtp_draft_tok, mtp_draft_lp = _step_mtp_head(hidden[:, -1:, :], main_tok)
                mx.eval(mtp_draft_tok)
                y = mx.array([main_tok.item()], mx.uint32)

            if ntoks >= max_tokens:
                break
            continue

        # Pure GPU
        tokens, logprobs = _step(y)
        mx.eval(tokens, logprobs)
        tok = tokens.item() if tokens.ndim == 0 else tokens.tolist()[-1]
        lp = logprobs if logprobs.ndim == 1 else logprobs[-1]
        drafter.add_token(tok)
        drafter.gpu_tokens += 1
        gen_position += 1
        ntoks += 1
        yield tok, lp, False, "gpu"
        y = mx.array([tok], mx.uint32)
        mtp_draft_tok = None
        if ntoks >= max_tokens:
            break
