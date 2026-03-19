"""
Adaptive Orchestrator for Four-Path Speculative Decoding
========================================================

The static four-path cascade (N-gram → ANE → MTP → GPU) wastes forward
passes when sources have low acceptance rates. The orchestrator learns
which source is performing well and adapts in real-time.

Key behaviors:
  - Tracks rolling acceptance rate per source (window of last 10 rounds)
  - Throttles sources that fall below acceptance threshold
  - Caps ANE draft batch size based on recent acceptance (don't propose 32
    tokens if only 17% land - propose 8 instead)
  - Blends sources: N-gram chain as spine, ANE fills remaining slots
  - Tracks cost: each failed verification = 1 wasted forward pass
  - Reports efficiency metrics

The goal: maximize tokens accepted per forward pass, not just maximize
tokens proposed.
"""

import os
import sys
import time
import functools
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
from mlx_lm.models import cache

# Imports are package-relative
from four_path.ngram import NgramPredictor, EMPTY
from four_path.three_path import ANELookahead, ane_generate_async


# ── Source Performance Tracker ─────────────────────────────────

@dataclass
class SourceTracker:
    """Rolling window performance tracker for a draft source."""
    name: str
    window_size: int = 10
    min_acceptance: float = 0.15   # throttle below this
    max_batch_scale: float = 1.0   # reduced when acceptance is low

    # Rolling window: (proposed, accepted) per round
    history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Lifetime stats
    total_proposed: int = 0
    total_accepted: int = 0
    total_rounds: int = 0
    total_wasted_passes: int = 0  # rounds where acceptance was 0

    @property
    def rolling_acceptance(self) -> float:
        if not self.history:
            return 0.5  # optimistic prior
        total_p = sum(p for p, a in self.history)
        total_a = sum(a for p, a in self.history)
        return total_a / total_p if total_p else 0.5

    @property
    def lifetime_acceptance(self) -> float:
        return self.total_accepted / self.total_proposed if self.total_proposed else 0

    @property
    def is_throttled(self) -> bool:
        """Should we reduce this source's batch size?"""
        if len(self.history) < 3:
            return False  # not enough data
        return self.rolling_acceptance < self.min_acceptance

    @property
    def recommended_batch_size(self) -> int:
        """Adaptive batch size based on rolling acceptance."""
        base = 32
        if len(self.history) < 3:
            return base
        rate = self.rolling_acceptance
        if rate > 0.5:
            return base          # full batch - source is hot
        elif rate > 0.3:
            return max(16, base)  # decent - keep going
        elif rate > 0.15:
            return 8              # mediocre - small batches
        else:
            return 4              # poor - minimal proposals

    def record(self, proposed: int, accepted: int):
        self.history.append((proposed, accepted))
        self.total_proposed += proposed
        self.total_accepted += accepted
        self.total_rounds += 1
        if accepted == 0 and proposed > 0:
            self.total_wasted_passes += 1

    @property
    def tokens_per_pass(self) -> float:
        """Average accepted tokens per verification pass. Higher = better efficiency."""
        return self.total_accepted / self.total_rounds if self.total_rounds else 0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "total_proposed": self.total_proposed,
            "total_accepted": self.total_accepted,
            "lifetime_acceptance": self.lifetime_acceptance,
            "rolling_acceptance": self.rolling_acceptance,
            "rounds": self.total_rounds,
            "wasted_passes": self.total_wasted_passes,
            "tokens_per_pass": self.tokens_per_pass,
            "recommended_batch": self.recommended_batch_size,
            "throttled": self.is_throttled,
        }


# ── Orchestrator ──────────────────────────────────────────────

class Orchestrator:
    """
    Adaptive four-path draft orchestrator.

    Instead of a fixed cascade, the orchestrator:
    1. Checks N-gram first (always - it's free)
    2. If N-gram fires, optionally extends with ANE tokens (blending)
    3. If N-gram misses, checks ANE with adaptive batch size
    4. If both miss, falls back to MTP
    5. Adjusts strategy based on rolling performance

    Blending: when N-gram produces a short chain (say 5 tokens) but
    the ANE has longer predictions, append ANE tokens to fill up to
    the batch size. This catches cases where N-gram grabs the opening
    boilerplate and ANE predicts the continuation.
    """

    def __init__(self, ngram_n: int = 8, table_size: int = 4 * 1024 * 1024,
                 enable_blending: bool = True):
        self.ngram = NgramPredictor(n=ngram_n, table_size=table_size)
        self.ane_lookahead: Optional[ANELookahead] = None
        self.all_tokens: List[int] = []
        self.ane_position: int = 0
        self.enable_blending = enable_blending

        # Per-source trackers
        self.trackers = {
            "ngram": SourceTracker("ngram"),
            "ane": SourceTracker("ane", min_acceptance=0.12),
            "mtp": SourceTracker("mtp", min_acceptance=0.05),
        }
        self.gpu_tokens = 0
        self.total_rounds = 0
        self.total_forward_passes = 0  # includes wasted ones

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

    def get_draft(self, max_tokens: int, tokenizer=None) -> Tuple[List[int], List[str]]:
        """
        Get the best available draft tokens with adaptive sizing.
        Returns (tokens, sources) where sources[i] identifies each token's origin.

        Strategy:
        1. Try N-gram chain
        2. If N-gram chain is short and blending is enabled, extend with ANE
        3. If N-gram misses entirely, use ANE (with adaptive batch size)
        4. If both miss, return empty (caller falls to MTP)
        """
        self.total_rounds += 1
        draft_tokens = []
        sources = []

        # ── N-gram chain ──
        ngram_batch = min(max_tokens, self.trackers["ngram"].recommended_batch_size)
        ngram_chain = self.ngram.draft_chain(
            self.all_tokens, max_tokens=ngram_batch, min_tokens=1
        )

        if ngram_chain:
            draft_tokens = list(ngram_chain)
            sources = ["ngram"] * len(ngram_chain)

            # ── Blending: extend short N-gram chains with ANE ──
            if (self.enable_blending
                and len(ngram_chain) < max_tokens // 2
                and not self.trackers["ane"].is_throttled):

                remaining = max_tokens - len(ngram_chain)
                ane_batch = min(remaining, self.trackers["ane"].recommended_batch_size)
                ane_tokens = self._get_ane_tokens(ane_batch, tokenizer)

                if ane_tokens:
                    draft_tokens.extend(ane_tokens)
                    sources.extend(["ane"] * len(ane_tokens))

            return draft_tokens, sources

        # ── ANE only (N-gram missed) ──
        if not self.trackers["ane"].is_throttled:
            ane_batch = min(max_tokens, self.trackers["ane"].recommended_batch_size)
            ane_tokens = self._get_ane_tokens(ane_batch, tokenizer)

            if ane_tokens:
                return ane_tokens, ["ane"] * len(ane_tokens)

        # ── Both missed → caller uses MTP ──
        return [], []

    def _get_ane_tokens(self, num_tokens: int, tokenizer=None) -> List[int]:
        if not self.ane_lookahead:
            return []
        if not self.ane_lookahead.is_ready():
            self.ane_lookahead.wait(timeout=0.01)
        if not self.ane_lookahead.is_ready() or self.ane_lookahead.error:
            return []
        tokens = self.ane_lookahead.tokenize(tokenizer)
        result = []
        for i in range(num_tokens):
            pos = self.ane_position + i
            if pos < len(tokens):
                result.append(tokens[pos])
            else:
                break
        return result

    def record_result(self, sources: List[str], n_accepted: int):
        """Record acceptance results, attributing to each source."""
        # Determine which sources had tokens in the accepted range
        source_counts = {}
        for i, src in enumerate(sources):
            if src not in source_counts:
                source_counts[src] = {"proposed": 0, "accepted": 0}
            source_counts[src]["proposed"] += 1
            if i < n_accepted:
                source_counts[src]["accepted"] += 1

        for src, counts in source_counts.items():
            if src in self.trackers:
                self.trackers[src].record(counts["proposed"], counts["accepted"])

        self.total_forward_passes += 1

    def record_mtp(self, proposed: int, accepted: int):
        self.trackers["mtp"].record(proposed, accepted)
        self.total_forward_passes += 1

    def record_gpu_only(self):
        self.gpu_tokens += 1
        self.total_forward_passes += 1

    def advance_ane(self, n: int):
        self.ane_position += n

    @property
    def efficiency(self) -> float:
        """Tokens generated per forward pass. Standard generation = 1.0."""
        total_tokens = sum(t.total_accepted for t in self.trackers.values()) + self.gpu_tokens
        return total_tokens / self.total_forward_passes if self.total_forward_passes else 1.0

    def summary(self) -> dict:
        total_accepted = sum(t.total_accepted for t in self.trackers.values())
        return {
            "total_rounds": self.total_rounds,
            "total_forward_passes": self.total_forward_passes,
            "efficiency": self.efficiency,
            "gpu_only_tokens": self.gpu_tokens,
            "sources": {name: t.summary() for name, t in self.trackers.items()},
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n  Orchestrator Summary")
        print(f"  ───────────────────")
        print(f"  Forward passes: {s['total_forward_passes']}")
        print(f"  Efficiency: {s['efficiency']:.2f} tokens/pass (1.0 = no spec decode)")
        print()
        print(f"  {'Source':<8} {'Proposed':>10} {'Accepted':>10} {'Accept%':>10} {'Tok/Pass':>10} {'Batch':>8} {'Throttled':>10}")
        print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")
        for name, ts in s["sources"].items():
            print(f"  {name:<8} {ts['total_proposed']:>10} {ts['total_accepted']:>10} "
                  f"{ts['lifetime_acceptance']:>9.0%} {ts['tokens_per_pass']:>9.1f} "
                  f"{ts['recommended_batch']:>8} {'YES' if ts['throttled'] else 'no':>10}")
        print(f"  {'gpu':<8} {'-':>10} {s['gpu_only_tokens']:>10}")


# ── Orchestrated Generate Step ────────────────────────────────

def orchestrated_generate_step(
    prompt: mx.array,
    model: nn.Module,
    orchestrator: Orchestrator,
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
    Orchestrated four-path generation with adaptive source selection.
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

    def _step_standard(y, n_predict=1):
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

    # Feed prompt to orchestrator
    orchestrator.feed_prompt(prompt.tolist())

    # First token
    tokens, logprobs = _step_standard(y)
    mx.eval(tokens, logprobs)
    first_token = tokens.item() if tokens.ndim == 0 else tokens.tolist()[-1]
    orchestrator.add_token(first_token)
    orchestrator.record_gpu_only()

    ntoks = 1
    y = mx.array([first_token], mx.uint32)
    yield first_token, logprobs if logprobs.ndim == 1 else logprobs[-1], False, "gpu"

    if ntoks >= max_tokens:
        return

    mtp_draft_tok = None
    mtp_draft_lp = None

    while True:
        num_draft = min(max_tokens - ntoks, num_draft_tokens)

        # Ask orchestrator for best draft
        draft_tokens, sources = orchestrator.get_draft(num_draft, tokenizer=tokenizer)

        if draft_tokens:
            # ── Batch verify orchestrated drafts ──
            draft_mx = mx.array(draft_tokens, mx.uint32)
            verify_input = mx.concatenate([y, draft_mx])
            tokens, logprobs = _step_standard(verify_input, len(draft_tokens) + 1)
            mx.eval(tokens, logprobs)
            tokens_list = tokens.tolist()

            n_accepted = 0
            while n_accepted < len(draft_tokens):
                if tokens_list[n_accepted] != draft_tokens[n_accepted]:
                    break
                orchestrator.add_token(tokens_list[n_accepted])
                n_accepted += 1
                ntoks += 1
                lp = logprobs[n_accepted - 1] if logprobs.ndim > 1 else logprobs
                src = sources[n_accepted - 1] if n_accepted - 1 < len(sources) else "unknown"
                yield tokens_list[n_accepted - 1], lp, True, src
                if ntoks >= max_tokens:
                    break

            orchestrator.record_result(sources, n_accepted)

            # Advance ANE position for accepted ANE tokens
            ane_accepted = sum(1 for i in range(n_accepted) if i < len(sources) and sources[i] == "ane")
            if ane_accepted:
                orchestrator.advance_ane(ane_accepted)

            if ntoks >= max_tokens:
                break

            # Emit rejection/bonus token
            reject_tok = tokens_list[n_accepted]
            reject_lp = logprobs[n_accepted] if logprobs.ndim > 1 else logprobs
            orchestrator.add_token(reject_tok)
            orchestrator.gpu_tokens += 1
            ntoks += 1
            yield reject_tok, reject_lp, False, "gpu"

            n_to_trim = len(draft_tokens) - n_accepted
            if n_to_trim > 0:
                cache.trim_prompt_cache(model_cache, n_to_trim)

            y = mx.array([reject_tok], mx.uint32)
            mtp_draft_tok = None
            if ntoks >= max_tokens:
                break
            continue

        # ── MTP fallback (orchestrator returned empty) ──
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

                    orchestrator.add_token(mtp_draft_tok.item())
                    orchestrator.record_mtp(1, 1)
                    ntoks += 1
                    yield mtp_draft_tok.item(), mtp_draft_lp, True, "mtp"
                    if ntoks >= max_tokens:
                        break

                    orchestrator.add_token(toks[1].item())
                    orchestrator.gpu_tokens += 1
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

                    orchestrator.add_token(toks[0].item())
                    orchestrator.record_mtp(1, 0)
                    orchestrator.gpu_tokens += 1
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

                orchestrator.add_token(main_tok.item())
                orchestrator.record_gpu_only()
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

        # ── Pure GPU fallback ──
        tokens, logprobs = _step_standard(y)
        mx.eval(tokens, logprobs)
        tok = tokens.item() if tokens.ndim == 0 else tokens.tolist()[-1]
        lp = logprobs if logprobs.ndim == 1 else logprobs[-1]
        orchestrator.add_token(tok)
        orchestrator.record_gpu_only()
        ntoks += 1
        yield tok, lp, False, "gpu"
        y = mx.array([tok], mx.uint32)
        mtp_draft_tok = None
        if ntoks >= max_tokens:
            break
