"""
Three-Path Speculative Decoding on Apple Silicon
=================================================

CPU  (N-gram hash)   → pattern-based draft tokens    (nanoseconds)
ANE  (1.7B CoreML)   → neural lookahead draft tokens (parallel, pre-generated)
GPU  (9B MLX)        → verification + generation     (the real model)

Architecture:
  1. ANE starts generating a full continuation in a background thread
  2. CPU N-gram table feeds from prompt + generated tokens
  3. GPU generates with speculative decoding, using BOTH sources:
     - N-gram matches (verbatim boilerplate)
     - ANE lookahead (semantic predictions the N-gram misses)
  4. Draft tokens are merged: N-gram first (free), ANE fills gaps

The ANE generates at 57 tok/s vs GPU's 25 tok/s - it's always ahead.
Both draft sources produce tokens during time the GPU would otherwise waste.
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

# ── ANE Lookahead ─────────────────────────────────────────────

ANE_SOCKET = "/tmp/orion-ane-server.sock"


def ane_generate_async(prompt_text: str, max_tokens: int = 256,
                       socket_path: str = ANE_SOCKET) -> "ANELookahead":
    """
    Launch ANE generation in a background thread.
    Returns an ANELookahead object that accumulates tokens as they're generated.
    """
    lookahead = ANELookahead()
    thread = threading.Thread(
        target=_ane_worker,
        args=(prompt_text, max_tokens, socket_path, lookahead),
        daemon=True,
    )
    thread.start()
    lookahead._thread = thread
    return lookahead


def _ane_worker(prompt_text: str, max_tokens: int, socket_path: str,
                lookahead: "ANELookahead"):
    """Background worker that calls the ANE server and tokenizes output."""
    try:
        # Call ANE server via Unix socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(120.0)
        sock.connect(socket_path)

        request = {
            "cmd": "analyze",
            "prompt": prompt_text,
            "max_tokens": max_tokens,
        }
        msg = json.dumps(request).encode("utf-8")
        sock.sendall(struct.pack("!I", len(msg)) + msg)

        # Read response
        raw_len = b""
        while len(raw_len) < 4:
            chunk = sock.recv(4 - len(raw_len))
            if not chunk:
                break
            raw_len += chunk

        if len(raw_len) == 4:
            resp_len = struct.unpack("!I", raw_len)[0]
            raw_resp = b""
            while len(raw_resp) < resp_len:
                chunk = sock.recv(resp_len - len(raw_resp))
                if not chunk:
                    break
                raw_resp += chunk

            resp = json.loads(raw_resp.decode("utf-8"))
            if resp.get("status") == "ok":
                lookahead._set_result(resp.get("result", ""), resp.get("elapsed_ms", 0))
            else:
                lookahead._set_error(resp.get("error", "unknown"))
        sock.close()

    except Exception as e:
        lookahead._set_error(str(e))


class ANELookahead:
    """
    Holds the ANE's pre-generated continuation.
    Tokenized lazily when the GPU tokenizer is available.
    """

    def __init__(self):
        self.text: Optional[str] = None
        self.tokens: Optional[List[int]] = None
        self.elapsed_ms: float = 0
        self.error: Optional[str] = None
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _set_result(self, text: str, elapsed_ms: float):
        self.text = text
        self.elapsed_ms = elapsed_ms
        self._ready.set()

    def _set_error(self, error: str):
        self.error = error
        self._ready.set()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def wait(self, timeout: float = 30.0) -> bool:
        return self._ready.wait(timeout)

    def tokenize(self, tokenizer) -> List[int]:
        """Tokenize the ANE output with the GPU model's tokenizer."""
        if self.tokens is not None:
            return self.tokens
        if not self._ready.is_set() or self.text is None:
            return []
        # Re-tokenize ANE output with the target model's tokenizer
        # This handles tokenizer alignment between Qwen3-1.7B and Qwen3.5-9B
        self.tokens = tokenizer.encode(self.text)
        return self.tokens

    def get_draft_at(self, position: int) -> int:
        """Get the ANE's predicted token at a given generation position."""
        if self.tokens is None or position >= len(self.tokens):
            return EMPTY
        return self.tokens[position]


# ── Three-Path Drafter ────────────────────────────────────────

class ThreePathDrafter:
    """
    Merges draft tokens from CPU (N-gram) and ANE (neural lookahead).

    Priority:
    1. N-gram match (if available) - highest confidence, verbatim patterns
    2. ANE lookahead (if available) - semantic predictions
    3. No draft - GPU generates normally

    Tracks which source each draft came from for analysis.
    """

    def __init__(self, ngram_n: int = 8, table_size: int = 4 * 1024 * 1024):
        self.ngram = NgramPredictor(n=ngram_n, table_size=table_size)
        self.ane_lookahead: Optional[ANELookahead] = None
        self.all_tokens: List[int] = []
        self.ane_position: int = 0  # current position in ANE's output

        # Stats
        self.total_drafted = 0
        self.total_accepted = 0
        self.ngram_drafted = 0
        self.ngram_accepted = 0
        self.ane_drafted = 0
        self.ane_accepted = 0
        self.gpu_only = 0
        self.rounds = 0

    def feed_prompt(self, tokens: List[int]):
        """Feed prompt into N-gram table."""
        self.all_tokens = list(tokens)
        self.ngram.feed(tokens)

    def set_ane_lookahead(self, lookahead: ANELookahead):
        """Attach the ANE's pre-generated continuation."""
        self.ane_lookahead = lookahead

    def add_token(self, token: int):
        """Record a generated token and update N-gram table."""
        self.all_tokens.append(token)
        n = self.ngram.n
        if len(self.all_tokens) >= n + 1:
            self.ngram.feed(self.all_tokens[-(n + 1):])

    def draft(self, num_tokens: int, tokenizer=None) -> Tuple[List[int], List[str]]:
        """
        Produce draft tokens from merged sources.
        Returns (draft_tokens, sources) where sources[i] is 'ngram' or 'ane'.
        """
        self.rounds += 1
        draft_tokens = []
        sources = []

        # Try N-gram chain first
        ngram_chain = self.ngram.draft_chain(
            self.all_tokens, max_tokens=num_tokens, min_tokens=1
        )

        if ngram_chain:
            # N-gram produced a chain - use it
            draft_tokens = ngram_chain
            sources = ["ngram"] * len(ngram_chain)
            self.ngram_drafted += len(ngram_chain)
        else:
            # N-gram missed - try ANE lookahead
            if self.ane_lookahead and tokenizer:
                if not self.ane_lookahead.is_ready():
                    self.ane_lookahead.wait(timeout=0.01)  # brief wait

                if self.ane_lookahead.is_ready() and self.ane_lookahead.error is None:
                    ane_tokens = self.ane_lookahead.tokenize(tokenizer)
                    # Get tokens from ANE starting at current position
                    for i in range(num_tokens):
                        pos = self.ane_position + i
                        if pos < len(ane_tokens):
                            draft_tokens.append(ane_tokens[pos])
                            sources.append("ane")
                        else:
                            break
                    self.ane_drafted += len(draft_tokens)

        self.total_drafted += len(draft_tokens)
        return draft_tokens, sources

    def record_accepted(self, n_accepted: int, sources: List[str]):
        """Record acceptance stats per source."""
        self.total_accepted += n_accepted
        for i in range(n_accepted):
            if i < len(sources):
                if sources[i] == "ngram":
                    self.ngram_accepted += 1
                elif sources[i] == "ane":
                    self.ane_accepted += 1

    def advance_ane_position(self, n: int):
        """Advance the ANE lookahead cursor."""
        self.ane_position += n

    def summary(self) -> dict:
        return {
            "rounds": self.rounds,
            "total_drafted": self.total_drafted,
            "total_accepted": self.total_accepted,
            "acceptance_rate": self.total_accepted / self.total_drafted if self.total_drafted else 0,
            "ngram_drafted": self.ngram_drafted,
            "ngram_accepted": self.ngram_accepted,
            "ngram_acceptance": self.ngram_accepted / self.ngram_drafted if self.ngram_drafted else 0,
            "ane_drafted": self.ane_drafted,
            "ane_accepted": self.ane_accepted,
            "ane_acceptance": self.ane_accepted / self.ane_drafted if self.ane_drafted else 0,
            "gpu_only_tokens": self.gpu_only,
            "ngram_table_occupancy": self.ngram.occupancy,
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n  Three-Path Drafter Summary")
        print(f"  ─────────────────────────")
        print(f"  Rounds: {s['rounds']}")
        print(f"  Total: {s['total_drafted']} drafted, {s['total_accepted']} accepted ({s['acceptance_rate']:.1%})")
        print(f"  N-gram (CPU):  {s['ngram_drafted']} drafted, {s['ngram_accepted']} accepted ({s['ngram_acceptance']:.1%})")
        print(f"  ANE (neural):  {s['ane_drafted']} drafted, {s['ane_accepted']} accepted ({s['ane_acceptance']:.1%})")
        print(f"  GPU only: {s['gpu_only_tokens']} tokens (no draft available)")


# ── Three-Path Generate Step ──────────────────────────────────

def three_path_generate_step(
    prompt: mx.array,
    model: nn.Module,
    drafter: ThreePathDrafter,
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
    Three-path speculative generation.

    Yields: (token_id, logprobs, from_draft, source)
    where source is 'ngram', 'ane', or 'gpu'
    """
    y = prompt.astype(mx.uint32)

    if prompt_cache is None:
        model_cache = cache.make_prompt_cache(model)
    else:
        model_cache = prompt_cache

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _process_and_sample(logits):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        return y, logprobs

    def _step(y, n_predict=1):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=model_cache)
            logits = logits[:, -n_predict:, :]
            quantize_cache_fn(model_cache)
            return _process_and_sample(logits.squeeze(0))

    # Prefill prompt
    with mx.stream(generation_stream):
        while y.size > prefill_step_size:
            model(y[:prefill_step_size][None], cache=model_cache)
            quantize_cache_fn(model_cache)
            mx.eval([c.state for c in model_cache])
            y = y[prefill_step_size:]
            mx.clear_cache()

    # Feed prompt to N-gram table
    prompt_tokens = prompt.tolist()
    drafter.feed_prompt(prompt_tokens)

    # First token - always from GPU
    tokens, logprobs = _step(y)
    mx.eval(tokens, logprobs)
    first_token = tokens.item() if tokens.ndim == 0 else tokens.tolist()[-1]
    drafter.add_token(first_token)

    ntoks = 0
    y = mx.array([first_token], mx.uint32)
    ntoks += 1
    yield first_token, logprobs if logprobs.ndim == 1 else logprobs[-1], False, "gpu"

    if ntoks >= max_tokens:
        return

    while True:
        num_draft = min(max_tokens - ntoks, num_draft_tokens)
        draft_tokens, sources = drafter.draft(num_draft, tokenizer=tokenizer)

        if not draft_tokens:
            # No drafts from either source - single GPU token
            tokens, logprobs = _step(y)
            mx.eval(tokens, logprobs)
            tok = tokens.item() if tokens.ndim == 0 else tokens.tolist()[-1]
            lp = logprobs if logprobs.ndim == 1 else logprobs[-1]
            drafter.add_token(tok)
            drafter.gpu_only += 1
            ntoks += 1
            yield tok, lp, False, "gpu"
            y = mx.array([tok], mx.uint32)
            if ntoks >= max_tokens:
                break
            continue

        # Build verification batch
        draft_mx = mx.array(draft_tokens, mx.uint32)
        verify_input = mx.concatenate([y, draft_mx])
        num_to_verify = len(draft_tokens)

        # Single GPU forward pass verifies all drafts
        tokens, logprobs = _step(verify_input, num_to_verify + 1)
        mx.eval(tokens, logprobs)
        tokens_list = tokens.tolist()

        n_accepted = 0
        while n_accepted < num_to_verify:
            target_tok = tokens_list[n_accepted]
            draft_tok = draft_tokens[n_accepted]
            lp = logprobs[n_accepted] if logprobs.ndim > 1 else logprobs

            if target_tok != draft_tok:
                break

            drafter.add_token(target_tok)
            n_accepted += 1
            ntoks += 1
            src = sources[n_accepted - 1] if n_accepted - 1 < len(sources) else "unknown"
            yield target_tok, lp, True, src

            if ntoks >= max_tokens:
                break

        drafter.record_accepted(n_accepted, sources)

        # Advance ANE position by how many tokens we actually generated
        if any(s == "ane" for s in sources[:n_accepted]):
            drafter.advance_ane_position(n_accepted)

        if ntoks >= max_tokens:
            break

        # Emit rejection/bonus token from GPU
        reject_tok = tokens_list[n_accepted]
        reject_lp = logprobs[n_accepted] if logprobs.ndim > 1 else logprobs
        drafter.add_token(reject_tok)
        ntoks += 1
        yield reject_tok, reject_lp, False, "gpu"

        if ntoks >= max_tokens:
            break

        # Rewind KV cache
        n_to_trim = num_to_verify - n_accepted
        if n_to_trim > 0:
            cache.trim_prompt_cache(model_cache, n_to_trim)

        y = mx.array([reject_tok], mx.uint32)
