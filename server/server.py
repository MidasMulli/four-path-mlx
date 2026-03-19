#!/usr/bin/env python3
"""
Speculative Decoding Server for Midas
======================================

OpenAI-compatible HTTP server that uses multi-path speculative decoding
instead of standard generation. Drop-in replacement for mlx_lm server.

Paths (auto-detected):
  CPU  N-gram hash   → pattern-based draft chains       (always on)
  ANE  1.7B CoreML   → neural lookahead draft tokens    (if ANE server running)
  MTP  GPU head      → hidden-state prediction          (if MTP model loaded)
  GPU  9B backbone   → verification + generation        (always on)

On stock mlx-lm + standard model: N-gram + ANE + GPU = 1.5-1.85x
With MTP model + fork:            all four paths      = 2-4x
"""

import json
import os
import sys
import time
import uuid
import logging
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock

import mlx.core as mx

# Add ngram-engine to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from four_path.ngram import NgramPredictor
from four_path.three_path import ane_generate_async, ANELookahead
from four_path.generate import FourPathDrafter, four_path_generate_step
from four_path.ane_sync import ANESyncDrafter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Suppress noisy loggers
for name in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(name).setLevel(logging.WARNING)

# ── Config ──────────────────────────────────────────────────────

MODEL_PATH = os.environ.get(
    "SPEC_MODEL",
    "mlx-community/Qwen3.5-9B-MLX-4bit",
)
MTP_WEIGHTS_PATH = os.environ.get(
    "SPEC_MTP_WEIGHTS",
    os.path.expanduser("~/models/Qwen3.5-9B-MLX-4bit-MTP"),
)
PORT = int(os.environ.get("SPEC_PORT", "8899"))
HOST = os.environ.get("SPEC_HOST", "127.0.0.1")
NGRAM_N = int(os.environ.get("SPEC_NGRAM_N", "8"))
NUM_DRAFT = int(os.environ.get("SPEC_DRAFT_TOKENS", "32"))

# ── Global state ────────────────────────────────────────────────

model = None
tokenizer = None
model_lock = Lock()
has_mtp = False
ane_available = False

# Persistent N-gram table - survives across requests, accumulates patterns
# from all generated tokens in the session. Resets on server restart.
persistent_ngram = None


def load_model():
    global model, tokenizer, has_mtp
    from mlx_lm import load
    log.info(f"Loading model: {MODEL_PATH}")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_PATH)
    elapsed = time.perf_counter() - t0
    log.info(f"Model loaded in {elapsed:.1f}s")

    # MTP disabled for server use - return_hidden path is too slow for interactive
    # (manual forward pass lacks MLX optimizations). MTP works in standalone benchmarks
    # where it's called on the already-optimized code path.
    # TODO: fix by caching hidden states from standard forward pass instead of re-running
    has_mtp = False
    log.info("MTP: disabled (return_hidden overhead - fix pending)")

    # Warmup
    warmup = mx.array(tokenizer.encode("Hello"), mx.uint32)
    from mlx_lm.generate import generate_step
    for _ in generate_step(warmup, model, max_tokens=5):
        pass
    log.info("Warmup complete")

    # Initialize persistent N-gram table
    global persistent_ngram
    persistent_ngram = NgramPredictor(n=NGRAM_N)
    log.info(f"Persistent N-gram table initialized (n={NGRAM_N})")


def check_ane():
    global ane_available
    try:
        import socket as sock_mod
        s = sock_mod.socket(sock_mod.AF_UNIX, sock_mod.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect("/tmp/orion-ane-server.sock")
        s.close()
        ane_available = True
        log.info("ANE server: connected")
    except Exception:
        ane_available = False
        log.info("ANE server: not available (N-gram only)")


def apply_chat_template(messages, tools=None):
    """Apply the model's chat template to messages with thinking disabled."""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Some templates don't support all kwargs
        kwargs.pop("tools", None)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)


# ── Speculative Generation ──────────────────────────────────────

def _build_ane_context(messages):
    """Build clean ANE context from raw messages (before chat template).
    Uses last assistant response + current user message as plain text.
    No tool schemas, no chat markup - just conversation content."""
    parts = []
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            parts.insert(0, msg["content"][:1500])
            break
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            parts.append(msg["content"][:1500])
            break
    return "\n\n".join(parts) if parts else ""


def spec_generate(prompt_text, prompt_tokens, max_tokens=2048, temperature=0.7, has_tools=False, messages=None):
    """Multi-path speculative generation.
    Tool prompts: N-gram + synchronized ANE (warm KV cache, per-round draft) + GPU
    Non-tool prompts: N-gram + ANE lookahead + GPU (domain tasks, buffer approach)
    N-gram persists across requests for session-level pattern matching."""
    # ANE sync: 0.4% acceptance. Qwen3-1.7B and Qwen3.5-9B are different families
    # with fundamentally different token distributions. Context length isn't the issue.
    # Needs same-family draft model (Qwen3.5-1.7B doesn't exist yet, or 4B on Pro).
    use_ane_sync = False
    use_ane_lookahead = not has_tools and ane_available
    ane_context = _build_ane_context(messages or []) if use_ane_lookahead else ""

    if has_mtp:
        return _generate_four_path(prompt_text, prompt_tokens, max_tokens, temperature,
                                    use_ane=use_ane_lookahead, ane_context=ane_context)
    return _generate_three_path(prompt_text, prompt_tokens, max_tokens, temperature,
                                 use_ane=use_ane_lookahead, ane_context=ane_context,
                                 use_ane_sync=use_ane_sync, messages=messages)


def _find_content_start(prompt_text, prompt_tokens):
    """Find where user content starts in the token stream (skip system prompt + tool defs).
    Only feed user content to the N-gram table to avoid drafting from tool schema boilerplate."""
    markers = ["<|im_start|>user", "<|user|>", "### User:", "[INST]"]
    last_pos = -1
    for marker in markers:
        pos = prompt_text.rfind(marker)
        if pos > last_pos:
            last_pos = pos
    if last_pos <= 0:
        return len(prompt_tokens) * 6 // 10
    prefix_tokens = tokenizer.encode(prompt_text[:last_pos])
    return len(prefix_tokens)


def _generate_four_path(prompt_text, prompt_tokens, max_tokens=2048, temperature=0.7, use_ane=True, ane_context=""):
    """Four-path: N-gram (CPU) + MTP (GPU head) + ANE (1.7B) + GPU (9B)."""
    lang_model = model.language_model if hasattr(model, "language_model") else model
    prompt = mx.array(prompt_tokens, mx.uint32)
    drafter = FourPathDrafter(ngram_n=NGRAM_N)
    ngram_start = _find_content_start(prompt_text, prompt_tokens)
    log.info(f"ngram_feed_start={ngram_start}/{len(prompt_tokens)} ({ngram_start*100//len(prompt_tokens)}% skipped)")

    # Launch ANE
    if ane_available and use_ane and ane_context:
        # ANE gets clean conversation text - no tool defs, no chat markup
        ane_lookahead = ane_generate_async(ane_context, max_tokens=max_tokens)
        drafter.set_ane_lookahead(ane_lookahead)
    else:
        ane_lookahead = None

    # Sampler
    if temperature <= 0:
        sampler = lambda x: mx.argmax(x, axis=-1)
    else:
        def sampler(logprobs):
            return mx.random.categorical(logprobs / temperature)

    # EOS tokens
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    if hasattr(tokenizer, "eos_token_ids") and tokenizer.eos_token_ids:
        eos_ids.update(tokenizer.eos_token_ids)

    output_tokens = []
    sources = {"ngram": 0, "ane": 0, "mtp": 0, "gpu": 0}

    t0 = time.perf_counter()
    for tok, lp, from_draft, source in four_path_generate_step(
        prompt, lang_model, drafter, tokenizer=tokenizer,
        num_draft_tokens=NUM_DRAFT, max_tokens=max_tokens,
        sampler=sampler if temperature > 0 else None,
        ngram_feed_start=ngram_start,
    ):
        output_tokens.append(tok)
        sources[source] = sources.get(source, 0) + 1
        if tok in eos_ids:
            break

    elapsed = time.perf_counter() - t0

    if ane_lookahead:
        ane_lookahead.wait(2)

    text = tokenizer.decode(output_tokens)
    if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
        text = text.replace(tokenizer.eos_token, "")

    return {
        "text": text.strip(),
        "tokens": output_tokens,
        "n_tokens": len(output_tokens),
        "elapsed": elapsed,
        "tok_per_sec": len(output_tokens) / elapsed if elapsed > 0 else 0,
        "sources": sources,
    }


def _generate_three_path(prompt_text, prompt_tokens, max_tokens=2048, temperature=0.7,
                          use_ane=True, ane_context="", use_ane_sync=False, messages=None):
    """Three-path: N-gram (CPU) + ANE (1.7B) + GPU (9B).
    use_ane_sync: if True, use synchronized per-round ANE with warm KV cache."""
    import functools
    from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
    from mlx_lm.models import cache

    prompt = mx.array(prompt_tokens, mx.uint32)

    # Use persistent N-gram table - accumulates patterns across all requests
    ngram = persistent_ngram

    # Feed user content from this request (skip system/tool boilerplate)
    content_start = _find_content_start(prompt_text, prompt_tokens)
    ngram.feed(prompt_tokens[content_start:])
    all_tokens = list(prompt_tokens)

    # Launch ANE lookahead (buffer approach for non-tool prompts)
    ane_lookahead = None
    ane_position = 0
    if ane_available and use_ane and ane_context:
        ane_lookahead = ane_generate_async(ane_context, max_tokens=max_tokens)

    # Synchronized ANE (per-round with warm KV cache, for tool prompts)
    ane_sync = None
    if use_ane_sync:
        ane_sync = ANESyncDrafter()
        # With 1024-ctx model, send the last user message for ANE prefill.
        # The ANE applies its own chat template, so send raw content.
        user_text = _build_ane_context(messages or [])
        if user_text:
            if ane_sync.prefill(user_text):
                log.info(f"ANE sync prefilled in {ane_sync.prefill_ms:.0f}ms, 1024-ctx model")
            else:
                ane_sync = None

    # Create cache
    model_cache = cache.make_prompt_cache(model)

    quantize_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=64,
        kv_bits=None,
    )

    def _sample(logits):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if temperature <= 0:
            tok = mx.argmax(logprobs, axis=-1)
        else:
            tok = mx.random.categorical(logprobs / temperature)
        return tok, logprobs

    def _forward(y, n_predict=1):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=model_cache)
            logits = logits[:, -n_predict:, :]
            quantize_fn(model_cache)
            return _sample(logits.squeeze(0))

    # EOS tokens
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    if hasattr(tokenizer, "eos_token_ids") and tokenizer.eos_token_ids:
        eos_ids.update(tokenizer.eos_token_ids)

    # Repetition filter - reject drafts that create output loops
    def _would_repeat(tok, window=120):
        """Reject tokens that create a repeated 5-gram in recent output."""
        if len(output_tokens) < 5:
            return False
        cand = tuple(output_tokens[-4:]) + (tok,)
        recent = output_tokens[-window:]
        for i in range(len(recent) - 5):
            if tuple(recent[i:i+5]) == cand:
                return True
        return False

    # Stats
    sources = {"ngram": 0, "ane": 0, "gpu": 0}
    ane_proposed = 0
    ane_rounds = 0
    output_tokens = []

    # Prefill
    y = prompt.astype(mx.uint32)
    prefill_step = 2048
    with mx.stream(generation_stream):
        while y.size > prefill_step:
            model(y[:prefill_step][None], cache=model_cache)
            quantize_fn(model_cache)
            mx.eval([c.state for c in model_cache if hasattr(c, "state")])
            y = y[prefill_step:]
            mx.clear_cache()

    # First token
    tokens, logprobs = _forward(y)
    mx.eval(tokens, logprobs)
    first_tok = tokens.item() if tokens.ndim == 0 else tokens.tolist()[-1]
    all_tokens.append(first_tok)
    output_tokens.append(first_tok)
    sources["gpu"] += 1

    if first_tok in eos_ids or len(output_tokens) >= max_tokens:
        text = tokenizer.decode(output_tokens)
        return _result(text, output_tokens, sources, time.perf_counter())

    y = mx.array([first_tok], mx.uint32)
    n = NGRAM_N

    t0 = time.perf_counter()

    while len(output_tokens) < max_tokens:
        num_draft = min(max_tokens - len(output_tokens), NUM_DRAFT)

        # ── Try N-gram chain ──
        ngram_chain = ngram.draft_chain(all_tokens, max_tokens=num_draft, min_tokens=1)

        if ngram_chain:
            draft_mx = mx.array(ngram_chain, mx.uint32)
            verify_input = mx.concatenate([y, draft_mx])
            tokens, logprobs = _forward(verify_input, len(ngram_chain) + 1)
            mx.eval(tokens, logprobs)
            tokens_list = tokens.tolist()

            n_accepted = 0
            while n_accepted < len(ngram_chain):
                if tokens_list[n_accepted] != ngram_chain[n_accepted]:
                    break
                tok = tokens_list[n_accepted]
                if _would_repeat(tok):
                    break  # Reject to prevent repetition loop
                all_tokens.append(tok)
                output_tokens.append(tok)
                sources["ngram"] += 1
                n_accepted += 1
                if tok in eos_ids or len(output_tokens) >= max_tokens:
                    break

            if output_tokens and output_tokens[-1] in eos_ids or len(output_tokens) >= max_tokens:
                break

            # Rejection/bonus token
            reject_tok = tokens_list[n_accepted]
            all_tokens.append(reject_tok)
            output_tokens.append(reject_tok)
            sources["gpu"] += 1

            # Update N-gram
            if len(all_tokens) >= n + 1:
                ngram.feed(all_tokens[-(n + 1):])

            # Trim cache for rejected drafts
            n_trim = len(ngram_chain) - n_accepted
            if n_trim > 0:
                cache.trim_prompt_cache(model_cache, n_trim)

            y = mx.array([reject_tok], mx.uint32)
            if reject_tok in eos_ids or len(output_tokens) >= max_tokens:
                break
            continue

        # ── Try ANE tokens ──
        if ane_lookahead and ane_lookahead.is_ready() and not ane_lookahead.error:
            ane_tokens = ane_lookahead.tokenize(tokenizer)
            ane_draft = []
            for i in range(num_draft):
                pos = ane_position + i
                if pos < len(ane_tokens):
                    ane_draft.append(ane_tokens[pos])
                else:
                    break

            if ane_draft:
                ane_proposed += len(ane_draft)
                ane_rounds += 1
                draft_mx = mx.array(ane_draft, mx.uint32)
                verify_input = mx.concatenate([y, draft_mx])
                tokens, logprobs = _forward(verify_input, len(ane_draft) + 1)
                mx.eval(tokens, logprobs)
                tokens_list = tokens.tolist()

                n_accepted = 0
                while n_accepted < len(ane_draft):
                    if tokens_list[n_accepted] != ane_draft[n_accepted]:
                        break
                    tok = tokens_list[n_accepted]
                    if _would_repeat(tok):
                        break  # Reject to prevent repetition loop
                    all_tokens.append(tok)
                    output_tokens.append(tok)
                    sources["ane"] += 1
                    n_accepted += 1
                    if tok in eos_ids or len(output_tokens) >= max_tokens:
                        break

                ane_position += n_accepted

                if output_tokens[-1] in eos_ids or len(output_tokens) >= max_tokens:
                    break

                reject_tok = tokens_list[n_accepted]
                all_tokens.append(reject_tok)
                output_tokens.append(reject_tok)
                sources["gpu"] += 1

                if len(all_tokens) >= n + 1:
                    ngram.feed(all_tokens[-(n + 1):])

                n_trim = len(ane_draft) - n_accepted
                if n_trim > 0:
                    cache.trim_prompt_cache(model_cache, n_trim)

                y = mx.array([reject_tok], mx.uint32)
                if reject_tok in eos_ids or len(output_tokens) >= max_tokens:
                    break
                continue

        # ── Try synchronized ANE draft (per-round, warm KV cache) ──
        if ane_sync and ane_sync.active:
            # Launch ANE decode in background while GPU generates
            ane_future = ane_sync.draft_one_async()

            # GPU generates its token (this takes ~50ms on 9B)
            tokens, logprobs = _forward(y)
            mx.eval(tokens, logprobs)
            gpu_tok = tokens.item() if tokens.ndim == 0 else tokens.tolist()[-1]

            # Check if ANE predicted the same token
            ane_tok = ane_future.wait(timeout=0.1)
            if ane_tok is not None and ane_tok == gpu_tok:
                # ANE got it right - count as ANE-drafted
                all_tokens.append(gpu_tok)
                output_tokens.append(gpu_tok)
                sources["ane"] += 1
                ane_proposed += 1

                if len(all_tokens) >= n + 1:
                    ngram.feed(all_tokens[-(n + 1):])
                y = mx.array([gpu_tok], mx.uint32)
                if gpu_tok in eos_ids or len(output_tokens) >= max_tokens:
                    break
                continue
            else:
                # ANE missed - use GPU token, count ANE proposal
                if ane_tok is not None:
                    ane_proposed += 1
                all_tokens.append(gpu_tok)
                output_tokens.append(gpu_tok)
                sources["gpu"] += 1

                if len(all_tokens) >= n + 1:
                    ngram.feed(all_tokens[-(n + 1):])
                y = mx.array([gpu_tok], mx.uint32)
                if gpu_tok in eos_ids or len(output_tokens) >= max_tokens:
                    break
                continue

        # ── Standard single-token generation (no ANE) ──
        tokens, logprobs = _forward(y)
        mx.eval(tokens, logprobs)
        tok = tokens.item() if tokens.ndim == 0 else tokens.tolist()[-1]
        all_tokens.append(tok)
        output_tokens.append(tok)
        sources["gpu"] += 1

        if len(all_tokens) >= n + 1:
            ngram.feed(all_tokens[-(n + 1):])

        y = mx.array([tok], mx.uint32)
        if tok in eos_ids or len(output_tokens) >= max_tokens:
            break

    if ane_lookahead:
        ane_lookahead.wait(2)

    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(output_tokens)

    # Strip EOS from text
    if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
        text = text.replace(tokenizer.eos_token, "")

    tok_per_sec = len(output_tokens) / elapsed if elapsed > 0 else 0

    # Feed output tokens into persistent N-gram table for next request
    if output_tokens:
        persistent_ngram.feed(output_tokens)

    # Log ANE acceptance rate
    if ane_proposed > 0:
        accept_rate = sources["ane"] / ane_proposed * 100
        log.info(f"ANE: {sources['ane']}/{ane_proposed} accepted ({accept_rate:.1f}%) over {ane_rounds} rounds")

    return {
        "text": text.strip(),
        "tokens": output_tokens,
        "n_tokens": len(output_tokens),
        "elapsed": elapsed,
        "tok_per_sec": tok_per_sec,
        "sources": sources,
        "ane_detail": {"proposed": ane_proposed, "accepted": sources["ane"], "rounds": ane_rounds},
    }


def _result(text, tokens, sources, t0):
    elapsed = time.perf_counter() - t0
    return {
        "text": tokenizer.decode(tokens).strip(),
        "tokens": tokens,
        "n_tokens": len(tokens),
        "elapsed": elapsed,
        "tok_per_sec": len(tokens) / elapsed if elapsed > 0 else 0,
        "sources": sources,
    }


# ── HTTP Handler ────────────────────────────────────────────────

class SpecHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        log.info(format % args)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg, status=400):
        self._json({"error": msg}, status)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            self._json({
                "object": "list",
                "data": [{
                    "id": MODEL_PATH,
                    "object": "model",
                    "owned_by": "spec-decode-server",
                }]
            })
        elif self.path == "/health":
            self._json({
                "status": "ok",
                "model": MODEL_PATH,
                "mtp": has_mtp,
                "ane": ane_available,
                "ngram_n": NGRAM_N,
            })
        else:
            self._err("Not found", 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat()
        elif self.path == "/v1/completions":
            self._handle_text()
        else:
            self._err("Not found", 404)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n))

    def _handle_chat(self):
        try:
            body = self._read_body()
        except Exception as e:
            return self._err(f"Invalid JSON: {e}")

        messages = body.get("messages", [])
        if not messages:
            return self._err("No messages provided")

        max_tokens = body.get("max_tokens", 2048)
        temperature = body.get("temperature", 0.7)
        tools = body.get("tools") or None

        try:
            prompt_text = apply_chat_template(messages, tools=tools)
            prompt_tokens = tokenizer.encode(prompt_text)
        except Exception as e:
            return self._err(f"Template error: {e}")

        with model_lock:
            try:
                result = spec_generate(
                    prompt_text, prompt_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    has_tools=bool(tools),
                    messages=messages,
                )
            except Exception as e:
                log.error(f"Generation error: {e}", exc_info=True)
                return self._err(f"Generation error: {e}", 500)

        text = result["text"]

        # Parse tool calls
        tool_calls = _parse_tool_calls(text) if tools else None
        content = _strip_tool_markup(text) if tool_calls else text
        if tool_calls and not content.strip():
            content = None

        choice = {
            "index": 0,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "message": {
                "role": "assistant",
                "content": content,
                "reasoning": "",
                "tool_calls": tool_calls or [],
            },
        }

        src = result.get("sources", {})
        drafted = src.get("ngram", 0) + src.get("ane", 0) + src.get("mtp", 0)
        total = result["n_tokens"]

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": MODEL_PATH,
            "created": int(time.time()),
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": total,
                "total_tokens": len(prompt_tokens) + total,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "x_four_path": {
                "tok_per_sec": round(result["tok_per_sec"], 1),
                "sources": src,
                "draft_ratio": round(drafted / total, 3) if total else 0,
            },
        }

        log.info(
            f"{total} tok / {result['elapsed']:.2f}s = {result['tok_per_sec']:.1f} tok/s | "
            f"ngram={src.get('ngram',0)} mtp={src.get('mtp',0)} ane={src.get('ane',0)} gpu={src.get('gpu',0)} "
            f"({drafted}/{total} drafted)"
        )
        self._json(response)

    def _handle_text(self):
        try:
            body = self._read_body()
        except Exception as e:
            return self._err(f"Invalid JSON: {e}")

        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.7)
        prompt_tokens = tokenizer.encode(prompt)

        with model_lock:
            result = spec_generate(prompt, prompt_tokens, max_tokens, temperature)

        self._json({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "model": MODEL_PATH,
            "created": int(time.time()),
            "choices": [{"text": result["text"], "index": 0, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": result["n_tokens"],
                "total_tokens": len(prompt_tokens) + result["n_tokens"],
            },
        })


def _parse_tool_calls(text):
    """Parse Qwen-style <tool_call> tags from model output."""
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    calls = []
    for match in matches:
        try:
            data = json.loads(match)
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data.get("name", ""),
                    "arguments": json.dumps(data.get("arguments", {}))
                    if isinstance(data.get("arguments"), dict)
                    else str(data.get("arguments", "{}")),
                },
            })
        except json.JSONDecodeError:
            continue
    return calls or None


def _strip_tool_markup(text):
    """Remove tool call tags from text."""
    return re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip()


# ── Main ────────────────────────────────────────────────────────

def main():
    load_model()
    check_ane()

    paths = ["N-gram (CPU)", "GPU (9B)"]
    if ane_available:
        paths.insert(1, "ANE (1.7B)")
    if has_mtp:
        paths.insert(-1, "MTP (head)")

    server = HTTPServer((HOST, PORT), SpecHandler)
    log.info(f"Speculative decode server on {HOST}:{PORT}")
    log.info(f"  Model: {MODEL_PATH}")
    log.info(f"  Active paths: {' + '.join(paths)}")
    log.info(f"  N-gram n={NGRAM_N}, draft tokens={NUM_DRAFT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
