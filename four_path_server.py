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
from threading import Lock, Thread
import threading

import mlx.core as mx

# Add ngram-engine to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from ngram_cascade import CascadingNgramPredictor as NgramPredictor
from three_path import ane_generate_async, ANELookahead
from four_path import FourPathDrafter, four_path_generate_step
from ane_sync import ANESyncDrafter
from prompt_lookup import prompt_lookup_draft
from ssm_checkpoint import checkpoint_ssm_state, restore_ssm_state
from ane_drafter import ANEDrafter
from gdn_drafter import GDNCoreMLDrafter
from amx_drafter import AMXDrafter

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
gdn_drafter = None  # GDN CoreML drafter (same-tokenizer, Qwen3.5-0.8B on ANE)
lightweight_model = None  # Lightweight model for easy queries (ANE routing layer)
amx_drafter = None       # CPU/AMX draft model (llama.cpp server, zero GPU contention)

# Persistent N-gram table - survives across requests, accumulates patterns
# from all generated tokens in the session. Resets on server restart.
persistent_ngram = None


# ── Hardware-aware routing ─────────────────────────────────────
# Classifies query complexity and routes to the appropriate model:
#   Easy: lightweight model (0.8B on ANE, 2W) answers directly
#   Hard: full model (9B/70B on GPU, 20W) generates
# Saves GPU power for queries that don't need it.

EASY_PATTERNS = [
    # Greetings and simple acknowledgments
    r'^(hi|hello|hey|thanks|thank you|ok|yes|no|sure|got it)\s*[.!?]?\s*$',
    # Simple factual questions
    r'^(what is|what\'s|define|who is|when was|where is)\s+\w+(\s+\w+){0,3}\??$',
    # Short commands
    r'^(list|show|tell me|give me)\s+\d+\s+\w+',
]

def _classify_query_complexity(messages):
    """Classify whether a query needs the full GPU model or can be handled lightweight.
    Returns 'easy' or 'hard'. Conservative — defaults to 'hard' when uncertain."""
    import re

    if not messages:
        return 'hard'

    # Get the last user message
    last_user = None
    for m in reversed(messages):
        if m.get('role') == 'user':
            last_user = m.get('content', '')
            break

    if not last_user:
        return 'hard'

    # Multi-turn conversations always go to full model
    user_count = sum(1 for m in messages if m.get('role') == 'user')
    if user_count > 1:
        return 'hard'

    # Tool-using requests always go to full model
    if any(m.get('tool_calls') for m in messages):
        return 'hard'

    # Long queries go to full model
    if len(last_user.split()) > 30:
        return 'hard'

    # Check easy patterns
    for pattern in EASY_PATTERNS:
        if re.match(pattern, last_user.strip(), re.IGNORECASE):
            return 'easy'

    return 'hard'


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
    persistent_ngram = NgramPredictor(levels=(8, 6, 4, 2))
    log.info(f"Cascading N-gram initialized (levels=8,6,4,2)")

    # Pre-seeding disabled — vault contains enricher artifacts (entity graphs,
    # similarity scores, metadata) that corrupt output. N-gram learns from
    # its own output across requests. Pre-seed with curated text only.
    log.info("N-gram pre-seed: disabled (use curated corpus, not raw vault)")


def _get_eos_ids():
    """Get all stop token IDs for the loaded tokenizer.
    Qwen3.5 has eos_token_id=248046 (<|im_end|>) but the model also generates
    <|endoftext|> (248044) as a stop signal. Both must be caught."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    # eos_token_ids can be int or list depending on tokenizer version
    raw = getattr(tokenizer, "eos_token_ids", None)
    if raw is not None:
        if isinstance(raw, int):
            ids.add(raw)
        elif hasattr(raw, "__iter__"):
            ids.update(raw)
    # Explicitly add <|endoftext|> and <|im_end|> for Qwen chat models
    for special in ["<|endoftext|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(special)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return ids


def _clean_output(text):
    """Strip special tokens and trailing chat markup from generated text."""
    for tok_str in ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]:
        text = text.replace(tok_str, "")
    # Strip trailing role markers like "user\n" or "assistant\n" that leak after im_start
    text = re.sub(r'\s*(user|assistant|system)\s*$', '', text)
    return text.strip()


def check_ane():
    global ane_available
    # Try per-round ANE drafter first (new, text-level matching)
    drafter = ANEDrafter(K=3)
    if drafter.connect():
        ane_available = True
        log.info("ANE server: connected (per-round drafter, K=3)")
    else:
        ane_available = False
        log.info("ANE server: not available (PLD + N-gram only)")


def _init_amx_drafter():
    """Start AMX/CPU draft model (llama.cpp server, zero GPU contention).
    Measured: 160 tok/s generation on M5 Air. 0.5% GPU interference."""
    global amx_drafter
    # AMX drafter disabled on <32GB — subprocess memory pressure hurts GPU throughput
    import platform
    mem_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    if mem_gb < 32:
        log.info("AMX drafter: disabled (%.0fGB RAM — needs 32GB+)", mem_gb)
        return
    if not os.path.isfile(AMXDrafter().gguf_path):
        log.info("AMX drafter: GGUF not found, skipping")
        return
    if not os.path.isfile(AMXDrafter().llama_cli.replace('llama-cli', 'llama-server')):
        log.info("AMX drafter: llama-server not built, skipping (cd llama.cpp/build && make llama-server)")
        return
    drafter = AMXDrafter()
    if drafter.start():
        amx_drafter = drafter
        log.info("AMX drafter: started (Qwen3-0.6B CPU-only, 160 tok/s, port %d)", drafter.port)
    else:
        log.info("AMX drafter: failed to start")


def check_gdn():
    """Load GDN CoreML drafter (same-tokenizer Qwen3.5-0.8B on ANE).

    Disabled on M5 Air 16GB: 0.8B at 24ms/tok is only 1.75x faster than 9B at 42ms/tok,
    not enough to overcome verification overhead. Measured 0.66-0.94x (slower than baseline).

    Enabled on Pro with 70B target: 0.8B at 24ms/tok vs 70B at ~200ms/tok = 8x faster,
    firmly in the spec decode sweet spot. Expected 1.3-1.8x speedup at 60% acceptance.

    Set SPEC_GDN_ENABLE=1 to force enable (for benchmarking or with slower target models).
    """
    global gdn_drafter
    gdn_model_dir = os.path.expanduser("~/models/Qwen3.5-0.8B-coreml")
    force_enable = os.environ.get("SPEC_GDN_ENABLE", "0") == "1"

    if not os.path.isdir(gdn_model_dir):
        log.info("GDN CoreML: model not found at %s", gdn_model_dir)
        return

    # GDN drafter disabled on 16GB machines — CoreML model starves GPU of bandwidth.
    # On Air: 205 tok/s without drafter, 5.6 tok/s with. 37x slowdown from memory pressure.
    # Enable only on Pro (64GB) where memory is not contested.
    import platform
    mem_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    if not force_enable and mem_gb < 32:
        log.info("GDN CoreML: disabled (%.0fGB RAM — needs 32GB+, set SPEC_GDN_ENABLE=1 to force)", mem_gb)
        return

    try:
        drafter = GDNCoreMLDrafter(gdn_model_dir, context_length=256)
        drafter.load()
        gdn_drafter = drafter
        log.info("GDN CoreML: loaded (Qwen3.5-0.8B, same tokenizer, 24ms/tok)")

        # Also use as the lightweight routing model for easy queries
        global lightweight_model
        lightweight_model = drafter
        log.info("Routing layer: lightweight model active (easy queries → 0.8B, GPU idle)")
    except Exception as e:
        log.warning("GDN CoreML: failed to load: %s", e)
        gdn_drafter = None


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


def spec_generate(prompt_text, prompt_tokens, max_tokens=2048, temperature=0.7, has_tools=False, messages=None, repetition_penalty=1.3):
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
                                    use_ane=use_ane_lookahead, ane_context=ane_context,
                                    repetition_penalty=repetition_penalty)
    return _generate_three_path(prompt_text, prompt_tokens, max_tokens, temperature,
                                 use_ane=use_ane_lookahead, ane_context=ane_context,
                                 use_ane_sync=use_ane_sync, messages=messages,
                                 repetition_penalty=repetition_penalty)


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


def _generate_four_path(prompt_text, prompt_tokens, max_tokens=2048, temperature=0.7, use_ane=True, ane_context="", repetition_penalty=1.3):
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

    # Sampler with repetition penalty
    output_tokens = []

    if temperature <= 0:
        def sampler(logits):
            if repetition_penalty != 1.0 and output_tokens:
                recent = list(set(output_tokens[-256:]))
                if recent:
                    indices = mx.array(recent, dtype=mx.int32)
                    vals = logits[indices] if logits.ndim == 1 else logits[..., indices]
                    penalized = mx.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
                    if logits.ndim == 1:
                        logits[indices] = penalized
                    else:
                        logits[..., indices] = penalized
            return mx.argmax(logits, axis=-1)
    else:
        def sampler(logits):
            if repetition_penalty != 1.0 and output_tokens:
                recent = list(set(output_tokens[-256:]))
                if recent:
                    indices = mx.array(recent, dtype=mx.int32)
                    vals = logits[indices] if logits.ndim == 1 else logits[..., indices]
                    penalized = mx.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
                    if logits.ndim == 1:
                        logits[indices] = penalized
                    else:
                        logits[..., indices] = penalized
            return mx.random.categorical(logits / temperature)

    eos_ids = _get_eos_ids()

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

    text = _clean_output(tokenizer.decode(output_tokens))

    return {
        "text": text,
        "tokens": output_tokens,
        "n_tokens": len(output_tokens),
        "elapsed": elapsed,
        "tok_per_sec": len(output_tokens) / elapsed if elapsed > 0 else 0,
        "sources": sources,
    }


def _generate_three_path(prompt_text, prompt_tokens, max_tokens=2048, temperature=0.7,
                          use_ane=True, ane_context="", use_ane_sync=False, messages=None,
                          repetition_penalty=1.3):
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

    # Per-round ANE drafter — disabled until same-family model available (Qwen3.5-1.7B/4B).
    # Cross-vocab (Qwen3 151K vs Qwen3.5 248K) causes token boundary misalignment
    # that degrades output quality. The 60% acceptance measured in teacher-forcing is
    # real but doesn't transfer to per-round token-level matching.
    # Re-enable when: same-tokenizer draft model on ANE, OR text-level verification.
    ane_drafter = None

    # Create cache
    model_cache = cache.make_prompt_cache(model)

    quantize_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=64,
        kv_bits=None,
    )

    def _sample(logits):
        # Apply repetition penalty to logits for tokens in recent output
        if repetition_penalty != 1.0 and output_tokens:
            recent = list(set(output_tokens[-256:]))
            if recent:
                indices = mx.array(recent, dtype=mx.int32)
                vals = logits[indices] if logits.ndim == 1 else logits[..., indices]
                penalized = mx.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
                if logits.ndim == 1:
                    logits[indices] = penalized
                else:
                    logits[..., indices] = penalized
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

    def _forward_async(y, n_predict=1):
        """Submit forward pass and return lazy result (IAM async pipeline).
        Use mx.async_eval to overlap GPU compute with Python draft preparation."""
        with mx.stream(generation_stream):
            logits = model(y[None], cache=model_cache)
            logits = logits[:, -n_predict:, :]
            quantize_fn(model_cache)
            tokens, logprobs = _sample(logits.squeeze(0))
        mx.async_eval(tokens, logprobs)
        return tokens, logprobs

    eos_ids = _get_eos_ids()

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
    sources = {"prompt_lookup": 0, "ngram": 0, "amx": 0, "ane": 0, "gpu": 0}
    ane_proposed = 0
    ane_rounds = 0
    output_tokens = []
    # For prompt lookup: skip system prompt + tool definitions (prevents schema echoing),
    # and also skip the tail (prevents question echoing). Only match against
    # user/assistant conversation content in the middle of the prompt.
    pld_tail_skip = 80
    pld_start = content_start  # reuse N-gram's content start (skips tools + system)
    pld_end = max(0, len(prompt_tokens) - pld_tail_skip)
    prompt_token_list = list(prompt_tokens[pld_start:pld_end]) if pld_end > pld_start + 20 else []
    PLD_MAX_DRAFT = 16  # Cap per-match draft length

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

    # Kick off first ANE draft (runs during first GPU round)
    if ane_drafter and ane_drafter.active:
        ane_drafter.draft_async()

    # GDN CoreML drafter: prefill with last few prompt tokens + first token
    # Only prefill the tail (CTX=64), not the whole prompt
    if gdn_drafter and gdn_drafter.loaded:
        gdn_drafter.reset()
        tail_tokens = prompt_tokens[-60:] if len(prompt_tokens) > 60 else prompt_tokens
        for tok in tail_tokens:
            gdn_drafter._step(tok)
        gdn_drafter.draft_async(K=8, last_token=first_tok)

    t0 = time.perf_counter()

    # ── Logit cache for N-gram pre-filtering ──
    # After GPU verification, save the logprobs from the last position.
    # The N-gram can check proposed tokens against these before sending
    # a full chain to GPU verification.
    _last_gpu_logprobs = None  # mx.array of shape [vocab_size]
    _last_gpu_top_k = None     # set of top-K token IDs

    def _cache_gpu_logprobs(logprobs_row):
        """Save the GPU's prediction for the next position."""
        nonlocal _last_gpu_logprobs, _last_gpu_top_k
        _last_gpu_logprobs = logprobs_row
        # Cache top-64 token IDs for fast N-gram filtering
        top_indices = mx.argpartition(logprobs_row, kth=-64)[-64:]
        _last_gpu_top_k = set(top_indices.tolist())

    def _filter_ngram_chain(chain):
        """Pre-filter N-gram chain using cached GPU logprobs.
        Trims chain at the first token the GPU considers unlikely.
        Returns trimmed chain (may be shorter or empty)."""
        if _last_gpu_top_k is None or not chain:
            return chain  # No cached logprobs — pass through unfiltered
        # Only check the FIRST token against GPU's top-K prediction
        # If the first token isn't in the GPU's top-64 — chain is probably wrong
        if chain[0] not in _last_gpu_top_k:
            return []  # First token unlikely — skip entire chain
        return chain

    def _verify_draft(draft_chain, source_name):
        """Batch verify a draft chain. Returns (n_accepted, should_break).
        SSM state corruption from rejected tokens is negligible (measured:
        100% token overlap vs baseline). Checkpoint/restore available in
        ssm_checkpoint.py for Pro/70B or long chains where it matters."""
        nonlocal y

        draft_mx = mx.array(draft_chain, mx.uint32)
        verify_input = mx.concatenate([y, draft_mx])
        # IAM: use async forward — GPU starts immediately, Python continues
        tokens, logprobs = _forward_async(verify_input, len(draft_chain) + 1)

        # ── CONCURRENT: While GPU verifies, kick off AMX + ANE drafts ──
        # GPU is computing asynchronously. Use this time to start drafters
        # for the NEXT round, so drafts are ready when GPU finishes.
        _speculative_next_token = draft_chain[-1] if draft_chain else None

        # AMX drafter: kick off CPU-based draft during GPU verify
        if amx_drafter and amx_drafter.loaded:
            if not amx_drafter._draft_thread or not amx_drafter._draft_thread.is_alive():
                # Build context from recent tokens — speculate from last draft token
                spec_context = tokenizer.decode(all_tokens[-128:] + list(draft_chain))
                amx_drafter.draft_async(spec_context, tokenizer, K=min(num_draft, 8))

        # ANE drafter
        if gdn_drafter and gdn_drafter.loaded and _speculative_next_token is not None:
            if not gdn_drafter._draft_thread or not gdn_drafter._draft_thread.is_alive():
                gdn_drafter.draft_async(K=NUM_DRAFT, last_token=_speculative_next_token)

        # Collect result (blocks until GPU finishes)
        mx.eval(tokens, logprobs)
        tokens_list = tokens.tolist()

        n_accepted = 0
        while n_accepted < len(draft_chain):
            if tokens_list[n_accepted] != draft_chain[n_accepted]:
                break
            tok = tokens_list[n_accepted]
            if _would_repeat(tok):
                break
            n_accepted += 1

        reject_tok = tokens_list[n_accepted]
        n_rejected = len(draft_chain) - n_accepted

        # ── Cache GPU logprobs for N-gram pre-filtering ──
        # The logprobs at the last position predict the next token.
        # Save them so the N-gram can check its proposals cheaply.
        if logprobs.ndim >= 1:
            _cache_gpu_logprobs(logprobs[-1] if logprobs.ndim == 2 else logprobs)

        # Trim attention KV caches for rejected tokens
        if n_rejected > 0:
            cache.trim_prompt_cache(model_cache, n_rejected)

        # Update token lists
        done = False
        for i in range(n_accepted):
            tok = draft_chain[i]
            all_tokens.append(tok)
            output_tokens.append(tok)
            sources[source_name] += 1
            if tok in eos_ids or len(output_tokens) >= max_tokens:
                done = True
                break

        if not done:
            all_tokens.append(reject_tok)
            output_tokens.append(reject_tok)
            sources["gpu"] += 1
            if reject_tok in eos_ids or len(output_tokens) >= max_tokens:
                done = True

        if len(all_tokens) >= n + 1:
            ngram.feed(all_tokens[-(n + 1):])

        y = mx.array([all_tokens[-1]], mx.uint32)
        return n_accepted, done

    while len(output_tokens) < max_tokens:
        num_draft = min(max_tokens - len(output_tokens), NUM_DRAFT)

        # ── Priority 0: Prompt lookup (zero cost) ──
        pld_chain = prompt_lookup_draft(prompt_token_list, all_tokens, max_draft=min(num_draft, PLD_MAX_DRAFT))
        if pld_chain:
            _, done = _verify_draft(pld_chain, "prompt_lookup")
            if done:
                break
            continue

        # ── Priority 1: N-gram chain (pre-filtered by GPU logprobs) ──
        ngram_chain = ngram.draft_chain(all_tokens, max_tokens=num_draft, min_tokens=1)
        ngram_chain = _filter_ngram_chain(ngram_chain)

        if ngram_chain:
            _, done = _verify_draft(ngram_chain, "ngram")
            if done:
                break
            continue

        # ── Priority 2: AMX CPU draft (zero GPU contention, 160 tok/s) ──
        if amx_drafter and amx_drafter.loaded:
            amx_draft = amx_drafter.get_draft(timeout=0.005)
            if amx_draft:
                amx_draft = _filter_ngram_chain(amx_draft)  # reuse logprob filter
                if amx_draft:
                    n_acc, done = _verify_draft(amx_draft, "amx")
                    if done:
                        break
                    # Kick off next AMX draft with updated context
                    context_text = tokenizer.decode(all_tokens[-256:])
                    amx_drafter.draft_async(context_text, tokenizer, K=num_draft)
                    continue
            # No draft ready — kick off for next round
            if not amx_drafter._draft_thread or not amx_drafter._draft_thread.is_alive():
                context_text = tokenizer.decode(all_tokens[-256:])
                amx_drafter.draft_async(context_text, tokenizer, K=num_draft)

        # ── Priority 3: GDN CoreML draft (same tokenizer, 60%+ acceptance) ──
        # Only enabled when SPEC_GDN_ENABLE=1 (disabled on M5 Air, enabled on Pro)
        if gdn_drafter and gdn_drafter.loaded:
            gdn_draft = gdn_drafter.get_draft(timeout=0.001)
            if gdn_draft:
                ane_proposed += len(gdn_draft)
                n_acc, done = _verify_draft(gdn_draft, "ane")
                # Sync drafter: rewind rejected, feed correction
                n_rej = len(gdn_draft) - n_acc
                if n_rej > 0:
                    gdn_drafter.rewind(n_rej)
                gdn_drafter._step(all_tokens[-1])
                gdn_drafter.draft_async(K=8, last_token=all_tokens[-1])
                if done:
                    break
                continue
            # Draft not ready — kick off for next round if not already running
            if not gdn_drafter._draft_thread:
                gdn_drafter.draft_async(K=8, last_token=all_tokens[-1])

        # ── Priority 3: Per-round ANE draft (text-level, 60%+ acceptance) ──
        # Check if ANE draft from PREVIOUS round is ready (non-blocking)
        if ane_drafter and ane_drafter.active:
            draft_text = ane_drafter.get_draft_text(timeout=0)

            if draft_text:
                # Re-tokenize with 9B tokenizer for verification
                ane_target_tokens = tokenizer.encode(draft_text, add_special_tokens=False)

                # Guard: only use if re-tokenized text round-trips cleanly
                # (prevents garbled output from cross-vocab boundary misalignment)
                if ane_target_tokens:
                    roundtrip = tokenizer.decode(ane_target_tokens)
                    if roundtrip.strip() == draft_text.strip() and len(ane_target_tokens) <= 8:
                        ane_proposed += len(ane_target_tokens)
                        ane_rounds += 1
                        n_acc, done = _verify_draft(ane_target_tokens, "ane")

                        # Sync ANE: rewind all, feed back correct text
                        correct_text = tokenizer.decode(
                            all_tokens[-(n_acc + 1):]
                        ) if n_acc >= 0 else tokenizer.decode([all_tokens[-1]])
                        ane_drafter.on_partial_reject(ane_drafter.K, correct_text)

                        if done:
                            break

                        # Launch next ANE draft (runs during next GPU step)
                        ane_drafter.draft_async()
                        continue

            # ANE not ready or failed roundtrip — launch for next round and fall through to GPU
            if not ane_drafter._draft_thread:
                ane_drafter.draft_async()

        # ── Standard single-token generation (no draft sources available) ──
        # IAM async: submit GPU work, then do CPU draft prep while GPU computes
        tokens, logprobs = _forward_async(y)
        # CPU work while GPU is computing (async_eval pipelining)
        # This overlaps draft preparation with GPU forward pass
        _next_pld = prompt_lookup_draft(prompt_token_list, all_tokens, max_draft=min(num_draft, PLD_MAX_DRAFT))
        _next_ngram = ngram.draft_chain(all_tokens, max_tokens=num_draft, min_tokens=1) if not _next_pld else None
        # Now collect GPU result
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

    elapsed = time.perf_counter() - t0
    text = _clean_output(tokenizer.decode(output_tokens))

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
        "text": _clean_output(tokenizer.decode(tokens)),
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
                "gdn_coreml": gdn_drafter is not None and gdn_drafter.loaded,
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
        repetition_penalty = body.get("repetition_penalty", 1.1)
        tools = body.get("tools") or None

        # Normalize messages for Qwen's chat template
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                normalized = []
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        # Qwen template expects arguments as DICT, not JSON string
                        args = func.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        normalized.append({
                            "type": "function",
                            "function": {
                                "name": func.get("name", ""),
                                "arguments": args,
                            }
                        })
                    else:
                        normalized.append(tc)
                msg["tool_calls"] = normalized
                # Template also needs content=None not empty string
                if not msg.get("content"):
                    msg["content"] = None

        # ── Hardware-aware routing: easy queries skip the GPU ──
        complexity = _classify_query_complexity(messages)
        if complexity == 'easy' and lightweight_model is not None:
            log.info(f"Routing: EASY → lightweight 0.8B (GPU idle)")
            try:
                lw_prompt = apply_chat_template(messages, tools=None)
                lw_tokens = tokenizer.encode(lw_prompt)

                # Generate with the lightweight GDN model
                lw = lightweight_model
                lw.reset()
                # Prefill
                for tok in lw_tokens:
                    lw._step(tok)
                # Generate
                t0 = time.perf_counter()
                out_tokens = []
                eos = _get_eos_ids()
                for _ in range(min(max_tokens, 200)):  # cap lightweight at 200 tokens
                    logits = lw._step(out_tokens[-1] if out_tokens else lw_tokens[-1])
                    tok = int(logits[0, 0].argmax())
                    if tok in eos:
                        break
                    out_tokens.append(tok)
                elapsed = time.perf_counter() - t0
                lw_text = tokenizer.decode(out_tokens)
                lw_tps = len(out_tokens) / elapsed if elapsed > 0 else 0

                log.info(f"Lightweight: {len(out_tokens)} tok / {elapsed:.2f}s = {lw_tps:.1f} tok/s")

                return self._json_response({
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "model": "lightweight-0.8B-ANE",
                    "created": int(time.time()),
                    "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": lw_text,
                                    "reasoning": "", "tool_calls": []}}],
                    "usage": {"prompt_tokens": len(lw_tokens),
                        "completion_tokens": len(out_tokens),
                        "total_tokens": len(lw_tokens) + len(out_tokens)},
                    "x_routing": {"complexity": "easy", "model": "lightweight-0.8B",
                                  "tok_per_sec": round(lw_tps, 1)}
                })
            except Exception as e:
                log.info(f"Lightweight routing failed, falling back to GPU: {e}")

        if complexity == 'easy' and lightweight_model is None:
            log.info(f"Routing: EASY query but no lightweight model loaded → GPU")

        try:
            prompt_text = apply_chat_template(messages, tools=tools)
            prompt_tokens = tokenizer.encode(prompt_text)
            log.info(f"Prompt: {len(prompt_tokens)} tokens ({len(messages)} messages, tools={'yes' if tools else 'no'})")
        except Exception as e:
            # Log the problematic messages for debugging
            for i, m in enumerate(messages):
                if m.get("tool_calls"):
                    log.error(f"Template error at msg[{i}]: role={m.get('role')} tool_calls={json.dumps(m['tool_calls'])[:300]}")
            return self._err(f"Template error: {e}")

        with model_lock:
            try:
                result = spec_generate(
                    prompt_text, prompt_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    has_tools=bool(tools),
                    messages=messages,
                    repetition_penalty=repetition_penalty,
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

        # Report to Heartbeat dashboard (fire-and-forget in background thread)
        def _report_heartbeat():
            try:
                import urllib.request as _ur
                _data = json.dumps({
                    "tok_per_sec": round(result["tok_per_sec"], 1),
                    "sources": src,
                    "draft_ratio": round(drafted / total, 3) if total else 0,
                    "total_tokens": total,
                    "elapsed": round(result["elapsed"], 3),
                }).encode()
                _req = _ur.Request(
                    "http://127.0.0.1:8423/api/inference/report",
                    data=_data, headers={"Content-Type": "application/json"},
                )
                _ur.urlopen(_req, timeout=1)
            except Exception:
                pass
        threading.Thread(target=_report_heartbeat, daemon=True).start()

    def _handle_text(self):
        try:
            body = self._read_body()
        except Exception as e:
            return self._err(f"Invalid JSON: {e}")

        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.7)
        repetition_penalty = body.get("repetition_penalty", 1.1)
        prompt_tokens = tokenizer.encode(prompt)

        with model_lock:
            result = spec_generate(prompt, prompt_tokens, max_tokens, temperature,
                                   repetition_penalty=repetition_penalty)

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
    """Parse tool calls from model output. Handles two Qwen formats:
    1. JSON: <tool_call>{"name":"func","arguments":{...}}</tool_call>
    2. XML:  <tool_call><function=func><parameter=key>value</parameter></function></tool_call>
    Also handles truncated output (missing closing tags)."""
    calls = []

    # Format 1: JSON inside <tool_call> tags
    json_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    for match in re.findall(json_pattern, text, re.DOTALL):
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

    if calls:
        return calls

    # Format 2: XML parameter format (with or without closing tags)
    # Matches: <function=name> or <function=name\n — '>' is optional (9B sometimes omits it)
    xml_pattern = r'<function=(\w+)>?(.*?)(?:</function>|$)'
    for func_match in re.finditer(xml_pattern, text, re.DOTALL):
        func_name = func_match.group(1)
        params_text = func_match.group(2)

        # Extract parameters — try XML tags first, then JSON fallback
        arguments = {}
        param_pattern = r'<parameter=(\w+)>\s*(.*?)\s*(?:</parameter>|(?=<parameter=)|$)'
        for param_match in re.finditer(param_pattern, params_text, re.DOTALL):
            key = param_match.group(1)
            value = param_match.group(2).strip()
            # Try to parse as number/bool
            if value.isdigit():
                arguments[key] = int(value)
            elif value.lower() in ("true", "false"):
                arguments[key] = value.lower() == "true"
            else:
                arguments[key] = value
        # Fallback: model sometimes dumps JSON instead of XML parameters
        if not arguments:
            json_match = re.search(r'\{[^}]+\}', params_text)
            if json_match:
                try:
                    arguments = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        if func_name:
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(arguments),
                },
            })

    return calls or None


def _strip_tool_markup(text):
    """Remove tool call tags from text (both JSON and XML formats)."""
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    # Also strip unclosed tool_call blocks (truncated output)
    text = re.sub(r'<tool_call>.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'<function=\w+>.*?(?:</function>|$)', '', text, flags=re.DOTALL)
    return text.strip()


# ── Main ────────────────────────────────────────────────────────

def main():
    load_model()
    check_ane()
    check_gdn()
    _init_amx_drafter()

    paths = ["N-gram (CPU)", "GPU (9B)"]
    if ane_available:
        paths.insert(1, "ANE (1.7B)")
    if gdn_drafter:
        paths.insert(-1, "GDN CoreML (0.8B)")
    if amx_drafter:
        paths.insert(-1, "AMX (0.6B CPU)")
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
