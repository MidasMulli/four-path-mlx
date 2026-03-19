#!/usr/bin/env python3
"""
Three-Path Benchmark: Standard vs N-gram vs Three-Path (CPU+ANE+GPU)

Measures wall-clock tok/s and draft acceptance rates across all three modes
on real ISDA prompts. The definitive benchmark for the heterogeneous
speculative decoding architecture.

Usage:
    # Ensure ANE server is running and MLX server is NOT running
    ~/.mlx-env/bin/python benchmark_three_path.py
"""

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))

MODEL_ID = "mlx-community/Qwen3.5-9B-MLX-4bit"
SAMPLES_DIR = Path(__file__).parent.parent / "isda-classifier" / "samples"

# ISDA context for N-gram priming
def load_isda_context(max_chars=8000):
    f = SAMPLES_DIR / "aerocentury-isda-2002.txt"
    if f.exists():
        return f.read_text()[:max_chars]
    return ""

ISDA_CONTEXT = load_isda_context()

TEST_PROMPTS = [
    {
        "name": "clause_analysis",
        "desc": "Analytical (novel output)",
        "prompt": (
            f"Reference ISDA Agreement excerpt:\n{ISDA_CONTEXT}\n\n---\n\n"
            "Analyze the following ISDA Master Agreement clause and identify: "
            "(1) the clause type, (2) any non-standard provisions, (3) risk implications.\n\n"
            "Section 5(a)(vi) Cross-Default. If \"Cross-Default\" is specified as applying to "
            "the party, the occurrence or existence of (A) a default, event of default or other "
            "similar condition or event (however described) in respect of such party."
        ),
        "max_tokens": 256,
    },
    {
        "name": "schedule_gen",
        "desc": "Boilerplate generation (high repetition)",
        "prompt": (
            f"Reference ISDA Agreement excerpt:\n{ISDA_CONTEXT}\n\n---\n\n"
            "Generate a standard ISDA Schedule Part 1 for a derivatives transaction between "
            "a US bank (Party A) and a UK hedge fund (Party B). Include: Termination Currency, "
            "Cross-Default threshold, Credit Event Upon Merger provisions.\n\n"
            "Schedule to the 2002 ISDA Master Agreement\n\nPart 1. Termination Provisions.\n\n"
        ),
        "max_tokens": 512,
    },
    {
        "name": "csa_draft",
        "desc": "CSA paragraph drafting (mixed)",
        "prompt": (
            f"Reference ISDA Agreement excerpt:\n{ISDA_CONTEXT}\n\n---\n\n"
            "Draft a Credit Support Annex (CSA) Paragraph 13 for a bilateral derivatives "
            "relationship. Party A is a US G-SIB bank, Party B is a European pension fund. "
            "Include: eligible collateral, valuation percentages, thresholds, minimum transfer "
            "amounts, and notification times. Use standard market terms.\n\n"
            "Paragraph 13. Elections and Variables.\n\n"
        ),
        "max_tokens": 512,
    },
]


def load_model():
    from mlx_lm import load
    print(f"Loading model: {MODEL_ID}")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    return model, tokenizer


def check_ane_server():
    """Check if ANE server is available."""
    try:
        sys.path.insert(0, "/Users/midas/Desktop/cowork/orion-ane/memory")
        from ane_server import ANEClient
        client = ANEClient()
        resp = client.ping()
        print(f"  ANE server: alive (uptime {resp['uptime']:.0f}s)")
        return True
    except Exception as e:
        print(f"  ANE server: not available ({e})")
        return False


# ── Benchmark functions ────────────────────────────────────────

def bench_standard(model, tokenizer, prompt_tokens, max_tokens):
    """Baseline: standard one-token-at-a-time."""
    from mlx_lm.generate import generate_step
    prompt = mx.array(prompt_tokens, mx.uint32)
    tokens_out = []
    t0 = time.perf_counter()
    for token, logprobs in generate_step(prompt, model, max_tokens=max_tokens):
        tokens_out.append(token)
    elapsed = time.perf_counter() - t0
    return {
        "method": "standard",
        "tokens": len(tokens_out),
        "elapsed": elapsed,
        "tok_per_sec": len(tokens_out) / elapsed,
    }


def bench_ngram(model, tokenizer, prompt_tokens, max_tokens, ngram_n=8):
    """N-gram only: CPU draft path."""
    from ngram_generate import NgramDrafter, ngram_speculative_generate_step
    prompt = mx.array(prompt_tokens, mx.uint32)
    drafter = NgramDrafter(n=ngram_n)
    tokens_out = []
    drafted = 0
    t0 = time.perf_counter()
    for token, logprobs, from_draft in ngram_speculative_generate_step(
        prompt, model, drafter, num_draft_tokens=32, max_tokens=max_tokens,
    ):
        tokens_out.append(token)
        if from_draft:
            drafted += 1
    elapsed = time.perf_counter() - t0
    return {
        "method": f"ngram_n{ngram_n}",
        "tokens": len(tokens_out),
        "elapsed": elapsed,
        "tok_per_sec": len(tokens_out) / elapsed,
        "drafted": drafted,
        "draft_ratio": drafted / len(tokens_out) if tokens_out else 0,
        "drafter": drafter.summary(),
    }


def bench_three_path(model, tokenizer, prompt_tokens, prompt_text, max_tokens,
                     ngram_n=8, ane_available=True):
    """Three-path: CPU N-gram + ANE neural + GPU verify."""
    from three_path import ThreePathDrafter, three_path_generate_step, ane_generate_async

    prompt = mx.array(prompt_tokens, mx.uint32)
    drafter = ThreePathDrafter(ngram_n=ngram_n)

    # Launch ANE lookahead in background
    ane_lookahead = None
    if ane_available:
        # Send a condensed version of the prompt to ANE (it has limited context)
        # Extract just the task instruction, not the full ISDA reference
        task_start = prompt_text.find("---\n\n")
        ane_prompt = prompt_text[task_start + 5:] if task_start >= 0 else prompt_text[-500:]
        ane_lookahead = ane_generate_async(ane_prompt, max_tokens=max_tokens)
        drafter.set_ane_lookahead(ane_lookahead)

    tokens_out = []
    source_counts = {"ngram": 0, "ane": 0, "gpu": 0}

    t0 = time.perf_counter()
    for token, logprobs, from_draft, source in three_path_generate_step(
        prompt, model, drafter, tokenizer=tokenizer,
        num_draft_tokens=32, max_tokens=max_tokens,
    ):
        tokens_out.append(token)
        source_counts[source] = source_counts.get(source, 0) + 1

    elapsed = time.perf_counter() - t0

    # Wait for ANE to finish so we can report its timing
    ane_elapsed = 0
    if ane_lookahead:
        ane_lookahead.wait(timeout=5)
        ane_elapsed = ane_lookahead.elapsed_ms

    summary = drafter.summary()

    return {
        "method": "three_path",
        "tokens": len(tokens_out),
        "elapsed": elapsed,
        "tok_per_sec": len(tokens_out) / elapsed,
        "sources": source_counts,
        "ngram_drafted": summary["ngram_drafted"],
        "ngram_accepted": summary["ngram_accepted"],
        "ane_drafted": summary["ane_drafted"],
        "ane_accepted": summary["ane_accepted"],
        "gpu_only": summary["gpu_only_tokens"],
        "ane_elapsed_ms": ane_elapsed,
        "drafter_summary": summary,
    }


# ── Main ──────────────────────────────────────────────────────

def run():
    model, tokenizer = load_model()
    ane_available = check_ane_server()

    # Warmup
    print("\nWarming up GPU...")
    warmup_prompt = mx.array(tokenizer.encode("Hello"), mx.uint32)
    from mlx_lm.generate import generate_step
    for _ in generate_step(warmup_prompt, model, max_tokens=5):
        pass
    print("  Done\n")

    print("=" * 90)
    print("THREE-PATH SPECULATIVE DECODING BENCHMARK")
    print(f"  GPU: Qwen3.5-9B-MLX-4bit  |  ANE: Qwen3-1.7B CoreML  |  CPU: N-gram hash (n=8)")
    print("=" * 90)

    all_results = []

    for test in TEST_PROMPTS:
        name = test["name"]
        desc = test["desc"]
        max_tokens = test["max_tokens"]
        prompt_text = test["prompt"]
        prompt_tokens = tokenizer.encode(prompt_text)

        print(f"\n{'─' * 90}")
        print(f"  {name} - {desc}")
        print(f"  Prompt: {len(prompt_tokens):,} tokens, Generate: {max_tokens} tokens")
        print(f"{'─' * 90}")

        # 1. Standard baseline
        print(f"\n  [1/3] Standard (GPU only)...")
        std = bench_standard(model, tokenizer, prompt_tokens, max_tokens)
        print(f"        {std['tokens']} tokens, {std['tok_per_sec']:.1f} tok/s")

        # 2. N-gram only
        print(f"  [2/3] N-gram (CPU + GPU)...")
        ngram = bench_ngram(model, tokenizer, prompt_tokens, max_tokens, ngram_n=8)
        ng_speedup = ngram["tok_per_sec"] / std["tok_per_sec"]
        print(f"        {ngram['tokens']} tokens, {ngram['tok_per_sec']:.1f} tok/s "
              f"({ng_speedup:.2f}x) - {ngram['draft_ratio']:.0%} drafted")

        # 3. Three-path
        print(f"  [3/3] Three-Path (CPU + ANE + GPU)...")
        three = bench_three_path(
            model, tokenizer, prompt_tokens, prompt_text, max_tokens,
            ngram_n=8, ane_available=ane_available,
        )
        tp_speedup = three["tok_per_sec"] / std["tok_per_sec"]
        src = three["sources"]
        print(f"        {three['tokens']} tokens, {three['tok_per_sec']:.1f} tok/s "
              f"({tp_speedup:.2f}x)")
        print(f"        Sources: N-gram={src.get('ngram',0)}, ANE={src.get('ane',0)}, GPU={src.get('gpu',0)}")
        if three["ane_elapsed_ms"]:
            print(f"        ANE lookahead: {three['ane_elapsed_ms']}ms")

        all_results.append({
            "test": name,
            "desc": desc,
            "prompt_tokens": len(prompt_tokens),
            "standard": {
                "tok_per_sec": std["tok_per_sec"],
                "tokens": std["tokens"],
            },
            "ngram": {
                "tok_per_sec": ngram["tok_per_sec"],
                "speedup": ng_speedup,
                "draft_ratio": ngram["draft_ratio"],
            },
            "three_path": {
                "tok_per_sec": three["tok_per_sec"],
                "speedup": tp_speedup,
                "sources": src,
                "ngram_accepted": three["ngram_accepted"],
                "ane_accepted": three["ane_accepted"],
                "gpu_only": three["gpu_only"],
                "ane_elapsed_ms": three["ane_elapsed_ms"],
            },
        })

    # ── Summary table ─────────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print(f"\n  {'Test':<20} {'Standard':>10} {'N-gram':>10} {'3-Path':>10} {'NG Speed':>10} {'3P Speed':>10} {'ANE hits':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for r in all_results:
        print(f"  {r['test']:<20} "
              f"{r['standard']['tok_per_sec']:>9.1f} "
              f"{r['ngram']['tok_per_sec']:>9.1f} "
              f"{r['three_path']['tok_per_sec']:>9.1f} "
              f"{r['ngram']['speedup']:>9.2f}x "
              f"{r['three_path']['speedup']:>9.2f}x "
              f"{r['three_path']['ane_accepted']:>10}")

    print(f"\n  Processor roles:")
    print(f"    CPU  - N-gram hash table: catches verbatim boilerplate patterns")
    print(f"    ANE  - 1.7B neural lookahead: catches semantic predictions")
    print(f"    GPU  - 9B verification + novel token generation")

    # Save
    out_path = Path(__file__).parent / "three_path_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run()
