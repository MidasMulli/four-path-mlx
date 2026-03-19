#!/usr/bin/env python3
"""
Wall-Clock Benchmark: N-gram Speculative Decoding vs Standard Generation

Measures actual tokens/second on real ISDA prompts, comparing:
  1. Standard generation (baseline)
  2. N-gram speculative generation (CPU draft path)

This is the test that matters - not theoretical hit rates, but real speedup.

Usage:
    # Make sure mlx-lm server is NOT running (we load the model directly)
    ~/.mlx-env/bin/python benchmark_wallclock.py
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

# Test prompts - real ISDA analysis tasks
TEST_PROMPTS = [
    {
        "name": "clause_analysis",
        "prompt": (
            "Analyze the following ISDA Master Agreement clause and identify: "
            "(1) the clause type, (2) any non-standard provisions, (3) risk implications.\n\n"
            "Section 5(a)(vi) Cross-Default. If \"Cross-Default\" is specified as applying to "
            "the party, the occurrence or existence of (A) a default, event of default or other "
            "similar condition or event (however described) in respect of such party or any "
            "Credit Support Provider of such party under one or more Specified Indebtedness "
            "in an aggregate amount of not less than the Threshold Amount (as specified in the "
            "Schedule) which has resulted in such Specified Indebtedness becoming, or becoming "
            "capable at such time of being declared, due and payable under such agreements or "
            "instruments, before it would otherwise have been due and payable."
        ),
        "max_tokens": 256,
    },
    {
        "name": "clause_comparison",
        "prompt": (
            "Compare the following two ISDA provisions and explain the practical differences "
            "for a derivatives dealer:\n\n"
            "Provision A - Events of Default Section 5(a)(i): The failure by the party to make, "
            "when due, any payment under this Agreement or delivery under Section 2(a)(i) or "
            "2(e) required to be made by it if such failure is not remedied on or before the "
            "first Local Business Day after notice of such failure is given to the party.\n\n"
            "Provision B - Events of Default Section 5(a)(i) (Modified): The failure by the "
            "party to make, when due, any payment under this Agreement or delivery under "
            "Section 2(a)(i) or 2(e) required to be made by it if such failure is not remedied "
            "on or before the third Local Business Day after notice of such failure is given to "
            "the party.\n\nAnalysis:"
        ),
        "max_tokens": 256,
    },
    {
        "name": "schedule_generation",
        "prompt": (
            "Generate a standard ISDA Schedule Part 1 for a derivatives transaction between "
            "a US bank (Party A) and a UK hedge fund (Party B). Include: Termination Currency, "
            "Cross-Default threshold, Credit Event Upon Merger provisions, and Automatic Early "
            "Termination elections. Use standard market terms.\n\nSchedule to the 2002 ISDA "
            "Master Agreement\n\nPart 1. Termination Provisions.\n\n"
        ),
        "max_tokens": 512,
    },
]


def load_model():
    """Load the Qwen 3.5 9B model."""
    from mlx_lm import load
    print(f"Loading model: {MODEL_ID}")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    return model, tokenizer


def add_isda_context(prompt: str, n_context_tokens: int = 2000) -> str:
    """
    Prepend real ISDA text as context. This is key - the N-gram engine
    benefits from having seen similar text in the prompt.
    """
    sample_file = SAMPLES_DIR / "aerocentury-isda-2002.txt"
    if sample_file.exists():
        context = sample_file.read_text()[:8000]  # ~2000 tokens
        return f"Reference ISDA Agreement excerpt:\n{context}\n\n---\n\nTask: {prompt}"
    return prompt


def benchmark_standard(model, tokenizer, prompt_tokens, max_tokens, warmup=False):
    """Standard one-token-at-a-time generation."""
    from mlx_lm.generate import generate_step

    prompt = mx.array(prompt_tokens, mx.uint32)

    tokens_out = []
    t0 = time.perf_counter()

    for token, logprobs in generate_step(prompt, model, max_tokens=max_tokens):
        tokens_out.append(token)

    elapsed = time.perf_counter() - t0
    return {
        "tokens": len(tokens_out),
        "elapsed": elapsed,
        "tok_per_sec": len(tokens_out) / elapsed if elapsed > 0 else 0,
        "text": tokenizer.decode(tokens_out) if not warmup else "",
    }


def benchmark_ngram(model, tokenizer, prompt_tokens, max_tokens, ngram_n=16,
                    num_draft=32, warmup=False):
    """N-gram speculative generation."""
    from ngram_generate import NgramDrafter, ngram_speculative_generate_step

    prompt = mx.array(prompt_tokens, mx.uint32)
    drafter = NgramDrafter(n=ngram_n)

    tokens_out = []
    drafted_count = 0
    normal_count = 0

    t0 = time.perf_counter()

    for token, logprobs, from_draft in ngram_speculative_generate_step(
        prompt, model, drafter,
        num_draft_tokens=num_draft,
        max_tokens=max_tokens,
    ):
        tokens_out.append(token)
        if from_draft:
            drafted_count += 1
        else:
            normal_count += 1

    elapsed = time.perf_counter() - t0

    return {
        "tokens": len(tokens_out),
        "elapsed": elapsed,
        "tok_per_sec": len(tokens_out) / elapsed if elapsed > 0 else 0,
        "drafted": drafted_count,
        "normal": normal_count,
        "draft_ratio": drafted_count / len(tokens_out) if tokens_out else 0,
        "drafter_summary": drafter.summary(),
        "text": tokenizer.decode(tokens_out) if not warmup else "",
    }


def run_benchmarks():
    model, tokenizer = load_model()

    # Warmup pass
    print("\nWarming up...")
    warmup_tokens = tokenizer.encode("Hello, world!")
    benchmark_standard(model, tokenizer, warmup_tokens, max_tokens=10, warmup=True)
    print("  Standard warmup done")

    # Also warmup ngram path
    benchmark_ngram(model, tokenizer, warmup_tokens, max_tokens=10, warmup=True)
    print("  N-gram warmup done")

    print(f"\n{'=' * 80}")
    print("WALL-CLOCK BENCHMARK: Standard vs N-gram Speculative Decoding")
    print(f"Model: {MODEL_ID}")
    print(f"{'=' * 80}")

    all_results = []

    for test in TEST_PROMPTS:
        name = test["name"]
        max_tokens = test["max_tokens"]

        # Add ISDA context to prompt (simulates real usage with reference text)
        full_prompt = add_isda_context(test["prompt"])
        prompt_tokens = tokenizer.encode(full_prompt)

        print(f"\n{'─' * 80}")
        print(f"Test: {name}")
        print(f"Prompt tokens: {len(prompt_tokens):,}, Max generate: {max_tokens}")
        print(f"{'─' * 80}")

        # Standard generation
        print("\n  Standard generation...")
        std = benchmark_standard(model, tokenizer, prompt_tokens, max_tokens)
        print(f"    {std['tokens']} tokens in {std['elapsed']:.2f}s = {std['tok_per_sec']:.1f} tok/s")

        # N-gram speculative generation at different n values
        for ngram_n in [8, 12, 16, 24]:
            print(f"\n  N-gram speculative (n={ngram_n})...")
            ngram = benchmark_ngram(model, tokenizer, prompt_tokens, max_tokens,
                                     ngram_n=ngram_n, num_draft=32)
            speedup = ngram["tok_per_sec"] / std["tok_per_sec"] if std["tok_per_sec"] > 0 else 0

            print(f"    {ngram['tokens']} tokens in {ngram['elapsed']:.2f}s = {ngram['tok_per_sec']:.1f} tok/s")
            print(f"    Drafted: {ngram['drafted']}/{ngram['tokens']} ({ngram['draft_ratio']:.1%})")
            print(f"    Speedup: {speedup:.2f}x")

            all_results.append({
                "test": name,
                "method": f"ngram_n{ngram_n}",
                "ngram_n": ngram_n,
                "prompt_tokens": len(prompt_tokens),
                "generated_tokens": ngram["tokens"],
                "elapsed": ngram["elapsed"],
                "tok_per_sec": ngram["tok_per_sec"],
                "drafted": ngram["drafted"],
                "draft_ratio": ngram["draft_ratio"],
                "speedup_vs_standard": speedup,
            })

        all_results.append({
            "test": name,
            "method": "standard",
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": std["tokens"],
            "elapsed": std["elapsed"],
            "tok_per_sec": std["tok_per_sec"],
        })

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  {'Test':<25} {'Method':<15} {'Tok/s':>8} {'Drafted':>10} {'Speedup':>10}")
    print(f"  {'─'*25} {'─'*15} {'─'*8} {'─'*10} {'─'*10}")

    for r in all_results:
        drafted = f"{r.get('draft_ratio', 0):.0%}" if "draft_ratio" in r else "-"
        speedup = f"{r.get('speedup_vs_standard', 1.0):.2f}x" if "speedup_vs_standard" in r else "baseline"
        print(f"  {r['test']:<25} {r['method']:<15} {r['tok_per_sec']:>7.1f} {drafted:>10} {speedup:>10}")

    # Save
    out_path = Path(__file__).parent / "wallclock_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_benchmarks()
