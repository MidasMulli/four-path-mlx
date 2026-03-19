#!/usr/bin/env python3
"""
Four-Path Benchmark: The full heterogeneous speculative decoding stack.

  CPU  N-gram  → pattern chains
  ANE  1.7B    → neural lookahead
  MTP  head    → hidden-state prediction (free)
  GPU  9B      → verification + novel tokens

Usage:
    ~/.mlx-env/bin/python benchmark_four_path.py
"""

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))

MODEL_ID = os.path.expanduser("~/models/Qwen3.5-9B-MLX-4bit-MTP")
SAMPLES_DIR = Path(__file__).parent.parent / "isda-classifier" / "samples"


def load_isda_context(max_chars=8000):
    f = SAMPLES_DIR / "aerocentury-isda-2002.txt"
    return f.read_text()[:max_chars] if f.exists() else ""

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
        "desc": "Boilerplate generation",
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
        "desc": "CSA drafting (mixed)",
        "prompt": (
            f"Reference ISDA Agreement excerpt:\n{ISDA_CONTEXT}\n\n---\n\n"
            "Draft a Credit Support Annex (CSA) Paragraph 13 for a bilateral derivatives "
            "relationship. Party A is a US G-SIB bank, Party B is a European pension fund.\n\n"
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
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s, MTP: {hasattr(model, 'mtp_forward')}")
    return model, tokenizer


def check_ane():
    try:
        sys.path.insert(0, "/Users/midas/Desktop/cowork/orion-ane/memory")
        from ane_server import ANEClient
        ANEClient().ping()
        print("  ANE server: alive")
        return True
    except:
        print("  ANE server: not available")
        return False


def bench_standard(model, tokenizer, prompt_tokens, max_tokens):
    from mlx_lm.generate import generate_step
    prompt = mx.array(prompt_tokens, mx.uint32)
    out = []
    t0 = time.perf_counter()
    for tok, _ in generate_step(prompt, model, max_tokens=max_tokens):
        out.append(tok)
    elapsed = time.perf_counter() - t0
    return {"tok_per_sec": len(out) / elapsed, "tokens": len(out)}


def bench_four_path(model, tokenizer, prompt_tokens, prompt_text, max_tokens, ngram_n=8):
    from four_path import FourPathDrafter, four_path_generate_step
    from three_path import ane_generate_async

    prompt = mx.array(prompt_tokens, mx.uint32)
    drafter = FourPathDrafter(ngram_n=ngram_n)

    # Launch ANE
    task_start = prompt_text.find("---\n\n")
    ane_prompt = prompt_text[task_start + 5:] if task_start >= 0 else prompt_text[-500:]
    ane_lookahead = ane_generate_async(ane_prompt, max_tokens=max_tokens)
    drafter.set_ane_lookahead(ane_lookahead)

    out = []
    sources = {"ngram": 0, "ane": 0, "mtp": 0, "gpu": 0}

    t0 = time.perf_counter()
    for tok, lp, from_draft, source in four_path_generate_step(
        prompt, model, drafter, tokenizer=tokenizer,
        num_draft_tokens=32, max_tokens=max_tokens,
    ):
        out.append(tok)
        sources[source] = sources.get(source, 0) + 1
    elapsed = time.perf_counter() - t0

    if ane_lookahead:
        ane_lookahead.wait(5)

    return {
        "tok_per_sec": len(out) / elapsed,
        "tokens": len(out),
        "sources": sources,
        "summary": drafter.summary(),
    }


def run():
    model, tokenizer = load_model()
    ane_ok = check_ane()

    print("\nWarming up...")
    warmup = mx.array(tokenizer.encode("Hello"), mx.uint32)
    from mlx_lm.generate import generate_step
    for _ in generate_step(warmup, model, max_tokens=5):
        pass
    print("  Done\n")

    print("=" * 100)
    print("FOUR-PATH SPECULATIVE DECODING BENCHMARK")
    print("  N-gram (CPU) + MTP (GPU head) + ANE (1.7B neural) + GPU (9B backbone)")
    print("=" * 100)

    all_results = []

    for test in TEST_PROMPTS:
        name = test["name"]
        desc = test["desc"]
        max_tokens = test["max_tokens"]
        prompt_text = test["prompt"]
        prompt_tokens = tokenizer.encode(prompt_text)

        print(f"\n{'─' * 100}")
        print(f"  {name} — {desc}")
        print(f"  Prompt: {len(prompt_tokens):,} tokens → Generate: {max_tokens}")
        print(f"{'─' * 100}")

        # Standard baseline
        print(f"\n  Standard (GPU only)...")
        std = bench_standard(model, tokenizer, prompt_tokens, max_tokens)
        baseline = std["tok_per_sec"]
        print(f"    {baseline:.1f} tok/s")

        # Four-path
        print(f"  Four-Path (CPU + MTP + ANE + GPU)...")
        fp = bench_four_path(model, tokenizer, prompt_tokens, prompt_text, max_tokens)
        speedup = fp["tok_per_sec"] / baseline
        src = fp["sources"]
        total_drafted = src.get("ngram", 0) + src.get("ane", 0) + src.get("mtp", 0)
        print(f"    {fp['tok_per_sec']:.1f} tok/s ({speedup:.2f}x)")
        print(f"    Sources: N-gram={src.get('ngram',0)}, MTP={src.get('mtp',0)}, "
              f"ANE={src.get('ane',0)}, GPU={src.get('gpu',0)}")
        print(f"    Draft ratio: {total_drafted}/{fp['tokens']} ({total_drafted/fp['tokens']:.0%})")

        all_results.append({
            "test": name, "desc": desc,
            "standard_tps": baseline,
            "four_path_tps": fp["tok_per_sec"],
            "speedup": speedup,
            "sources": src,
            "summary": fp["summary"],
        })

    # Summary
    print(f"\n\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"\n  {'Test':<20} {'Standard':>10} {'4-Path':>10} {'Speedup':>10}  │ {'N-gram':>8} {'MTP':>8} {'ANE':>8} {'GPU':>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}  │ {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for r in all_results:
        src = r["sources"]
        print(f"  {r['test']:<20} {r['standard_tps']:>9.1f} {r['four_path_tps']:>9.1f} {r['speedup']:>9.2f}x  │ "
              f"{src.get('ngram',0):>8} {src.get('mtp',0):>8} {src.get('ane',0):>8} {src.get('gpu',0):>8}")

    print(f"\n  Four processors, four prediction methods, one verification target.")
    print(f"  Each catches what the others miss.")

    out_path = Path(__file__).parent / "four_path_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run()
