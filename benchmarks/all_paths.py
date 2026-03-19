#!/usr/bin/env python3
"""
All-Paths Benchmark: Every speculative decoding method head-to-head

  1. Standard (GPU only)
  2. N-gram (CPU + GPU)
  3. MTP (GPU, native MTP head)
  4. Three-Path (CPU N-gram + ANE lookahead + GPU)

All on the same ISDA prompts with the same model.

Usage:
    # ANE server running, MLX server NOT running
    ~/.mlx-env/bin/python benchmark_all_paths.py
"""

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))

# Use MTP model
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
    has_mtp = hasattr(model, 'mtp_forward')
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s, MTP: {has_mtp}")
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


# ── Benchmark Methods ──────────────────────────────────────────

def bench_standard(model, tokenizer, prompt_tokens, max_tokens):
    from mlx_lm.generate import generate_step
    prompt = mx.array(prompt_tokens, mx.uint32)
    tokens_out = []
    t0 = time.perf_counter()
    for token, _ in generate_step(prompt, model, max_tokens=max_tokens):
        tokens_out.append(token)
    elapsed = time.perf_counter() - t0
    return {"tokens": len(tokens_out), "elapsed": elapsed,
            "tok_per_sec": len(tokens_out) / elapsed}


def bench_mtp(model, tokenizer, prompt_tokens, max_tokens):
    from mlx_lm.generate import stream_generate
    prompt = mx.array(prompt_tokens, mx.uint32)
    tokens_out = []
    drafted = 0
    t0 = time.perf_counter()
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, mtp=True):
        tokens_out.append(resp.token)
        if hasattr(resp, 'from_draft') and resp.from_draft:
            drafted += 1
    elapsed = time.perf_counter() - t0
    return {"tokens": len(tokens_out), "elapsed": elapsed,
            "tok_per_sec": len(tokens_out) / elapsed,
            "drafted": drafted,
            "draft_ratio": drafted / len(tokens_out) if tokens_out else 0}


def bench_ngram(model, tokenizer, prompt_tokens, max_tokens, ngram_n=8):
    from ngram_generate import NgramDrafter, ngram_speculative_generate_step
    prompt = mx.array(prompt_tokens, mx.uint32)
    drafter = NgramDrafter(n=ngram_n)
    tokens_out = []
    drafted = 0
    t0 = time.perf_counter()
    for token, _, from_draft in ngram_speculative_generate_step(
        prompt, model, drafter, num_draft_tokens=32, max_tokens=max_tokens,
    ):
        tokens_out.append(token)
        if from_draft:
            drafted += 1
    elapsed = time.perf_counter() - t0
    return {"tokens": len(tokens_out), "elapsed": elapsed,
            "tok_per_sec": len(tokens_out) / elapsed,
            "drafted": drafted,
            "draft_ratio": drafted / len(tokens_out) if tokens_out else 0}


def bench_three_path(model, tokenizer, prompt_tokens, prompt_text, max_tokens, ngram_n=8):
    from three_path import ThreePathDrafter, three_path_generate_step, ane_generate_async
    prompt = mx.array(prompt_tokens, mx.uint32)
    drafter = ThreePathDrafter(ngram_n=ngram_n)

    task_start = prompt_text.find("---\n\n")
    ane_prompt = prompt_text[task_start + 5:] if task_start >= 0 else prompt_text[-500:]
    ane_lookahead = ane_generate_async(ane_prompt, max_tokens=max_tokens)
    drafter.set_ane_lookahead(ane_lookahead)

    tokens_out = []
    sources = {"ngram": 0, "ane": 0, "gpu": 0}
    t0 = time.perf_counter()
    for token, _, from_draft, source in three_path_generate_step(
        prompt, model, drafter, tokenizer=tokenizer,
        num_draft_tokens=32, max_tokens=max_tokens,
    ):
        tokens_out.append(token)
        sources[source] = sources.get(source, 0) + 1
    elapsed = time.perf_counter() - t0

    if ane_lookahead:
        ane_lookahead.wait(5)

    drafted = sources.get("ngram", 0) + sources.get("ane", 0)
    return {"tokens": len(tokens_out), "elapsed": elapsed,
            "tok_per_sec": len(tokens_out) / elapsed,
            "drafted": drafted,
            "draft_ratio": drafted / len(tokens_out) if tokens_out else 0,
            "sources": sources,
            "summary": drafter.summary()}


# ── Main ──────────────────────────────────────────────────────

def run():
    model, tokenizer = load_model()
    ane_ok = check_ane()

    # Warmup
    print("\nWarming up...")
    warmup = mx.array(tokenizer.encode("Hello"), mx.uint32)
    from mlx_lm.generate import generate_step
    for _ in generate_step(warmup, model, max_tokens=5):
        pass
    print("  Done\n")

    print("=" * 95)
    print("ALL-PATHS SPECULATIVE DECODING BENCHMARK")
    print("  GPU: Qwen3.5-9B-MLX-4bit + MTP  |  ANE: Qwen3-1.7B CoreML  |  CPU: N-gram (n=8)")
    print("=" * 95)

    all_results = []

    for test in TEST_PROMPTS:
        name = test["name"]
        desc = test["desc"]
        max_tokens = test["max_tokens"]
        prompt_text = test["prompt"]
        prompt_tokens = tokenizer.encode(prompt_text)

        print(f"\n{'─' * 95}")
        print(f"  {name} — {desc}")
        print(f"  Prompt: {len(prompt_tokens):,} tokens → Generate: {max_tokens}")
        print(f"{'─' * 95}")

        result = {"test": name, "desc": desc, "prompt_tokens": len(prompt_tokens)}

        # 1. Standard
        print(f"\n  [1/4] Standard (GPU only)...")
        std = bench_standard(model, tokenizer, prompt_tokens, max_tokens)
        print(f"        {std['tok_per_sec']:.1f} tok/s")
        result["standard"] = std["tok_per_sec"]
        baseline = std["tok_per_sec"]

        # 2. MTP
        print(f"  [2/4] MTP (GPU + MTP head)...")
        mtp = bench_mtp(model, tokenizer, prompt_tokens, max_tokens)
        print(f"        {mtp['tok_per_sec']:.1f} tok/s ({mtp['tok_per_sec']/baseline:.2f}x) "
              f"— {mtp['draft_ratio']:.0%} from MTP head")
        result["mtp"] = {"tok_per_sec": mtp["tok_per_sec"],
                         "speedup": mtp["tok_per_sec"] / baseline,
                         "draft_ratio": mtp["draft_ratio"]}

        # 3. N-gram
        print(f"  [3/4] N-gram (CPU + GPU)...")
        ng = bench_ngram(model, tokenizer, prompt_tokens, max_tokens, ngram_n=8)
        print(f"        {ng['tok_per_sec']:.1f} tok/s ({ng['tok_per_sec']/baseline:.2f}x) "
              f"— {ng['draft_ratio']:.0%} from N-gram")
        result["ngram"] = {"tok_per_sec": ng["tok_per_sec"],
                           "speedup": ng["tok_per_sec"] / baseline,
                           "draft_ratio": ng["draft_ratio"]}

        # 4. Three-path
        if ane_ok:
            print(f"  [4/4] Three-Path (CPU + ANE + GPU)...")
            tp = bench_three_path(model, tokenizer, prompt_tokens, prompt_text, max_tokens)
            print(f"        {tp['tok_per_sec']:.1f} tok/s ({tp['tok_per_sec']/baseline:.2f}x) "
                  f"— N-gram={tp['sources'].get('ngram',0)}, ANE={tp['sources'].get('ane',0)}, GPU={tp['sources'].get('gpu',0)}")
            result["three_path"] = {"tok_per_sec": tp["tok_per_sec"],
                                    "speedup": tp["tok_per_sec"] / baseline,
                                    "sources": tp["sources"]}

        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n\n{'=' * 95}")
    print("SUMMARY — All Paths Compared")
    print(f"{'=' * 95}")
    print(f"\n  {'Test':<20} {'Standard':>10} {'MTP':>10} {'N-gram':>10} {'3-Path':>10}  │ {'Best':>10} {'Speedup':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}  │ {'─'*10} {'─'*10}")

    for r in all_results:
        std_tps = r["standard"]
        mtp_tps = r.get("mtp", {}).get("tok_per_sec", 0)
        ng_tps = r.get("ngram", {}).get("tok_per_sec", 0)
        tp_tps = r.get("three_path", {}).get("tok_per_sec", 0)

        methods = {"Standard": std_tps, "MTP": mtp_tps, "N-gram": ng_tps, "3-Path": tp_tps}
        best_name = max(methods, key=methods.get)
        best_tps = methods[best_name]
        best_speedup = best_tps / std_tps

        print(f"  {r['test']:<20} {std_tps:>9.1f} {mtp_tps:>9.1f} {ng_tps:>9.1f} {tp_tps:>9.1f}  │ {best_name:>10} {best_speedup:>9.2f}x")

    print(f"\n  Token sources:")
    print(f"    CPU  N-gram hash — verbatim boilerplate, nanosecond lookup")
    print(f"    GPU  MTP head    — model's own hidden-state prediction, free with forward pass")
    print(f"    ANE  1.7B neural — semantic lookahead, parallel generation")
    print(f"    GPU  Backbone    — 9B verification + novel tokens")

    print(f"\n  Note: MTP and N-gram use different generation loops (MTP: 1 draft/round,")
    print(f"        N-gram: chain drafts). Four-path combined = next build.")

    out_path = Path(__file__).parent / "all_paths_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run()
