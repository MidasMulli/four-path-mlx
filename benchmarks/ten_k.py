#!/usr/bin/env python3
"""
10-K Filing Benchmark: Four-Path Speculative Decoding on SEC Filings

Tests the four-path architecture on bank 10-K filings — highly structured
regulatory documents with massive boilerplate overlap across companies.

Two phases:
  1. N-gram hit rate analysis (cross-document pattern transfer)
  2. Full four-path generation benchmark

Usage:
    ~/.mlx-env/bin/python benchmark_10k.py
"""

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))
from ngram_predict import NgramPredictor, EMPTY

MODEL_ID = os.path.expanduser("~/models/Qwen3.5-9B-MLX-4bit-MTP")
SAMPLES_10K = Path(__file__).parent / "10k-samples"


def load_tokenizer():
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {MODEL_ID}")
    return AutoTokenizer.from_pretrained(MODEL_ID)


def load_model():
    from mlx_lm import load
    print(f"Loading model: {MODEL_ID}")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    return model, tokenizer


def check_ane():
    try:
        sys.path.insert(0, "/Users/midas/Desktop/cowork/orion-ane/memory")
        from ane_server import ANEClient
        ANEClient().ping()
        return True
    except:
        return False


# ── Phase 1: N-gram Hit Rate Analysis ──────────────────────────

def phase1_ngram_analysis(tokenizer):
    print("\n" + "=" * 90)
    print("PHASE 1: N-gram Hit Rate Analysis on 10-K Filings")
    print("=" * 90)

    files = sorted(SAMPLES_10K.glob("*.txt"))
    if not files:
        print("  ERROR: No 10-K files found")
        return

    # Tokenize all files (truncate to keep manageable — first 50K tokens each)
    MAX_TOKENS = 50000
    all_docs = []
    for f in files:
        text = f.read_text()
        tokens = tokenizer.encode(text)[:MAX_TOKENS]
        all_docs.append((f.stem, tokens))
        print(f"  {f.stem}: {len(tokens):,} tokens (from {len(text):,} chars)")

    total = sum(len(t) for _, t in all_docs)
    print(f"  Total: {total:,} tokens across {len(all_docs)} filings\n")

    # Test 1: Single-document repetition at various n-gram sizes
    print("  ── Single-Document Accuracy (internal repetition) ──")
    print(f"  {'Doc':<15} {'n=4':>8} {'n=6':>8} {'n=8':>8} {'n=12':>8} {'n=16':>8}")
    print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for name, tokens in all_docs:
        results = []
        for n in [4, 6, 8, 12, 16]:
            pred = NgramPredictor(n=n)
            correct = 0
            total_lookups = 0
            for i in range(n, len(tokens)):
                total_lookups += 1
                predicted = pred.predict(tokens[:i])
                if predicted != EMPTY and predicted == tokens[i]:
                    correct += 1
                if i >= n:
                    pred.feed(tokens[i-n:i+1])
            acc = correct / total_lookups if total_lookups else 0
            results.append(acc)
        print(f"  {name:<15} {results[0]:>7.1%} {results[1]:>7.1%} {results[2]:>7.1%} {results[3]:>7.1%} {results[4]:>7.1%}")

    # Test 2: Cross-document prediction (the big one)
    print(f"\n  ── Cross-Document Prediction (boilerplate transfer) ──")
    print(f"  Train on 3 banks, predict on the 4th\n")

    for target_idx in range(len(all_docs)):
        target_name, target_tokens = all_docs[target_idx]
        train_docs = [d for i, d in enumerate(all_docs) if i != target_idx]

        for n in [4, 8, 12]:
            pred = NgramPredictor(n=n)
            # Feed training docs
            for _, tokens in train_docs:
                pred.feed(tokens)

            # Predict on target
            correct = 0
            total_lookups = 0
            for i in range(n, len(target_tokens)):
                total_lookups += 1
                predicted = pred.predict(target_tokens[:i])
                if predicted != EMPTY and predicted == target_tokens[i]:
                    correct += 1
            acc = correct / total_lookups if total_lookups else 0
            if n == 4:
                print(f"  Target: {target_name:<12}  n=4: {acc:.1%}", end="")
            elif n == 8:
                print(f"  n=8: {acc:.1%}", end="")
            else:
                print(f"  n=12: {acc:.1%}")

    # Test 3: Compare ISDA vs 10-K cross-doc rates
    print(f"\n  ── Domain Comparison (cross-doc accuracy at n=8) ──")
    # 10-K rate: train on 3, predict on 4th (use first target)
    pred_10k = NgramPredictor(n=8)
    for _, tokens in all_docs[1:]:
        pred_10k.feed(tokens)
    correct_10k = sum(1 for i in range(8, len(all_docs[0][1]))
                      if pred_10k.predict(all_docs[0][1][:i]) == all_docs[0][1][i])
    rate_10k = correct_10k / (len(all_docs[0][1]) - 8)

    # ISDA rate from prior benchmark
    isda_rate = 0.287  # from benchmark_results.json, n=8 cross-doc

    print(f"  10-K filings (bank regulatory): {rate_10k:.1%}")
    print(f"  ISDA agreements (legal):        {isda_rate:.1%}")
    if rate_10k > isda_rate:
        print(f"  → 10-Ks are {rate_10k/isda_rate:.1f}x more repetitive than ISDA!")
    else:
        print(f"  → ISDA is {isda_rate/rate_10k:.1f}x more repetitive than 10-Ks")


# ── Phase 2: Four-Path Generation Benchmark ────────────────────

def phase2_four_path(model, tokenizer, ane_ok):
    print("\n\n" + "=" * 90)
    print("PHASE 2: Four-Path Generation on 10-K Analysis Tasks")
    print("=" * 90)

    # Load first 10-K as context
    jpm_text = (SAMPLES_10K / "jpm-10k.txt").read_text()[:8000]

    prompts = [
        {
            "name": "risk_analysis",
            "desc": "Risk factor analysis (analytical)",
            "prompt": (
                f"Reference 10-K excerpt:\n{jpm_text}\n\n---\n\n"
                "Identify and categorize the top 5 risk factors disclosed in this 10-K filing. "
                "For each risk, state: (1) the risk category (credit, market, operational, regulatory, "
                "liquidity), (2) specific exposure described, (3) potential financial impact."
            ),
            "max_tokens": 512,
        },
        {
            "name": "reg_disclosure",
            "desc": "Regulatory disclosure drafting (boilerplate)",
            "prompt": (
                f"Reference 10-K excerpt:\n{jpm_text}\n\n---\n\n"
                "Draft the regulatory capital disclosure section for a G-SIB bank's 10-K filing. "
                "Include: Basel III CET1 ratio, Tier 1 capital ratio, Total capital ratio, "
                "Supplementary Leverage Ratio, TLAC requirements, and stress capital buffer. "
                "Use standard SEC disclosure language.\n\n"
                "Regulatory Capital\n\n"
            ),
            "max_tokens": 512,
        },
        {
            "name": "md_and_a",
            "desc": "MD&A section drafting (mixed)",
            "prompt": (
                f"Reference 10-K excerpt:\n{jpm_text}\n\n---\n\n"
                "Draft the Management's Discussion and Analysis section covering net interest "
                "income performance for a major US bank. Include: NII trends, rate sensitivity, "
                "deposit mix shifts, loan growth, and forward-looking statements with appropriate "
                "safe harbor language.\n\n"
                "Management's Discussion and Analysis of Financial Condition and Results of Operations\n\n"
                "Net Interest Income\n\n"
            ),
            "max_tokens": 512,
        },
    ]

    from mlx_lm.generate import generate_step

    results = []

    for test in prompts:
        name = test["name"]
        desc = test["desc"]
        max_tokens = test["max_tokens"]
        prompt_text = test["prompt"]
        prompt_tokens = tokenizer.encode(prompt_text)

        print(f"\n{'─' * 90}")
        print(f"  {name} — {desc}")
        print(f"  Prompt: {len(prompt_tokens):,} tokens → Generate: {max_tokens}")
        print(f"{'─' * 90}")

        # Standard
        print(f"\n  Standard...")
        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        out = []
        t0 = time.perf_counter()
        for tok, _ in generate_step(prompt_mx, model, max_tokens=max_tokens):
            out.append(tok)
        baseline = len(out) / (time.perf_counter() - t0)
        print(f"    {baseline:.1f} tok/s")

        # Four-path
        print(f"  Four-Path...")
        from four_path import FourPathDrafter, four_path_generate_step
        from three_path import ane_generate_async

        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        drafter = FourPathDrafter(ngram_n=8)

        ane_lookahead = None
        if ane_ok:
            task_start = prompt_text.find("---\n\n")
            ane_prompt = prompt_text[task_start + 5:] if task_start >= 0 else prompt_text[-500:]
            ane_lookahead = ane_generate_async(ane_prompt, max_tokens=max_tokens)
            drafter.set_ane_lookahead(ane_lookahead)

        out = []
        sources = {"ngram": 0, "ane": 0, "mtp": 0, "gpu": 0}
        t0 = time.perf_counter()
        for tok, lp, from_draft, source in four_path_generate_step(
            prompt_mx, model, drafter, tokenizer=tokenizer,
            num_draft_tokens=32, max_tokens=max_tokens,
        ):
            out.append(tok)
            sources[source] = sources.get(source, 0) + 1
        elapsed = time.perf_counter() - t0
        tps = len(out) / elapsed
        speedup = tps / baseline

        if ane_lookahead:
            ane_lookahead.wait(5)

        src = sources
        drafted = src.get("ngram", 0) + src.get("ane", 0) + src.get("mtp", 0)
        print(f"    {tps:.1f} tok/s ({speedup:.2f}x)")
        print(f"    Sources: N-gram={src['ngram']}, MTP={src['mtp']}, ANE={src['ane']}, GPU={src['gpu']}")

        results.append({
            "test": name, "desc": desc,
            "standard_tps": baseline,
            "four_path_tps": tps,
            "speedup": speedup,
            "sources": sources,
        })

    # Summary
    print(f"\n\n{'=' * 90}")
    print("10-K FILING RESULTS")
    print(f"{'=' * 90}")
    print(f"\n  {'Test':<20} {'Standard':>10} {'4-Path':>10} {'Speedup':>10}  │ {'N-gram':>8} {'MTP':>8} {'ANE':>8} {'GPU':>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}  │ {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for r in results:
        s = r["sources"]
        print(f"  {r['test']:<20} {r['standard_tps']:>9.1f} {r['four_path_tps']:>9.1f} {r['speedup']:>9.2f}x  │ "
              f"{s.get('ngram',0):>8} {s.get('mtp',0):>8} {s.get('ane',0):>8} {s.get('gpu',0):>8}")

    out_path = Path(__file__).parent / "10k_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


def run():
    tokenizer = load_tokenizer()
    phase1_ngram_analysis(tokenizer)

    # Phase 2 needs the full model
    print("\n  Loading full model for generation benchmark...")
    model, tokenizer = load_model()
    ane_ok = check_ane()
    phase2_four_path(model, tokenizer, ane_ok)


if __name__ == "__main__":
    run()
