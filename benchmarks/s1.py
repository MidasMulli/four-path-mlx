#!/usr/bin/env python3
"""
S-1 Pre-IPO Filing Benchmark: Four-Path on Registration Statements

S-1s are the most formulaic SEC filings — risk factors, use of proceeds,
dilution sections are practically templated across tech IPOs.

Usage:
    ~/.mlx-env/bin/python benchmark_s1.py
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
SAMPLES_S1 = Path(__file__).parent / "s1-samples"
SAMPLES_10K = Path(__file__).parent / "10k-samples"
SAMPLES_ISDA = Path(__file__).parent.parent / "isda-classifier" / "samples"


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


def run():
    model, tokenizer = load_model()
    ane_ok = check_ane()
    print(f"  ANE: {'alive' if ane_ok else 'not available'}")

    # ── Phase 1: Cross-document N-gram analysis ──────────────────
    print(f"\n{'=' * 90}")
    print("PHASE 1: Cross-Document N-gram Repetition — S-1 vs 10-K vs ISDA")
    print(f"{'=' * 90}")

    MAX_TOKENS = 50000

    domains = {}
    for label, sample_dir, glob_pat in [
        ("S-1", SAMPLES_S1, "*.txt"),
        ("10-K", SAMPLES_10K, "*.txt"),
        ("ISDA", SAMPLES_ISDA, "*.txt"),
    ]:
        files = sorted(sample_dir.glob(glob_pat))
        docs = []
        for f in files:
            tokens = tokenizer.encode(f.read_text())[:MAX_TOKENS]
            docs.append((f.stem, tokens))
        domains[label] = docs
        total = sum(len(t) for _, t in docs)
        print(f"  {label}: {len(docs)} docs, {total:,} tokens")

    print(f"\n  Cross-document accuracy (train on all-but-one, predict on held-out):")
    print(f"  {'Domain':<10} {'n=4':>8} {'n=8':>8} {'n=12':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8}")

    for label, docs in domains.items():
        if len(docs) < 2:
            continue
        results = []
        for n in [4, 8, 12]:
            # Average across all leave-one-out splits
            accs = []
            for target_idx in range(len(docs)):
                pred = NgramPredictor(n=n)
                for i, (_, tokens) in enumerate(docs):
                    if i != target_idx:
                        pred.feed(tokens)
                target_tokens = docs[target_idx][1]
                correct = 0
                lookups = 0
                for j in range(n, min(len(target_tokens), 20000)):  # cap for speed
                    lookups += 1
                    p = pred.predict(target_tokens[:j])
                    if p != EMPTY and p == target_tokens[j]:
                        correct += 1
                accs.append(correct / lookups if lookups else 0)
            results.append(sum(accs) / len(accs))
        print(f"  {label:<10} {results[0]:>7.1%} {results[1]:>7.1%} {results[2]:>7.1%}")

    # ── Phase 2: Four-Path Generation ────────────────────────────
    print(f"\n{'=' * 90}")
    print("PHASE 2: Four-Path Generation on S-1 Tasks")
    print(f"{'=' * 90}")

    # Use Reddit S-1 as reference context
    s1_files = sorted(SAMPLES_S1.glob("*.txt"))
    ref_text = s1_files[0].read_text()[:8000] if s1_files else ""
    ref_name = s1_files[0].stem if s1_files else "unknown"

    prompts = [
        {
            "name": "risk_factors",
            "desc": "Risk factor drafting (highest boilerplate)",
            "prompt": (
                f"Reference S-1 excerpt ({ref_name}):\n{ref_text}\n\n---\n\n"
                "Draft the Risk Factors section for a pre-IPO technology company's S-1 filing. "
                "Include risks related to: revenue concentration, competition, regulatory "
                "environment, data privacy, intellectual property, key personnel, and market "
                "conditions. Use standard SEC disclosure language.\n\n"
                "RISK FACTORS\n\n"
                "Investing in our Class A common stock involves a high degree of risk. You should "
                "carefully consider the risks and uncertainties described below, together with all "
                "of the other information in this prospectus, before deciding to invest in our "
                "Class A common stock.\n\n"
            ),
            "max_tokens": 512,
        },
        {
            "name": "use_of_proceeds",
            "desc": "Use of Proceeds (very formulaic)",
            "prompt": (
                f"Reference S-1 excerpt ({ref_name}):\n{ref_text}\n\n---\n\n"
                "Draft the Use of Proceeds section for a technology company's S-1 filing. "
                "The company is raising approximately $500 million in its initial public offering.\n\n"
                "USE OF PROCEEDS\n\n"
                "We estimate that the net proceeds to us from this offering will be approximately "
                "$465.0 million, or approximately $535.3 million if the underwriters exercise their "
                "option to purchase additional shares in full, based on an assumed initial public "
                "offering price of $"
            ),
            "max_tokens": 512,
        },
        {
            "name": "dilution",
            "desc": "Dilution section (mathematical/formulaic)",
            "prompt": (
                f"Reference S-1 excerpt ({ref_name}):\n{ref_text}\n\n---\n\n"
                "Draft the Dilution section for a technology company's S-1 filing.\n\n"
                "DILUTION\n\n"
                "If you invest in our Class A common stock in this offering, your ownership "
                "interest will be immediately diluted to the extent of the difference between "
                "the initial public offering price per share and the pro forma as adjusted net "
                "tangible book value per share after this offering.\n\n"
            ),
            "max_tokens": 512,
        },
    ]

    from mlx_lm.generate import generate_step
    from four_path import FourPathDrafter, four_path_generate_step
    from three_path import ane_generate_async

    all_results = []

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
        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        out = []
        t0 = time.perf_counter()
        for tok, _ in generate_step(prompt_mx, model, max_tokens=max_tokens):
            out.append(tok)
        baseline = len(out) / (time.perf_counter() - t0)
        print(f"  Standard: {baseline:.1f} tok/s")

        # Four-path
        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        drafter = FourPathDrafter(ngram_n=8)
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

        if ane_ok and ane_lookahead:
            ane_lookahead.wait(5)

        print(f"  4-Path:   {tps:.1f} tok/s ({speedup:.2f}x)")
        print(f"  Sources:  N-gram={sources['ngram']}, MTP={sources['mtp']}, ANE={sources['ane']}, GPU={sources['gpu']}")

        all_results.append({
            "test": name, "desc": desc,
            "standard_tps": baseline,
            "four_path_tps": tps,
            "speedup": speedup,
            "sources": sources,
        })

    # Summary
    print(f"\n\n{'=' * 90}")
    print("S-1 FILING RESULTS")
    print(f"{'=' * 90}")
    print(f"\n  {'Test':<20} {'Standard':>10} {'4-Path':>10} {'Speedup':>10}  │ {'N-gram':>8} {'MTP':>8} {'ANE':>8} {'GPU':>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}  │ {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for r in all_results:
        s = r["sources"]
        print(f"  {r['test']:<20} {r['standard_tps']:>9.1f} {r['four_path_tps']:>9.1f} {r['speedup']:>9.2f}x  │ "
              f"{s.get('ngram',0):>8} {s.get('mtp',0):>8} {s.get('ane',0):>8} {s.get('gpu',0):>8}")

    # Cross-domain comparison
    print(f"\n  ── Domain Comparison (all four-path results) ──")
    print(f"  {'Domain':<12} {'Best Speedup':>14} {'Task':>30}")
    print(f"  {'─'*12} {'─'*14} {'─'*30}")
    print(f"  {'ISDA':<12} {'2.10x':>14} {'CSA drafting':>30}")
    print(f"  {'10-K':<12} {'4.00x':>14} {'MD&A drafting':>30}")

    best_s1 = max(all_results, key=lambda r: r["speedup"])
    print(f"  {'S-1':<12} {best_s1['speedup']:>13.2f}x {best_s1['test']:>30}")

    out_path = Path(__file__).parent / "s1_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run()
