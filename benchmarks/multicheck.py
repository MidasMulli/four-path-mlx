#!/usr/bin/env python3
"""
Multi-Check Benchmark: Does checking alternative sources at rejection
points improve throughput?

Key metric: multicheck_recoveries — how often did an alternative source
have the right token that the primary source missed?
"""

import json, os, sys, time
from pathlib import Path
import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))

MODEL_ID = os.path.expanduser("~/models/Qwen3.5-9B-MLX-4bit-MTP")
SAMPLES_10K = Path(__file__).parent / "10k-samples"
SAMPLES_S1 = Path(__file__).parent / "s1-samples"
SAMPLES_ISDA = Path(__file__).parent.parent / "isda-classifier" / "samples"


def load_model():
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
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
    print(f"Model loaded, ANE: {'alive' if ane_ok else 'unavailable'}")

    from mlx_lm.generate import generate_step
    warmup = mx.array(tokenizer.encode("Hello"), mx.uint32)
    for _ in generate_step(warmup, model, max_tokens=5): pass

    ref_10k = (SAMPLES_10K / "jpm-10k.txt").read_text()[:8000]
    ref_s1 = sorted(SAMPLES_S1.glob("*.txt"))[0].read_text()[:8000]
    ref_isda = sorted(SAMPLES_ISDA.glob("*.txt"))[0].read_text()[:8000]

    tests = [
        ("10k_reg", "Regulatory disclosure", f"Reference:\n{ref_10k}\n\n---\n\nDraft regulatory capital disclosure for a G-SIB.\n\nRegulatory Capital\n\n", 1024),
        ("s1_risk", "S-1 risk factors", f"Reference:\n{ref_s1}\n\n---\n\nDraft Risk Factors for a tech company S-1.\n\nRISK FACTORS\n\n", 1024),
        ("isda_csa", "ISDA CSA draft", f"Reference:\n{ref_isda}\n\n---\n\nDraft CSA Paragraph 13.\n\nParagraph 13. Elections and Variables.\n\n", 1024),
        ("analysis", "Risk analysis", f"Reference:\n{ref_10k}\n\n---\n\nAs a credit analyst, evaluate this bank's risk profile and recommend mitigations.\n\n", 512),
        ("code", "Code generation", "Write a Python SEC EDGAR parser that extracts risk factors and stores them in SQLite.", 512),
    ]

    print(f"\n{'=' * 110}")
    print("MULTI-CHECK vs STATIC FOUR-PATH")
    print(f"{'=' * 110}")

    results = []

    for name, desc, prompt_text, max_tokens in tests:
        prompt_tokens = tokenizer.encode(prompt_text)
        print(f"\n{'─' * 110}")
        print(f"  {name} — {desc} ({len(prompt_tokens):,} prompt → {max_tokens} gen)")
        print(f"{'─' * 110}")

        # Standard
        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        out = []
        t0 = time.perf_counter()
        for tok, _ in generate_step(prompt_mx, model, max_tokens=max_tokens):
            out.append(tok)
        baseline = len(out) / (time.perf_counter() - t0)

        # Static four-path
        from four_path import FourPathDrafter, four_path_generate_step
        from three_path import ane_generate_async

        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        static_d = FourPathDrafter(ngram_n=8)
        if ane_ok:
            tp = prompt_text.find("---\n\n")
            ane_p = prompt_text[tp+5:] if tp >= 0 else prompt_text[-500:]
            lh = ane_generate_async(ane_p, max_tokens=max_tokens)
            static_d.set_ane_lookahead(lh)

        out = []
        static_src = {"ngram": 0, "ane": 0, "mtp": 0, "gpu": 0}
        t0 = time.perf_counter()
        for tok, _, _, src in four_path_generate_step(
            prompt_mx, model, static_d, tokenizer=tokenizer,
            num_draft_tokens=32, max_tokens=max_tokens,
        ):
            out.append(tok)
            static_src[src] = static_src.get(src, 0) + 1
        static_tps = len(out) / (time.perf_counter() - t0)
        if static_d.ane_lookahead: static_d.ane_lookahead.wait(5)

        # Multi-check
        from multicheck import MultiCheckDrafter, multicheck_generate_step

        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        mc_d = MultiCheckDrafter(ngram_n=8)
        if ane_ok:
            tp = prompt_text.find("---\n\n")
            ane_p = prompt_text[tp+5:] if tp >= 0 else prompt_text[-500:]
            lh = ane_generate_async(ane_p, max_tokens=max_tokens)
            mc_d.set_ane_lookahead(lh)

        out = []
        mc_src = {"ngram": 0, "ane": 0, "mtp": 0, "gpu": 0}
        t0 = time.perf_counter()
        for tok, _, _, src in multicheck_generate_step(
            prompt_mx, model, mc_d, tokenizer=tokenizer,
            num_draft_tokens=32, max_tokens=max_tokens,
        ):
            out.append(tok)
            mc_src[src] = mc_src.get(src, 0) + 1
        mc_tps = len(out) / (time.perf_counter() - t0)
        if mc_d.ane_lookahead: mc_d.ane_lookahead.wait(5)

        mc_summary = mc_d.summary()
        recoveries = mc_summary["multicheck_recoveries"]
        attempts = mc_summary["multicheck_attempts"]
        recovery_rate = mc_summary["multicheck_rate"]

        static_speedup = static_tps / baseline
        mc_speedup = mc_tps / baseline
        delta = (mc_tps - static_tps) / static_tps * 100

        print(f"  Standard:    {baseline:.1f} tok/s")
        print(f"  Static 4P:   {static_tps:.1f} tok/s ({static_speedup:.2f}x)  "
              f"N={static_src['ngram']} A={static_src['ane']} M={static_src['mtp']} G={static_src['gpu']}")
        print(f"  Multi-check: {mc_tps:.1f} tok/s ({mc_speedup:.2f}x)  "
              f"N={mc_src['ngram']} A={mc_src['ane']} M={mc_src['mtp']} G={mc_src['gpu']}")
        print(f"  Recoveries:  {recoveries}/{attempts} ({recovery_rate:.0%}) — "
              f"times ANE/MTP had the right token at rejection")
        print(f"  Δ Multi-check vs Static: {delta:+.1f}%")

        results.append({
            "test": name, "baseline": baseline,
            "static_tps": static_tps, "static_speedup": static_speedup,
            "mc_tps": mc_tps, "mc_speedup": mc_speedup,
            "delta_pct": delta,
            "recoveries": recoveries, "attempts": attempts,
            "recovery_rate": recovery_rate,
            "static_sources": static_src, "mc_sources": mc_src,
        })

    # Summary
    print(f"\n\n{'=' * 110}")
    print("SUMMARY")
    print(f"{'=' * 110}")
    print(f"\n  {'Test':<14} {'Std':>7} {'Static':>8} {'MCheck':>8} {'S-Spd':>7} {'MC-Spd':>7} {'Δ%':>7} {'Recoveries':>12}")
    print(f"  {'─'*14} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*12}")
    for r in results:
        print(f"  {r['test']:<14} {r['baseline']:>6.1f} {r['static_tps']:>7.1f} {r['mc_tps']:>7.1f} "
              f"{r['static_speedup']:>6.2f}x {r['mc_speedup']:>6.2f}x {r['delta_pct']:>+6.1f}% "
              f"{r['recoveries']:>4}/{r['attempts']:<4} ({r['recovery_rate']:.0%})")

    out_path = Path(__file__).parent / "multicheck_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run()
