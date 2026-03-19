#!/usr/bin/env python3
"""
Orchestrator Benchmark: Adaptive vs Static Four-Path

Compares:
  1. Standard (baseline)
  2. Static four-path (fixed cascade)
  3. Orchestrated four-path (adaptive routing + blending)

Key metric: tokens per forward pass (efficiency).
The orchestrator should reduce wasted verification passes.

Usage:
    ~/.mlx-env/bin/python benchmark_orchestrator.py
"""

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))

MODEL_ID = os.path.expanduser("~/models/Qwen3.5-9B-MLX-4bit-MTP")
SAMPLES_10K = Path(__file__).parent / "10k-samples"
SAMPLES_S1 = Path(__file__).parent / "s1-samples"
SAMPLES_ISDA = Path(__file__).parent.parent / "isda-classifier" / "samples"


def load_model():
    from mlx_lm import load
    print(f"Loading model: {MODEL_ID}")
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
    print(f"  ANE: {'alive' if ane_ok else 'unavailable'}")

    # Warmup
    from mlx_lm.generate import generate_step
    warmup = mx.array(tokenizer.encode("Hello"), mx.uint32)
    for _ in generate_step(warmup, model, max_tokens=5):
        pass

    # Test prompts across domains
    ref_10k = (SAMPLES_10K / "jpm-10k.txt").read_text()[:8000]
    ref_s1 = sorted(SAMPLES_S1.glob("*.txt"))[0].read_text()[:8000] if list(SAMPLES_S1.glob("*.txt")) else ""
    ref_isda = sorted(SAMPLES_ISDA.glob("*.txt"))[0].read_text()[:8000] if list(SAMPLES_ISDA.glob("*.txt")) else ""

    tests = [
        {
            "name": "10k_reg_disc",
            "desc": "10-K regulatory disclosure (boilerplate)",
            "prompt": f"Reference:\n{ref_10k}\n\n---\n\nDraft the regulatory capital disclosure section for a G-SIB bank's 10-K. Include Basel III CET1, Tier 1, Total capital ratios, SLR, TLAC.\n\nRegulatory Capital\n\n",
            "max_tokens": 1024,
        },
        {
            "name": "s1_risk",
            "desc": "S-1 risk factors (high boilerplate)",
            "prompt": f"Reference:\n{ref_s1}\n\n---\n\nDraft Risk Factors for a technology company's S-1.\n\nRISK FACTORS\n\nInvesting in our Class A common stock involves a high degree of risk.\n\n",
            "max_tokens": 1024,
        },
        {
            "name": "isda_csa",
            "desc": "ISDA CSA drafting (domain boilerplate)",
            "prompt": f"Reference:\n{ref_isda}\n\n---\n\nDraft Credit Support Annex Paragraph 13 for a bilateral derivatives relationship.\n\nParagraph 13. Elections and Variables.\n\n",
            "max_tokens": 1024,
        },
        {
            "name": "analysis",
            "desc": "Risk analysis (novel/analytical)",
            "prompt": f"Reference:\n{ref_10k}\n\n---\n\nAs a credit analyst, evaluate this bank's risk profile. Identify the 5 most material risks, assess probability and severity, and recommend risk mitigation strategies.\n\n",
            "max_tokens": 512,
        },
        {
            "name": "code",
            "desc": "Python code generation (no domain text)",
            "prompt": "Write a Python class that connects to the SEC EDGAR API, downloads 10-K filings for a given CIK, extracts risk factors using regex, and stores them in a SQLite database. Include proper error handling and rate limiting.",
            "max_tokens": 512,
        },
    ]

    print(f"\n{'=' * 110}")
    print("ORCHESTRATOR BENCHMARK: Static vs Adaptive Four-Path")
    print(f"{'=' * 110}")

    all_results = []

    for test in tests:
        name = test["name"]
        desc = test["desc"]
        max_tokens = test["max_tokens"]
        prompt_text = test["prompt"]
        prompt_tokens = tokenizer.encode(prompt_text)

        print(f"\n{'─' * 110}")
        print(f"  {name} - {desc}")
        print(f"  Prompt: {len(prompt_tokens):,} tokens → Generate: {max_tokens}")
        print(f"{'─' * 110}")

        result = {"test": name, "desc": desc, "prompt_tokens": len(prompt_tokens)}

        # ── Standard ──
        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        out = []
        t0 = time.perf_counter()
        for tok, _ in generate_step(prompt_mx, model, max_tokens=max_tokens):
            out.append(tok)
        baseline = len(out) / (time.perf_counter() - t0)
        print(f"  Standard:     {baseline:.1f} tok/s")
        result["standard"] = baseline

        # ── Static four-path ──
        from four_path import FourPathDrafter, four_path_generate_step
        from three_path import ane_generate_async

        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        drafter = FourPathDrafter(ngram_n=8)
        if ane_ok:
            task_start = prompt_text.find("---\n\n")
            ane_prompt = prompt_text[task_start + 5:] if task_start >= 0 else prompt_text[-500:]
            ane_lh = ane_generate_async(ane_prompt, max_tokens=max_tokens)
            drafter.set_ane_lookahead(ane_lh)

        out = []
        sources_static = {"ngram": 0, "ane": 0, "mtp": 0, "gpu": 0}
        t0 = time.perf_counter()
        for tok, lp, _, source in four_path_generate_step(
            prompt_mx, model, drafter, tokenizer=tokenizer,
            num_draft_tokens=32, max_tokens=max_tokens,
        ):
            out.append(tok)
            sources_static[source] = sources_static.get(source, 0) + 1
        static_tps = len(out) / (time.perf_counter() - t0)
        static_speedup = static_tps / baseline

        if ane_ok and drafter.ane_lookahead:
            drafter.ane_lookahead.wait(5)

        static_summary = drafter.summary()
        print(f"  Static 4P:    {static_tps:.1f} tok/s ({static_speedup:.2f}x) - "
              f"N={sources_static['ngram']} A={sources_static['ane']} M={sources_static['mtp']} G={sources_static['gpu']}")
        result["static"] = {"tps": static_tps, "speedup": static_speedup, "sources": sources_static}

        # ── Orchestrated four-path ──
        from orchestrator import Orchestrator, orchestrated_generate_step

        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        orch = Orchestrator(ngram_n=8, enable_blending=True)
        if ane_ok:
            task_start = prompt_text.find("---\n\n")
            ane_prompt = prompt_text[task_start + 5:] if task_start >= 0 else prompt_text[-500:]
            ane_lh = ane_generate_async(ane_prompt, max_tokens=max_tokens)
            orch.set_ane_lookahead(ane_lh)

        out = []
        sources_orch = {"ngram": 0, "ane": 0, "mtp": 0, "gpu": 0}
        t0 = time.perf_counter()
        for tok, lp, _, source in orchestrated_generate_step(
            prompt_mx, model, orch, tokenizer=tokenizer,
            num_draft_tokens=32, max_tokens=max_tokens,
        ):
            out.append(tok)
            sources_orch[source] = sources_orch.get(source, 0) + 1
        orch_tps = len(out) / (time.perf_counter() - t0)
        orch_speedup = orch_tps / baseline

        if ane_ok and orch.ane_lookahead:
            orch.ane_lookahead.wait(5)

        orch_summary = orch.summary()
        orch_eff = orch_summary["efficiency"]
        print(f"  Orchestrated: {orch_tps:.1f} tok/s ({orch_speedup:.2f}x) - "
              f"N={sources_orch['ngram']} A={sources_orch['ane']} M={sources_orch['mtp']} G={sources_orch['gpu']} "
              f"| {orch_eff:.1f} tok/pass")
        result["orchestrated"] = {
            "tps": orch_tps, "speedup": orch_speedup,
            "sources": sources_orch, "efficiency": orch_eff,
            "summary": orch_summary,
        }

        # Improvement
        if static_tps > 0:
            improvement = (orch_tps - static_tps) / static_tps * 100
            print(f"  Δ Orchestrated vs Static: {improvement:+.1f}%")
            result["improvement_pct"] = improvement

        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n\n{'=' * 110}")
    print("SUMMARY")
    print(f"{'=' * 110}")
    print(f"\n  {'Test':<18} {'Std':>8} {'Static':>8} {'Orch':>8} {'S-Speed':>9} {'O-Speed':>9} {'Δ%':>7} {'Tok/Pass':>9}")
    print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*9} {'─'*9} {'─'*7} {'─'*9}")

    for r in all_results:
        std = r["standard"]
        s = r.get("static", {})
        o = r.get("orchestrated", {})
        delta = r.get("improvement_pct", 0)
        eff = o.get("efficiency", 0)
        print(f"  {r['test']:<18} {std:>7.1f} {s.get('tps',0):>7.1f} {o.get('tps',0):>7.1f} "
              f"{s.get('speedup',0):>8.2f}x {o.get('speedup',0):>8.2f}x {delta:>+6.1f}% {eff:>8.1f}")

    # Detailed orchestrator stats for last test
    print(f"\n  ── Orchestrator Details (last test) ──")
    if all_results and "orchestrated" in all_results[-1]:
        orch_data = all_results[-1]["orchestrated"].get("summary", {})
        sources_data = orch_data.get("sources", {})
        for name, ts in sources_data.items():
            throttled = "THROTTLED" if ts.get("throttled") else ""
            print(f"    {name:<8} batch={ts.get('recommended_batch',0):>3} "
                  f"accept={ts.get('lifetime_acceptance',0):.0%} "
                  f"tok/pass={ts.get('tokens_per_pass',0):.1f} {throttled}")

    out_path = Path(__file__).parent / "orchestrator_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run()
