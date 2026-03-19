#!/usr/bin/env python3
"""
Stress Test: Answering the hard questions about four-path speculative decoding.

1. Is 4x just copy-paste? — Measure edit distance between output and reference
2. Does speedup hold at 2048 tokens? — Test at 256, 512, 1024, 2048
3. What are the rejection rates? — Track proposals vs acceptances per source
4. Is MTP correct? — Verify MTP-accepted tokens match standard generation
5. Are benchmarks diverse enough? — Test on genuinely different task types

Usage:
    ~/.mlx-env/bin/python benchmark_stress.py
"""

import json
import os
import sys
import time
from pathlib import Path
from difflib import SequenceMatcher

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


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Is 4x just copy-paste?
# ═══════════════════════════════════════════════════════════════════

def test_copy_paste(model, tokenizer):
    """Measure how much of the output is verbatim from the reference."""
    print(f"\n{'=' * 90}")
    print("TEST 1: Is 4x just copy-paste?")
    print("  Measures: what % of output appears verbatim in the reference prompt")
    print(f"{'=' * 90}")

    ref_text = (SAMPLES_10K / "jpm-10k.txt").read_text()[:8000]

    prompts = [
        ("md_and_a", "Draft the MD&A section covering net interest income for a major US bank."),
        ("risk_analysis", "Analyze the top 5 risk factors in this filing and assess their severity."),
        ("comparison", "Compare this bank's regulatory capital position to industry standards."),
    ]

    for name, task in prompts:
        full_prompt = f"Reference 10-K excerpt:\n{ref_text}\n\n---\n\nTask: {task}\n\n"
        prompt_tokens = tokenizer.encode(full_prompt)

        # Generate 512 tokens with standard generation
        from mlx_lm.generate import generate_step
        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        out_tokens = []
        for tok, _ in generate_step(prompt_mx, model, max_tokens=512):
            out_tokens.append(tok)
        output_text = tokenizer.decode(out_tokens)

        # Measure overlap with reference
        # Method: find longest common substrings
        matcher = SequenceMatcher(None, ref_text.lower(), output_text.lower())
        matching_blocks = matcher.get_matching_blocks()

        # Count characters in output that appear in matching blocks >= 20 chars
        verbatim_chars = sum(b.size for b in matching_blocks if b.size >= 20)
        total_chars = len(output_text)
        verbatim_pct = verbatim_chars / total_chars if total_chars else 0

        # Also count exact 4-gram overlaps
        ref_words = ref_text.lower().split()
        out_words = output_text.lower().split()
        ref_4grams = set()
        for i in range(len(ref_words) - 3):
            ref_4grams.add(tuple(ref_words[i:i+4]))
        out_4gram_hits = 0
        out_4gram_total = max(len(out_words) - 3, 1)
        for i in range(len(out_words) - 3):
            if tuple(out_words[i:i+4]) in ref_4grams:
                out_4gram_hits += 1
        gram4_pct = out_4gram_hits / out_4gram_total

        print(f"\n  {name}:")
        print(f"    Output: {len(out_tokens)} tokens, {total_chars} chars")
        print(f"    Verbatim from reference (≥20 char spans): {verbatim_pct:.1%}")
        print(f"    4-gram overlap with reference: {gram4_pct:.1%}")
        print(f"    First 200 chars: {output_text[:200].strip()}")

    print(f"\n  Interpretation: if verbatim % is high, the speedup is from echoing.")
    print(f"  If low, the model is generating novel text that N-grams still catch")
    print(f"  (because the model learned similar patterns, not because it copied).")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Does speedup hold at longer generation?
# ═══════════════════════════════════════════════════════════════════

def test_length_scaling(model, tokenizer, ane_ok):
    """Test at 256, 512, 1024, 2048 tokens."""
    print(f"\n{'=' * 90}")
    print("TEST 2: Speedup vs Generation Length")
    print("  Does N-gram hit rate decay as generation gets longer?")
    print(f"{'=' * 90}")

    ref_text = (SAMPLES_10K / "jpm-10k.txt").read_text()[:8000]
    prompt_text = (
        f"Reference 10-K excerpt:\n{ref_text}\n\n---\n\n"
        "Draft a comprehensive regulatory capital disclosure section for a G-SIB bank's "
        "10-K filing. Include Basel III ratios, stress testing results, TLAC requirements, "
        "countercyclical buffers, and forward-looking capital plans.\n\n"
        "Regulatory Capital and Liquidity\n\n"
    )
    prompt_tokens = tokenizer.encode(prompt_text)

    from mlx_lm.generate import generate_step
    from four_path import FourPathDrafter, four_path_generate_step
    from three_path import ane_generate_async

    print(f"\n  {'Length':>8} {'Standard':>10} {'4-Path':>10} {'Speedup':>10} {'N-gram%':>10} {'ANE%':>10} {'GPU%':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for max_tokens in [256, 512, 1024, 2048]:
        # Standard
        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        out = []
        t0 = time.perf_counter()
        for tok, _ in generate_step(prompt_mx, model, max_tokens=max_tokens):
            out.append(tok)
        baseline = len(out) / (time.perf_counter() - t0)
        actual_tokens = len(out)

        # Four-path
        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        drafter = FourPathDrafter(ngram_n=8)
        if ane_ok:
            task_start = prompt_text.find("---\n\n")
            ane_prompt = prompt_text[task_start + 5:] if task_start >= 0 else prompt_text[-500:]
            ane_lh = ane_generate_async(ane_prompt, max_tokens=max_tokens)
            drafter.set_ane_lookahead(ane_lh)

        out = []
        sources = {"ngram": 0, "ane": 0, "mtp": 0, "gpu": 0}
        t0 = time.perf_counter()
        for tok, lp, from_draft, source in four_path_generate_step(
            prompt_mx, model, drafter, tokenizer=tokenizer,
            num_draft_tokens=32, max_tokens=max_tokens,
        ):
            out.append(tok)
            sources[source] = sources.get(source, 0) + 1
        tps = len(out) / (time.perf_counter() - t0)
        speedup = tps / baseline

        if ane_ok and hasattr(drafter, 'ane_lookahead') and drafter.ane_lookahead:
            drafter.ane_lookahead.wait(5)

        total = len(out)
        ng_pct = sources["ngram"] / total if total else 0
        ane_pct = sources["ane"] / total if total else 0
        gpu_pct = sources["gpu"] / total if total else 0

        print(f"  {actual_tokens:>8} {baseline:>9.1f} {tps:>9.1f} {speedup:>9.2f}x "
              f"{ng_pct:>9.0%} {ane_pct:>9.0%} {gpu_pct:>9.0%}")


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Rejection rates per source
# ═══════════════════════════════════════════════════════════════════

def test_rejection_rates(model, tokenizer, ane_ok):
    """Track proposals vs acceptances for each draft source."""
    print(f"\n{'=' * 90}")
    print("TEST 3: Rejection Rates per Source")
    print("  How many tokens does each source propose vs how many get accepted?")
    print(f"{'=' * 90}")

    ref_text = (SAMPLES_S1 / sorted(SAMPLES_S1.glob("*.txt"))[0].name).read_text()[:8000]
    prompt_text = (
        f"Reference S-1 excerpt:\n{ref_text}\n\n---\n\n"
        "Draft the Risk Factors section for a technology company's S-1 filing.\n\n"
        "RISK FACTORS\n\n"
    )
    prompt_tokens = tokenizer.encode(prompt_text)

    from four_path import FourPathDrafter, four_path_generate_step
    from three_path import ane_generate_async

    prompt_mx = mx.array(prompt_tokens, mx.uint32)
    drafter = FourPathDrafter(ngram_n=8)
    if ane_ok:
        task_start = prompt_text.find("---\n\n")
        ane_prompt = prompt_text[task_start + 5:] if task_start >= 0 else prompt_text[-500:]
        ane_lh = ane_generate_async(ane_prompt, max_tokens=512)
        drafter.set_ane_lookahead(ane_lh)

    out = []
    for tok, lp, from_draft, source in four_path_generate_step(
        prompt_mx, model, drafter, tokenizer=tokenizer,
        num_draft_tokens=32, max_tokens=512,
    ):
        out.append(tok)

    if ane_ok and drafter.ane_lookahead:
        drafter.ane_lookahead.wait(5)

    s = drafter.summary()
    print(f"\n  {'Source':<10} {'Proposed':>10} {'Accepted':>10} {'Rejected':>10} {'Accept%':>10} {'Rounds':>8}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    for src in ["ngram", "ane", "mtp"]:
        proposed = s.get(f"{src}_drafted", 0)
        accepted = s.get(f"{src}_accepted", 0)
        rejected = proposed - accepted
        accept_pct = accepted / proposed if proposed else 0
        rounds = s.get(f"{src}_rounds", 0)
        print(f"  {src:<10} {proposed:>10} {accepted:>10} {rejected:>10} {accept_pct:>9.0%} {rounds:>8}")

    print(f"\n  GPU-only tokens (no draft available): {s.get('gpu_only', 0)}")
    print(f"  Total tokens generated: {len(out)}")


# ═══════════════════════════════════════════════════════════════════
# TEST 4: MTP correctness verification
# ═══════════════════════════════════════════════════════════════════

def test_mtp_correctness(model, tokenizer):
    """Verify MTP-generated output matches standard generation."""
    print(f"\n{'=' * 90}")
    print("TEST 4: MTP Correctness Verification")
    print("  Does MTP generation produce the same tokens as standard (greedy)?")
    print(f"{'=' * 90}")

    from mlx_lm.generate import generate_step, stream_generate

    prompts = [
        "What are the key provisions of a standard ISDA Master Agreement?",
        "Explain the Basel III capital requirements for G-SIB banks.",
        "Draft a short risk factor about cybersecurity for an S-1 filing.",
    ]

    for prompt in prompts:
        # Standard greedy
        prompt_mx = mx.array(tokenizer.encode(prompt), mx.uint32)
        std_tokens = []
        for tok, _ in generate_step(prompt_mx, model, max_tokens=100):
            std_tokens.append(tok)

        # MTP greedy
        mtp_tokens = []
        for resp in stream_generate(model, tokenizer, prompt, max_tokens=100, mtp=True):
            mtp_tokens.append(resp.token)

        # Compare
        match_len = 0
        for i in range(min(len(std_tokens), len(mtp_tokens))):
            if std_tokens[i] == mtp_tokens[i]:
                match_len += 1
            else:
                break

        total = min(len(std_tokens), len(mtp_tokens))
        # Check full sequence match too
        full_match = std_tokens[:total] == mtp_tokens[:total]

        print(f"\n  Prompt: {prompt[:60]}...")
        print(f"    Standard: {len(std_tokens)} tokens")
        print(f"    MTP:      {len(mtp_tokens)} tokens")
        print(f"    Match:    first {match_len} identical, full match: {full_match}")
        if not full_match and match_len < total:
            diverge_std = tokenizer.decode([std_tokens[match_len]]) if match_len < len(std_tokens) else "EOS"
            diverge_mtp = tokenizer.decode([mtp_tokens[match_len]]) if match_len < len(mtp_tokens) else "EOS"
            print(f"    Diverges at token {match_len}: std='{diverge_std}' vs mtp='{diverge_mtp}'")


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Task diversity
# ═══════════════════════════════════════════════════════════════════

def test_diversity(model, tokenizer, ane_ok):
    """Test on genuinely different task types — not just document drafting."""
    print(f"\n{'=' * 90}")
    print("TEST 5: Task Diversity — Non-Drafting Tasks")
    print("  Does four-path help on tasks that AREN'T document generation?")
    print(f"{'=' * 90}")

    ref_text = (SAMPLES_10K / "jpm-10k.txt").read_text()[:6000]

    diverse_prompts = [
        {
            "name": "numerical_qa",
            "desc": "Extract specific numbers from filing",
            "prompt": f"Reference:\n{ref_text}\n\n---\n\nList every dollar amount and percentage mentioned in this excerpt. Format: one per line, with the context.",
        },
        {
            "name": "summarize",
            "desc": "Summarize (not draft) the filing",
            "prompt": f"Reference:\n{ref_text}\n\n---\n\nWrite a 3-paragraph executive summary of this 10-K excerpt for a board presentation. Be concise and use plain language, not SEC boilerplate.",
        },
        {
            "name": "critique",
            "desc": "Critical analysis (adversarial to boilerplate)",
            "prompt": f"Reference:\n{ref_text}\n\n---\n\nAs a short-seller, identify 3 red flags or weaknesses in this disclosure that management may be downplaying. Be specific and skeptical.",
        },
        {
            "name": "code_gen",
            "desc": "Generate Python code (no financial boilerplate)",
            "prompt": "Write a Python function that parses SEC EDGAR XBRL filings and extracts all monetary values with their context labels. Include error handling.",
        },
    ]

    from mlx_lm.generate import generate_step
    from four_path import FourPathDrafter, four_path_generate_step
    from three_path import ane_generate_async

    print(f"\n  {'Task':<20} {'Standard':>10} {'4-Path':>10} {'Speedup':>10} {'N-gram':>8} {'MTP':>6} {'ANE':>6} {'GPU':>6}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*6} {'─'*6} {'─'*6}")

    for test in diverse_prompts:
        prompt_tokens = tokenizer.encode(test["prompt"])
        max_tokens = 256

        # Standard
        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        out = []
        t0 = time.perf_counter()
        for tok, _ in generate_step(prompt_mx, model, max_tokens=max_tokens):
            out.append(tok)
        baseline = len(out) / (time.perf_counter() - t0)

        # Four-path
        prompt_mx = mx.array(prompt_tokens, mx.uint32)
        drafter = FourPathDrafter(ngram_n=8)
        if ane_ok:
            task_start = test["prompt"].find("---\n\n")
            ane_prompt = test["prompt"][task_start + 5:] if task_start >= 0 else test["prompt"][-500:]
            ane_lh = ane_generate_async(ane_prompt, max_tokens=max_tokens)
            drafter.set_ane_lookahead(ane_lh)

        out = []
        sources = {"ngram": 0, "ane": 0, "mtp": 0, "gpu": 0}
        t0 = time.perf_counter()
        for tok, lp, from_draft, source in four_path_generate_step(
            prompt_mx, model, drafter, tokenizer=tokenizer,
            num_draft_tokens=32, max_tokens=max_tokens,
        ):
            out.append(tok)
            sources[source] = sources.get(source, 0) + 1
        tps = len(out) / (time.perf_counter() - t0)
        speedup = tps / baseline

        if ane_ok and drafter.ane_lookahead:
            drafter.ane_lookahead.wait(5)

        s = sources
        print(f"  {test['name']:<20} {baseline:>9.1f} {tps:>9.1f} {speedup:>9.2f}x "
              f"{s['ngram']:>8} {s['mtp']:>5} {s['ane']:>5} {s['gpu']:>5}")


def run():
    model, tokenizer = load_model()
    ane_ok = check_ane()
    print(f"  ANE: {'alive' if ane_ok else 'unavailable'}\n")

    # Warmup
    from mlx_lm.generate import generate_step
    warmup = mx.array(tokenizer.encode("Hello"), mx.uint32)
    for _ in generate_step(warmup, model, max_tokens=5):
        pass

    test_copy_paste(model, tokenizer)
    test_length_scaling(model, tokenizer, ane_ok)
    test_rejection_rates(model, tokenizer, ane_ok)
    test_mtp_correctness(model, tokenizer)
    test_diversity(model, tokenizer, ane_ok)

    print(f"\n\n{'=' * 90}")
    print("ALL STRESS TESTS COMPLETE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    run()
