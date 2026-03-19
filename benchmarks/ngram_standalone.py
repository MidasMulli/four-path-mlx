#!/usr/bin/env python3
"""
N-gram Prediction Benchmark on ISDA Text

Simulates an inference loop: tokenize the document, walk through it token by
token, feed n-grams into the hash table, and check how often the predicted
next token matches the actual next token.

This measures the CPU prediction path's hit rate on real legal boilerplate —
the foundation for the three-path heterogeneous speculative decoding architecture.

Usage:
    ~/.mlx-env/bin/python benchmark.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Use the MLX env's transformers
sys.path.insert(0, os.path.dirname(__file__))
from ngram_predict import NgramPredictor, EMPTY

SAMPLES_DIR = Path(__file__).parent.parent / "isda-classifier" / "samples"
MODEL_ID = "mlx-community/Qwen3.5-9B-MLX-4bit"


def load_tokenizer():
    """Load the Qwen3.5 tokenizer (same one used for inference)."""
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {MODEL_ID}")
    return AutoTokenizer.from_pretrained(MODEL_ID)


def load_isda_texts() -> list[tuple[str, str]]:
    """Load all ISDA .txt files. Returns [(filename, text), ...]."""
    texts = []
    for f in sorted(SAMPLES_DIR.glob("*.txt")):
        texts.append((f.name, f.read_text()))
    return texts


def benchmark_single_document(predictor: NgramPredictor, tokens: list[int]) -> dict:
    """
    Simulate inference on a single document.

    Walk through the token sequence. At each position:
    1. Try to predict the next token from the hash table
    2. Feed the actual n-gram into the table
    3. Track hit rate

    This simulates what happens during real generation — the table is built
    incrementally from the prompt + generated tokens.
    """
    n = predictor.n
    if len(tokens) < n + 1:
        return {"error": "document too short"}

    predictor.reset()
    predictor.stats = type(predictor.stats)()  # fresh stats

    # Feed the first n tokens as seed (can't predict without context)
    # Then walk through the rest
    for i in range(n, len(tokens)):
        context = tokens[:i]
        actual = tokens[i]

        # Try to predict
        predictor.evaluate(context, actual)

        # Feed the new n-gram (context[i-n:i] -> actual)
        if i >= n:
            predictor.feed(tokens[i - n: i + 1])

    return {
        "tokens": len(tokens),
        "lookups": predictor.stats.lookups,
        "hits": predictor.stats.hits,
        "correct": predictor.stats.correct,
        "hit_rate": predictor.stats.hit_rate,
        "accuracy": predictor.stats.accuracy,
        "occupancy": predictor.occupancy,
        "feed_time_ms": predictor.stats.feed_time_ms,
        "lookup_time_ms": predictor.stats.lookup_time_ms,
    }


def benchmark_chain_prediction(predictor: NgramPredictor, tokens: list[int],
                                 chain_interval: int = 100) -> dict:
    """
    Simulate chain drafting — every `chain_interval` tokens, attempt a
    draft chain and measure how many tokens are correct.
    """
    n = predictor.n
    if len(tokens) < n + 1:
        return {"error": "document too short"}

    predictor.reset()
    predictor.stats = type(predictor.stats)()

    total_drafted = 0
    total_accepted = 0
    chain_attempts = 0

    # Feed entire document first (simulating prompt processing)
    predictor.feed(tokens)

    # Now simulate drafting at various points in the document
    for start in range(n, len(tokens) - 64, chain_interval):
        context = tokens[:start]
        chain = predictor.draft_chain(context, max_tokens=64, min_tokens=1)

        if chain:
            chain_attempts += 1
            # Check how many consecutive tokens match
            accepted = 0
            for j, draft_tok in enumerate(chain):
                if start + j < len(tokens) and draft_tok == tokens[start + j]:
                    accepted += 1
                else:
                    break
            total_drafted += len(chain)
            total_accepted += accepted

    return {
        "chain_attempts": chain_attempts,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "acceptance_rate": total_accepted / total_drafted if total_drafted else 0,
        "avg_chain_len": total_drafted / chain_attempts if chain_attempts else 0,
        "avg_accepted": total_accepted / chain_attempts if chain_attempts else 0,
    }


def benchmark_cross_document(predictor: NgramPredictor, all_tokens: list[list[int]]) -> dict:
    """
    Feed documents 1..N-1 into the table, then predict on document N.
    Measures cross-document pattern transfer — how much boilerplate
    repeats across different ISDA agreements.
    """
    n = predictor.n
    predictor.reset()
    predictor.stats = type(predictor.stats)()

    # Feed all but the last document
    for tokens in all_tokens[:-1]:
        predictor.feed(tokens)

    # Predict on the last document
    target = all_tokens[-1]
    for i in range(n, len(target)):
        context = target[:i]
        actual = target[i]
        predictor.evaluate(context, actual)

    return {
        "training_docs": len(all_tokens) - 1,
        "target_tokens": len(all_tokens[-1]),
        "lookups": predictor.stats.lookups,
        "hits": predictor.stats.hits,
        "correct": predictor.stats.correct,
        "hit_rate": predictor.stats.hit_rate,
        "accuracy": predictor.stats.accuracy,
        "cross_doc_prediction": predictor.stats.accuracy,
    }


def run_benchmarks():
    tokenizer = load_tokenizer()
    texts = load_isda_texts()

    if not texts:
        print("ERROR: No ISDA text files found in", SAMPLES_DIR)
        return

    print(f"\nTokenizing {len(texts)} ISDA agreements...")
    all_tokens = []
    for name, text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.append(tokens)
        print(f"  {name}: {len(tokens):,} tokens")

    total_tokens = sum(len(t) for t in all_tokens)
    print(f"  Total: {total_tokens:,} tokens across {len(texts)} documents\n")

    # =========================================================
    # Benchmark 1: Single-document hit rate at various n-gram sizes
    # =========================================================
    print("=" * 70)
    print("BENCHMARK 1: Single-Document Token Prediction")
    print("  Simulates inference — table builds incrementally as tokens are processed")
    print("=" * 70)

    n_values = [4, 6, 8, 12, 16, 24]
    # Use the largest document for this test
    largest_doc = max(all_tokens, key=len)
    largest_name = texts[all_tokens.index(largest_doc)][0]
    print(f"  Document: {largest_name} ({len(largest_doc):,} tokens)\n")

    print(f"  {'N':>4}  {'Lookups':>10}  {'Hits':>10}  {'Correct':>10}  {'Hit Rate':>10}  {'Accuracy':>10}  {'Feed ms':>10}  {'Look ms':>10}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    b1_results = []
    for n in n_values:
        predictor = NgramPredictor(n=n)
        result = benchmark_single_document(predictor, largest_doc)
        b1_results.append({"n": n, **result})
        print(f"  {n:>4}  {result['lookups']:>10,}  {result['hits']:>10,}  {result['correct']:>10,}  "
              f"{result['hit_rate']:>9.1%}  {result['accuracy']:>9.1%}  "
              f"{result['feed_time_ms']:>9.1f}  {result['lookup_time_ms']:>9.1f}")

    # =========================================================
    # Benchmark 2: Chain drafting (simulates speculative decode)
    # =========================================================
    print(f"\n{'=' * 70}")
    print("BENCHMARK 2: Draft Chain Prediction")
    print("  Feed entire document, then draft chains at intervals")
    print("  Measures: how many consecutive tokens can be predicted correctly")
    print("=" * 70)

    print(f"  Document: {largest_name}\n")
    print(f"  {'N':>4}  {'Chains':>10}  {'Drafted':>10}  {'Accepted':>10}  {'Accept%':>10}  {'Avg Chain':>10}  {'Avg Accept':>10}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    b2_results = []
    for n in n_values:
        predictor = NgramPredictor(n=n)
        result = benchmark_chain_prediction(predictor, largest_doc, chain_interval=50)
        b2_results.append({"n": n, **result})
        print(f"  {n:>4}  {result['chain_attempts']:>10,}  {result['total_drafted']:>10,}  "
              f"{result['total_accepted']:>10,}  {result['acceptance_rate']:>9.1%}  "
              f"{result['avg_chain_len']:>9.1f}  {result['avg_accepted']:>9.1f}")

    # =========================================================
    # Benchmark 3: Cross-document prediction
    # =========================================================
    print(f"\n{'=' * 70}")
    print("BENCHMARK 3: Cross-Document Prediction (Boilerplate Transfer)")
    print("  Feed 5 ISDA agreements into table, predict on the 6th")
    print("  Measures: how much legal boilerplate repeats across agreements")
    print("=" * 70)

    target_name = texts[-1][0]
    print(f"  Training on: {', '.join(t[0] for t in texts[:-1])}")
    print(f"  Predicting on: {target_name} ({len(all_tokens[-1]):,} tokens)\n")

    print(f"  {'N':>4}  {'Lookups':>10}  {'Hits':>10}  {'Correct':>10}  {'Hit Rate':>10}  {'Accuracy':>10}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    b3_results = []
    for n in n_values:
        predictor = NgramPredictor(n=n)
        result = benchmark_cross_document(predictor, all_tokens)
        b3_results.append({"n": n, **result})
        print(f"  {n:>4}  {result['lookups']:>10,}  {result['hits']:>10,}  {result['correct']:>10,}  "
              f"{result['hit_rate']:>9.1%}  {result['accuracy']:>9.1%}")

    # =========================================================
    # Benchmark 4: All documents combined
    # =========================================================
    print(f"\n{'=' * 70}")
    print("BENCHMARK 4: Per-Document Hit Rates (n=8)")
    print("  Single-document prediction for each agreement")
    print("=" * 70)

    n = 8
    print(f"\n  {'Document':<45}  {'Tokens':>8}  {'Accuracy':>10}")
    print(f"  {'─'*45}  {'─'*8}  {'─'*10}")

    b4_results = []
    for (name, _), tokens in zip(texts, all_tokens):
        predictor = NgramPredictor(n=n)
        result = benchmark_single_document(predictor, tokens)
        b4_results.append({"name": name, **result})
        print(f"  {name:<45}  {len(tokens):>8,}  {result['accuracy']:>9.1%}")

    # =========================================================
    # Save results
    # =========================================================
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_tokenizer": MODEL_ID,
        "total_tokens": total_tokens,
        "documents": len(texts),
        "benchmark_1_single_doc": b1_results,
        "benchmark_2_chain_draft": b2_results,
        "benchmark_3_cross_doc": b3_results,
        "benchmark_4_per_doc": b4_results,
    }

    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # =========================================================
    # Summary
    # =========================================================
    best_single = max(b1_results, key=lambda r: r["accuracy"])
    best_cross = max(b3_results, key=lambda r: r["accuracy"])
    best_chain = max(b2_results, key=lambda r: r["acceptance_rate"])

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Best single-doc accuracy:  n={best_single['n']}, {best_single['accuracy']:.1%}")
    print(f"  Best cross-doc accuracy:   n={best_cross['n']}, {best_cross['accuracy']:.1%}")
    print(f"  Best chain acceptance:     n={best_chain['n']}, {best_chain['acceptance_rate']:.1%}")
    print(f"  Total tokens benchmarked:  {total_tokens:,}")
    print()
    print("  If single-doc accuracy > 20%: N-gram CPU path is viable for ISDA")
    print("  If cross-doc accuracy > 10%:  Boilerplate transfer works across agreements")
    print("  If chain acceptance > 30%:    Draft chains are long enough for spec decode")


if __name__ == "__main__":
    run_benchmarks()
