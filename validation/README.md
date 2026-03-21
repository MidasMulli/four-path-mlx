# K=32 Batch Verification Validation

Statistical validation of the batch verification plateau on Apple Silicon M5.

## Key Finding

Verifying K=32 draft tokens costs the same as K=16 on M5 Air. K=8 is actually MORE expensive than K=16 due to the GatedDeltaNet SSM batch processing threshold.

## Results (600 measurements: 20 runs x 5 prompts x 6 K values)

| K | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | n |
|---|-----------|----------|----------|----------|---|
| 1 | 39.3 | 0.7-6.1 | 36.5 | 64.5 | 100 |
| 4 | 78.4 | 0.2-18.8 | 71.1 | 143.7 | 100 |
| 8 | 149.7 | 0.9-10.4 | 146.5 | 194.9 | 100 |
| 16 | 117.5 | 0.4-15.8 | 113.9 | 160.2 | 100 |
| 24 | 120.3 | 0.5-9.3 | 118.4 | 161.7 | 100 |
| 32 | 117.7 | 0.8-16.4 | 112.1 | 157.7 | 100 |

Std ranges reflect per-prompt variation (5 prompts, each with their own std from 20 runs).

### Plateau confirmation

- K=32/K=16 ratio: **1.002x** (effectively identical)
- K=24/K=16 ratio: 1.024x
- All three within noise floor

### K=8 inversion

K=8 (149.7ms) costs **27.4% more** than K=16 (117.5ms). The GatedDeltaNet SSM batch processing mode kicks in between K=8 and K=16 — below this threshold, the sequential recurrence dominates; above it, the parallel projections amortize the cost.

### Throughput ceiling (100% draft acceptance)

| K | Ceiling (tok/s) | vs 23.0 tok/s baseline |
|---|-----------------|----------------------|
| 1 | 25 | 1.1x |
| 4 | 51 | 2.2x |
| 8 | 53 | 2.3x |
| 16 | 136 | 5.9x |
| 24 | 200 | 8.7x |
| 32 | 272 | 11.8x |

### Wall-clock validation (autoregressive spec decode, K=32)

CSA drafting with warm N-gram table:
- **143.9 tok/s** (6.26x over 23.0 tok/s baseline)
- 525 tokens in 3.65 seconds
- 91.2% of tokens from N-gram drafts
- 45 verification rounds for 525 tokens (~11.7 tokens accepted per round)

### Acceptance rates (autoregressive, NOT teacher-forcing)

| Prompt Type | Median | Min | Max | Condition |
|-------------|--------|-----|-----|-----------|
| CSA full draft | 31.2% | 0% | 100% | warm |
| Netting provisions | 22.2% | 0% | 100% | warm |
| Termination events | 9.4% | 0% | 100% | warm |
| Events of default | 6.2% | 0% | 100% | warm |
| ISDA comparison | 0.0% | 0% | 59.4% | warm |

High variance is expected: N-gram chains either match perfectly (when the model echoes boilerplate patterns) or miss completely (when the model generates novel text). The wall-clock throughput reflects the real-world mix.

## Methodology

The benchmark (`gate_zero.py`) runs 5 tests:

1. **Baseline** (10 runs): Single-token autoregressive generation, 100 tokens each
2. **Verification wall time** (20 runs x 5 prompts x 6 K values): Generate 150 context tokens, then time a single batch verification pass. Includes warmup verification pass (untimed) before each measurement to prime Metal pipelines.
3. **Acceptance rates** (5 boilerplate-inducing prompts): Generate 500 tokens autoregressively, build N-gram table from generated text, then draft and verify chains at 20+ positions. Both warm (corpus + generated) and cold (corpus only) N-gram tables tested.
4. **Wall-clock throughput**: Full autoregressive spec decode loop at K=32 with warm N-gram, measuring real end-to-end tok/s.
5. **Reconciliation**: Maps verification costs back to single-token costs, explains the K=32 plateau mechanism.

Each verification measurement:
- Fresh model cache (no state leakage between runs)
- 150 context tokens generated before timing (realistic cache state)
- N-gram drafts from real ISDA/10-K/S-1 corpus (8 files, 87K tokens)
- Untimed warmup verification pass before timed pass
- mx.eval() synchronization before and after timing

## Reproducing

```bash
# Requires: MLX, mlx-lm, Qwen3.5-9B-MLX-4bit
# The model downloads automatically on first run (~5GB)

pip install mlx mlx-lm

# Copy ngram_predict.py from the parent repo if not already present
cp ../four_path/ngram.py ngram_predict.py  # or use the one in this directory

# Run (takes 30-40 minutes on M5 Air 16GB)
python gate_zero.py

# Results saved to gate_zero_results.json
```

The script requires sample corpus files in `isda-samples/`, `10k-samples/`, `s1-samples/` (same as the main benchmarks). If these directories don't exist, it falls back to synthetic ISDA text (lower N-gram hit rates, but verification timing is unaffected).

## Hardware

- MacBook Air M5, 16GB unified memory, 10 GPU cores
- macOS 26.3 (Tahoe)
- MLX 0.31.1, mlx-lm 0.31.1
- Model: mlx-community/Qwen3.5-9B-MLX-4bit

## Files

- `gate_zero.py` — Benchmark script (reproducible)
- `gate_zero_results.json` — Raw results (K means, baseline, wall-clock CSA, gate verdicts)
- `gate_zero_output.txt` — Full console output from the validation run
