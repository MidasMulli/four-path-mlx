# Four-Path Speculative Decoding on Apple Silicon

Four prediction sources, three processors, one generate loop. 2-4.6x speedup on financial document tasks. Same model, same weights, same output.

![Benchmark results on real SEC filings](docs/benchmark.png)

Tested on a MacBook Air M5 16GB running Qwen3.5-9B-MLX-4bit against real 10-K filings from EDGAR and ISDA Master Agreements.

## How it works

Standard LLM inference uses one processor (GPU) generating one token at a time. This system uses four draft sources across three processors to predict tokens before the GPU verifies them:

| Source | Processor | What it catches | Cost |
|--------|-----------|----------------|------|
| **N-gram hash** | CPU | Verbatim echoes from prompt context | Nanoseconds |
| **ANE 1.7B** | Neural Engine | Semantic predictions from a small model | 57 tok/s, parallel |
| **MTP head** | GPU (free) | Next-next token from hidden states | Comes with the forward pass |
| **GPU 9B** | GPU | Everything else (novel tokens) | Full forward pass |

The sources fire in priority order. If N-gram has a chain, verify the whole chain in one forward pass. If ANE has tokens, verify those. If MTP predicted a token, verify that. GPU generates normally only when nothing cheaper had a prediction.

On domain-specific text, most of what an LLM generates is predictable by something much cheaper than a 9 billion parameter forward pass.

## Results

Benchmarked on real SEC filings pulled from EDGAR and ISDA derivatives documentation:

| Task | Tokens | tok/s | Speedup | Draft % |
|------|--------|-------|---------|---------|
| Single Filing Analysis | 71 | 20.7 | 0.99x | 55% |
| ISDA Clause Analysis | 256 | 29.5 | 1.48x | 84% |
| Cross-Company Comparison | 1,024 | 50.2 | 2.39x | 72% |
| Adversarial Critique | 1,024 | 69.6 | 3.31x | 87% |
| Document Drafting (10-K) | 2,048 | 96.6 | 4.60x | 94% |

Baseline is ~21 tok/s on stock mlx-lm with the same model.

**What determines the speedup:** generation length and repetition structure of the input. ISDA Master Agreements have 60% cross-document repetition at n=8. 10-K filings are lower but within-prompt echoing fuels the hash table on long generations. Code generation with no reference context gets ~1x.

**Not copy-paste:** 0% verbatim overlap between generated output and reference documents (verified via SequenceMatcher). The model generates novel text. The draft sources just predict what it's going to say.

## Architecture

```
                    ┌─────────────┐
                    │  Prompt      │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
        │  N-gram   │ │  ANE  │ │    GPU    │
        │  CPU hash │ │ 1.7B  │ │  9B + MTP │
        │  table    │ │CoreML │ │  backbone │
        └─────┬─────┘ └───┬───┘ └─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │   Verify    │
                    │  (GPU 9B)   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Output    │
                    └─────────────┘
```

## Quick start

### As an MLX server (drop-in replacement for mlx-lm)

```bash
# Clone
git clone https://github.com/MidasMulli/four-path-mlx.git
cd four-path-mlx

# Start the server (port 8899, same API as mlx-lm server)
python server/server.py

# It auto-detects available paths:
#   N-gram: always on
#   ANE: if orion-ane server is running on /tmp/orion-ane-server.sock
#   MTP: if ~/models/Qwen3.5-9B-MLX-4bit-MTP weights exist
#   GPU: always on
```

The server exposes `/v1/chat/completions` and `/v1/models`. Any OpenAI-compatible client works. Your existing code doesn't change.

### In Python

```python
from mlx_lm import load
from four_path.generate import FourPathDrafter, four_path_generate_step
from four_path.ngram import NgramPredictor
import mlx.core as mx

model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-4bit")

prompt = "Draft an ISDA Schedule Part 1..."
tokens = tokenizer.encode(prompt)
prompt_mx = mx.array(tokens, mx.uint32)

drafter = FourPathDrafter(ngram_n=8)

for tok, logprobs, from_draft, source in four_path_generate_step(
    prompt_mx, model, drafter, tokenizer=tokenizer,
    max_tokens=1024,
):
    print(tokenizer.decode([tok]), end="", flush=True)
```

### Adding MTP (optional, +0.3x on generation tasks)

```python
from four_path.mtp_patch import patch_mtp

model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-4bit")
patch_mtp(model, "~/models/Qwen3.5-9B-MLX-4bit-MTP")
# model now has mtp_forward() — four_path_generate_step uses it automatically
```

MTP weights are from the Qwen3.5-9B model's own MTP heads, converted via mlx-lm. The patch loads them at runtime onto stock mlx-lm without needing any fork.

## Running the benchmarks

```bash
# Real-world benchmark (pulls 10-K filings from EDGAR)
python benchmarks/realworld.py

# Standalone ISDA benchmarks (requires sample files in benchmarks/samples/)
python benchmarks/four_path.py

# Full ablation (all individual paths + combinations)
python benchmarks/all_paths.py
```

## Key findings

1. **Naive cascading beats smart routing** on consumer hardware. We tested adaptive orchestration (throttle underperforming sources) and multi-check recovery (re-verify at rejection points). Both hurt performance. When verification costs ~50ms, the overhead of routing logic exceeds the savings.

2. **Speedup is task-dependent, not domain-dependent.** The same model on the same domain can range from 1x to 4.6x depending on how much of the output echoes the input. Boilerplate generation and structured output get the most benefit.

3. **The ANE matters on analytical tasks.** When N-gram has less fuel (novel analysis, shorter context), the ANE's 1.7B neural predictions become the primary draft source. On ISDA clause analysis, ANE contributed 154 tokens vs N-gram's 60.

4. **MTP catches what others miss.** The model's own MTP head uses hidden states from the current forward pass. It has the highest per-token accuracy of any draft source but only produces one token per round. It fills gaps between N-gram chains and ANE predictions.

## Hardware

Built and tested on MacBook Air M5 (16GB unified memory, 10 GPU cores, Metal 4). The ANE path requires a CoreML model loaded on the Apple Neural Engine via [orion-ane](https://github.com/MidasMulli/orion-ane).

On 64GB M5 Pro with a 70B target model, verification cost increases from ~50ms to ~200ms per token. This makes each accepted draft token worth more wall-clock time, potentially shifting the break-even point for adaptive orchestration.

## Prior work

- [orion-ane](https://github.com/MidasMulli/orion-ane) — Three-tier architecture (CPU extraction + ANE enrichment + GPU reasoning), persistent ANE server, memory daemon
- N-gram hash table based on llama.cpp's ngram-mod algorithm (LCG hash, 4M entries)
- MTP heads from [ml-explore/mlx-lm PR #990](https://github.com/ml-explore/mlx-lm/pull/990) (AirRunner), loaded at runtime via `mtp_patch.py`

## License

MIT
