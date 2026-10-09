# Measured controlled pilot

Experiment commit: `da9992f807c8d32923679640ba67813c9ba5364e`; RTX 5070, seed 17. Each trainable variant completed eight SFT updates: four local Fable examples and four generated verified examples. All 11 held-out generated tasks passed for every variant. This shows no reasoning gain.

| Experiment | Trainable parameters | Training peak GiB | Input train tokens/s | Native-cache decode tokens/s | Test correct |
|---|---:|---:|---:|---:|---:|
| frozen-d1 | 0 | 7.114 | 0.0 | 22.9 | 11/11 |
| gos-d1 | 273,256 | 7.313 | 307.7 | 22.3 | 11/11 |
| gos-d2 | 273,256 | 7.313 | 265.3 | 21.6 | 11/11 |
| gos-d4 | 273,256 | 7.313 | 224.3 | 20.1 | 11/11 |
| gos-without-auxiliary_critics | 273,256 | 7.313 | 171.5 | 20.6 | 11/11 |
| gos-without-edge_routing | 273,256 | 7.313 | 229.0 | 20.4 | 11/11 |
| gos-without-holographic_scratchpad | 273,188 | 7.313 | 247.3 | 20.8 | 11/11 |
| gos-without-recurrent_memory | 269,000 | 7.313 | 241.4 | 20.7 | 11/11 |
| gos-without-thermal_recursion | 273,256 | 7.313 | 239.3 | 20.8 | 11/11 |
| lora-d1 | 1,298,432 | 7.445 | 162.5 | 19.4 | 11/11 |
| mlp-d1 | 273,256 | 7.313 | 169.4 | 22.9 | 11/11 |

Peak memory is PyTorch allocated memory; driver/device totals differ. Timings are short, single-run measurements, with checkpoint writing included in training. All variants use the same source/data hashes. Independent exhaustive cross-partition checks found no lexical/trace leakage in the selected 106 rows. This does not certify semantic or pretraining contamination freedom.

All GoS exports reproduced the trained head logits exactly. The recommended reference artifact is the depth-four run, selected by the predefined configuration, not by test accuracy: [custom GoS adapter GGUF](../runs/controlled-final-v1/gos-d4/gos-adapter.gguf). It requires the original local NF4 Gemma backbone and [Python loader](../generate_gguf_adapter.py). It does not run in stock llama.cpp or Ollama.

Full outputs, manifests, checkpoints, per-variant configuration, cache measurements, and exact source snapshots are in `runs/controlled-final-v1/`. This directory is intentionally ignored by Git. Aggregated machine-readable measurements are in [controlled-results-v1.json](controlled-results-v1.json).

Cache eviction and INT4 change the predictive distribution. Even unevicted custom attention differs numerically from full-sequence SDPA on the real BF16/NF4 model. Consult measured KL, logit errors, and top-1 agreement before choosing a memory policy. Tiny FP32 equivalence tests are not an NF4 quality guarantee.

For reproducible commands, verifier scope, ablation switches, recovery and limitations, see [CONTROLLED_BENCHMARKS.md](../docs/CONTROLLED_BENCHMARKS.md).

Regression validation: **58 tests passed**, including all 43 original tests. The conventional baseline also has a [standard LoRA adapter GGUF](../runs/controlled-final-v1/lora-d1/lora-adapter.gguf), requiring a matching full Gemma GGUF.
