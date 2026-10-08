# Verification record

Validated on the user's NVIDIA GeForce RTX 5070, 12,227 MiB total VRAM,
with driver 615.71.09 and PyTorch 2.14.0+cu130. GPU access requires execution
outside this session's sandbox; the host driver and CUDA runtime work.

- Integration suite: **11 passed**, including CUDA BF16 and FP16 training and
  generation, exact CPU epoch resume, and local Hugging Face feature extraction.
- Standalone architecture smoke test: passed; synthetic loss decreased from
  2.5552 to 0.0291. This verifies optimization, not generalization.
- Full CLI demo: prepared 38 training and 10 validation prompts, trained 30
  epochs with BF16 on CUDA, evaluated, and generated text.
- Full CLI resume: restored model, optimizer/scaler, random state, and progress
  from epoch 30; completed epochs 31 and 32 with optimizer step advancing
  from 150 to 160.
- Demo training peak PyTorch allocation: **78,769,664 bytes (75.12 MiB)**;
  reserved memory: **92,274,688 bytes (88 MiB)**. These figures cover the tiny
  builtin demo and do not estimate a pretrained 7B/8B backbone's memory.
- Best checkpoint selected by validation cross-entropy: loss 0.35426 and
  generated exact match 7/10 on the validation split. Latest checkpoint at
  epoch 32: validation loss 0.58125 and exact match 3/10. These small, shared
  truth-table examples are pipeline validation, not a reasoning benchmark.
- Generated example from the best checkpoint: `1 AND 0 =` → `0`, followed by EOS.

Local artifacts:

- `runs/5070-verified/training/best.pt`
- `runs/5070-verified/training/last.pt`
- `runs/5070-verified/training/metrics.jsonl`
- `runs/5070-verified/evaluation.json` (best checkpoint)
- `runs/5070-verified/predictions.jsonl` (best checkpoint)

No pretrained model weights or external datasets were downloaded. Real
pretrained-backbone quantization was not exercised: bitsandbytes is not
installed in this Python environment. The optional Hugging Face integration
was checked with locally constructed tiny GPT-2 weights and a local tokenizer.

## Unified trainer verification — October 8, 2026

The separate `.venv-training` environment now contains PEFT 0.20.0, TRL 1.14.2,
Accelerate 1.15.0, datasets 5.1.0, and bitsandbytes 0.50.2 with the existing
Transformers 5.17.0 / CUDA PyTorch installation.

- Full suite: **23 passed**, including both the original and unified trainers.
- Real locally constructed Gemma4 weights loaded through bitsandbytes NF4 on
  the RTX 5070; gradients reached the custom GoS head while the core stayed frozen.
- Real TRL supervised updates, adapter checkpoint round trips, and optimizer
  resume passed. Real streaming TRL GRPO updates and grouped evaluation passed.
- The full production entry-point workflow was exercised with synthetic local
  JSONL and a tiny local NF4 Gemma: SFT → saved adapter → GRPO → saved adapter →
  final test likelihood and generated-answer evaluation.
- Source ratios, deterministic task grouping, masked assistant-only windows,
  causal invariance, and checkpointed gradient equivalence passed.
- Optional Docker verifier flags were checked with mocks; no containers or
  candidate dataset code were executed and no Docker image was downloaded.
- No external datasets or model weights were downloaded. The real Gemma 12B
  model and the public corpora have not been trained or memory-profiled here.

See `UNIFIED_TRAINING.md` for the production recipes and limitations.

## 2026-10-08 — memory, cognition, tool and runtime extensions

- RTX 5070 / SM120 / PyTorch 2.14.0+cu130 verified directly; GPU driver is available.
- Full synthetic NF4 Gemma4 pipeline completed SFT → GRPO → test with holographic
  scratchpad, thermal looping, recurrent slots, matrix branches, dual friction,
  and an INT4 Triton H2O inference cache. The complete pipeline also passed with
  DeepSpeed checkpointing and CPU activation offload.
- DeepSpeed uses single-process Gloo coordination. A debug run isolated a stall
  to NCCL process-group teardown, after successful gradient checks. Gloo resolved
  teardown; CPU offload itself passed. CPU offload stays optional.
- Forced PyTorch Flash SDPA passed on the 5070. FlashAttention-3 is guarded out
  because upstream targets Hopper. A Hopper execution was not verified.
- Triton INT4 packing matched the PyTorch reference exactly across 30 BF16
  random seeds, including half-step boundaries. Fixed reciprocal-scale rounding
  after discovering a one-ULP scale discrepancy that could change a 4-bit code.
- Paired modality fusion, causal chunk invariance, masked modality fallback,
  semantic/operational critic paths, actuator supervision, recursive halting,
  PEFT checkpoint round trips, GQA head-specific eviction, absolute positions,
  hybrid sliding/full attention and Gemma shared-KV layers are checked.
- Tool output is bounded and reinjected into the next model turn. Container
  restrictions and tool gating are tested with mocks. Live tool execution is
  blocked by this session's Docker socket permissions; no host-code fallback.
- Quantization timings and storage measurements: `verification/gpu-diagnostics.json`.
  Per-stage synthetic integration metrics: `verification/integration-results.json`.
  Reproduce the full path with `scripts/verify_offline_pipeline.py` (optional
  `--deepspeed`). It creates tiny random weights locally and reads no remote data.
- Random tiny-model test reward is zero; these are pipeline checks, not reasoning
  benchmarks. Real 12B/4096-token capacity and answer quality remain unmeasured.

Final suite: **43 passed** on the RTX 5070, including GPU/NF4/Triton and isolated DeepSpeed checks.
