# GoS memory, execution and runtime extensions

The supplied Gemma 5070 recipes enable the holographic scratchpad, recurrent
latent compaction, thermal recursion, matrix branches, dual friction, coherence
parameters and actuator parameters. These are experimental architectural
changes, not evidence of improved reasoning accuracy or established research
novelty. No external model weights or datasets were downloaded for verification.

## Recursive memory and stopping

`gos_memory.py` implements real unitary circular convolution. Learned Fourier
phases bind each node's value to its role; conjugate phases unbind it. The pad
superposes bound values with a decayed recurrent write. Single-entry retrieval
is exact to numerical precision; multiple entries introduce interference.
FFT operations run in float32 even with BF16 adapter parameters.

The compactor recurrently attends over its previous slots and the new graph
state tokens. It keeps four slots by default, rather than storing all latent
node tokens from all reasoning rounds. This compacts **recursive reasoning
history per causal token**; it does not merge input text tokens or transformer
KV entries. Scratchpad and slots are initialized inside every forward call.
They cannot leak across examples, supervised-token chunks or checkpoint replay.

Thermal control measures normalized routing entropy, its change between rounds,
and RMS movement of graph states. Temperature modulates evidence attention and
graph routing. A token stops only when entropy change and movement both meet
the configured tolerances, after `min_rounds`. Twelve rounds is the default
maximum. Training and inference use identical discrete stopping decisions;
there is no claimed gradient through the threshold. Stable states can still
be incorrect. `TokenGraph.features_and_risk(..., diagnostics=True)` exposes
round counts, entropy and temperature without storing mutable diagnostics.

## Coherence, branches and friction

`gos_cognition.py` accepts pre-aligned modality vectors and fuses them with the
text anchor using masked attention. SFT can consume optional local row fields:

- `modal_states`: `[rendered_token_count, modalities, backbone_hidden_width]`.
- `modal_mask`: `[rendered_token_count, modalities]`, true for available evidence.

Vectors must already be mapped to the backbone hidden width by an external
encoder and aligned causally: a position must not include future answer
information. This repository does not download or train image/audio encoders.
Masks, shapes and finite values are checked; semantic/temporal alignment is the
data producer's responsibility. Windowing preserves the alignment. A weighted
cosine coherence loss accompanies the token loss. With ordinary text-only data,
the coherence branch is dormant. The text GRPO rollout does not consume these
extra vectors. Direct head scoring supports `modal_states`/`modal_mask` too.

Independent node matrix branches precede graph transitions. CUDA streams can
launch the branches independently during inference; a barrier precedes fusion.
Training uses the same branch matrices in one batched equation. On the actual
5070, batched matrices were faster for the tested small graph, so
`async_branches` defaults to false. Set it true to experiment, then profile.

Dual friction has separate semantic and operational logits. Their modeled
combined failure probability is `1 - (1-p_semantic)*(1-p_operational)` and feeds
routing with detached critic predictions. Explicit execution outcomes train the
operational stream; verified completion rewards train the semantic stream.
Those completion rewards remain coarse labels, and do not prove a particular
node's logic correct. Predicted low friction is never used as its own RL reward.

## Attention-based eviction and INT4

`gos_h2o.py` implements an H2O-style policy: accumulate received attention mass,
retain heavy hitters per KV head, and always retain recent tokens. GQA query
heads are averaged within each KV group. The primary method is
[H2O, NeurIPS 2023](https://arxiv.org/abs/2306.14048). This is an implementation
variant, not the authors' original kernels or an exact reproduction.

The scoped Gemma 4 attention backend owns its own bounded K/V tensors; HF
`DynamicCache` is disabled. Both prompt prefill and decoding stream one token at
a time. Absolute RoPE positions survive eviction. Sliding-attention layers mask
using absolute positions. Full and sliding layers, including shared-KV Gemma
layers, are supported. Shared-KV layers keep separate per-layer retained caches;
this duplicates storage across those layers. Text-only, causal, unpadded batch
size one is supported. Beam search and concurrent sessions on one model are not.
The backend and cache state are restored/released on success or exception.

The default budget is 256 heavy + 256 recent tokens per head. Persistent storage
stays at that budget after eviction; attention temporarily sees budget + 1.
Select `inference_cache: "sliding"` for pure FIFO sliding-window eviction: only
`h2o_recent_tokens` is retained, with zero heavy slots. Eviction changes model
outputs and must be measured on held-out tasks. There is no quality guarantee.

`gos_int4.py` packs two symmetric 4-bit codes into each byte, with float32 scales
per 32-dimensional group. `gos_kernels.py` provides actual Triton pack/unpack
kernels; a PyTorch reference handles CPU and parity checks. K/V persist packed
between steps. Attention temporarily dequantizes the bounded cache; this is not
an INT4 attention matrix-multiply kernel. Scores/positions/scales and decode
workspace still consume memory. Tiny dimensions can lose the storage benefit
to padding/scales. Quantization introduces additional approximation.

SFT and GRPO policy scoring keep full causal context, and GRPO sampling keeps
the same context. H2O/sliding/INT4 are used by generated test answers and the
standalone generation CLI. Enabling an inference cache never silently changes
the GRPO behavior policy without corresponding likelihood replay.

## Tool traces and actuator gating

Local `tool_interactions` rows are converted into assistant tool calls and tool
result turns. Supply `code`, `sandbox_raw_return`, and optional explicit
`success`/`is_error`; an optional final `answer` supplies a negative gate target.
The attachment's command-only rows are retained as trace data, but arbitrary
shell commands are not executable through the runtime's Python-only interface.
Plain error strings are not guessed into outcome labels.

Actuator targets are attached to explicitly labeled assistant turn endpoints.
The trained gate emits a probability; `--tools` executes a decoded structured
Python call only when that probability meets the threshold (default 0.85).
The call format is:

```text
<tool_call>{"name":"python","code":"print(1 + 1)"}</tool_call>
```

`gos_tools.py` pauses between decoded turns, executes in a restricted Docker
container, and reinjects bounded raw output/error text into the next context.
This is a trained end-of-attempt gate, not a claim that one hidden vector can
be decoded directly into a whole program, or that side effects occur inside
latent checkpoint recomputation. Tool execution success is distinct from
answer correctness. The separate GRPO test verifier requires actual unit tests.

Containers have no network, host mounts, writable root, added capabilities or
root user. CPU, memory, process count, wall time and captured output are bounded.
Images are never pulled automatically. Install a Python image separately and
configure `LRR_VERIFIER_IMAGE` if needed. Docker permissions must work first.
On this machine Docker is present but this session cannot access its socket;
execution/reinjection logic is tested with synthetic/mocked outcomes, not a live
container. There is no unsandboxed subprocess fallback.

## Kernels and checkpointing

The frozen backbone uses PyTorch SDPA. A forced Flash SDPA check passed on the
RTX 5070 (SM120). This is not FlashAttention-3. The upstream
[FlashAttention-3 implementation](https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release)
targets Hopper H100/H800. `attention_backend: "flash_attention_3"` has a hardware
and installation guard; it is rejected on this GPU. Its compatible-Hopper path
is wired through Transformers but was not tested on Hopper hardware.

The production default is non-reentrant PyTorch checkpointing around small graph
and vocabulary chunks. The optional DeepSpeed backend wraps those chunks only;
it does not apply ZeRO or shard the frozen NF4 base. See `gos_runtime.py` and the
[DeepSpeed checkpoint API](https://deepspeed.readthedocs.io/en/stable/activation-checkpointing.html).
The DeepSpeed non-reentrant API supports detached frozen features. Optional
`checkpoint_cpu_offload: true` moves checkpoint inputs to CPU. Single-process
coordination uses Gloo: NCCL teardown stalled in this runtime even after correct
gradients, and Gloo fixed the lifecycle problem. GPU gradient parity with
PyTorch checkpointing passed, including CPU offload. No memory saving is assumed
from adding DeepSpeed to a one-GPU frozen model. Install the `deepspeed` project
extra and set `checkpoint_backend: "deepspeed"` to select it.

## Run locally

```sh
.venv-training/bin/python train_unified.py --config configs/gemma5070.json --check-config
.venv-training/bin/python generate_gos.py --adapter runs/gemma-gos/adapter --prompt "Write a Python palindrome checker" --cache h2o --kv-quantization int4 --kv-kernel triton
.venv-training/bin/python generate_gos.py --adapter runs/gemma-gos/adapter --prompt "Compute 15 factorial using Python" --tools
.venv-training/bin/python diagnose_gos.py
.venv-training/bin/python -m pytest -q
```

Training still requires the existing explicit local-data/model-cache workflow
in `UNIFIED_TRAINING.md`. Added parameter sets require a newly trained adapter;
old adapters load with their original settings. No learned accuracy is implied
by the synthetic pipeline checks.

The checked measurements are saved in `verification/`. In the 512-token,
128-dimension synthetic cache check, packed storage was 163,840 bytes versus
524,288 bytes for BF16 (including INT4 scales, excluding attention workspace).
This is a cache-storage measurement, not a whole-model VRAM reduction.
The offline production-path integration is reproducible with:

```sh
.venv-training/bin/python scripts/verify_offline_pipeline.py --output runs/offline-check
.venv-training/bin/python scripts/verify_offline_pipeline.py --deepspeed --output runs/offline-ds-check
```

Both commands generate tiny random Gemma weights and synthetic rows locally.
They do not fetch model weights or datasets and do not measure learned accuracy.
