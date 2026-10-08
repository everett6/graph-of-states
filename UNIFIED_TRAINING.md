# Unified Gemma + GoS training

The new entry point is `train_unified.py`. It uses PEFT for a custom nonlinear
GoS adapter, TRL for SFT/GRPO, and Accelerate through TRL for placement, mixed
precision, accumulation, and optimizer/checkpoint orchestration. The earlier
`train_lrr.py` remains available as the independent byte-decoder prototype.

No external datasets or model weights were downloaded during implementation.
Tests construct tiny random Gemma weights and synthetic rows locally. Missing
training packages were installed into `.venv-training`, which shares the
existing working CUDA PyTorch installation.

## Configuration without reading data or models

```sh
.venv-training/bin/python train_unified.py --config configs/gemma5070.json --check-config
```

This validates and prints the entire recipe. It never opens a dataset or loads
model weights. Production runs default to local model files and reject remote
dataset reads unless explicitly enabled.

The three ready-to-edit recipes are:

- `configs/gemma5070.json`: supervised distillation and feedback traces.
- `configs/gemma5070-grpo.json`: GRPO, in a separate output directory.
- `configs/gemma5070-test.json`: final held-out test evaluation.

## Sources and mixture

| Source | Default split | Weight | Overall allocation |
| --- | --- | --- | --- |
| `bespokelabs/Bespoke-Stratos-17k` | `train` | 2 | 50% |
| `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` | `full` | 1 | 25% |
| `m-a-p/CodeFeedback-Filtered-Instruction` | `train` | 1 | 25% |

Stratos and Fable together make up the requested 75% reasoning share. Feedback
gets 25%. Weights are configurable positive integers; validation enforces the
3:1 reasoning/feedback ratio. Sources emit according to a fixed weighted
schedule, so the ratio is exact over complete scheduling cycles. Allocation is
by training windows for SFT and assistant response opportunities for GRPO, not
by token count. Training repeats exhausted sources to preserve the ratio;
validation and test streams stop rather than repeat examples.

The [Fable dataset card](https://huggingface.co/datasets/MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x/blob/main/README.md)
lists `lite` and `full` splits. Only `full` is selected; overlapping variants
should not be added as independent corpora. Its long agent traces are handled
as overlapping token windows rather than truncated to their first 4,096 tokens.

Supported schemas include Stratos `conversations` with `from`/`value`, Fable
`messages` with strings or text/thinking/tool blocks, OpenAI-style messages and
tool calls, and `query`/`answer` or `prompt`/`answer` pairs. An OpenThinker-style
local export can replace Stratos using the same schema adapters; specify an
actual repository/config/split if using a different remote source.

The [Filtered-Instruction dataset](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction)
provides question/answer pairs; it does not supply live execution feedback or
unit tests by itself. Richer local tool-result traces are supported separately.
The code preserves explicit `is_error`/`success` labels and never invents
pass/fail outcomes from error-looking words.

## 80/10/10 split without leakage across conversation windows

A normalized initial user task determines a source-independent SHA-256 group.
A seeded hash assigns its group to train (buckets 0–79), validation (80–89), or
test (90–99). Membership is assigned **before** tokenization/windowing. All
later tool results, revisions, and windows of that task remain in its partition,
including matching tasks appearing in different datasets.

This is a deterministic streaming **80/10/10 allocation in expectation**. Exact
finite counts can differ from these percentages; enforcing exact counts would
require a complete grouped indexing pass. It prevents exact task overlap,
not semantic near-duplicates or contamination already present in a pretrained
backbone. Pin source revisions in the config for reproducible future runs.

Only training and validation partitions are instantiated during SFT/GRPO. The
test partition is instantiated by `--stage test`, after model selection. Test
results do not drive early model selection. That command reports both
assistant-token loss and greedy generated-answer rewards, and writes
`test-predictions.jsonl`. By default rewards are exact final-answer matches;
a code verifier changes them to measured test success.

## Local input files

Replace each source's `repo` with, or add, an explicit local `path`:

```json
{"name":"fable","path":"/absolute/path/to/fable.jsonl","weight":1}
```

The file is read line by line, with one raw conversation/pair object per line.
No conversion script or bulk in-memory dataset is required. Keep all three
weights in the recipe. Set `model` to an existing trainable Hugging Face model
folder or a model ID already cached locally. GGUF inference files are not
Hugging Face training checkpoints.

```sh
.venv-training/bin/python train_unified.py --config configs/gemma5070.json --stage sft
```

To enable streaming later when you actually want remote data:

```sh
.venv-training/bin/accelerate launch --num_processes 1 train_unified.py --config configs/gemma5070.json --stage sft --allow-remote-data
```

Streaming transfers dataset bytes over the network while iterating. It avoids
materializing the full corpus, but it is not zero data transfer. Without the
flag, remote reads are refused. Model downloading is separately disabled by
default and requires `--allow-model-download`. Dataset authentication uses the
normal Hugging Face environment; the code never prints access tokens.

## Memory and execution extensions

The supplied recipes now enable the new graph extensions and bounded INT4 inference cache.
See [Memory and runtime extensions](MEMORY_AND_RUNTIME.md) for the causal scope,
hardware support, tool trace format, backend choices, and measured limitations.
Training/GRPO keep the context policy described below; generated test answers can use H2O or sliding eviction.

## Memory guardrails

- Production loading requires bitsandbytes **4-bit NF4**, double quantization,
  and BF16 compute. No fallback to a dense model is allowed.
- Every backbone parameter, embedding, and original LM-head weight is frozen.
  A runtime audit permits gradients only inside `GoSReasoningLayer`.
- PEFT manages custom adapter parameters; ordinary LoRA is not applied to
  transformer layers. The graph's residual up-projection starts at zero, so
  attaching an untrained GoS adapter initially preserves backbone logits.
- Maximum total sequence length is **4,096**. SFT windows are capped at that
  value; GRPO defaults to at most 3,072 prompt + 1,024 completion tokens.
- Every training vocabulary projection and graph computation is checkpointed
  in chunks of 32 supervised tokens. This avoids allocating the complete
  `[4096, Gemma vocabulary]` logits tensor. Chunk size is configurable.
- Microbatch size is one. Gradient accumulation defaults to eight.
- Frozen decoder calls use `no_grad`, return only final hidden states, and
  disable KV caching and attention-history outputs.
- GRPO generates group members serially and disables KV caching; this trades
  speed for memory. Group size defaults to four. vLLM and a second reference
  model are disabled (`beta=0`). Token policy GRPO is used; graph nodes are
  differentiable soft hypotheses, not separately sampled discrete policies.
- PyTorch's per-process allocator fraction is capped at 90% of GPU capacity.
  Allocated/reserved peak memory is recorded in stage metrics.

Accelerate is not a general memory-leak cure and this recipe does not silently
offload blocks to CPU. The frozen backbone is placed explicitly on one GPU.
Quantized weight residency, embeddings, allocator overhead, and runtime workspace
still count. Neither a fixed 7.5 GB residency nor a fixed 2 GB checkpointing
saving is assumed. Capacity of the real 12B model at 4,096 tokens must be
measured when its weights and data are available.

The backbone's `gradient_checkpointing` setting is intentionally disabled
because it never has an autograd graph. Checkpointing is enforced around the
**active graph and output projection** in `selected_logps`; it is not disabled
for the trainable head.

## What the GoS head changes

For each causal decoder state, a projection creates multiple role-conditioned
nodes. Shared recursive transitions use directed compatibility scores that
compare the proposed changes of source and destination nodes. Soft routing and
residual gates update hypotheses. Mean and variance fusion produces a latent
correction, which is projected back into Gemma's hidden width. Gemma's original
frozen output matrix then produces token logits.

The graph never sees later token states. It preserves Gemma's decoder and
vocabulary instead of learning a new language decoder from scratch. The adapter
is nonlinear and cannot be merged into a conventional LoRA matrix; use
`load_gos_adapter`, which re-registers its PEFT mapping. Only adapter weights
are saved; frozen embedding matrices and the backbone are excluded.

## SFT, GRPO, and friction supervision

SFT applies cross-entropy only to assistant content and its stop tokens. User,
system, and tool-result text are context. Overlapping context is masked so its
assistant targets are not charged twice. Long traces stay in one split.
Templates must be text-only and prefix-stable; incompatible renderings fail
rather than silently train with incorrect masks.

An explicit tool outcome supervises friction at the end of the preceding
assistant attempt. Unlabeled question/answer pairs train generation but supply
no invented critic targets. GRPO trains the critic on verified outcomes of
sampled completions. This is coarse completion-level supervision, not a
node-specific proof of correctness.

GRPO uses TRL's group-relative policy objective. Its reward is verified success,
equivalent to `1 - measured_friction`, with `measured_friction = 1 - success`.
Predicted neural friction never serves as its own correctness reward.

Start GRPO from the trained SFT adapter:

```sh
.venv-training/bin/python train_unified.py --config configs/gemma5070-grpo.json --stage grpo --adapter runs/gemma-gos/adapter --allow-remote-data
```

Omit `--allow-remote-data` when all sources have local paths. Both recipes must
use the same model and graph dimensions. GRPO's sampled tokens change future
causal states and hence graph evolution; it does not assume that each graph
node has a separately observable execution reward.

The runnable baseline reward strips `<think>...</think>` and compares the final
answer exactly. It is useful for exact-answer math/logic. Semantic code testing
requires a verifier. Set `reward_function` to `module:function`; it receives
`completions`, `answer`, and dataset metadata and must return one finite success
score in `[0,1]` per completion.

An optional implementation, `docker_code_reward:code_test_reward`, accepts local
rows with a `tests` string such as `assert add(2, 3) == 5`. It executes candidate
code in a restricted Docker container with no network or host mounts, resource
limits, and a read-only filesystem. It requires an already installed Python
image; **it never pulls images**. Neither provided public source automatically
supplies these tests. Do not infer code correctness from compilation alone.

## Checkpoints and final test

TRL writes optimizer/scheduler/RNG checkpoints at `save_steps`. Custom save hooks
include `gos_config.json` and omit frozen embeddings. Resume with the same
recipe, revisions, and streaming source order:

```sh
.venv-training/bin/python train_unified.py --config configs/gemma5070.json --stage sft --resume runs/gemma-gos/checkpoint-100 --allow-remote-data
```

Use a separate test output directory and the final selected adapter:

```sh
.venv-training/bin/python train_unified.py --config configs/gemma5070-test.json --stage test --adapter runs/gemma-gos-grpo/adapter --allow-remote-data
```

Test evaluation is bounded by `eval_samples`, default 128. Increase it for your
final experiment. It does not run automatically at the end of training.

## Verification

```sh
.venv-training/bin/python -m pytest -q
```

The suite uses synthetic schemas and tiny local weights. It checks mixture
ratios, group splits, thinking/tool normalization, 4,096-token windows,
assistant masking, frozen gradients, causal invariance, PEFT adapter round trips,
chunked gradient equivalence, real TRL SFT/resume and streaming GRPO updates,
verified critic targets, NF4 GPU backpropagation, and verifier isolation flags.
No accuracy or VRAM claims for the real 12B model follow from these checks.
