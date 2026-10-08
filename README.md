# Latent Recursive Reconstruction

For frozen Gemma NF4 training with PEFT, TRL SFT/GRPO, Fable/Stratos/CodeFeedback mixing,
and task-grouped 80/10/10 splits, see [Unified training](UNIFIED_TRAINING.md).

For holographic memory, entropy-controlled recursion, H2O/sliding eviction, INT4 Triton caches,
coherence, matrix branches, dual friction and tool gating, see [Memory and runtime extensions](MEMORY_AND_RUNTIME.md).

A runnable PyTorch model with frozen prompt features, a recursive Graph-of-States
reasoner, and an autoregressive byte decoder. Includes JSONL preparation,
train/validation splitting, minibatch training, mixed precision, gradient
accumulation, checkpoint resume, evaluation, and generation.

## Start with the complete offline demo

Run from this directory with your existing PyTorch environment:

```sh
python train_lrr.py demo --device cuda --precision bf16 --epochs 30 --output runs/my-demo
```

Use `--device cpu --precision fp32` for CPU. The demo creates its own 48-example
Boolean logic dataset, caches features, trains, evaluates the checkpoint with
the lowest validation cross-entropy, writes predictions, and generates an answer.
It needs no network access, downloaded models, or additional dependencies.

The builtin encoder is a small **random frozen transformer** used to verify the
pipeline. It has no pretrained world knowledge. Its exact weights are bundled
in the checkpoint for portable inference. A real reasoning experiment should
use pretrained backbone features and a substantially larger dataset.

## Train on your own dataset

Use one JSON object per line:

```json
{"prompt":"Compute 17 * 23.","answer":"391","passed":true}
{"prompt":"Return a Python function that adds two arguments.","answer":"def add(a, b):\n    return a + b"}
```

`passed` is optional. Explicit failures are excluded, duplicates are removed,
and conflicting answers for the same prompt are rejected. Answers are supervised
training targets. Include reasoning steps in `answer` if you want to teach them.

First verify your data with the builtin encoder:

```sh
python train_lrr.py prepare --data examples/logic.jsonl --backbone builtin --device cuda --output runs/logic-data --validation-fraction 0.2
python train_lrr.py train --features runs/logic-data/train --validation-features runs/logic-data/val --output runs/logic-model --device cuda --precision bf16 --epochs 30
```

For pretrained features, use an existing Hugging Face-format model directory:

```sh
python train_lrr.py prepare --data train.jsonl --backbone /absolute/path/to/local-backbone --local-files-only --quantize-4bit --device cuda --output runs/my-data --max-prompt-tokens 512 --max-answer-tokens 512
python train_lrr.py train --features runs/my-data/train --validation-features runs/my-data/val --output runs/my-model --device cuda --precision bf16 --width 256 --nodes 4 --rounds 8 --checkpoint-rounds --batch-size 2 --accumulation-steps 8 --epochs 20 --learning-rate 0.001 --warmup-steps 100
```

The local backbone directory must include compatible model weights, config, and
input tokenizer. Optional Hugging Face support uses `transformers`, `accelerate`,
and `safetensors`; 4-bit loading additionally uses `bitsandbytes`. Installation
metadata is in `pyproject.toml`, with extras `backbone`, `quantization`, and
`test`. Preserve your working CUDA PyTorch installation when setting up an
environment. No model downloads or dependency installations are performed by
the demo. The prepare command can also accept a remote model ID when network
access is explicitly desired; omit `--local-files-only` in that case.

Preparation exits before training starts, releasing backbone VRAM. Final hidden
states are cached in FP16, one example per file. Disk use grows with prompt
length and backbone width. The decoder always uses a compact UTF-8 vocabulary
of 259 entries; its targets do not depend on the backbone tokenizer.

The limits reject overlong answers rather than teach truncated solutions.
Overlong prompts also fail unless `--truncate-prompts` is selected. Prompt
truncation can remove the actual question, so set the limits deliberately.
Prepared train/validation splits contain disjoint prompts, and the trainer
checks for prompt leakage between separately supplied caches.

## Resume, evaluate, and generate

Checkpoints are written atomically at the end of every epoch:

- `last.pt`: latest model, optimizer, precision scaler, random state, and progress.
- `best.pt`: checkpoint with lowest validation token loss, or training loss when
  no validation cache is provided.
- `metrics.jsonl`: epoch training loss, validation loss, token accuracy,
  generated exact match, and CUDA peak memory.
- `run.json`: command settings, architecture, and parameter count.

Resume with the same optimizer, batch, precision, seed, and dataset settings;
`--epochs` is the total target epoch count. Architecture comes from the checkpoint.
For the simple `logic-model` example above:

```sh
python train_lrr.py train --features runs/logic-data/train --validation-features runs/logic-data/val --output runs/logic-model --resume runs/logic-model/last.pt --device cuda --precision bf16 --epochs 40
python train_lrr.py evaluate --checkpoint runs/logic-model/best.pt --features runs/logic-data/val --device cuda --predictions runs/logic-model/predictions.jsonl --output runs/logic-model/evaluation.json
python train_lrr.py generate --checkpoint runs/logic-model/best.pt --prompt '1 AND 0 =' --device cuda --max-new-tokens 16
```

For a pretrained backbone, generation loads the original backbone to extract
prompt features, releases it, and then generates through the trained graph and
decoder. Keep the original model files available. `--backbone-device` allows
separate feature extraction placement. A checkpoint trained with 4-bit features
requires a compatible CUDA backbone runtime; quantization is never silently
changed. Full training checkpoints contain optimizer state and are larger than
weights-only files.

Generation uses the trained fixed graph depth by default. `--dynamic-halt`
enables an experimental per-example stability heuristic; evaluate its impact
before deployment. It halts on small node movement, not proven correctness.

## The graph transition

Each bounded node has a learned role and independently attends to prompt
evidence. A shared proposal network creates a displacement from its previous
state. Directed edge scores depend on the destination candidate, source
candidate, and **the difference between their displacements**. This tests the
hypothesis that relationships between proposed changes help route competing
hypotheses better than static similarity alone.

Incoming softmax weights combine edge compatibility with detached critic
reliability. Residual gates move each candidate toward its incoming message.
Final fusion includes both the node mean and coordinate-wise variance, exposing
a summary of disagreement to the decoder. Parameter sharing lets the transition
repeat without allocating a new parameter set for every round. Activation
checkpointing trades recomputation for lower training memory.

Nodes have bounded count and learned roles. Soft routing reduces influence;
it does not perform hard node deletion. Variance is a feature, not a calibrated
uncertainty measure. Scientific novelty and reasoning gains require experiments.

## Learning objective and current scope

The supported training objective is token cross-entropy on accepted answers,
including rejection-sampling fine-tuning on externally verified successes.
Gradients update the graph, feature projection, and decoder. Backbone features
are detached, and the backbone is frozen.

The core exposes optional coarse outcome supervision for the risk predictor.
The supplied supervised trainer leaves that predictor at uniform 0.5, so risk
has no effect on routing until supervised. The predictor does not inspect
candidate generated answers, and is not a logical contradiction oracle.
Policy-gradient RL and execution-based rewards are future extensions; the CLI
does not execute dataset code.

The decoder starts from random weights. Freezing a knowledgeable backbone does
not automatically transfer its fluent text generation into this decoder.
Training and evaluation on your actual tasks determine whether this design is
useful. The builtin demo verifies execution and learning mechanics, not broad
math/code reasoning or superiority over the original backbone.

## Checks

```sh
python -m pytest -q
python gos_engine_architecture.py --smoke-test
```

Tests cover Unicode/EOS handling, data rejection and duplicate checks, disjoint
splits, padding invariance, conditional learning and graph gradients, exact CPU
epoch resume, token-weighted gradient accumulation, checkpointing, optional
halting, local Hugging Face loading, and CUDA BF16/FP16 training and generation.
CUDA checks skip when the process cannot see the GPU. The Hugging Face check
creates local tiny random weights and downloads nothing.

For research evaluation, compare original-backbone answers, pooled-feature
conditioning, a one-round graph, and the recursive graph on held-out tasks.
Ablate displacement-based edges and variance fusion. Report multiple seeds,
latency, verified answer success, and measured memory with comparable budgets.
No fixed percentage memory savings or 12 GB capacity guarantee is assumed.
