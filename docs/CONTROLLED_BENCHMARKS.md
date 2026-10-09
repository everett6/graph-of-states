# Controlled GoS experiments

`benchmark_gos.py` runs a frozen local Gemma text backbone, conventional attention-projection LoRA, an exactly parameter-matched two-layer residual MLP, and GoS at configurable recursion depths. Every variant starts in a fresh process with the same original weights, seed, source rows, partition, and update budget. Five optional leave-one-component-out ablations disable recurrent slots, holographic scratchpad, edge routing, thermal recursion, and auxiliary critic routing. Existing training defaults remain unchanged.

## Run and recover

Install the existing `unified` dependencies and the new `benchmark` extra. No models or datasets are downloaded by the harness. The provided configuration points to this machine's existing cached Gemma 12B weights and local Fable files; replace those paths on another machine.

```bash
.venv-training/bin/python benchmark_gos.py \
  --config configs/benchmark-controlled-v1.json --output runs/my-controlled-run
# After interruption, use the identical configuration and output directory:
.venv-training/bin/python benchmark_gos.py \
  --config configs/benchmark-controlled-v1.json --output runs/my-controlled-run --resume
```

Checkpoints atomically save trainable tensors, AdamW state, step, Python/NumPy/Torch/CUDA RNG states, and configuration/data digests. Recovery rejects different configurations or datasets. This supports recovery within the originally planned step budget; changing that budget requires a new experiment. Load only trusted local PyTorch checkpoints. Frozen weights are referenced, not copied into every checkpoint.

`comparison.json` combines all experiment reports. Each report records software/CUDA/GPU information, source hashes, Git revision and branch, trainable parameter count, training loss, validation likelihood, consumed training sources, generated answers/verdicts, peak allocated and reserved GPU memory, latency, generation throughput, supervised/input training throughput, and cache profiling. `dataset_manifest.json` contains the actual selected examples, grouping assignments, and references. These files stay under ignored `runs/`; keep that directory to reproduce the results. Source snapshots accompany the recorded run.

The MLP has two dense projections and partial learned input/output biases to exactly fill the graph's parameter budget; every bias enters the forward computation. Conventional LoRA targets `q_proj` and `v_proj`, uses rank 2 by default, and is reported with its own parameter count. It is not claimed to be parameter-matched. GoS/MLP freeze decoder features; conventional LoRA differentiates through the checkpointed decoder. All trainable variants use the same token objective. This pilot does not add a separate critic-supervision loss; the critic ablation concerns routing, not evidence of a trained logical verifier.

## Local verified suites and rewards

Set `synthetic_tasks` to false and `verified_tasks` to a list of local JSONL paths for substantive experiments. Each row requires `prompt`, `answer` (both strings), and `task_spec`. Optional `source` and `trace_id` preserve grouping. References must pass their verifier before training. Entire verified references must fit the configured sequence budget; they are never silently truncated. Existing pretokenized Fable windows are capped to the configured length and only window zero is selected.

```json
{"prompt":"Return one half as a number.","answer":"0.5","task_spec":{"kind":"math","expected":"1/2","tolerance":"0"}}
{"prompt":"Return a JSON object with count equal to two.","answer":"{\"count\":2}","task_spec":{"kind":"structured","expected":{"count":2},"schema":{"type":"object","required":["count"],"additionalProperties":false,"properties":{"count":{"type":"integer"}}}}}
{"prompt":"Implement solve(x) returning x plus one.","answer":"def solve(x):\n return x+1","task_spec":{"kind":"coding","verifier":"restricted_python","cases":[{"args":[-2],"expected":-1},{"args":[5],"expected":6}]}}
```

Mathematics uses bounded AST arithmetic and exact rational comparison with an explicit tolerance. It does not verify arbitrary symbolic identities or proofs. Structured answers must parse as strict JSON, satisfy an optional schema, and match the typed reference; duplicate keys, nonfinite values, and bool/number substitutions fail. Coding uses the existing isolated Docker unit-test verifier by default (`tests` in `task_spec`). Unavailable execution infrastructure is an unavailable verifier, not a successful answer. The explicit `restricted_python` option interprets a bounded pure-function subset without host `exec` or `eval`; arbitrary Python, loops, calls, imports and I/O are rejected. Passing these finite cases is not a proof of general program correctness.

For the existing TRL trainer set `reward_function` to `verified_rewards:task_verified_reward` and supply `task_spec` in each source row. The unified data path preserves it through GRPO and generated-answer evaluation. Scores are verified success (0/1), never the model's own friction score. No numeric/JSON reference can be inferred safely from arbitrary distilled prose, so source rows need explicit specifications.

## Leakage prevention

The benchmark combines selected local sources before assigning splits. It groups exact normalized prompts, source-scoped trace IDs, and all prompt pairs with normalized word/punctuation 3-gram Jaccard similarity at or above the configured threshold (default 0.8). Shared-shingle indexing finds every pair meeting this metric; connected components prevent transitive leaks. An order-independent representative plus seed hashes whole groups into 80/10/10 buckets in expectation. Small samples need not have exact percentages.

The audit records group counts, detected near-duplicate pairs, partition counts, and cross-split matches. The tests exercise near matches, repeated traces, row-order invariance, and recovery. This guarantees separation only for the selected corpus and defined lexical metric. It does not detect all paraphrases, semantic template overlap, or contamination already present in Gemma pretraining. The original streaming splitter retains its existing behavior and does not acquire a global near-duplicate guarantee; use the audited benchmark path when that guarantee is needed. Exhaustive candidate verification can be quadratic for highly repetitive corpora, so source row limits are deliberate.

## Cache measurements and GRPO

Existing GRPO behavior remains serial generation with no inference cache. `grpo_use_cache: true` opts into the exact, unevicted native Transformers cache. `grpo_feature_cache_entries` and `grpo_feature_cache_bytes` enable a bounded CPU LRU of detached frozen decoder features during repeated completion scoring. Adapter logits and gradients are always recomputed. Caching rejects a trainable decoder; clear/recreate it if frozen weights are manually changed. Host hashing/transfers and low hit rates can outweigh its savings. It is not prefix KV sharing or batched generation.

The profiler calls the actual `sequential_rollout` hook and completion-likelihood scorer with the same seeded prompts for all four native-cache/feature-cache combinations. Reports include hit/miss counts, output/log-probability hashes, latency, tokens/second and allocated memory. These short timings have no confidence interval and do not establish a general speedup.

Native cache and full-context greedy results are measured on the same held-out tasks. `cache_quality.py` also measures teacher-forced top-1 agreement, KL divergence, and maximum logit error for unevicted H2O, evicting H2O, a sliding window, and packed INT4. It reports a native incremental baseline separately: low-precision streaming kernels can differ from full-sequence kernels even before eviction. H2O and INT4 remain approximate inference-only options and are not enabled for GRPO training/scoring. Test tolerances on tiny FP32 models do not certify bitwise equivalence on NF4 12B weights.

Peak GPU values are PyTorch allocator counters, not total device/driver usage. Training timing includes per-step checkpoint serialization and gradient clipping, excludes initial model loading, and counts both full input tokens and supervised target tokens separately. Inference latency covers synchronized generation; tokenization and the initial transfer are outside the timed interval. Cold starts, long contexts, statistical repeated measurements, energy and multimodal inference need separate experiments.

## GGUF exports

GoS is a nonlinear recurrent head. Its faithful export is `gos-adapter.gguf`, a custom `gos_adapter` GGUF architecture containing every trained head tensor and versioned topology, graph, and original-backbone metadata. It requires the original local NF4 Gemma backbone and this Python runtime. It is **not a standalone model and is not executable in stock llama.cpp or Ollama**. Both synthetic tests and the actual trained export check exact head-logit round trips. Do not rename it to imply standard Gemma support or merge it as ordinary LoRA.

```bash
.venv-training/bin/python generate_gguf_adapter.py \
  --adapter runs/my-controlled-run/gos-d4/gos-adapter.gguf \
  --prompt 'Calculate 17 * 9. Return only the number.'
```

The conventional LoRA baseline can additionally be converted by the installed llama.cpp `convert_lora_to_gguf.py` with a native Gemma4 text configuration. That file is a standard **LoRA adapter GGUF**, still requiring a matching full Gemma GGUF. Its conversion is separate from the custom GoS export. Stock execution of nonlinear GoS requires a new runtime architecture or a separately validated distillation; neither is claimed here.

## Interpretation

The checked configuration is an eight-update, 192-token smoke pilot on local Fable plus generated elementary tasks, with one held-out task per category and one seed. It verifies that training, comparison, profiling, recovery and export work on the card. It is not a published coding/math benchmark, not convergence training, and not evidence of reasoning gains. Report the frozen baseline, all ablations, truncations, sample counts, verification coverage and parameter budgets. For efficacy claims use harder independently sourced suites, larger held-out sets, multiple seeds, equal data/compute budgets, confidence intervals and contamination controls. Never select a checkpoint or ablation based on these test outcomes.
