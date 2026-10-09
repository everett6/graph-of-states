# Adaptive passes through shared GoS layers

The GoS head now supports an explicit adaptive mode. `graph_rounds` is its maximum pass budget, not a number of distinct layers. Each causal token state repeatedly uses the same graph transition weights and recurrent memory. It stops after both state motion and routing-entropy change stay below their thresholds for `halting_patience` consecutive passes, once `min_rounds` is reached. Remaining states continue up to the maximum. Halted states are removed from subsequent graph batches, rather than merely masking their outputs. Batch compaction reduces executed token-passes; it does not guarantee lower wall-clock latency for every batch size.

Default runtime overrides use 2–20 passes, patience 2, motion tolerance 0.05, and entropy-change tolerance 0.01. Temperature adaptation remains an independent `thermal` option. Stopping tests run identically during training, likelihood scoring and generation. Gradients flow through the passes executed; no gradient or optimality claim is made through the discrete stopping choice. Memory resets per invocation, preserving causal token independence and cache/chunk semantics.

Use the existing trained GGUF with a runtime override:

```bash
.venv-training/bin/python generate_gguf_adapter.py \
  --adapter runs/controlled-final-v1/gos-d4/gos-adapter.gguf \
  --prompt 'Calculate 17 * 9. Return only the number.' \
  --adaptive-reasoning --min-passes 2 --max-passes 20
```

The same arguments work with `generate_gos.py` for normal PEFT adapters. `--fixed-passes 4` explicitly disables adaptive and thermal early stopping. Without pass arguments, both commands preserve the policy stored in the original adapter. `--motion-tolerance`, `--entropy-tolerance`, and `--halting-patience` tune the adaptive rule. Supplying min/max pass arguments enables adaptive mode. Runtime overrides do not rewrite the input artifact.

`runs/controlled-final-v1/gos-d4/gos-adaptive-adapter.gguf` is a separate policy-only export of the existing trained weights with a stored 2–20-pass adaptive policy. It uses the same custom Python runtime and original NF4 Gemma backbone; it is not standalone llama.cpp/Ollama GGUF. No extra weights or learned halting classifier were introduced, so existing checkpoints remain structurally compatible. Save an updated graph configuration alongside a PEFT adapter, or provide matching rounds/enhancements when exporting, to persist a runtime override.

For training with adaptive passes from the start:

```bash
.venv-training/bin/python benchmark_gos.py \
  --config configs/benchmark-adaptive-v1.json --output runs/adaptive-training-v1
```

This benchmark configuration uses the same existing local legacy Gemma weights and pretokenized Fable sources as the measured pilot, with no downloads. It retains a short eight-update smoke budget; increase it for substantive training. `configs/gemma5070-adaptive.json` also exposes the policy to the existing unified trainer for a compatible Transformers backbone and configured local conversation sources. Neither configuration grants download permission. It sets `graph_rounds: 20` and `graph_enhancements` fields `adaptive: true`, `min_rounds: 2`, `halting_patience: 2`, and `motion_tolerance: 0.05`. Original fixed-depth and legacy thermal settings remain unchanged. The runtime API is `set_reasoning_budget(model, minimum=2, maximum=20, adaptive=True, patience=2)`.

`TokenGraph.features_and_risk(..., diagnostics=True)` reports per-token `rounds`, `converged`, `hit_budget`, and total `executed_token_passes`. In the real-GPU smoke check, the sample prompt had pass counts ranging from 13 to 20 at motion tolerance 0.05. Stricter thresholds 0.01 and 0.025 exhausted all 20 passes. Generation returned `153`. These are compute-policy observations, not task-difficulty or correctness estimates.

Tests cover different pass counts within one batch, skipping halted states, minimum/maximum bounds, patience resets, gradients, full-budget equivalence to the existing fixed path, chunk independence, unchanged weights, invalid policies, and GGUF round trips. Convergence is a heuristic: a stable state can be wrong, and an unstable state can already contain a correct answer. The existing pilot weights were trained at four passes; extending them to twenty is an unvalidated depth change beyond this smoke check. Retraining with the adaptive configuration and controlled held-out comparisons is needed before claiming a reasoning or efficiency improvement.
