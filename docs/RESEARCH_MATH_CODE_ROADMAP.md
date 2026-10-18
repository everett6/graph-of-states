# Research review: making GoS good at math and code

Date: 2026-10-17. This file checks an external set of recommendations against what the repository already has
(commit `f05aaa2`) and against recent literature. Nothing here has been run; it is a plan and an evidence review.

## 1. Bottom line

1. **About half of the advice is already built.** `verified_rewards.py` does numeric-equivalence math checks,
   restricted-Python and Docker unit tests. `benchmark_gos.py` already compares frozen, LoRA, a parameter-matched
   MLP, GoS at depths 1/2/4 and five leave-one-out ablations, on the same data, seed and update budget.
   `split_audit.py` audits leakage. Adaptive pass budgets exist.
2. **The real gap is evidence, not machinery.** The controlled pilot scored 11/11 for every variant, including
   the frozen backbone ([CONTROLLED_RESULTS.md](../verification/CONTROLLED_RESULTS.md)). That test cannot tell
   any two variants apart. Everything else depends on an eval set with headroom.
3. **Expect small effects.** GoS is a ~273K-parameter head on a frozen 12B model. The literature says
   bolted-on latent recursion gives little; gains came from looping during pretraining or from
   per-step supervision (section 3). The honest prior is "GoS ≈ MLP baseline" until data says otherwise.
4. **The strongest cheap gain on a 5070 is probably not architectural:** verifier-guided best-of-N plus
   repair on the frozen backbone. GoS has to beat that at matched compute to justify itself.

## 2. Advice vs. repository

| Recommendation | Status in repo | What is actually missing |
|---|---|---|
| Replace exact-match reward | Done (see §6): configs now use `verified_rewards:grpo_verified_reward` | — |
| Math-aware verification | `math_reward` parses numbers/fractions only | No symbolic answers (`\frac{}{}`, `\sqrt`, π, intervals, sets, tuples, matrices). Add a SymPy / `math-verify` path; keep the restricted evaluator for numbers. |
| Code verification with tests | Restricted-Python + Docker (`docker_code_reward.py`) | No failure feedback into a repair turn; no property-based tests; no time/memory limits per case reported back. |
| Matched comparison module | Done: frozen / LoRA / matched MLP / GoS d1–d4 / ablations | Single seed; saturated test; LoRA baseline is q/v only, rank 2 (weak baseline, see 3.4). |
| Node specialization | Not measured (`diagnose_gos.py` is GPU/kernel diagnostics) | Node-state similarity (CKA/cosine), routing entropy per node, node-drop ablation on accuracy. |
| Candidate search, retry, adaptive compute | `gos_tools.gated_tool_turns` (Python tool loop), adaptive passes | No best-of-N, no verifier-ranked selection, no repair loop driven by test output. |
| Dedicated math/code data | Mix is Bespoke-Stratos / Fable / CodeFeedback | Bespoke-Stratos is already mostly R1-distilled math and code, so the advice overstates this. The missing part is **verifiable** training items (answer key or tests) for RL. |

Adaptive passes are a warning sign, not a feature yet: in `verification/adaptive-results-v1.json`, at the
default tolerance 0.05 only 12 of 26 probe tokens converged, the rest hit the 20-pass cap, and the weights were
trained with at most 4 passes. Running beyond trained depth is untested extrapolation.

## 3. What the literature says

### 3.1 Latent recursion vs. explicit reasoning
- **Huginn-3.5B** (depth-recurrent): raising recurrent steps 4→32 moved GSM8K only from 3.1% to 4.9%, while
  explicit CoT on the same model reached 25–38% ([arXiv 2507.02199](https://arxiv.org/html/2507.02199v2)).
- **Ouro / LoopLM**: looping works when built into pretraining (7.7T tokens, entropy-regularized depth).
  The reported 1.4B–2.6B models match much larger models, but this needs pretraining scale
  ([summary](https://www.themoonlight.io/review/scaling-latent-reasoning-via-looped-language-models)).
- **LOTUS** (June 2026): latent CoT lags explicit CoT above ~1B, and the gap grows with scale. A looped model
  closed it at 3B only with per-latent-step supervision against gold CoT tokens; ablations say both the loop and
  the step supervision are needed ([arXiv 2606.31779](https://arxiv.org/abs/2606.31779)).
- **TRM (7M params)**: tiny recursive nets do very well on ARC/Sudoku/Maze with small data
  ([arXiv 2510.04871](https://arxiv.org/abs/2510.04871)). Note that these are fixed-format grid puzzles
  trained from scratch, not free-form math text from a frozen LM.

**Implication for GoS:** recursion trained only by next-token loss on the final answer is the weak setup.
If GoS recursion is kept, give each pass a target (deep supervision as in TRM/LOTUS), and test it on a task
where the frozen model's explicit CoT is not already enough.

### 3.2 What RL with verifiable rewards really buys
- RLVR mostly raises pass@1 by sharpening what the base model can already sample; at large k (≈256) base models
  match or beat the RL model, and distillation is what added new reasoning patterns
  ([Yue et al., NeurIPS 2025](https://arxiv.org/abs/2504.13837v1)). **Report pass@k curves, not just pass@1.**
- Random or spurious rewards raised Qwen math scores, partly through contamination
  ([arXiv 2507.10532](https://arxiv.org/html/2507.10532v3)) and partly through clipping/entropy effects that
  also appear in Llama ([arXiv 2512.16912](https://arxiv.org/html/2512.16912v3)).
  **Add a random-reward control and a format-only control** so a gain can be credited to correctness.

### 3.3 Test-time compute
- Compute-optimal allocation (per-difficulty choice between parallel sampling, revisions and PRM search) beat
  best-of-N by about 4× in compute, and could beat a 14× larger model where the base had non-trivial success
  ([Snell et al., arXiv 2408.03314](https://arxiv.org/abs/2408.03314v1)).
  **Best-of-N with the existing verifier is the baseline any GoS "thinking" must beat at equal FLOPs.**

### 3.4 Adapter baselines
- Rank-1 LoRA matched full fine-tuning in RL; attention-only LoRA underperforms, MLP layers must be included;
  LoRA wants ~10× the full-FT learning rate ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/),
  [TRL docs](https://huggingface.co/docs/trl/main/en/lora_without_regret)).
  So (a) parameter count is not the bottleneck for RL, and (b) the current q/v-only rank-2 LoRA is a weaker
  baseline than a fair one. Add an all-linear LoRA with a tuned learning rate before claiming GoS beats LoRA.

### 3.5 Benchmarks and contamination
- MathArena found strong contamination signs on AIME 2024 and near-duplicates online even for AIME 2025
  ([arXiv 2505.23281](https://arxiv.org/pdf/2505.23281)). Use contests released after the Gemma checkpoint's
  cutoff, such as **AIME 2026 / HMMT Feb 2026**.
- **FrontierMath** is held privately by Epoch AI; it is not a practical target. **SWE-bench Verified** needs a
  strong agentic model and many Docker images; out of reach for a 12B NF4 model on one 12 GB GPU right now.
- **miniF2F (Lean 4)** is feasible as a separate track but needs Lean + Mathlib and a prover-tuned model.
  For reference, Goedel-Prover-V2-8B reports ~84.6% pass@32
  ([ICLR 2026](https://iclr.cc/virtual/2026/poster/10007912)); a general 12B Gemma head will not get close.

## 4. Compute reality on the RTX 5070

Measured single-stream decode is ~20–23 tok/s. AIME 2026 (30 problems) × 8 samples × ~4K tokens ≈ 1M tokens
≈ 12 hours per variant. With 4+ variants and 3 seeds that is weeks. Before any large eval:
batched generation (or vLLM/llama.cpp server with the adapter) is a prerequisite, and short-answer sets with
headroom (e.g. MATH level 4–5 numeric subset, post-cutoff LiveCodeBench easy/medium) should come first.

## 5. Recommended order

1. **Fix the measuring stick.** Build a held-out set where the frozen model scores roughly 20–60%: ~300 numeric
   math items (MATH L4–5 + AIME 2026/HMMT 2026) and ~150 post-cutoff coding items with tests. Run
   `split_audit.py` on it. Run a power check: with paired items, detecting a 5-point gain needs a few hundred
   items; report paired bootstrap CIs or McNemar, 3 seeds.
2. **Wire the verified reward into GRPO** and extend `math_reward` with a SymPy/`math-verify` symbolic path.
   Add unit tests for `1/2 == 0.5 == 2/4`, `\frac{1}{2}`, `\sqrt{8} == 2\sqrt{2}`, wrong-answer rejection.
3. **Establish the non-architectural ceiling:** frozen backbone + verifier best-of-N (N=1,4,8,16) + one repair
   turn using test failures. This is the bar.
4. **Rerun the controlled comparison on the new set:** frozen, all-linear LoRA (tuned LR), matched MLP, GoS d1,
   GoS d4; SFT then GRPO with identical budgets; plus random-reward and format-only controls. Report pass@1,
   pass@k, tokens, latency, peak memory.
5. **Only then node diagnostics and deep supervision:** node CKA/cosine, routing entropy, node-drop accuracy;
   try per-pass supervision. Keep or cut recursion based on step 4 numbers.
6. **Optional tracks:** miniF2F with a Lean verifier; multi-file coding. Skip FrontierMath and SWE-bench for now.

**Decision rule worth fixing in advance:** if GoS d4 does not beat the matched MLP and all-linear LoRA by a
margin whose 95% CI excludes zero on the step-1 set, stop adding graph features and put effort into search +
verification on the plain adapter.

## 6. Status (2026-10-17)

- **Step 2 done.** `symbolic_math.py` adds a SymPy equivalence check that never `eval`s model text (whitelisted
  AST walker; handles `\frac`, `\sqrt[n]{}`, π, i, implicit products, tuples, sets, `x=…`, `1,000`, `%`, `°`;
  intervals/matrices are rejected as uncheckable). `math_reward` keeps the exact-rational path for plain numbers
  and falls back to SymPy; `\boxed{}` extraction now handles nested braces (the old regex rejected
  `\boxed{\frac{1}{2}}`). `verified_rewards:grpo_verified_reward` is now the reward in all
  `configs/gemma5070*.json`: task_spec verifier if the row has one, else SymPy check against the reference's
  `\boxed{}` answer (Bespoke-Stratos solutions carry one), else the old exact-text match (most CodeFeedback rows;
  those still give almost no GRPO signal).
- **Step 1 builder done, data not yet fetched.** `build_eval_set.py` + `configs/eval-heldout-v1.json` build rows
  with JSON-string task specs from local files: AIME 2025/2026, HMMT Feb 2025/2026 (post-cutoff; Gemma 4 card
  says Jan 2025), MATH-500 level 4–5 (pre-cutoff, report separately), LiveCodeBench v6 LeetCode functional
  problems after 2025-01-31 (public tests only; pickled private tests are never loaded). Near-duplicates against
  `--against` training files are dropped by 3-gram Jaccard ≥ 0.5. Headroom is not yet measured.
- Code rows need the `python:3.12-slim` Docker image, which is not installed.
