# Data

All data lives under `data/` (git-ignored). Configs and builders are in the repo, so every file below can be
rebuilt. Every row carries a `task_spec` that a verifier checks; nothing is graded by text match alone.

## Evaluation (never trained on)

| File | Rows | Contents |
|---|---|---|
| `data/eval/heldout-v1.jsonl` | 416 | AIME 2025/2026, HMMT Feb 2025/2026 (post-cutoff), MATH-500 L4–5 (pre-cutoff), 49 LCB v6 LeetCode problems on public tests only. Used by the headroom pass and the pilot. |
| `data/eval/heldout-v2.jsonl` | 489 | v1 math sources minus 9 MATH-500 rows that are near-duplicates of MATH-train problems, plus all 131 LCB v6 problems released after 2025-01-31 (49 LeetCode, 82 AtCoder) graded on public **and hidden** tests (median 40 per problem). Used by `configs/benchmark-heldout-v2.json`. |
| `data/lean/minif2f-test.jsonl` | 243 | miniF2F test (AI-MO corrected), statements that compile on our Mathlib. |
| `data/lean/matholympiadbench.jsonl` | 297 | Goedel MathOlympiadBench, statements that compile on our Mathlib. |

Gemma 4's pretraining cutoff is January 2025 (model card), so post-cutoff sources are the clean signal;
MATH-500 and miniF2F are older and are reported separately.

## Training

| File | Rows | Contents |
|---|---|---|
| `data/train/math-train-v1.jsonl` | 6,795 | MATH train worked solutions whose own `\boxed{}` answer verifies. `math-train-640.jsonl` (6,129) fits 640 tokens. |
| `data/train/code-train-v1.jsonl` | 587 | LCB v1–v4 problems released before 2025 (LeetCode functional, AtCoder/Codeforces stdin/stdout) with ≥3 hidden tests each (up to 64 KB of cases). No reference solutions: for RL and agent trajectory collection, not SFT. |
| `data/train/lean/lean-workbook-v1.jsonl` | 22,661 | Goedel-Prover's Lean Workbook proofs re-checked on our Mathlib (default heartbeats, axiom audit), statements screened against miniF2F and MathOlympiadBench (3-gram Jaccard ≥ 0.5 dropped). |
| `data/train/agent/*.jsonl` | queued | Verified agent transcripts from the frozen model (code, MATH, Lean) for rejection-sampling SFT. |

Rebuild:

```bash
.venv-training/bin/python build_eval_set.py --config configs/eval-heldout-v2.json --output data/eval/heldout-v2.jsonl --against data/train/math-train-v1.jsonl data/train/code-train-v1.jsonl
.venv-training/bin/python build_eval_set.py --config configs/train-code-v1.json --output data/train/code-train-v1.jsonl --against data/eval/heldout-v1.jsonl
.venv-training/bin/python scripts/build_lean_train.py --input data/train/lean/lean_workbook_proofs.parquet --output data/train/lean/lean-workbook-v1.jsonl --against data/lean/minif2f-test.jsonl data/lean/matholympiadbench.jsonl --workers 3
.venv-training/bin/python scripts/filter_token_budget.py --input data/train/lean/lean-workbook-v1.jsonl --output data/train/lean/lean-workbook-640.jsonl --base-path <gemma dir> --budget 640
```

Sources: MathArena (AIME/HMMT), HuggingFaceH4/MATH-500, EleutherAI/hendrycks_math (train split),
livecodebench/code_generation_lite (test.jsonl–test4.jsonl for training, test6.jsonl for evaluation),
AI-MO/minif2f_test, Goedel-LM MathOlympiadBench and Lean-workbook-proofs.

## Verifiers

- **Math**: exact rational, then SymPy equivalence through an AST whitelist (no `eval` of model text).
- **Sandbox**: Docker if usable, else bubblewrap (no network, read-only `/usr`, 1 GB / 12 s CPU limits).
- **Code, functional (training set, held-out v2)**: the sandbox calls `Solution().<method>(*args)` and returns
  each result as JSON; the host compares type-strictly (`True != 1`, floats within 1e-6). Expected outputs
  never enter the sandbox, and an object whose `__eq__` always returns True cannot pass (it does pass an
  in-sandbox `assert got == expected`; tested).
- **Code, functional (held-out v1 only)**: the older assertion script inside the sandbox, with a per-run nonce
  against `os._exit(42)` forgeries. Fine for scoring frozen or SFT models; do not use it as an RL reward.
- **Code, stdin/stdout**: each test case runs in a forked child with real stdin/stdout pipes (so `input()`,
  `open(0)` and `os.write(1)` all work); the host compares outputs (whitespace tokens, numbers within 1e-6).
- **LCB hidden tests** are zlib+pickle encoded; they are decoded with an unpickler that refuses every global,
  so only plain strings and containers can be built.
- **Lean**: see `docs/FORMAL_PROOFS.md`.
