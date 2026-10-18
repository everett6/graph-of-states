# Formal proofs (Lean 4 + Mathlib)

## Setup (done on this machine)

- `~/.elan` (elan 4.2.4), toolchain `leanprover/lean4:v4.36.0-rc1`.
- `lean_env/`: Lake project pinned to Mathlib `v4.36.0-rc1` with the prebuilt cache (`.lake/`, ~7 GB, git-ignored).
  Rebuild elsewhere: `cd lean_env && lake exe cache get`.
- One check ≈ 2–6 s and ≈ 7 GB RSS (importing Mathlib dominates), so run ≤ 3 checks in parallel on 29 GB RAM.
- Importing Mathlib dominates, so `check_proofs` puts many candidates in one file (16 trivial proofs ≈ the cost
  of one). A batch pass is only a candidate and is re-checked alone; a batch failure is re-checked alone when
  another candidate could have interfered (environment-changing text, errors outside every theorem,
  timeout). Verdicts are therefore identical to single checks (tested). `verify_many` uses it for best-of-N
  and GRPO. `confirm=False` skips the re-check and is used only to filter trusted dataset proofs.

## What counts as a proof (`lean_verifier.py`)

The verifier writes the file itself: `import Mathlib`, the dataset's `open` lines and definitions, then the
target theorem **renamed to `__gos_target` with the exact dataset statement**, then the model's tactic block,
then `#print axioms __gos_target`. A proof passes only if Lean exits cleanly with no errors and the axiom audit
lists nothing beyond `propext`, `Classical.choice`, `Quot.sound`. Textual bans (`sorry`, `admit`, `axiom`,
`native_decide`, `unsafe`, `implemented_by`, `#exit`, `#eval`, macros/elaborators) fail fast; the axiom audit
catches anything that slips through (tested: `sorryAx` smuggling, `#exit` after a failing tactic).
Lean runs under bubblewrap: no network, read-only toolchain and Mathlib, private `/tmp`.

`verify(completion, {'kind': 'lean', 'statement': ..., 'opens': ...})` uses it, so Lean tasks work in GRPO
rewards, best-of-N and the agent loop. In the agent loop the model can write `<lean>…</lean>` to check an
attempt and read Lean's errors (line numbers relative to its proof) before answering, the self-correction loop
used by Goedel-Prover-V2. Lean is the real success criterion, so this is not test leakage.

## Benchmarks

| Rung | Data | What success would show |
|---|---|---|
| 1 | miniF2F test (AI-MO corrected, 244) | competition-level formal proving; Goedel-Prover-V2-8B reports ~84.6% pass@32 |
| 2 | MathOlympiadBench (Goedel, 360 human-verified olympiad formalizations) | olympiad-level proofs |
| 3 | PutnamBench | university competition proofs |
| 4 | Formal Conjectures (DeepMind) open statements | a checked proof here is a genuinely new result |

Statements that do not elaborate on this Mathlib are dropped and counted (`scripts/build_lean_tasks.py`).

## Training data

`scripts/build_lean_train.py` re-checks Goedel-Prover's 29.7k Lean Workbook proofs on this Mathlib (default
heartbeats, axiom audit; ~20% fail, mostly renamed lemmas such as `div_le_iff` and heartbeat limits) and drops
statements that are near-duplicates of miniF2F or MathOlympiadBench (anonymized statement, goal or informal
statement; this catches renamed twins such as IMO 1964 P2, which Lean Workbook contains three times). Rows use
the evaluation prompt format,
so they serve as SFT targets and as RL / agent-collection tasks.

## Queued runs (`scripts/gpu_queue_lean.sh`)

Full miniF2F with the Lean tool takes ~30 h per model on this GPU (NF4, sequential turns), so the queue runs
seeded subsets first: Goedel-Prover-V2-8B on 40 problems (harness sanity check against its published
~84.6% pass@32), Gemma 4 12B on 80, both pass@2 with up to 2 Lean checks. Then it collects verified agent
transcripts on training tasks (code, MATH, Lean Workbook).

## About the Millennium problems

No current system can prove a Millennium Prize problem, and better tooling will not change that on its own.
What this track provides is the prerequisite: every claimed proof is machine-checked, so progress up the
ladder above is real rather than persuasive text. A natural-language "proof" of an open problem from any model
should be treated as unverified until it is formalized and checked. Of the Millennium problems, only some have
complete formal statements in Lean today; formalizing a statement faithfully is itself research work.
