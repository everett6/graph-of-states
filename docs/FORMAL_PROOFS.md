# Formal proofs (Lean 4 + Mathlib)

## Setup (done on this machine)

- `~/.elan` (elan 4.2.4), toolchain `leanprover/lean4:v4.36.0-rc1`.
- `lean_env/`: Lake project pinned to Mathlib `v4.36.0-rc1` with the prebuilt cache (`.lake/`, ~7 GB, git-ignored).
  Rebuild elsewhere: `cd lean_env && lake exe cache get`.
- One check ≈ 2–6 s and ≈ 7 GB RSS (importing Mathlib dominates), so run ≤ 3 checks in parallel on 29 GB RAM.

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

## About the Millennium problems

No current system can prove a Millennium Prize problem, and better tooling will not change that on its own.
What this track provides is the prerequisite: every claimed proof is machine-checked, so progress up the
ladder above is real rather than persuasive text. A natural-language "proof" of an open problem from any model
should be treated as unverified until it is formalized and checked. Of the Millennium problems, only some have
complete formal statements in Lean today; formalizing a statement faithfully is itself research work.
