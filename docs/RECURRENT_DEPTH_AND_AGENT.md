# Recurrent depth and agent loop

## Why

The original GoS adapter sits on the LM head: it refines one token's final hidden state and never sees other
tokens. It can reshape a next-token distribution but cannot add computation over the problem. Results so far tie
it with a parameter-matched MLP. Published gains from "thinking in depth" come from looping layers **inside**
the network (Huginn, Ouro/LoopLM, retrofitted recurrence), where each pass attends over the whole context.

## Recurrent block (`gos_recurrent.py`)

Gemma 4 12B has 48 layers. Layers `[start, end)` (default 18–24: five sliding-window layers and one global
layer, 12.5% of depth) run `passes` times with the same weights:

```
e   = output of layers [0, start)                         # prelude, frozen
s_1 = block(e)                                            # pass 1 = original computation
s_r = s_{r-1} + gate * (block(s_{r-1} + inject(e)) - s_{r-1})   # r >= 2
out = layers [end, 48)(s_passes)                          # coda, frozen
```

- `gate` (per-channel) and the rank-64 `inject` projection start at zero, so a fresh model is **exactly** the
  base model at any pass count (tested); training decides how much extra passes contribute.
- LoRA (rank 16) on the block's attention and MLP projections is shared by all passes.
- Every pass has its own KV-cache slots; cached generation equals full recomputation (tested with sliding
  windows), so generation cost is ~+12.5% per extra pass.
- Training samples the pass count per step (1..max) so the model works at every depth; inference can run
  any count, including more than trained (extrapolation is measured, not assumed).
- Each extra pass adds 12.5% FLOPs, so compare at matched compute: the benchmark evaluates the **same
  weights** at 1 and 4 passes, and includes `block-lora18-24-p1` (identical LoRA, never re-passed).

## Agent loop (`agent_loop.py`)

The model may write `<run>…</run>`; generation stops there, the code runs in the sandbox (Docker if usable,
else rootless bubblewrap: no network, read-only `/usr`, CPU/memory/file limits), and the output (≤2,000
chars) comes back as the next turn, up to `--max-tools` runs. A reply without `<run>` is the final answer,
checked by the task verifier. No test cases are exposed to the agent, so agentic and single-shot scores
are comparable. `--collect` writes verified successful transcripts; the benchmark trains on them one assistant
turn at a time, conditioned on exactly the context the model saw (Gemma's live prompt differs from its
rendered history). This is rejection-sampling fine-tuning on the model's own verified tool use.

## Planned runs (after the frozen headroom pass)

1. GPU tests; short smoke run of every benchmark arm.
2. Frozen baselines: best-of-N (thinking off/on) and the agent loop on the held-out set.
3. Agent trajectories on training tasks (frozen model, verified successes only): 587 LCB problems with hidden
   tests, 400 MATH problems, 300 Lean Workbook statements (`scripts/gpu_queue_lean.sh`).
4. Comparison on MATH SFT (+ agent transcripts): frozen, all-linear LoRA, MLP, GoS d1/d4, GoS deep supervision,
   block LoRA p1, recurrent p4 (evaluated at 1 and 4 passes), each single-shot and in the agent loop.
