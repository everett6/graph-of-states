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

### Deep-recurrence training recipe (from retrofitted recurrence, arXiv 2511.07384)

McLeish et al. convert pretrained decoders into recurrent ones and report better math at a fixed compute budget
than post-training the original model, *if* depth is trained with (a) pass counts drawn from a Poisson-lognormal
whose mean is raised over a curriculum (linear and 1-sqrt matched; 1-sqrt over the first 75% of training was
used for math) and (b) backprop through only the last k recurrences (they used 8 at mean 32). Test-time gains
saturated around 16–32 recurrences; recurring deeper than trained on harder problems stayed open.
Both are recurrent-spec options here (benchmark and `rl_recurrent.py`):

| key | meaning |
|---|---|
| `sampler` | `categorical` (default; `pass_weights` over 1..`max_passes`, unchanged draws) or `poisson_lognormal` |
| `mean_passes`, `sigma` | Poisson-lognormal mean (passes = 1 + Poisson(e^τ), τ ~ N(log(m−1) − σ²/2, σ)), σ default 0.5; draws clip to `max_passes` |
| `curriculum_steps`, `curriculum` | raise the mean from 1 to `mean_passes` over these steps, `linear` or `1-sqrt` |
| `backprop_passes` | k: passes before the last k run without autograd (memory flat in pass count; gradient equals detaching the state entering the window, tested) |

`pass_trace(model, ids)` reports each extra pass's relative state change, to see whether deep pass counts
converge (shrinking) or diverge (growing). `configs/benchmark-depth-v2.json` compares, on identical rows:
frozen, block LoRA p1, uniform 1–4 passes (evaluated at 1/4/8) and the curriculum recipe (mean 6, max 12,
last 4 passes trained; evaluated at 1/4/8/12). `scripts/gpu_queue_depth.sh` runs it after the pilot, then the
Lean/agent queue.

## Agent loop (`agent_loop.py`)

The model may write `<run>…</run>`; generation stops there, the code runs in the sandbox (Docker if usable,
else rootless bubblewrap: no network, read-only `/usr`, CPU/memory/file limits), and the output (≤2,000
chars) comes back as the next turn, up to `--max-tools` runs. A reply without `<run>` is the final answer,
checked by the task verifier. No test cases are exposed to the agent, so agentic and single-shot scores
are comparable. `--collect` writes verified successful transcripts; the benchmark trains on them one assistant
turn at a time, conditioned on exactly the context the model saw (Gemma's live prompt differs from its
rendered history). This is rejection-sampling fine-tuning on the model's own verified tool use.

## Agentic RL (`agent_rl.py`)

With `agent_max_tools > 0`, GRPO rollouts use the same `<run>` protocol: generation stops at `</run>`, the
code runs in the sandbox, and the result is spliced in as a user turn rendered by the model's own chat
template. Those tool-result tokens get `env_mask` 0, so TRL leaves them out of the policy loss; the model is
trained only on tokens it produced. Rewards grade `agent_final` (the last assistant turn), so text printed
by a program cannot be graded as the answer. Configs: `configs/gemma5070-grpo-agent.json` (MATH + LCB train,
3 runs, 2,048 completion tokens) and its random-reward control. Plain verified GRPO (no tools) is
`gemma5070-grpo-verified{,-random-control,-format-control}.json` over MATH, LCB and Lean Workbook.

Limits: these configs train the GoS head through `train_unified.py`; RL for the recurrent block is not
wired yet (it trains only through `benchmark_gos.py` SFT). Rollouts are sequential and unbatched, so a
step with 4 generations and tool calls takes minutes on the 5070.

## Planned runs (after the frozen headroom pass)

1. GPU tests; short smoke run of every benchmark arm.
2. Frozen baselines: best-of-N (thinking off/on) and the agent loop on the held-out set.
3. Agent trajectories on training tasks (frozen model, verified successes only): 587 LCB problems with hidden
   tests, 400 MATH problems, 300 Lean Workbook statements (`scripts/gpu_queue_lean.sh`).
4. Comparison on MATH SFT (+ agent transcripts): frozen, all-linear LoRA, MLP, GoS d1/d4, GoS deep supervision,
   block LoRA p1, recurrent p4 (evaluated at 1 and 4 passes), each single-shot and in the agent loop.
