# How Gemma 4 12B was trained, and what fits it

Sources: the model card bundled with the checkpoint (`README.md` in the HF snapshot) and the Gemma 4 technical
report (arXiv 2607.02770). The report gives architecture, data and quantization details. It describes
post-training only as "similar to Gemma 3" (distillation from a larger instruct teacher plus RL with
human-feedback and verifiable rewards), with thinking mode as the main addition. It gives no RL algorithm,
reward list or thinking budget.

## Facts that matter here

| Fact | Source | Consequence for GoS |
|---|---|---|
| 12B "Unified": 48 layers, encoder-free, 1024-token sliding window interleaved with global layers (last layer always global); global layers share K and V and use p-RoPE; 256K context | card, report §2.3 | Attention is already near-linear (40 of 48 layers windowed). Context is not the bottleneck; weight reads per token are. |
| Pretraining cutoff January 2025; decontaminated against benchmarks | card, report | Post-cutoff sources (AIME/HMMT 2025-26, LCB after 2025-01-31) are the clean held-out signal. |
| Instruct model reasons in **thinking mode** (`<|think|>` in the system turn, reasoning in `<|channel>thought ... <channel|>`). Reported with thinking: AIME 2026 77.5%, LiveCodeBench v6 72.0%, GPQA Diamond 78.8% | card | Our no-thinking 2048-token evals (AIME 2026 40%, LCB 48%) measure the model far below its tuned mode. The 16k thinking check measures the gap. Any recurrence or RL gain should also be checked with thinking on. |
| With thinking off, the 12B model still emits an empty thought block, which the chat template puts in the generation prompt | card, template | The training prefix and the live prompt already match (`enable_thinking=False`). |
| Instruct turns end with `<turn|>` (id 106), not `<eos>`; "fine-tuning must add the matching end token" | report | Benchmark SFT appended `<eos>`. `sft_end_token: "turn"` now supervises `<turn|>` (the depth round uses it; the pilot keeps the legacy default so its arms stay comparable). |
| Recommended sampling: temperature 1.0, top_p 0.95, top_k 64 | card | Our older runs used temperature 0.7 with no top-k/top-p. New runs (thinking check, self-distillation, depth-round and RL evals) use Google's settings. GRPO rollouts keep untruncated temperature-1 sampling, so the sampled tokens and the scored log-probs come from the same distribution. |
| Multi-turn history keeps only final answers, except tool-call turns, which keep their thinking | card | The agent loop runs with thinking off, so this does not change anything yet. If agent RL ever runs with thinking on, earlier thoughts must be dropped from history except around tool calls. |
| QAT Q4_0 checkpoint: about 7.65 GB at 4-bit, quality close to bf16 | report Table 3 | We load the unquantized QAT export in NF4. The weights were trained to tolerate 4-bit, so adapter training on top of it is well-matched. |

## What synergizes, ranked

1. **Use the mode it was tuned for.** Thinking on, enough tokens, recommended sampling. This is likely the largest
   single gain (the thinking check is queued first).
2. **Train on its own outputs, in its own format.** SFT on human-written MATH solutions collapsed the pilot's
   LoRA arm (52.8% to 15.2%). Self-distilled verified solutions (`scripts/self_distill.py`) ending in `<turn|>`
   stay on the model's distribution, as on-policy distillation and RL do in Google's own pipeline.
3. **RL with verifiable rewards on top.** This matches the RL stage Google used (Gemma 3-style). Queued after
   the depth round, from the curriculum recurrent adapter and a matched control.
4. **Recurrent depth (re-passing layers 18-24).** This is not part of Google's recipe. Gains have to show up in
   the paired depth round against block LoRA without re-passing.
