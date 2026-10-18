#!/usr/bin/env bash
# After scripts/gpu_queue.sh: formal-proof baselines on seeded miniF2F subsets (Lean tool in the loop), the frozen
# agent baseline on held-out v2, then verified agent-trajectory collection on training tasks (code, MATH, Lean
# Workbook) for rejection-sampling SFT. Roughly 40 GPU hours in total; stop it with: pkill -f gpu_queue_lean.sh
# Full miniF2F runs are ~30 h per model on this GPU, so subsets come first; rerun without --sample later.
cd "$(dirname "$0")/.."
PY=.venv-training/bin/python
GOEDEL=$(ls -d ~/.cache/huggingface/hub/models--Goedel-LM--Goedel-Prover-V2-8B/snapshots/*/ | head -1)
GEMMA=/home/everett/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-q4_0-unquantized/snapshots/b6ed86275a6a5735884e208bfed95b445a684ca2
log() { echo "$(date -Is) $*" | tee -a runs/queue/status.log; }
while pgrep -f gpu_queue.sh | grep -qv $$ || pgrep -f measure_headroom.py >/dev/null || pgrep -f "benchmark_gos.py --config" >/dev/null; do sleep 300; done
mkdir -p runs/queue data/train/agent
log "lean queue start"
common="--eval data/lean/minif2f-test.jsonl --max-tools 2 --n 2 --temperature 0.7"
goedel="--base-path $GOEDEL --thinking --max-new-tokens 3072 --context 8192"
gemma="--base-path $GEMMA --max-new-tokens 2048 --context 6144"
$PY agent_loop.py $common $goedel --limit 3 --output runs/lean-smoke-goedel > runs/queue/lean-smoke-goedel.log 2>&1
log "goedel smoke exit $?"
$PY agent_loop.py $common $gemma --limit 3 --output runs/lean-smoke-gemma > runs/queue/lean-smoke-gemma.log 2>&1
log "gemma smoke exit $?"
$PY agent_loop.py $common $goedel --sample 40 --output runs/minif2f40-goedel8b-agent > runs/queue/minif2f-goedel.log 2>&1
log "goedel minif2f40 exit $?"
$PY agent_loop.py $common $gemma --sample 80 --output runs/minif2f80-gemma12b-agent > runs/queue/minif2f-gemma.log 2>&1
log "gemma minif2f80 exit $?"
# Frozen agentic baseline on post-cutoff held-out problems (contest math + LCB v6 with hidden tests).
$PY agent_loop.py --eval data/eval/heldout-v2.jsonl --sources aime_2025,hmmt_feb_2025,aime_2026,hmmt_feb_2026,lcb_v6 \
  $gemma --n 1 --max-tools 3 --sample 120 --output runs/agent-heldout-v2-gemma12b > runs/queue/agent-heldout.log 2>&1
log "agent heldout-v2 baseline exit $?"
collect="$gemma --n 2 --temperature 0.8"
$PY agent_loop.py --eval data/train/code-train-v1.jsonl $collect --max-tools 3 --sample 150 --output runs/collect-code-gemma \
  --collect data/train/agent/code-gemma.jsonl > runs/queue/collect-code.log 2>&1
log "collect code exit $?"
$PY agent_loop.py --eval data/train/math-train-v1.jsonl $collect --max-tools 3 --sample 150 \
  --output runs/collect-math-gemma --collect data/train/agent/math-gemma.jsonl > runs/queue/collect-math.log 2>&1
log "collect math exit $?"
while pgrep -f build_lean_train.py >/dev/null; do sleep 300; done
$PY agent_loop.py --eval data/train/lean/lean-workbook-v1.jsonl $collect --max-tools 2 --sample 150 \
  --output runs/collect-lean-gemma --collect data/train/agent/lean-gemma.jsonl > runs/queue/collect-lean.log 2>&1
log "collect lean exit $?"
