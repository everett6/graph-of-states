#!/usr/bin/env bash
# After the pilot: the recurrent-depth round (configs/benchmark-depth-v2.json), then the Lean/agent queue
# (scripts/gpu_queue_lean.sh). One GPU job at a time. Stop it with: pkill -f gpu_queue_depth.sh
cd "$(dirname "$0")/.."
PY=.venv-training/bin/python
GEMMA=/home/everett/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-q4_0-unquantized/snapshots/b6ed86275a6a5735884e208bfed95b445a684ca2
log() { echo "$(date -Is) $*" | tee -a runs/queue/status.log; }
mkdir -p runs/queue
while pgrep -f gpu_queue.sh >/dev/null || pgrep -f measure_headroom.py >/dev/null || pgrep -f "benchmark_gos.py --config" >/dev/null; do sleep 300; done
[ -f runs/pilot-heldout-v1/comparison.json ] && $PY scripts/compare_heldout.py runs/pilot-heldout-v1 --baseline frozen-d4 \
  > runs/queue/pilot-compare.log 2>&1 && log "pilot compare written"
while pgrep -f build_lean_train.py >/dev/null; do sleep 300; done
if [ ! -s data/train/lean/lean-workbook-640.jsonl ]; then
  $PY scripts/filter_token_budget.py --input data/train/lean/lean-workbook-v1.jsonl --output data/train/lean/lean-workbook-640.jsonl \
    --base-path "$GEMMA" --budget 640 > runs/queue/filter-lean.log 2>&1
  log "lean 640 filter exit $?"
fi
log "depth round start"
$PY benchmark_gos.py --config configs/benchmark-depth-v2.json --output runs/depth-v2 --resume > runs/queue/depth-v2.log 2>&1
log "depth round exit $?"
$PY scripts/compare_heldout.py runs/depth-v2 --baseline block-lora18-24-p1/passes-1 > runs/queue/depth-v2-compare.log 2>&1
log "depth compare exit $?"
exec scripts/gpu_queue_lean.sh
