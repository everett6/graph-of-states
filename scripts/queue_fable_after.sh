#!/usr/bin/env bash
# Run the Fable 5.1 distillation comparison after the currently running GPU queue processes exit.
cd "$(dirname "$0")/.."
PY=.venv-training/bin/python
PIDS=("$@")
if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "Pass the active queue process IDs." >&2
  exit 2
fi
mkdir -p runs/queue
log() { echo "$(date -Is) $*" | tee -a runs/queue/status.log; }
for pid in "${PIDS[@]}"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
log "Fable 5.1 distillation comparison start"
$PY benchmark_gos.py --config configs/benchmark-fable-v1.json --output runs/fable-v1 --resume > runs/queue/fable-v1.log 2>&1
code=$?
log "Fable 5.1 distillation comparison exit $code"
if [ "$code" -eq 0 ]; then
  $PY scripts/compare_heldout.py runs/fable-v1 --baseline frozen-d4 > runs/queue/fable-v1-compare.log 2>&1
  log "Fable 5.1 paired comparison exit $?"
fi
