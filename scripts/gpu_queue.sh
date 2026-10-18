#!/usr/bin/env bash
# Sequential GPU jobs after the headroom pass. Each step logs to runs/queue/; a failed step does not stop later ones.
cd "$(dirname "$0")/.."
PY=.venv-training/bin/python
log() { echo "$(date -Is) $*" | tee -a runs/queue/status.log; }
while pgrep -f measure_headroom.py >/dev/null; do sleep 120; done
log "headroom finished; starting queue"
$PY -m pytest -q tests > runs/queue/pytest-gpu.log 2>&1; log "pytest exit $? ($(tail -1 runs/queue/pytest-gpu.log))"
$PY benchmark_gos.py --config configs/benchmark-smoke-v1.json --output runs/smoke-v1 > runs/queue/smoke.log 2>&1
log "smoke exit $?"
$PY benchmark_gos.py --config configs/benchmark-pilot-heldout-v1.json --output runs/pilot-heldout-v1 > runs/queue/pilot.log 2>&1
log "pilot exit $?"
