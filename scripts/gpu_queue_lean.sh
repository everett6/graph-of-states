#!/usr/bin/env bash
# Formal-proof baselines on miniF2F (Lean tool in the loop), after scripts/gpu_queue.sh finishes.
cd "$(dirname "$0")/.."
PY=.venv-training/bin/python
GOEDEL=$(ls -d ~/.cache/huggingface/hub/models--Goedel-LM--Goedel-Prover-V2-8B/snapshots/*/ | head -1)
GEMMA=/home/everett/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-q4_0-unquantized/snapshots/b6ed86275a6a5735884e208bfed95b445a684ca2
log() { echo "$(date -Is) $*" | tee -a runs/queue/status.log; }
while pgrep -f gpu_queue.sh | grep -qv $$ || pgrep -f measure_headroom.py >/dev/null; do sleep 300; done
log "lean queue start"
common="--eval data/lean/minif2f-test.jsonl --max-tools 2 --n 2 --temperature 0.7"
$PY agent_loop.py $common --base-path "$GOEDEL" --thinking --max-new-tokens 3072 --context 8192 --limit 3 --output runs/lean-smoke-goedel > runs/queue/lean-smoke-goedel.log 2>&1
log "goedel smoke exit $?"
$PY agent_loop.py $common --base-path "$GEMMA" --max-new-tokens 2048 --context 6144 --limit 3 --output runs/lean-smoke-gemma > runs/queue/lean-smoke-gemma.log 2>&1
log "gemma smoke exit $?"
$PY agent_loop.py $common --base-path "$GOEDEL" --thinking --max-new-tokens 3072 --context 8192 --output runs/minif2f-goedel8b-agent > runs/queue/minif2f-goedel.log 2>&1
log "goedel minif2f exit $?"
$PY agent_loop.py $common --base-path "$GEMMA" --max-new-tokens 2048 --context 6144 --output runs/minif2f-gemma12b-agent > runs/queue/minif2f-gemma.log 2>&1
log "gemma minif2f exit $?"
