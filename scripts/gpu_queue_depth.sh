#!/usr/bin/env bash
# After the pilot, one GPU job at a time:
#   1. paired pilot comparison;
#   2. long-thinking check (16k tokens, thinking on, 40 contest problems): frozen, plus the pilot's recurrent p4
#      adapter only if it did not lose to frozen (SFT on human-written solutions collapsed the LoRA arm);
#   3. self-distilled MATH-train SFT data (the frozen model's own verified solutions, <= 1024 tokens);
#   4. the recurrent-depth round on it (configs/benchmark-depth-v2.json) with paired and pass-scaling reports;
#   5. agentic GRPO (verified rewards) from the depth round's curriculum adapter and from its block-LoRA control,
#      each scored on the same held-out rows, then paired (runs/rl-eval);
#   6. the Lean/agent queue (scripts/gpu_queue_lean.sh).
# Stop it with: pkill -f gpu_queue_depth.sh
cd "$(dirname "$0")/.."
PY=.venv-training/bin/python
GEMMA=/home/everett/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-q4_0-unquantized/snapshots/b6ed86275a6a5735884e208bfed95b445a684ca2
log() { echo "$(date -Is) $*" | tee -a runs/queue/status.log; }
mkdir -p runs/queue
while pgrep -f gpu_queue.sh >/dev/null || pgrep -f measure_headroom.py >/dev/null || pgrep -f "benchmark_gos.py --config" >/dev/null; do sleep 300; done
$PY scripts/compare_heldout.py runs/pilot-heldout-v1 --baseline frozen-d4 > runs/queue/pilot-compare.log 2>&1
log "pilot compare exit $?"
think="--eval data/eval/heldout-v2.jsonl --base-path $GEMMA --sources aime_2025,hmmt_feb_2025,aime_2026,hmmt_feb_2026
  --per-source 10 --n 1 --ks 1 --thinking --max-new-tokens 16384 --context 18432 --batch 1
  --temperature 1.0 --top-k 64 --top-p 0.95"
$PY best_of_n.py $think --output runs/thinking16k-frozen > runs/queue/thinking16k-frozen.log 2>&1
log "thinking16k frozen exit $?"
RECURRENT=runs/pilot-heldout-v1/recurrent-block18-24-p4/adapter
if [ -f $RECURRENT/recurrence.pt ] && $PY -c "
import json, sys
table = json.load(open('runs/pilot-heldout-v1/paired_vs_frozen-d4.json'))
sys.exit(0 if table['recurrent-block18-24-p4/passes-4']['all']['diff'] >= -0.02 else 1)"; then
  $PY best_of_n.py $think --adapter $RECURRENT --adapter-kind recurrent --passes 4 \
    --output runs/thinking16k-recurrent-p4 > runs/queue/thinking16k-recurrent.log 2>&1
  log "thinking16k recurrent p4 exit $?"
else
  log "thinking16k recurrent p4 skipped (adapter missing or worse than frozen)"
fi
$PY scripts/self_distill.py --input data/train/math-train-v1.jsonl --output data/train/math-selfdistill-1024.jsonl \
  --base-path "$GEMMA" --sample 700 --budget 1024 > runs/queue/selfdistill-math.log 2>&1
log "self-distill math exit $? ($(wc -l < data/train/math-selfdistill-1024.jsonl) rows)"
log "depth round start"
$PY benchmark_gos.py --config configs/benchmark-depth-v2.json --output runs/depth-v2 --resume > runs/queue/depth-v2.log 2>&1
log "depth round exit $?"
$PY scripts/compare_heldout.py runs/depth-v2 --baseline frozen-d4 > runs/queue/depth-v2-compare.log 2>&1
log "depth compare exit $?"
$PY scripts/compare_heldout.py runs/depth-v2 --scaling > runs/queue/depth-v2-scaling.log 2>&1
log "depth scaling exit $?"
evalrl="--eval data/eval/heldout-v2.jsonl --base-path $GEMMA --sources aime_2025,hmmt_feb_2025,aime_2026,hmmt_feb_2026,lcb_v6,math500_l45
  --per-source 30 --n 2 --ks 1,2 --max-new-tokens 2048 --batch 2 --adapter-kind recurrent
  --temperature 1.0 --top-k 64 --top-p 0.95"
for arm in curriculum:recurrent-curriculum-m6:8 p1-control:block-lora18-24-p1:1; do
  IFS=: read name source passes <<< "$arm"
  config=configs/rl-recurrent-$name-agent.json
  out=runs/rl-recurrent-$name-agent
  if [ -f runs/depth-v2/$source/adapter/recurrence.pt ]; then
    [ -f $out/adapter/recurrence.pt ] || $PY rl_recurrent.py --config $config > runs/queue/rl-$name.log 2>&1
    log "rl $name exit $?"
    $PY best_of_n.py $evalrl --adapter $out/adapter --passes $passes --output runs/rl-eval/$name \
      > runs/queue/rl-eval-$name.log 2>&1
    log "rl eval $name exit $?"
  else
    log "rl $name skipped (no depth-round adapter)"
  fi
done
$PY scripts/compare_heldout.py runs/rl-eval --baseline p1-control > runs/queue/rl-compare.log 2>&1
log "rl compare exit $?"
exec scripts/gpu_queue_lean.sh
