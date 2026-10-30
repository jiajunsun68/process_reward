#!/bin/bash
set -e
cd /data4/sjj/catelogue/process_reward
PY=/data/sjj/process_reward_venv/bin/python
MODEL=/data4/sjj/.cache/modelscope/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/2de2439ea21be1dc5cb21f22f88af07e43393cbb
PROGRESS=/tmp/g3_cond_eval_progress.log
echo "start $(date)" > "$PROGRESS"
run() {
  NAME=$1; ADAPTER=$2
  echo "=== $NAME begin $(date) ===" >> "$PROGRESS"
  ADAPTER_ARGS=()
  if [ -n "$ADAPTER" ]; then ADAPTER_ARGS=(--adapter "$ADAPTER"); fi
  CUDA_VISIBLE_DEVICES=0 $PY evaluate_actions.py --mode conditional --model "$MODEL" "${ADAPTER_ARGS[@]}" \
    --device cuda:0 --output-dir "runs/$NAME" > "logs/$NAME.log" 2>&1
  echo "=== $NAME end $(date) ===" >> "$PROGRESS"
}
run G3-BASE-a01 ""
run G3-TRUE-eval-a01 runs/G3-DPO-TRUE-s42-a01/adapter
run G3-SHUFFLE-eval-a01 runs/G3-DPO-SHUFFLE-s42-a01/adapter
echo "ALL DONE $(date)" >> "$PROGRESS"
