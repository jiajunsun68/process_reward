#!/bin/bash
set -e
cd /data4/sjj/catelogue/process_reward
PY=/data/sjj/process_reward_venv/bin/python
MODEL=/data4/sjj/.cache/modelscope/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/2de2439ea21be1dc5cb21f22f88af07e43393cbb
PROGRESS=/tmp/g3_step_a_progress.log
echo "start $(date)" > "$PROGRESS"

run() {
  NAME=$1; ADAPTER=$2; REUSE=$3
  echo "=== $NAME begin $(date) ===" >> "$PROGRESS"
  ADAPTER_ARGS=()
  if [ -n "$ADAPTER" ]; then ADAPTER_ARGS=(--adapter "$ADAPTER"); fi
  $PY evaluate_actions.py --mode score --model "$MODEL" "${ADAPTER_ARGS[@]}" \
    --reuse-greedy-dir "$REUSE" \
    --device cuda:0 --output-dir "runs/$NAME" \
    > "/tmp/${NAME}.log" 2>&1
  echo "=== $NAME end $(date) ===" >> "$PROGRESS"
}

run G3-READ-BASE-a01 "" runs/G2R-BASE-a01
run G3-READ-I42-a01 runs/G2-I-s42-a01/adapter runs/G2R-I-s42-a01
run G3-READ-I43-a01 runs/G2-I-s43-a01/adapter runs/G2R-I-s43-a01
run G3-READ-C42-a01 runs/G2-C-s42-a01/adapter runs/G2R-C-s42-a01
run G3-READ-C43-a01 runs/G2-C-s43-a01/adapter runs/G2R-C-s43-a01

echo "ALL DONE $(date)" >> "$PROGRESS"
