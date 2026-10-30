#!/bin/bash
set -e
cd /data4/sjj/catelogue/process_reward
PY=/data/sjj/process_reward_venv/bin/python
MODEL=/data4/sjj/.cache/modelscope/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/2de2439ea21be1dc5cb21f22f88af07e43393cbb
echo "begin $(date)" > /tmp/g3_dpo_true_progress.log
CUDA_VISIBLE_DEVICES=0 $PY train_dpo.py \
  --train-file data/g3_conditional/true/train.jsonl \
  --eval-file data/g3_conditional/true/validation.jsonl \
  --output-dir runs/G3-DPO-TRUE-s42-a01 \
  --save-final-adapter --max-save-mib 512 \
  --model "$MODEL" --revision main --local-files-only \
  --max-steps -1 --epochs 1.0 \
  --max-length 2048 --max-action-tokens 64 \
  --microbatch 1 --grad-accum 8 \
  --learning-rate 1e-5 --beta 0.1 \
  --lora-targets qv --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
  --action-format tool-json --seed 42 \
  > /tmp/G3-DPO-TRUE-s42-a01.log 2>&1
echo "end $(date)" >> /tmp/g3_dpo_true_progress.log
