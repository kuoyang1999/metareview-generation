#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/mnt/nvme1/Kuo/Meta-Review

python scripts/python/eval.py \
  --base_model "./checkpoints/20250108_223544/checkpoint-1800_merged" \
  --max_samples 1 \
  --flash_attn