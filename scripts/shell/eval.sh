#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/mnt/nvme1/Kuo/Meta-Review

python scripts/python/eval.py \
  --base_model "./checkpoints/20241207_125507/checkpoint-19000_merged" \
  # --max_samples 4