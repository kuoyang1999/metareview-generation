#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:/mnt/nvme1/Kuo/Meta-Review

python scripts/python/inf.py \
  --base_model "./checkpoints/20250103_122115/checkpoint-500_merged"
  # --base_model "./checkpoints/sft_final/checkpoint-1000_merged"