#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:/mnt/nvme1/Kuo/Meta-Review

python scripts/python/inf.py \
  --base_model "./checkpoints/20250108_223544/checkpoint-1800_merged"
  # --base_model "./checkpoints/sft_final/checkpoint-1000_merged"