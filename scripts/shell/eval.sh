#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(dirname "$0")/../../

export CUDA_VISIBLE_DEVICES=0

python scripts/python/eval.py \
  --base_model "./checkpoints/meta-review/ICLR/20250214_044603/checkpoint-1000_merged" \
  --conference "ICLR"\
  --data_name_or_path "MetaGen" \
  --tag "1000_merged" \
  --max_samples 100 \
  --flash_attn \
  # --flash_attn
  # --max_samples 1 \