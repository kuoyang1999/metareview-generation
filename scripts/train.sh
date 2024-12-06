#!/usr/bin/env bash

torchrun --nproc_per_node=4 \
    scripts/train.py \
    --model_name_or_path /mnt/nvme1/models/Llama-2-7b-hf \
    --bf16 True \
    --output_dir ./checkpoints/12_5 \
    --model_max_length 16384 \
    --use_flash_attn True \
    --data_path /mnt/nvme1/datasets/LongAlpaca-16k-length/LongAlpaca-16k-length.json \
    --low_rank_training True \
    --num_train_epochs 200 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 98 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed configs/deepspeed_config.json