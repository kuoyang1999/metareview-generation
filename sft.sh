#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:/mnt/nvme1/Kuo/Meta-Review

torchrun --nproc_per_node=4 \
    sft.py \
    --bf16 True \
    --output_dir "./checkpoints/sft_final" \
    --use_flash_attn False \
    --low_rank_training True \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --model_max_length 16384 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed configs/ds_configs/stage3.json

    # --max_samples 8
    #     --data_name_or_path /mnt/nvme1/datasets/LongAlpaca-16k-length/LongAlpaca-16k-length.json \
    #    --model_max_length 16384 \
    #    --model_name_or_path /mnt/nvme1/models/Llama-2-7b-hf \
    #    --data_name_or_path "oaimli/PeerSum" \