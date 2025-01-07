#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/mnt/nvme1/Kuo/Meta-Review
export WANDB_API_KEY="fef0c9efbf5c8ed6f3fb3811b172280e040e1bba"

torchrun --nproc_per_node=3 \
    scripts/python/train.py \
    --bf16 True \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --output_dir "" \
    --use_flash_attn True \
    --low_rank_training True \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed configs/ds_configs/stage3.json \
    # --logging_level "debug" \

    # --max_samples 8
    #     --data_name_or_path /mnt/nvme1/datasets/LongAlpaca-16k-length/LongAlpaca-16k-length.json \
    #    --model_max_length 16384 \
    #    --data_name_or_path "oaimli/PeerSum" \

    # --max_samples 1\