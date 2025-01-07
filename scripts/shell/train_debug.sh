#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/mnt/nvme1/Kuo/Meta-Review

torchrun --nproc_per_node=1 \
    scripts/python/train.py \
    --max_samples 1 \
    --bf16 True \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --output_dir "" \
    --use_flash_attn True \
    --low_rank_training True \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 0 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 0 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed configs/ds_configs/stage3.json \
    --logging_level "debug" \

    # --max_samples 8
    #     --data_name_or_path /mnt/nvme1/datasets/LongAlpaca-16k-length/LongAlpaca-16k-length.json \
    #    --model_max_length 16384 \
    #    --data_name_or_path "oaimli/PeerSum" \

    # --max_samples 1\