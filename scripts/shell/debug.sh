#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(dirname "$0")/../../
export HF_TOKEN="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"
huggingface-cli login --token $HF_TOKEN
export WANDB_API_KEY="fef0c9efbf5c8ed6f3fb3811b172280e040e1bba"
export WANDB_BASE_URL="https://api.wandb.ai"
wandb login $WANDB_API_KEY --relogin

torchrun --nproc_per_node=3 \
    scripts/python/train.py \
    --max_samples 1 \
    --bf16 True \
    --data_name_or_path "test" \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --output_dir "" \
    --use_flash_attn True \
    --low_rank_training True \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
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
    --use_wandb False

    # --logging_level "debug" \