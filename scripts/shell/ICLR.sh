#!/usr/bin/env bash

# Clone the private repository
git clone https://kuoyang1999:${GITHUB_TOKEN}@github.com/kuoyang1999/metareview-generation.git

# Navigate into the cloned repository directory
cd metareview-generation

# Activate conda environment
eval "$(conda shell.bash hook)"
conda env create -f environment.yml
conda activate meta-review
pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:$(dirname "$0")/../../
# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found. Please create one based on .env.template"
    exit 1
fi

wandb login $WANDB_API_KEY --relogin

torchrun --nproc_per_node=8 \
    scripts/python/train.py \
    --bf16 True \
    --data_name_or_path "MetaGen" \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --output_dir "/sensei-fs/users/qdong/kuo/meta-review/" \
    --log_dir "/sensei-fs/users/qdong/kuo/meta-review/" \
    --use_flash_attn True \
    --low_rank_training True \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed configs/ds_configs/stage3.json \
    --conferences "ICLR"