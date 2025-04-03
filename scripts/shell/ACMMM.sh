export PYTHONPATH=$PYTHONPATH:$(dirname "$0")/../../
# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found. Please create one based on .env.template"
    exit 1
fi

wandb login $WANDB_API_KEY --relogin

torchrun --nproc_per_node=3 \
    scripts/python/train.py \
    --bf16 True \
    --data_name_or_path "MetaGen" \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --output_dir "" \
    --use_flash_attn True \
    --low_rank_training True \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed configs/ds_configs/stage3.json \
    --conferences "ACMMM"