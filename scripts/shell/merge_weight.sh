export PYTHONPATH=$PYTHONPATH:/mnt/nvme1/Kuo/Meta-Review

python scripts/python/merge_weight.py \
    --peft_model "./checkpoints/20250108_223544/checkpoint-1800" \
    --save_path "./checkpoints/20250108_223544/checkpoint-1800_merged"
    # --peft_model "./checkpoints/sft_final/checkpoint-1000" \
    # --save_path "./checkpoints/sft_final/checkpoint-1000_merged" \
