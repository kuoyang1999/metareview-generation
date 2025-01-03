export PYTHONPATH=$PYTHONPATH:/mnt/nvme1/Kuo/Meta-Review

python scripts/python/merge_weight.py \
    --peft_model "./checkpoints/20250103_122115/checkpoint-500" \
    --save_path "./checkpoints/20250103_122115/checkpoint-500_merged" \
    # --peft_model "./checkpoints/sft_final/checkpoint-1000" \
    # --save_path "./checkpoints/sft_final/checkpoint-1000_merged" \
