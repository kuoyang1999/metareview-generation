export PYTHONPATH=$PYTHONPATH:/mnt/nvme1/Kuo/Meta-Review

python merge_weight.py \
    --peft_model "./checkpoints/20241207_125507/checkpoint-19000" \
    --save_path "./checkpoints/20241207_125507/checkpoint-19000_merged" \
    # --peft_model "./checkpoints/sft_final/checkpoint-1000" \
    # --save_path "./checkpoints/sft_final/checkpoint-1000_merged" \
