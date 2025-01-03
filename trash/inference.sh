export PYTHONPATH=$PYTHONPATH:/mnt/nvme1/Kuo/Meta-Review

python inference.py \
    --base_model "./checkpoints/20241207_125507/checkpoint-19000_merged" \
    # --base_model "./checkpoints/sft_final/checkpoint-1000_merged"