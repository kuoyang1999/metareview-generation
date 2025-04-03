export PYTHONPATH=$PYTHONPATH:$(dirname "$0")/../../

python3 scripts/python/convert_merge_checkpoint.py \
        --checkpoint_dir ./checkpoints/meta-review/NEURIPS/20250214_121044/checkpoint-1600