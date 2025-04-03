#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(dirname "$0")/../../

# Load OpenAI API key from .env file
ENV_FILE="$(dirname "$0")/../../.env"
if [ -f "$ENV_FILE" ]; then
    # Read and export the API key directly, making sure to trim any whitespace
    export OPENAI_API_KEY=$(grep OPENAI_API_KEY "$ENV_FILE" | cut -d '=' -f2 | tr -d '[:space:]')
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY not found in .env file"
        exit 1
    fi
    echo "Successfully loaded OPENAI_API_KEY"
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=2

python scripts/python/gpt_eval.py \
  --base_model "gpt-4o-mini" \
  --conference "EMNLP" "ICLR" "TMLR" "ACMMM" "NEURIPS" \
  --data_name_or_path "MetaGen" \
  --tag "gpt-4o-mini" \
#   --max_samples 1
