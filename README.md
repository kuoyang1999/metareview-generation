# LLM Meta Review

**LLM Meta Review** is a project aimed at extending the context length of LLaMA-based models and fine-tuning them using low-rank adaptation (LoRA) for meta review tasks. It includes support for flash attention, custom datasets, and integration with DeepSpeed for efficient large-model training.

## Updates

- 2025-01-07: 🔥 To God Hua:
  - conda env create -f environment.yml
  - pip install -r requirements.txt
  - conda activate meta-review
  - pip install transformers==4.34.0
  - pip install deepspeed==0.15.4
  - sh scripts/shell/train.sh

## Features

- **Extended Context Length:** Train language models with up to 32k or more tokens of context.
- **LoRA Fine-tuning:** Efficiently fine-tune large language models with minimal extra parameters.
- **Flash Attention Support:** Leverage flash attention for faster training on compatible hardware (A100/H100).
- **Hugging Face Datasets Integration:** Easily load and preprocess datasets from the Hugging Face Hub.
- **DeepSpeed Integration:** Utilize ZeRO optimization and other features to handle large-scale training.
- **Custom Prompts:** Adapt prompt templates for different datasets (e.g., PeerSum with multiple reviews).

## Code Structure

```plaintext
Meta-Review/
├─ src/
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ datasets.py          # e.g. SupervisedDataset, PeerSumDataset
│  │  ├─ collators.py         # DataCollator classes, etc.
│  │  ├─ data_module.py       # make_supervised_data_module, etc.
│  │  └─ preprocessing.py     # preprocess function for tokenization
│  │
│  ├─ model/
│  │  ├─ __init__.py
│  │  ├─ base.py              # load_model_and_tokenizer
│  │  ├─ lora.py              # apply_lora_if_needed
│  │  └─ attn/
│  │     ├─ __init__.py
│  │     └─ llama_attn_replace_sft.py # Attention patching or FlashAttention-related code
│  │
│  ├─ training/
│  │  ├─ __init__.py
│  │  └─ trainer.py           # main training logic (HF Trainer or custom loops)
│  │
│  ├─ evaluation/
│  │  ├─ __init__.py
│  │  └─ eval_logic.py        # evaluation metrics, loops (perplexity, BLEU, ROUGE, etc.)
│  │
│  ├─ inference/
│  │  ├─ __init__.py
│  │  └─ infer.py             # inference or generation code
│  │
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ constants.py         # IGNORE_INDEX, special tokens, PROMPT_DICT, etc.
│  │  ├─ io_utils.py          # jload(), file I/O functions
│  │  └─ token_ops.py         # smart_tokenizer_and_embedding_resize, preprocess, etc.
│  │
│  ├─ tests/                  # unit tests
│  │  ├─ __init__.py
│  │  └─ test_data.py         # unit tests for data pipelines
│  │
│  └─ __init__.py             # optional top-level re-exports
│
├─ scripts/
│  ├─ python/
│  │  ├─ train.py             # main entry point for training (calls src/training/trainer)
│  │  ├─ eval.py              # main entry for evaluation (calls src/evaluation/eval_logic)
│  │  ├─ inference.py         # main entry for inference (uses src/inference/infer)
│  │  └─ merge_weight.py      # merges model weights or LoRA adaptors
│  └─ shell/
│     ├─ train.sh             # quick reference shell script to run train.py
│     ├─ eval.sh              # quick reference shell script to run eval.py
│     ├─ inference.sh         # quick reference shell script to run inference.py
│     ├─ merge_weight.sh      # shell script for merging
│     └─ run_docker.sh        # script to run Docker container with GPU support
│
├─ configs/
│  ├─ ds_configs/             # DeepSpeed config for 3 stages
│  └─ hyperparams.yaml        # example hyperparam config
│
├─ checkpoints/               # directory for saving model checkpoints
│
├─ logs/
│  ├─ wandb/                  # wandb logs (if used)
│  └─ other_logs/             # custom logs, TB logs, etc.
│
├─ environment.yml            # conda environment file
├─ requirements.txt           # pip requirements file
├─ Dockerfile                # CUDA-enabled container definition
├─ .dockerignore             # files to exclude from Docker context
├─ README.md                 # project overview & instructions
└─ LICENSE                   # license (e.g., MIT, Apache-2.0)
```

## Installation

### 1. Install PyTorch (Required)

Before setting up the environment, ensure you install a compatible PyTorch version. Visit the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for details. Below are some examples:

- **CUDA 12.1:**
  ```bash
  pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

### 1. Using `environment.yml` with Conda

If you have Conda installed, you can create an environment directly:

```bash
conda env create -f environment.yml
conda activate meta-review
```

### 2. Using `requirements.txt` (Pip)

```bash
conda create -n meta-review python=3.9 -y
conda activate meta-review
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Hugging Face Login

If you’re loading datasets or models from Hugging Face:

```bash
huggingface-cli login
```

Follow the instructions to paste your token.

### 4. Wandb Login

```bash
wandb login
```

By default, wandb logging is enabled. If you want to disable wandb logging, paste this in the training script or command line:

```bash
--use_wandb False
```

### 5. Set PYTHONPATH

Make sure `src` is on the Python path:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/Meta-Review
```

You can add this to your shell config or a setup script.

## Running the Training Script

### Running with the Training Script

The simplest way to start training is to use the provided training script from the root path:

```bash
sh scripts/train.py
```

Below is an example command using `torchrun` (for multi-GPU training) :

```bash
torchrun --nproc_per_node=4 \
    scripts/train.py \
    --bf16 True \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --output_dir "" \
    --use_flash_attn True \
    --low_rank_training True \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 98 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed configs/ds_configs/stage3.json
```

### Key Arguments:

- `--model_name_or_path`: Local or Hub model path (e.g., `meta-llama/Llama-2-7b-hf`).
- `--data_path`: Dataset name on HF Hub (e.g., `oaimli/PeerSum`) or local path.
- `--use_flash_attn`: Enables flash attention (requires appropriate GPU).
- `--low_rank_training`: Enables LoRA fine-tuning.
- `--deepspeed`: Uses the provided DeepSpeed config for efficient training.

## Debugging and Logging

- The dataset classes print a few example prompts and labels at initialization to help with debugging.
- Logs will be stored in `logs/training.log` if configured. Adjust logging settings in `training.py` or dataset classes as needed.
- If you encounter import issues (`ModuleNotFoundError`), ensure you’re running from the project root and `PYTHONPATH` is set correctly.

## Additional Notes

- Modify `max_samples`, `max_length`, or other arguments in `data_args` or dataset classes if needed.
- For larger models or longer training, consider more powerful hardware or additional DeepSpeed optimizations.
