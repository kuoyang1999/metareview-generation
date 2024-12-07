# LLM Meta Review

**LLM Meta Review** is a project aimed at extending the context length of LLaMA-based models and fine-tuning them using low-rank adaptation (LoRA) for meta review tasks. It includes support for flash attention, custom datasets, and integration with DeepSpeed for efficient large-model training.

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
│  ├─ __init__.py
│  ├─ utils.py                   # Utilities, constants, prompt dict
│  ├─ model.py                   # Model loading & LoRA configuration
│  ├─ training.py                # Main training logic using HF Trainer
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ datasets.py             # Dataset classes (SupervisedDataset, PeerSumDataset)
│  │  ├─ collators.py            # Data collators for supervised fine-tuning
│  │  ├─ preprocessing.py        # Preprocessing/tokenization routines
│  │  └─ data_module.py          # make_supervised_data_module function
│  └─ attn/
│     ├─ __init__.py
│     └─ llama_attn_replace_sft.py # Patches LLaMA attention for flash attn
│
├─ scripts/
│  └─ train.py                   # Entry point for training (calls `train()` from training.py)
│
├─ configs/
│  └─ deepspeed_config.json      # Example DeepSpeed config
│
├─ checkpoints/                  # Checkpoints saved here
│
├─ logs/
│  └─ wandb/                     # Wandb logs
│
├─ environment.yml               # Conda environment file (alternative install)
├─ requirements.txt              # Pip requirements file (alternative install)
└─ README.md                     # This documentation
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
