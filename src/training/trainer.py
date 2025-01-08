import json
import os
import logging
import datetime
import torch
import transformers
import wandb

from dataclasses import dataclass, field
from typing import Optional, Literal, List
from torch import nn
from transformers import Trainer, HfArgumentParser

# Local imports from your project
from src.data import make_supervised_data_module
from src.model import load_model_and_tokenizer, apply_lora_if_needed, replace_llama_attn
from src.utils import jload, IGNORE_INDEX, init_wandb

# Set random seed for reproducibility
transformers.set_seed(42)

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs("logs/logging", exist_ok=True)
log_file = f"logs/logging/training_{timestamp}.log"

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will still print to console
    ]
)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    model_type: Optional[str] = field(default="llama")

@dataclass
class DataArguments:
    data_name_or_path: str = field(
        default="oaimli/PeerSum",
        metadata={"help": "Hugging Face name or local path to the training data."}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use for training."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of steps for the warmup phase."},
    )
    logging_level: str = field(
        default="info",
        metadata={
            "help": "Logging level (debug, info, warning, error, critical)",
            "choices": ["debug", "info", "warning", "error", "critical"]
        }
    )

    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain full-attention."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA/low-rank adaptation."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable params besides LoRA weights if low-rank training."},
    )
    use_wandb: bool = field(
        default=True,
        metadata={"help": "Enable or disable Wandb logging."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "Rank of LoRA matrices."},
    )

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set logging level
    logging.getLogger().setLevel(training_args.logging_level.upper())
    
    # Setup default output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not training_args.output_dir:
        training_args.output_dir = f"checkpoints/{timestamp}"

    # Setup attention
    if model_args.model_type == "gpt-neox":
        raise NotImplementedError("GPT-NeoX models are not currently supported.")
    else:
        replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)    

    model, tokenizer = load_model_and_tokenizer(model_args, training_args)
    
    logging.debug(f"Self-attn class in layer[0]: {model.model.layers[0].self_attn.__class__}")

    # Build data module
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Ensure warmup steps is less than total steps
    total_steps = int(len(data_module["train_dataset"]) * training_args.num_train_epochs / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))
    if training_args.warmup_steps >= total_steps:
        training_args.warmup_steps = max(0, total_steps - 1)
        logging.info(f"Adjusted warmup steps to {training_args.warmup_steps} to be less than total steps {total_steps}")

    # Build config dictionary for logging or W&B
    config = {
        "model_name_or_path": model_args.model_name_or_path,
        "data_name_or_path": data_args.data_name_or_path,
        "max_samples": data_args.max_samples,
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "lr_scheduler_type": str(training_args.lr_scheduler_type),
        "use_flash_attn": training_args.use_flash_attn,
        "low_rank_training": training_args.low_rank_training,
        "save_steps": training_args.save_steps,
        "output_dir": training_args.output_dir,
        "lora_r": training_args.lora_r,
        # ... add more as needed
    }

    # Initialize Wandb if enabled

    wandb_enabled = init_wandb(training_args, config, timestamp)

    # Apply LoRA if needed
    model = apply_lora_if_needed(model, model_args, training_args)

    # Print dtype info for debugging
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        dtypes[dtype] = dtypes.get(dtype, 0) + p.numel()
    total = sum(dtypes.values())
    for k, v in dtypes.items():
        print(k, v, f"{v / total:.2%}")

    # Enable gradient checkpointing, input grads, disable cache
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    trainer = Trainer(model=model, args=training_args, **data_module)

    # Train!
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    # Save hyperparams/config
    args_file = os.path.join(training_args.output_dir, "settings.json")
    with open(args_file, "w") as f:
        json.dump(config, f, indent=4)

    # End Wandb session
    if wandb_enabled:
        wandb.finish()