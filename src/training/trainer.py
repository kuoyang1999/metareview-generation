import json
import os
import logging
import datetime
import torch
import transformers
import wandb

from dataclasses import dataclass, field
from typing import Optional
from torch import nn
from transformers import Trainer, HfArgumentParser

# Local imports from your project
from src.data import make_supervised_data_module
from src.model import load_model_and_tokenizer, apply_lora_if_needed, replace_llama_attn
from src.utils import jload, IGNORE_INDEX

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
        metadata={"help": "Maximum number of samples to use."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

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
        metadata={"help": "Enable Wandb logging (default: True)."},
    )

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup default output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not training_args.output_dir:
        training_args.output_dir = f"checkpoints/{timestamp}"

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
        # ... add more as needed
    }

    # Initialize Wandb if enabled
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if training_args.use_wandb and training_args.local_rank == 0:
        
        wandb.login(key=os.getenv("fef0c9efbf5c8ed6f3fb3811b172280e040e1bba"), relogin=True)

        wandb.init(
            project="meta-review",
            config=config,
            name=f"{timestamp}",
            dir="logs",
        )

    # Setup attention
    if model_args.model_type == "gpt-neox":
        raise NotImplementedError("GPT-NeoX models are not currently supported.")
    else:
        replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    # Build data module
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

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
    if training_args.use_wandb:
        wandb.finish()