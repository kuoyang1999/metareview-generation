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

    # --- New debug field ---
    debug: bool = field(
        default=False,
        metadata={"help": "Enable debug mode (default: False)."}
    )


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize the logger
    # If debug mode is enabled, set the log level to DEBUG, otherwise INFO
    log_level = logging.DEBUG if training_args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Optional: print out final arguments if in debug mode
    if training_args.debug:
        logger.debug("TrainingArguments: %s", training_args)
        logger.debug("ModelArguments: %s", model_args)
        logger.debug("DataArguments: %s", data_args)

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
        "debug_mode": training_args.debug,  # Include debug status in config
        # ... add more as needed
    }

    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Initialize Wandb if enabled
    if training_args.use_wandb and training_args.local_rank == 0:
        try:
            wandb.login(key=os.getenv("fef0c9efbf5c8ed6f3fb3811b172280e040e1bba"), relogin=True)
            wandb.init(
                project="meta-review",
                config=config,
                name=f"{timestamp}",
                dir="logs",
            )
        except Exception as e:
            logger.warning("Failed to login to WandB: %s", e)

    # Setup attention
    if model_args.model_type == "gpt-neox":
        raise NotImplementedError("GPT-NeoX models are not currently supported.")
    else:
        logger.debug("Replacing Llama attention with flash attention? %s", training_args.use_flash_attn)
        replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    # Load model and tokenizer
    logger.debug("Loading model and tokenizer ...")
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    # Build data module
    logger.debug("Building the data module ...")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Apply LoRA if needed
    logger.debug("Applying LoRA if needed ...")
    model = apply_lora_if_needed(model, model_args, training_args)

    # Debug print dtype distribution
    if training_args.debug:
        logger.debug("Printing parameter dtype distribution ...")
        dtypes = {}
        for _, p in model.named_parameters():
            dtype = p.dtype
            dtypes[dtype] = dtypes.get(dtype, 0) + p.numel()
        total = sum(dtypes.values())
        for k, v in dtypes.items():
            logger.debug("  dtype=%s, count=%d (%2.2f%%)", k, v, (v / total) * 100)

    # Enable gradient checkpointing, input grads, disable cache
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    trainer = Trainer(model=model, args=training_args, **data_module)

    # Debug: Optionally run a small forward pass before training
    if training_args.debug:
        logger.debug("Running a small dummy forward pass with random input for debugging ...")
        try:
            dummy_input = torch.randint(
                low=0, high=tokenizer.vocab_size, 
                size=(1, 8),  # batch_size=1, seq_len=8
                dtype=torch.long,
                device=trainer.model.device
            )
            _ = trainer.model(dummy_input)
            logger.debug("Dummy forward pass succeeded.")
        except Exception as e:
            logger.error("Dummy forward pass failed: %s", e)

    # Train!
    logger.info("Starting training ...")
    train_result = trainer.train()
    logger.info("Training completed.")

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    # Save hyperparams/config
    args_file = os.path.join(training_args.output_dir, "settings.json")
    with open(args_file, "w") as f:
        json.dump(config, f, indent=4)

    if training_args.use_wandb:
        wandb.finish()