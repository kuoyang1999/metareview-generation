import json
import os
import logging
import torch
import transformers
import wandb

from torch import nn
from transformers import Trainer

# Local imports from your project
from src.data import make_supervised_data_module
from src.model import load_model_and_tokenizer, apply_lora_if_needed, replace_llama_attn
from src.utils import init_wandb

def train(model_args, data_args, training_args, timestamp):
    # Setup attention
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
    }

    # Initialize Wandb if enabled
    wandb_enabled = init_wandb(training_args, config, timestamp, data_args)

    # Apply LoRA if needed
    model = apply_lora_if_needed(model, model_args, training_args)

    # Print dtype info only when using quantization
    if training_args.quantization != "none":
        logging.info("Printing model dtype information (quantized model):")
        dtypes = {}
        for _, p in model.named_parameters():
            dtype = p.dtype
            dtypes[dtype] = dtypes.get(dtype, 0) + p.numel()
        total = sum(dtypes.values())
        for k, v in dtypes.items():
            logging.info(f"{k}: {v} parameters ({v / total:.2%})")

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