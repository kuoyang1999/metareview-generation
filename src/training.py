import logging
import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional
from torch import nn
from transformers import Trainer, HfArgumentParser
from .data import make_supervised_data_module
from .model import load_model_and_tokenizer, apply_lora_if_needed
from .utils import IGNORE_INDEX
from .attn.llama_attn_replace_sft import replace_llama_attn
# from gptneox_attn_replace import replace_gpt_neox_attn # TODO: add this back when we have a model that uses it

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

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
        metadata={"help": "Additional trainable parameters except LoRA weights if low rank training."},
    )

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup attention
    if model_args.model_type == "gpt-neox":
        # TODO: add this back when we have a model that uses it, now raises an error
        # replace_gpt_neox_attn(training_args.use_flash_attn, training_args.use_full_attn)
        raise NotImplementedError("GPT-NeoX models are not currently supported")
    else:
        replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    model, tokenizer = load_model_and_tokenizer(model_args, training_args)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    model = apply_lora_if_needed(model, model_args, training_args)

    # dtype info
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        dtypes[dtype] = dtypes.get(dtype, 0) + p.numel()
    total = sum(dtypes.values())
    for k, v in dtypes.items():
        print(k, v, v / total)

    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    trainer = Trainer(model=model, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)