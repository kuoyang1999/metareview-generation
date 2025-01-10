import os
import logging
import datetime
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import HfArgumentParser

from src.training.trainer import train

# Set random seed for reproducibility
transformers.set_seed(42)

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

if __name__ == "__main__":
    # Configure logging
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs("logs/", exist_ok=True)
    log_file = f"logs/training_{timestamp}.log"

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Parse arguments and start training
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set logging level
    logging.getLogger().setLevel(training_args.logging_level.upper())
    
    if not training_args.output_dir:
        training_args.output_dir = f"checkpoints/{timestamp}"

    train(model_args, data_args, training_args, timestamp)