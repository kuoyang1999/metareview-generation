import os
import logging
import datetime
from dataclasses import dataclass, field
from typing import Optional, List
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
    years: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of years to filter the dataset by"}
    )
    conferences: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of conferences to filter the dataset by"}
    )
    subjects: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of subjects to filter the dataset by"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    log_dir: str = field(
        default="logs",
        metadata={"help": "Directory where training logs will be saved."},
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of steps for the warmup phase."},
    )
    quantization: str = field(
        default="none",
        metadata={
            "help": "Quantization type to use (4bit, 8bit, or none)",
            "choices": ["4bit", "8bit", "none"]
        }
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
    # Parse arguments first
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Configure logging
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create conference-specific directories if conferences are specified
    if data_args.conferences:
        conf_str = '_'.join(data_args.conferences)
        log_path = f"{training_args.log_dir}/{conf_str}"
        training_args.output_dir = f"{training_args.output_dir}/{conf_str}/{timestamp}"
    else:
        log_path = training_args.log_dir
        training_args.output_dir = f"{training_args.output_dir}/{timestamp}"
        
        
    os.makedirs(log_path, exist_ok=True)
    log_file = f"{log_path}/{timestamp}.log"

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set logging level
    logging.getLogger().setLevel(training_args.logging_level.upper())

    train(model_args, data_args, training_args, timestamp)