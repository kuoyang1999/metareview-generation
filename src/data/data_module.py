import logging
from typing import Dict
from transformers import PreTrainedTokenizer

from .datasets import SupervisedDataset, PeerSumDataset
from .collators import DataCollatorForSupervisedDataset

def make_supervised_data_module(tokenizer: PreTrainedTokenizer, data_args) -> Dict:
    """
    Creates a data module for supervised fine-tuning.
    Loads the specified dataset (local JSON or PeerSum),
    and returns a dictionary containing the train_dataset, eval_dataset, and data_collator.
    """
    logging.warning(f"Loading dataset from path: {data_args.data_path}")
    
    if data_args.data_path == "oaimli/PeerSum":
        train_dataset = PeerSumDataset(tokenizer=tokenizer, split="train", max_samples=data_args.max_samples)
        eval_dataset = None  # Modify if PeerSum has validation/test splits
    else:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, max_samples=data_args.max_samples)
        eval_dataset = None  # Add if you have evaluation data

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)