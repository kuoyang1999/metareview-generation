import logging
from typing import Dict
from transformers import PreTrainedTokenizer

from .datasets import SupervisedDataset, PeerSumDataset
from .collators import DataCollatorForSupervisedDataset
from ..test import PeerSumLongTest

def make_supervised_data_module(tokenizer: PreTrainedTokenizer, data_args) -> Dict:
    """
    Creates a data module for supervised fine-tuning.
    Loads the specified dataset (local JSON or PeerSum),
    and returns a dictionary containing the train_dataset, eval_dataset, and data_collator.
    """
    logging.warning(f"Loading dataset from path: {data_args.data_name_or_path}")
    
    if data_args.data_name_or_path == "oaimli/PeerSum":
        train_dataset = PeerSumDataset(tokenizer=tokenizer, split="train", max_samples=data_args.max_samples)
        # eval_dataset = PeerSumDataset(tokenizer=tokenizer, split="val", max_samples=data_args.max_eval_samples)
        eval_dataset = None
    elif data_args.data_name_or_path == "test":
        train_dataset = PeerSumLongTest(tokenizer=tokenizer, split="train", max_samples=data_args.max_samples)
        eval_dataset = None
    else:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_name_or_path=data_args.data_name_or_path, max_samples=data_args.max_samples)
        eval_dataset = None  # Add if you have evaluation data

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)