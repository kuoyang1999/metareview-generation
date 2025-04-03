import logging
from typing import Dict
from transformers import PreTrainedTokenizer

from .datasets import SupervisedDataset, PeerSumDataset, MetaGenDataset
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
        eval_dataset = None
        
    elif data_args.data_name_or_path == "MetaGen":
        # Create data filter dictionary from args
        data_filter = {}
        if hasattr(data_args, 'years') and data_args.years:
            data_filter['year'] = data_args.years
        if hasattr(data_args, 'conferences') and data_args.conferences:
            data_filter['conference'] = data_args.conferences
        if hasattr(data_args, 'subjects') and data_args.subjects:
            data_filter['subject'] = data_args.subjects
        data_filter = data_filter if data_filter else None
        
        train_dataset = MetaGenDataset(
            tokenizer=tokenizer, 
            max_samples=data_args.max_samples,
            data_filter=data_filter,
            split="train"
        )
        eval_dataset = None
        
    elif data_args.data_name_or_path == "test":
        train_dataset = PeerSumLongTest(tokenizer=tokenizer, split="train", max_samples=data_args.max_samples)
        eval_dataset = None
        
    else:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_name_or_path=data_args.data_name_or_path, max_samples=data_args.max_samples)
        eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)