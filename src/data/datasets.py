import logging
from typing import Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset

from ..utils import PROMPT_DICT, IGNORE_INDEX
from .preprocessing import preprocess

class SupervisedDataset(Dataset):
    """
    A dataset class for supervised fine-tuning from a local JSON file.
    Assumes data_path points to a .json or .jsonl file with 'instruction', 'input', 'output'.
    """
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_samples: Optional[int] = None):
        super().__init__()
        logging.warning("Loading local JSON data...")
        
        # Load data via utility function jload
        from ..utils import jload
        list_data_dict = jload(data_path)
        
        if max_samples is not None:
            list_data_dict = list_data_dict[:max_samples]

        logging.warning("Formatting inputs for SupervisedDataset...")

        prompt_input = PROMPT_DICT["prompt_input_llama2"]
        prompt_no_input = PROMPT_DICT["prompt_no_input_llama2"]
        
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing SupervisedDataset inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class PeerSumDataset(Dataset):
    """
    Dataset for PeerSum from Hugging Face: oaimli/PeerSum.
    Uses `review_contents` as the instruction and `meta_review` as the label.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, split: str = "train", max_samples: Optional[int] = None):
        super().__init__()
        logging.warning("Loading PeerSum dataset from Hugging Face...")
        
        dataset = load_dataset("oaimli/PeerSum", split=split)
        
        if max_samples is not None:
            dataset = dataset.select(range(max_samples))
        
        prompt_no_input = PROMPT_DICT["prompt_no_input_llama2"]
        
        sources = [
            prompt_no_input.format_map({"instruction": example["review_contents"]})
            for example in dataset
        ]
        targets = [f"{example['meta_review']}{tokenizer.eos_token}" for example in dataset]

        logging.warning("Tokenizing PeerSum dataset inputs... This may take some time...")
        
        logging.warning(f"First {min(3, len(sources))} examples:")
        for i in range(min(3, len(sources))):
            logging.warning(f"Example {i}:")
            logging.warning(f"Prompt: {sources[i]}")
            logging.warning(f"Label: {targets[i]}")
        
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])