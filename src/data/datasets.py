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
    Uses `paper_abstract` and `review_contents` to form the prompt,
    and `meta_review` as the label.
    Only samples where dataset attribute 'label' == 'train' will be used.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, split: str = "train", max_samples: Optional[int] = None):
        super().__init__()
        logging.warning("Loading PeerSum dataset...")

        # Load the full dataset for the specified split
        dataset = load_dataset("oaimli/PeerSum", split=split)

        # Filter the dataset to only include samples where label == "train"
        dataset = dataset.filter(lambda x: x.get('label', '') == 'train')

        if max_samples is not None:
            dataset = dataset.select(range(max_samples))

        sources = []
        targets = []

        for example in dataset:
            paper_abstract = example.get("paper_abstract", "")
            review_contents = example.get("review_contents", [])

            # Clean up reviews
            reviews = [r.strip() for r in review_contents if r.strip()]

            if reviews:
                numbered_reviews = [f"Review {i+1}:\n{rev}" for i, rev in enumerate(reviews)]
                joined_reviews = "\n\n".join(numbered_reviews)
            else:
                joined_reviews = "No reviews available."

            prompt = PROMPT_DICT["peersum_prompt"].format(
                paper_abstract=paper_abstract,
                review_contents=joined_reviews
            )

            meta_review = example.get("meta_review", "")
            target_text = f"{meta_review}{tokenizer.eos_token}"

            sources.append(prompt)
            targets.append(target_text)

        # Debug: print a few samples
        # for i in range(min(3, len(sources))):
        #     logging.info(f"Example {i}:")
        #     logging.info(f"Prompt (Instruction): {sources[i]}")
        #     logging.info(f"Label (Meta Review): {targets[i]}")

        logging.warning("Tokenizing PeerSum dataset inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])