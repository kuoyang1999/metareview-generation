import logging
from typing import Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset

from ..utils import PROMPT_DICT, IGNORE_INDEX
from ..data.preprocessing import preprocess


class PeerSumLongTest(Dataset):
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
        dataset = load_dataset("oaimli/PeerSum", split="all")

        # Filter the dataset to only include samples where label == split
        dataset = dataset.filter(lambda x: x.get('label', '') == split)

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

            # Create prompt and repeat 10 times
            prompt = PROMPT_DICT["peersum_prompt"].format(
                paper_abstract=paper_abstract * 10,
                review_contents=joined_reviews * 10
            )

            meta_review = example.get("meta_review", "")
            # Repeat meta review and eos token 10 times
            target_text = f"{meta_review * 10}{tokenizer.eos_token}"

            sources.append(prompt)
            targets.append(target_text)

        # Debug: print a few samples
        for i in range(min(3, len(sources))):
            logging.debug(f"Example {i}:")
            logging.debug(f"Prompt (Instruction): {sources[i]}")
            logging.debug(f"Label (Meta Review): {targets[i]}")

        logging.warning("Tokenizing PeerSum dataset inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])