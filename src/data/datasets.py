import logging
from typing import Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset

from ..utils import PROMPT_DICT, IGNORE_INDEX
from .preprocessing import preprocess

class MetaGenDataset(Dataset):
    """
    Dataset for MetaGen submissions. Each submission is expected to have the following structure:

    {
      "title": "string",
      "abstract": "string",
      "venue_id": "string",
      "conference": "string",
      "year": "string",
      "subject": "string",
      "decision": "string",
      "metareview": {
          "content": "string",
          // OR custom fields: "summary", "strengths", "weaknesses"
      },
      "threads": [
        {
          "reviewer": "string",
          "ratings": {
            "rating": "string",
            "confidence": "string",
            "ethics": "string",
            "recommendation": "string"
            // ... other fields
          },
          "conversations": [
            {
              "metadata": {
                "user": "string",
                "timestamp": number
              },
              "title": "string",
              "review": "string",
              "comment": "string"
              // ... other content fields
            }
          ]
        }
      ]
    }

    The prompt is constructed as follows (using LLaMA2-style markers):
    
    [INST] <<SYS>>
    You are an AC for "year" "conference" "subject". Provide a detailed meta-review for the paper titled "title".
    <</SYS>>

    Abstract of the paper:
    "abstract"

    Review Thread 1:
    user: message 1
    user: message 2
    ...
    Ratings: rating fields

    Review Thread 2:
    ...

    Generate a meta review:
    [/INST]
    
    The target text is extracted from the meta-review field.
    """
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer],
        max_samples: Optional[int] = None,
        data_filter: Optional[Dict[str, list]] = None,
        split: Optional[str] = "train",
    ):
        super().__init__()
        logging.warning("Loading MetaGen dataset...")
        
        dataset = load_dataset("kuoyang1999/MetaReviewGen-hf", split=split)
        
        # Apply filters if provided
        if data_filter:
            logging.warning(f"Filtering dataset with criteria: {data_filter}")
            filtered_dataset = []
            for example in dataset:
                # Check if example matches all provided filters
                matches = True
                for field, allowed_values in data_filter.items():
                    if field in ['year', 'conference', 'subject']:
                        example_value = example.get(field, '')
                        # Empty values match any filter criteria
                        if allowed_values and example_value and example_value not in allowed_values:
                            matches = False
                            break
                if matches:
                    filtered_dataset.append(example)
            
            dataset = filtered_dataset
            logging.warning(f"Dataset filtered to {len(dataset)} examples")
     
        if max_samples is not None:
            dataset = dataset[:max_samples]
            
        # Store raw examples for later use
        self.raw_examples = dataset
            
        # Skip tokenization if tokenizer is None (for GPT evaluation)
        if tokenizer is None:
            self.input_ids = None
            self.labels = None
            return
            
        logging.warning("Formatting inputs for MetaGen using custom prompt...")

        sources = []
        targets = []
        for example in dataset:
            prompt = self.build_prompt(example)
            target_text = self.build_target(example, tokenizer)
            sources.append(prompt)
            targets.append(target_text)

        # Debug: Save first example to file
        if logging.getLogger().level <= logging.DEBUG:
            debug_file = f"debug_metagen_{split}.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("="*50 + "\nFirst example source:\n" + "="*50 + "\n")
                f.write(sources[0])
                f.write("\n" + "="*50 + "\nFirst example target:\n" + "="*50 + "\n")
                f.write(targets[0])

        logging.warning("Tokenizing MetaGen dataset inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        # Log token length distribution
        input_lengths = [len(ids) for ids in self.input_ids]
        
        # Calculate statistics
        min_len = min(input_lengths)
        max_len = max(input_lengths)
        mean_len = sum(input_lengths)/len(input_lengths)
        
        # Calculate percentiles
        sorted_lengths = sorted(input_lengths)
        p25 = sorted_lengths[len(sorted_lengths)//4]
        p50 = sorted_lengths[len(sorted_lengths)//2]
        p75 = sorted_lengths[3*len(sorted_lengths)//4]
        p90 = sorted_lengths[9*len(sorted_lengths)//10]
        p95 = sorted_lengths[95*len(sorted_lengths)//100]
        p99 = sorted_lengths[99*len(sorted_lengths)//100]
        
        # Log the statistics
        logging.warning("Token length distribution:")
        logging.warning(f"Min length: {min_len}")
        logging.warning(f"Max length: {max_len}")
        logging.warning(f"Mean length: {mean_len:.2f}")
        logging.warning(f"25th percentile: {p25}")
        logging.warning(f"50th percentile (median): {p50}")
        logging.warning(f"75th percentile: {p75}")
        logging.warning(f"90th percentile: {p90}")
        logging.warning(f"95th percentile: {p95}")
        logging.warning(f"99th percentile: {p99}")
        
        # Log number of examples exceeding model_max_length
        model_max_length = 8192 * 4  # From train.py
        n_exceeding = sum(1 for l in input_lengths if l > model_max_length)
        if n_exceeding > 0:
            logging.warning(f"{n_exceeding} examples ({n_exceeding/len(input_lengths)*100:.2f}%) exceed model_max_length of {model_max_length}")

    def build_prompt(self, example: dict) -> str:
        # Extract basic fields
        title = example.get("title", "")
        abstract = example.get("abstract", "")
        year = example.get("year", "")
        conference = example.get("conference", "")
        subject = example.get("subject", "")

        # Build review threads string
        threads = example.get("threads", [])
        review_threads = ""
        for i, thread in enumerate(threads, start=1):
            reviewer = thread.get("reviewer", "Anonymous")
            review_threads += f"\nReview Thread {i}: {reviewer}\n"
            
            # Process each conversation in the thread
            conversations = thread.get("conversations", [])
            for conv in conversations:
                user = conv.get("metadata", {}).get("user", "Anonymous")
                messages = []
                # Get all fields except metadata
                for key, value in conv.items():
                    if key != "metadata" and value:  # Skip metadata and empty values
                        messages.append(f"{key}: {value}")
                if messages:
                    message_str = "\n".join(messages)
                    review_threads += f"{user}: \n{message_str}\n"
            # Append ratings if available
            ratings = thread.get("ratings", {})
            if ratings:
                ratings_str = "\n".join(f"{k}: {v}" for k, v in ratings.items() if v)
                review_threads += f"\nRatings:\n{ratings_str}\n"

        # Determine instruction based on metareview structure
        metareview_obj = example.get("metareview", {})
        if isinstance(metareview_obj, dict):
            non_empty_fields = {k: v for k, v in metareview_obj.items() if v}
            if len(non_empty_fields) > 1:
                fields_str = ", ".join(non_empty_fields.keys())
                generate_instruction = f"Generate a meta review with the following sections: {fields_str}:"
            else:
                generate_instruction = "Generate a meta review:"
        else:
            generate_instruction = "Generate a meta review:"

        # Use the template from PROMPT_DICT
        return PROMPT_DICT["metagen_prompt"].format(
            year=year,
            conference=conference,
            subject=subject,
            title=title,
            abstract=abstract,
            review_threads=review_threads,
            generate_instruction=generate_instruction
        )

    def build_target(self, example: dict, tokenizer: Optional[PreTrainedTokenizer] = None) -> str:
        metareview_obj = example.get("metareview", {})
        if isinstance(metareview_obj, dict):
            # Get all non-empty fields
            non_empty_fields = {k: v for k, v in metareview_obj.items() if v}
            
            if len(non_empty_fields) == 1:
                # If only one field, use its value directly
                meta_review_text = next(iter(non_empty_fields.values()))
            elif len(non_empty_fields) > 1:
                # If multiple fields, format as "key: value"
                parts = [f"{k}: {v}" for k, v in non_empty_fields.items()]
                meta_review_text = "\n".join(parts)
            else:
                meta_review_text = ""
        else:
            meta_review_text = metareview_obj or ""
            
        # Append the EOS token only if tokenizer is provided
        if tokenizer is not None:
            return f"{meta_review_text}{tokenizer.eos_token}"
        return meta_review_text

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}
    

class SupervisedDataset(Dataset):
    """
    A dataset class for supervised fine-tuning from a local JSON file.
    Assumes data_name_or_path points to a .json or .jsonl file with 'instruction', 'input', 'output'.
    """
    def __init__(self, data_name_or_path: str, tokenizer: PreTrainedTokenizer, max_samples: Optional[int] = None):
        super().__init__()
        logging.warning("Loading local JSON data...")
        
        # Load data via utility function jload
        from ..utils import jload
        list_data_dict = jload(data_name_or_path)
        
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

            prompt = PROMPT_DICT["peersum_prompt"].format(
                paper_abstract=paper_abstract,
                review_contents=joined_reviews
            )

            meta_review = example.get("meta_review", "")
            target_text = f"{meta_review}{tokenizer.eos_token}"

            sources.append(prompt)
            targets.append(target_text)

        logging.warning("Tokenizing PeerSum dataset inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])