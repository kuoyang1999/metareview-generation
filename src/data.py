import torch
from torch.utils.data import Dataset
from typing import Dict, Sequence
import logging
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer
from .utils import jload, preprocess, PROMPT_DICT, IGNORE_INDEX

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer):
        super().__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)
        
        # For debugging, only use first 100 examples
        list_data_dict = list_data_dict[:100]

        logging.warning("Formatting inputs...")

        prompt_input = PROMPT_DICT["prompt_input_llama2"]
        prompt_no_input = PROMPT_DICT["prompt_llama2"]
        
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" 
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)