from typing import Dict, Sequence
import copy
import transformers
import torch
import logging

from ..utils import IGNORE_INDEX

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
    Tokenize a list of strings using the given tokenizer.
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocess the data by tokenizing source and target strings.
    Mask out the source tokens in the labels, leaving only the target tokens.
    """
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    # Log token lengths if debug logging is enabled
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        for i, (input_id, source_len) in enumerate(zip(input_ids, sources_tokenized["input_ids_lens"])):
            total_len = len(input_id)
            target_len = total_len - source_len
            logging.debug(f"Example {i}: Total tokens = {total_len}, Source tokens = {source_len}, Target tokens = {target_len}")

    # Mask out the input portion by setting them to IGNORE_INDEX
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)