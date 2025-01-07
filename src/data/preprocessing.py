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
    Ensures targets are preserved in full while sources may be truncated if needed.

    Key behaviors:
    1. Targets are always preserved in their entirety
    2. Sources are truncated if needed to fit within remaining space
    3. Total sequence length never exceeds model's maximum length
    4. Labels are masked for source tokens (set to IGNORE_INDEX)

    Implementation details:
    - First tokenizes targets to determine their exact token lengths
    - Calculates remaining space for each source (model_max_length - target_length - 1)
    - Tokenizes sources with dynamic max_length based on remaining space
    - Combines truncated sources with full targets
    - Creates labels with source portion masked

    Args:
        sources: List of source strings to be preprocessed
        targets: List of target strings to be preprocessed
        tokenizer: Tokenizer to use for encoding texts

    Returns:
        Dict containing:
        - input_ids: Combined and tokenized source+target sequences
        - labels: Same as input_ids but with source tokens masked to IGNORE_INDEX
    """
    # First tokenize targets to know their lengths
    targets_tokenized = _tokenize_fn(targets, tokenizer)
    targets_lens = targets_tokenized["input_ids_lens"]
    
    # Calculate max available length for sources
    max_source_lengths = [
        tokenizer.model_max_length - target_len
        for target_len in targets_lens
    ]
    
    # Tokenize sources with dynamic max_length
    sources_tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_len,
            truncation=True,
        )
        for text, max_len in zip(sources, max_source_lengths)
    ]
    
    # Combine sources and targets
    input_ids = []
    labels = []
    for source_tok, target_tok, source_len, target_len in zip(
        sources_tokenized_list, 
        targets_tokenized["input_ids"], 
        [s.input_ids.ne(tokenizer.pad_token_id).sum().item() for s in sources_tokenized_list],
        targets_lens
    ):
        # Combine input_ids (remove padding from source)
        combined_input_ids = torch.cat([
            source_tok.input_ids[0][:source_len],
            target_tok[:target_len]
        ])
        input_ids.append(combined_input_ids)
        
        # Create labels with source portion masked
        combined_labels = combined_input_ids.clone()
        combined_labels[:source_len] = IGNORE_INDEX
        labels.append(combined_labels)

    # Log token lengths if debug logging is enabled for the first 3 or less examples
    # if logging.getLogger().isEnabledFor(logging.DEBUG):
    #     for i, (input_id, source_len, target_len) in enumerate(zip(input_ids, 
    #         [s.input_ids.ne(tokenizer.pad_token_id).sum().item() for s in sources_tokenized_list],
    #         targets_lens)):
    #         if i >= 3:
    #             break
    #         total_len = len(input_id)
    #         # Decode the tokens back to strings for debugging
    #         source_text = tokenizer.decode(input_id[:source_len])
    #         target_text = tokenizer.decode(input_id[source_len:source_len + target_len])
    #         logging.debug("-" * 80)
    #         logging.debug(f"Decoded example {i}:")
    #         logging.debug(f"  Total tokens = {total_len}, Source tokens = {source_len}, Target tokens = {target_len}")
    #         logging.debug(f"  Source text: {source_text}")
    #         logging.debug(f"  Target text: {target_text}")
    #         logging.debug("-" * 80)

    return dict(input_ids=input_ids, labels=labels)