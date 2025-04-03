import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import math
from tqdm import tqdm
import json
import os
from datetime import datetime
import transformers
import numpy as np
import argparse

from .output_formatter import format_submission, format_aggregate_metrics, save_evaluation_results
from .metrics import MetricsCalculator
from src.model import replace_llama_attn
from src.utils import (
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
    IGNORE_INDEX,
)
from src.utils.token_ops import smart_tokenizer_and_embedding_resize
from src.data.datasets import PeerSumDataset, MetaGenDataset

def build_generator(model, tokenizer, temperature, top_p, max_gen_len):
    """
    Build a simple generator function that takes tokenized prompt IDs and returns the generated text.
    """
    def response(prompt_ids):
        try:
            prompt_ids = prompt_ids.unsqueeze(0).to(model.device)  # Shape: (1, prompt_length)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                    # do_sample=False,
                )
            
            # The output_ids is already the full sequence tensor
            generated_ids = output_ids[0][prompt_ids.shape[1]:]  # Remove prompt tokens
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return None
    
    return response


def evaluate(args):
    """
    Main evaluation routine. 
    1) Replaces attention if needed, 
    2) loads model/tokenizer, 
    3) loads dataset,
    4) generates predictions, 
    5) computes metrics.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = "results"  # Base results directory

    # Replace attn
    if args.flash_attn:
        print("Replacing attention with flash attention")
        replace_llama_attn(inference=True)

    # Load config
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    # Possibly adjust RoPE scaling and max position embeddings
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        print(f"Setting RoPE scaling factor to {config.rope_scaling} for context length {args.context_size}")

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Check if token embeddings size is 32001
    if model.get_input_embeddings().weight.shape[0] != 32001:
        model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        print("Adding pad token")
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model.eval()
    
    # Initialize metrics calculator
    metrics_calculator = MetricsCalculator(cache_dir=args.cache_dir)
    
    # Define evaluation tags and their corresponding generators
    eval_tags = {
        args.tag if hasattr(args, 'tag') and args.tag else "base": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_gen_len": args.max_gen_len
        }
    }
    
    generators = {
        tag: build_generator(
            model, 
            tokenizer, 
            params["temperature"], 
            params["top_p"],
            params["max_gen_len"]
        ) for tag, params in eval_tags.items()
    }

    print(f"Evaluating with tags: {list(eval_tags.keys())}")
    
    all_metrics = []
    
    # Preload the full dataset once
    if args.data_name_or_path == "oaimli/PeerSum":
        full_dataset = PeerSumDataset(tokenizer=tokenizer, split=args.split, max_samples=args.max_samples)
    elif args.data_name_or_path == "MetaGen":
        # Load dataset with conference filter
        full_dataset = MetaGenDataset(
            tokenizer=tokenizer, 
            split=args.split, 
            max_samples=args.max_samples, 
            data_filter={"conference": args.conference}  # Pass all conferences at once
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.data_name_or_path}")
    
    # Process each conference separately
    for conf in args.conference:
        print(f"\nProcessing conference: {conf}")
        
        # Filter dataset for current conference
        if args.data_name_or_path == "MetaGen":
            # Filter the dataset for the current conference
            dataset_indices = [
                i for i, example in enumerate(full_dataset.raw_examples)
                if example.get("conference", "") == conf
            ]
            
            if not dataset_indices:
                print(f"\nNo examples found for conference: {conf}")
                continue
                
            # Create filtered dataset views
            input_ids = [full_dataset.input_ids[i] for i in dataset_indices]
            labels = [full_dataset.labels[i] for i in dataset_indices]
            raw_examples = [full_dataset.raw_examples[i] for i in dataset_indices]
            
            # Create a view of the dataset with filtered examples
            dataset = type('FilteredDataset', (), {
                'input_ids': input_ids,
                'labels': labels,
                'raw_examples': raw_examples,
                '__len__': lambda self: len(input_ids),
                '__getitem__': lambda self, idx: {
                    "input_ids": self.input_ids[idx],
                    "labels": self.labels[idx]
                }
            })()
        else:
            dataset = full_dataset  # For PeerSum, use the full dataset

        submissions = []
        skipped_count = 0
        skipped_lengths = []
        
        # Aggregate metrics per tag
        metrics_per_tag = {tag: [] for tag in eval_tags.keys()}

        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            input_ids = sample["input_ids"]
            labels = sample["labels"]

            # Determine source/target boundary
            target_start = next((i for i, label in enumerate(labels) if label != IGNORE_INDEX), len(labels))
            prompt_ids = input_ids[:target_start]
            target_ids = input_ids[target_start:]
            
            # Get prompt and reference text
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            reference = tokenizer.decode(target_ids, skip_special_tokens=True)
            
            # Skip if prompt is too long
            if len(prompt_ids) > args.context_size:
                skipped_count += 1
                skipped_lengths.append(len(prompt_ids))
                continue
            
            # If input length equals context length, replace the end of prompt with "Generate a meta-review:"
            if len(input_ids) == args.context_size:
                suffix = ".\n\nGenerate a meta-review:"
                # Encode and decode to ensure we get the exact token length of the suffix
                suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
                # Replace the end of prompt_ids with suffix_ids
                prompt_ids = torch.cat([prompt_ids[:-len(suffix_ids)], torch.tensor(suffix_ids)])
                # Update prompt text
                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            
            # Extract metadata from dataset if available
            metadata = {}
            if args.data_name_or_path == "MetaGen":
                # Get the original example from the raw examples list
                if hasattr(dataset, 'raw_examples'):
                    example = dataset.raw_examples[idx]
                    metadata = {
                        "title": example.get("title", ""),
                        "conference": example.get("conference", ""),
                        "year": example.get("year", ""),
                        "subject": example.get("subject", ""),
                        "venue_id": example.get("venue_id", "")
                    }
            
            # Generate predictions and compute metrics for each tag
            submission_predictions = {}
            submission_metrics = {}
            
            for tag, generator in generators.items():
                # Generate prediction
                prediction = generator(prompt_ids)
                if prediction is None or len(prediction.strip()) == 0:
                    continue
                    
                submission_predictions[tag] = prediction
                
                # Compute all metrics for this prediction
                current_metrics = metrics_calculator.compute_all_metrics(prediction, reference)
                submission_metrics[tag] = current_metrics
                metrics_per_tag[tag].append(current_metrics)
            
            # Format and append submission
            submission = format_submission(
                prompt_text=prompt_text,
                reference=reference,
                predictions=submission_predictions,
                metrics=submission_metrics,
                prompt_tokens=len(prompt_ids),
                reference_tokens=len(target_ids),
                metadata=metadata
            )
            submissions.append(submission)

        if skipped_lengths:
            print(f"Skipped prompt lengths: min={min(skipped_lengths)}, max={max(skipped_lengths)}, avg={sum(skipped_lengths)/len(skipped_lengths):.1f}")

        # Compute aggregate metrics
        metrics = format_aggregate_metrics(metrics_per_tag)
        all_metrics.extend(metrics)

        if submissions:  # Only save if we have submissions
            # Create a copy of args with only the current conference
            current_args = argparse.Namespace(**vars(args))
            current_args.conference = conf  # Set to current conference string instead of list
            
            # Save results for this conference
            output_file = save_evaluation_results(
                args=current_args,
                metrics=metrics,
                submissions=submissions,
                results_dir=results_dir,
                timestamp=timestamp,
                conference=conf,
                tag=args.tag
            )
            
            print(f"\nResults for {conf} saved to: {output_file}")
        else:
            print(f"\nNo valid submissions for conference: {conf}")
    
    return all_metrics