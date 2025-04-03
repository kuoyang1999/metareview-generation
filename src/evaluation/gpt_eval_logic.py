import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import math
from tqdm import tqdm
import json
import os
from datetime import datetime
import numpy as np
import argparse
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

from .output_formatter import format_submission, format_aggregate_metrics, save_evaluation_results
from .metrics import MetricsCalculator
from src.utils import IGNORE_INDEX
from src.data.datasets import PeerSumDataset, MetaGenDataset

def build_gpt_generator(model_name):
    """
    Build a generator function that uses OpenAI's GPT models.
    """
    # Check if we're using a custom API (based on API key format)
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    # Configure the client with appropriate settings
    client = OpenAI(
        api_key=api_key.strip(),  # Ensure no whitespace
        base_url=base_url,
        timeout=60.0,  # Increase timeout to 60 seconds
        max_retries=3  # Add built-in retries
    )

    @retry(
        wait=wait_random_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((APIError, APIConnectionError, RateLimitError, APITimeoutError)),
        reraise=True
    )
    def response(prompt_text):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
                timeout=60,  # Increase timeout to match client timeout
            )
            # Access the returned message
            return completion.choices[0].message.content.strip()
        except (APIError, APIConnectionError, RateLimitError, APITimeoutError) as e:
            print(f"\nOpenAI API Error: {str(e)}")
            print("Retrying...")
            raise  # Retry these specific errors
        except Exception as e:
            print(f"\nUnexpected error during GPT generation: {str(e)}")
            return None  # For any other errors, return None

    return response

def evaluate(args):
    """
    Main evaluation routine for GPT models. 
    1) Sets up OpenAI client
    2) loads dataset
    3) generates predictions
    4) computes metrics
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = "results"  # Base results directory

    # Set default context size for GPT models if not provided
    if not hasattr(args, 'context_size'):
        args.context_size = 16384  # GPT-3.5-turbo's max context length

    # Initialize OpenAI client
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    # Initialize metrics calculator
    metrics_calculator = MetricsCalculator(cache_dir=args.cache_dir)
    
    # Define evaluation tags and their corresponding generators
    eval_tags = {
        args.tag if hasattr(args, 'tag') and args.tag else "base": {
            "temperature": 0,
            "top_p": 0,
            "max_gen_len": 0
        }
    }
    
    generators = {
        tag: build_gpt_generator(
            args.base_model,
        ) for tag, params in eval_tags.items()
    }

    print(f"Evaluating with tags: {list(eval_tags.keys())}")
    
    all_metrics = []
    
    # Preload the full dataset once
    if args.data_name_or_path == "oaimli/PeerSum":
        full_dataset = PeerSumDataset(split=args.split, max_samples=args.max_samples)
    elif args.data_name_or_path == "MetaGen":
        # Load dataset with conference filter
        full_dataset = MetaGenDataset(
            tokenizer=None,  # We don't need tokenizer for GPT evaluation
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
            raw_examples = [full_dataset.raw_examples[i] for i in dataset_indices]
            
            # Create a view of the dataset with filtered examples
            dataset = type('FilteredDataset', (), {
                'raw_examples': raw_examples,
                '__len__': lambda self: len(raw_examples),
                '__getitem__': lambda self, idx: self.raw_examples[idx]
            })()
        else:
            dataset = full_dataset  # For PeerSum, use the full dataset

        submissions = []
        skipped_count = 0
        skipped_lengths = []
        failed_generations = 0
        
        # Aggregate metrics per tag
        metrics_per_tag = {tag: [] for tag in eval_tags.keys()}

        for idx in tqdm(range(len(dataset))):
            example = dataset[idx] if args.data_name_or_path == "MetaGen" else dataset.raw_examples[idx]
            
            # Build prompt text
            if args.data_name_or_path == "MetaGen":
                prompt_text = full_dataset.build_prompt(example)
                reference = full_dataset.build_target(example)
            else:
                # Handle PeerSum dataset
                prompt_text = PeerSumDataset.build_prompt(example)
                reference = example.get("meta_review", "")
            
            # Skip if prompt is too long (rough estimation based on characters)
            if len(prompt_text) > args.context_size * 4:  # Rough estimation of 4 chars per token
                skipped_count += 1
                skipped_lengths.append(len(prompt_text))
                continue
            
            # Extract metadata from dataset if available
            metadata = {}
            if args.data_name_or_path == "MetaGen":
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
                prediction = generator(prompt_text)
                if prediction is None or len(prediction.strip()) == 0:
                    failed_generations += 1
                    print(f"\nFailed to generate prediction for example {idx}")
                    continue
                    
                submission_predictions[tag] = prediction
                
                # Compute all metrics for this prediction
                current_metrics = metrics_calculator.compute_all_metrics(prediction, reference)
                submission_metrics[tag] = current_metrics
                metrics_per_tag[tag].append(current_metrics)
            
            # Only add submission if we have at least one successful prediction
            if submission_predictions:
                # Format and append submission
                submission = format_submission(
                    prompt_text=prompt_text,
                    reference=reference,
                    predictions=submission_predictions,
                    metrics=submission_metrics,
                    prompt_tokens=len(prompt_text.split()),  # Rough estimation
                    reference_tokens=len(reference.split()),  # Rough estimation
                    metadata=metadata
                )
                submissions.append(submission)

        if skipped_lengths:
            print(f"\nSkipped {skipped_count} examples due to length")
            print(f"Skipped prompt lengths: min={min(skipped_lengths)}, max={max(skipped_lengths)}, avg={sum(skipped_lengths)/len(skipped_lengths):.1f}")
        
        if failed_generations:
            print(f"\nFailed to generate predictions for {failed_generations} examples")

        # Compute aggregate metrics only if we have successful generations
        if any(metrics_per_tag.values()):
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
                print(f"Successfully processed {len(submissions)} examples")
            else:
                print(f"\nNo valid submissions for conference: {conf}")
        else:
            print(f"\nNo successful generations for conference: {conf}")
    
    return all_metrics 