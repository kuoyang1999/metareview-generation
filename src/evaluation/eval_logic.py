import torch
import math
from tqdm import tqdm
import json
import os
from datetime import datetime
import transformers
import evaluate as hf_evaluate

# Example import path for your own code
from src.model import replace_llama_attn
from src.utils import (
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
)
from src.data.datasets import PeerSumDataset

def build_generator(model, tokenizer, temperature, top_p, max_gen_len):
    """
    Build a simple generator function that takes a prompt and returns the generated text.
    """
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            # repetition_penalty=1.1,
        )
        out = tokenizer.decode(output[0], skip_special_tokens=True)
        out = out.split(prompt.lstrip("<s>"))[1].strip() if prompt.lstrip("<s>") in out else out
        return out
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
    # Create results directory if it doesn't exist
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Replace attn
    replace_llama_attn(inference=True, use_flash_attn=args.flash_attn)

    # Load config
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    # Possibly adjust RoPE scaling
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        print(f"Setting RoPE scaling factor to {scaling_factor} for context length {args.context_size}")

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Example for resizing: e.g. if you have added tokens up to ID 32000
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )
    
    # # Set pad token if not already set
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    generator = build_generator(
        model, 
        tokenizer, 
        temperature=args.temperature, 
        top_p=args.top_p,
        max_gen_len=args.max_gen_len,
    )
    
    # Load dataset
    print(f"Loading dataset {args.data_name_or_path} (split: {args.split})...")
    if args.data_name_or_path == "oaimli/PeerSum":
        dataset = PeerSumDataset(tokenizer=tokenizer, split=args.split, max_samples=args.max_samples)
    else:
        raise ValueError(f"Unsupported dataset: {args.data_name_or_path}")

    # Metrics
    rouge = hf_evaluate.load("rouge")

    print("Evaluating...")
    predictions = []
    references = []
    all_outputs = []  # Store all inputs, references, and predictions

    for idx, sample in enumerate(tqdm(dataset)):
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Decode prompt and target
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        valid_label_ids = [token_id for token_id in labels if token_id >= 0]
        reference = tokenizer.decode(valid_label_ids, skip_special_tokens=True)

        generated = generator(prompt=prompt)
        predictions.append(generated)
        references.append(reference)
        
        # Store detailed output
        all_outputs.append({
            "id": idx,
            "input": prompt,
            "reference": reference,
            "prediction": generated
        })

    # Compute metrics
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    
    # Calculate and print token lengths
    input_lengths = []
    for output in all_outputs:
        input_tokens = tokenizer(output["input"], return_tensors="pt")
        input_lengths.append(len(input_tokens.input_ids[0]))
    
    print("\nToken Length Statistics:")
    print(f"Average input length: {sum(input_lengths) / len(input_lengths):.1f} tokens")
    print(f"Max input length: {max(input_lengths)} tokens")
    print(f"Min input length: {min(input_lengths)} tokens")
    
    # Print results
    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    # Save all results
    output_file = os.path.join(results_dir, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "args": vars(args),
            "metrics": results,
            "outputs": all_outputs,
            "token_lengths": {
                "input_lengths": input_lengths,
                "average": sum(input_lengths) / len(input_lengths),
                "max": max(input_lengths),
                "min": min(input_lengths)
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return results