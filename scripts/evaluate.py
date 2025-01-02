import os
import sys
import math
import json
import torch
import argparse
from datetime import datetime
from datasets import load_dataset, load_metric
from tqdm import tqdm
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TextStreamer
)
from peft import PeftModel

from src.attn.llama_attn_replace_sft import replace_llama_attn
from src.utils import (
    PROMPT_DICT,
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
)
from src.data.datasets import PeerSumDataset

def parse_config():
    parser = argparse.ArgumentParser(description='Evaluation arguments')
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-hf", help='Base model path')
    parser.add_argument('--peft_model', type=str, default=None, help='Path to PEFT model')
    parser.add_argument('--data_name_or_path', type=str, default="oaimli/PeerSum", help='Dataset name or path')
    parser.add_argument('--split', type=str, default="test", help='Split to evaluate (e.g., test)')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='Cache directory')
    parser.add_argument('--context_size', type=int, default=-1, help='Context size for inference')
    parser.add_argument('--flash_attn', type=bool, default=False, help='Enable flash attention for inference')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--max_gen_len', type=int, default=512, help='Maximum generation length')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to evaluate')
    parser.add_argument('--save_dir', type=str, default="./results", help='Directory to save the evaluation results')
    return parser.parse_args()

def build_generator(model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=512, use_cache=True):
    # This generator function is adapted from inference.py
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # streamer = TextStreamer(tokenizer)
        
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=1,
            top_p=top_p,
            use_cache=use_cache,
            repetition_penalty=1.1,
            # streamer=streamer,
        )
        
        # Decode and remove the prompt portion
        out = tokenizer.decode(output[0], skip_special_tokens=True)
        # Attempt to strip prompt if present
        prompt_stripped = prompt.lstrip("<s>")
        if prompt_stripped in out:
            out = out.split(prompt_stripped, 1)[-1].strip()
        return out

    return response

def evaluate(args):

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Derive the model name for saving results
    model_name = os.path.basename(args.base_model)
    if not model_name:  # If base_model ends with a slash
        model_name = args.base_model.strip("/").split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f"{model_name}_{timestamp}_results.json")

    # Set up llama attn as per inference.py
    replace_llama_attn(inference=True, use_flash_attn=False)

    # Load and possibly modify config for RoPE scaling if context_size > orig_ctx_len
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    print(f"Loading base model from {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    )

    # If a PEFT model is provided, load it
    if args.peft_model is not None:
        model = PeftModel.from_pretrained(model, args.peft_model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    # Set pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    # Use torch.compile if available (as in inference.py)
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Build the response generator (from inference.py)
    respond = build_generator(
        model, tokenizer, 
        temperature=args.temperature, 
        top_p=args.top_p,
        max_gen_len=args.max_gen_len, 
        use_cache=True
    )

    # Load dataset
    print(f"Loading dataset {args.data_name_or_path} (split: {args.split})...")
    if args.data_name_or_path == "oaimli/PeerSum":
        dataset = PeerSumDataset(tokenizer=tokenizer, split=args.split, max_samples=20)
    else:
        raise ValueError(f"Unsupported dataset: {args.data_name_or_path}")

    # Load metric
    rouge = load_metric("rouge")

    print("Evaluating...")
    predictions = []
    references = []
    results_data = []  # To store per-sample results

    for sample in tqdm(dataset):
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Decode prompt and reference
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        valid_label_ids = [token_id for token_id in labels if token_id >= 0]
        reference = tokenizer.decode(valid_label_ids, skip_special_tokens=True)

        # Generate response
        generated = respond(prompt)

        # Compute per-sample ROUGE
        sample_rouge = rouge.compute(predictions=[generated], references=[reference], use_stemmer=True)
        rouge1 = sample_rouge["rouge1"].mid.fmeasure
        rouge2 = sample_rouge["rouge2"].mid.fmeasure
        rougeL = sample_rouge["rougeL"].mid.fmeasure

        # Save predictions and references for final aggregated metrics
        predictions.append(generated)
        references.append(reference)

        # Record sample results
        results_data.append({
            "prompt": prompt,
            "reference": reference,
            "generated": generated,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL
        })

    # Compute aggregated metrics
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    aggregated = {
        "rouge1": results["rouge1"].mid.fmeasure,
        "rouge2": results["rouge2"].mid.fmeasure,
        "rougeL": results["rougeL"].mid.fmeasure,
    }

    # Print aggregated results
    for key, value in aggregated.items():
        print(f"{key}: {value}")

    # Add aggregated metrics to the final output
    output_dict = {
        "model": args.base_model,
        "split": args.split,
        "samples": results_data,
        "aggregated": aggregated
    }

    # Save to JSON file
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    args = parse_config()
    evaluate(args)