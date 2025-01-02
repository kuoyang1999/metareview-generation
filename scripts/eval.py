import os
import torch
import argparse
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from src.attn.llama_attn_replace_sft import replace_llama_attn
from src.utils import (
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
)
from src.data.datasets import PeerSumDataset

def parse_config():
    parser = argparse.ArgumentParser(description='Evaluation arguments')
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-hf", help='Base model path')
    parser.add_argument('--data_name_or_path', type=str, default="oaimli/PeerSum", help='Dataset name from HF')
    parser.add_argument('--split', type=str, default="test", help='Split to evaluate (e.g., test)')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='Cache directory')
    parser.add_argument('--context_size', type=int, default=-1, help='Context size for inference')
    parser.add_argument('--flash_attn', type=bool, default=False, help='Enable flash attention for inference')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--max_gen_len', type=int, default=512, help='Maximum generation length')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to evaluate')
    return parser.parse_args()

def build_generator(model, tokenizer, temperature, top_p, max_gen_len):
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def evaluate(args):
    
    replace_llama_attn(inference=True， use_flash_attn=False)

    # Load model and tokenizer
    print(f"Loading base model from {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
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
    model = PeftModel.from_pretrained(model, args.peft_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    
     # Set pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # or use tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model.eval()
    generator = build_generator(model, tokenizer, args.temperature, args.top_p, args.max_gen_len)
    
    # Load dataset
    print(f"Loading dataset {args.data_name_or_path} (split: {args.split})...")
    if args.data_name_or_path == "oaimli/PeerSum":
        dataset = PeerSumDataset(tokenizer=tokenizer, split=args.split, max_samples=args.max_samples)
    else:
        raise ValueError(f"Unsupported dataset: {args.data_name_or_path}")

    # Metrics
    rouge = load_metric("rouge")

    print("Evaluating...")
    predictions = []
    references = []

    for sample in tqdm(dataset):
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Decode prompt and target for comparison
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        valid_label_ids = [token_id for token_id in labels if token_id >= 0]
        reference = tokenizer.decode(valid_label_ids, skip_special_tokens=True)

        # Generate response
        generated = generator(prompt)

        # Save predictions and references for metrics
        predictions.append(generated)
        references.append(reference)

    # Compute metrics
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    for key, value in results.items():
        print(f"{key}: {value.mid}")

if __name__ == "__main__":
    args = parse_config()
    evaluate(args)