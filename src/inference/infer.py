import sys
import math
import torch
import transformers
import logging
import json
import os
import evaluate as hf_evaluate
from datetime import datetime
from tqdm import tqdm
from transformers import BitsAndBytesConfig

from src.data.datasets import PeerSumDataset
from src.model import replace_llama_attn
from src.utils import PROMPT_DICT, IGNORE_INDEX
from src.utils.constants import (
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
)
from src.utils.token_ops import smart_tokenizer_and_embedding_resize
from src.utils.io_utils import read_quantization_config


def run_inference(args):
    """
    Main inference routine.
    1) Replace attention if needed,
    2) Load model/tokenizer with proper quantization,
    3) Possibly set RoPE scaling,
    4) Generate text for a sample prompt (or multiple).
    """
    # Create results directory if it doesn't exist
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Load metrics
    rouge = hf_evaluate.load("rouge")
    bleu = hf_evaluate.load("bleu")
    meteor = hf_evaluate.load("meteor")
    bertscore = hf_evaluate.load("bertscore")

    replace_llama_attn(inference=True, use_flash_attn=True)

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
        print(f"RoPE scaling factor: {scaling_factor} for context size {args.context_size} and original context size {orig_ctx_len}")
    # Read quantization config from checkpoint if it exists
    quant_config = read_quantization_config(args.base_model)
    load_kwargs = {
        "config": config,
        "cache_dir": args.cache_dir,
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda",
    }

    if quant_config:
        print("Using quantization config from checkpoint:")
        for k, v in quant_config.items():
            print(f"  {k}: {v}")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quant_config.get('load_in_4bit', False),
            load_in_8bit=quant_config.get('load_in_8bit', False),
            llm_int8_threshold=quant_config.get('llm_int8_threshold', 6.0),
            llm_int8_has_fp16_weight=quant_config.get('llm_int8_has_fp16_weight', False),
            bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
            bnb_4bit_compute_dtype=torch.bfloat16 if quant_config.get('bnb_4bit_compute_dtype') == 'bfloat16' else torch.float16,
        )

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **load_kwargs,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    # Handle special tokens
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # Resize tokenizer and model embeddings
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    model.eval()
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Move model to GPU if not already done by device_map
    if model.device.type != "cuda":
        model = model.cuda()

    dataset = PeerSumDataset(tokenizer=tokenizer, split="test", max_samples=3)
    
    all_outputs = []  # Store all inputs and predictions
    input_lengths = []

    for idx, sample in enumerate(tqdm(dataset)):
            input_ids = sample["input_ids"]
            labels = sample["labels"]

            # Find where the actual input ends (where IGNORE_INDEX stops)
            input_mask = (labels == IGNORE_INDEX)
            input_end_idx = input_mask.sum()
            
            # Get only the input portion for the prompt
            prompt_ids = input_ids[:input_end_idx]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            
            # Get reference from non-ignored labels
            valid_label_ids = [token_id for token_id in labels[input_end_idx:] if token_id >= 0]
            reference = tokenizer.decode(valid_label_ids, skip_special_tokens=True)

            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_gen_len,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    num_beams=1,
                    # repetition_penalty=1.1,
                )
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            if prompt in generated:
                generated = generated[len(prompt):].strip()
            
            # Get token length
            input_lengths.append(len(prompt_ids))

            # Store detailed output
            all_outputs.append({
                "id": idx,
                # "input": prompt,
                "reference": reference,
                "prediction": generated,
                # "full_generation": full_generated  # Store full generation for debugging
            })

    # Compute metrics
    predictions = [output["prediction"] for output in all_outputs]
    references = [output["reference"] for output in all_outputs]
    
    # ROUGE scores
    rouge_results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    
    # BLEU score
    bleu_results = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    
    # METEOR score
    meteor_results = meteor.compute(predictions=predictions, references=references)
    
    # BERTScore
    bertscore_results = bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli", batch_size=8)
    
    # Aggregate all metrics
    metrics_results = {
        "rouge": rouge_results,
        "bleu": bleu_results,
        "meteor": meteor_results,
        "bertscore": {
            "precision": sum(bertscore_results["precision"]) / len(bertscore_results["precision"]),
            "recall": sum(bertscore_results["recall"]) / len(bertscore_results["recall"]),
            "f1": sum(bertscore_results["f1"]) / len(bertscore_results["f1"])
        }
    }

    # Print all metrics
    print("\nEvaluation Metrics:")
    print("\nROUGE Scores:")
    for key, value in rouge_results.items():
        print(f"{key}: {value:.4f}")
    
    print("\nBLEU Score:")
    print(f"BLEU: {bleu_results['bleu']:.4f}")
    
    print("\nMETEOR Score:")
    print(f"METEOR: {meteor_results['meteor']:.4f}")
    
    print("\nBERTScore:")
    print(f"Precision: {metrics_results['bertscore']['precision']:.4f}")
    print(f"Recall: {metrics_results['bertscore']['recall']:.4f}")
    print(f"F1: {metrics_results['bertscore']['f1']:.4f}")

    # Print token length statistics
    print("\nToken Length Statistics:")
    print(f"Average input length: {sum(input_lengths) / len(input_lengths):.1f} tokens")
    print(f"Max input length: {max(input_lengths)} tokens")
    print(f"Min input length: {min(input_lengths)} tokens")

    # Save all results
    output_file = os.path.join(results_dir, "inference_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "args": vars(args),
            "metrics": metrics_results,
            "outputs": all_outputs,
            "token_lengths": {
                "input_lengths": input_lengths,
                "average": sum(input_lengths) / len(input_lengths),
                "max": max(input_lengths),
                "min": min(input_lengths)
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return all_outputs