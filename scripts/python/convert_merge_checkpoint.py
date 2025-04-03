import os
import torch
import argparse
import transformers
import warnings
from peft import PeftModel
from typing import Dict
import shutil
from pathlib import Path
import sys

# Filter out FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the project root to Python path to import zero_to_fp32
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

from src.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

from src.utils.constants import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, IGNORE_INDEX

from src.utils.token_ops import smart_tokenizer_and_embedding_resize

def parse_config():
    parser = argparse.ArgumentParser(description='Generate HF Checkpoint by Convert DeepSpeed checkpoint and merge LoRA weights')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to checkpoint directory (e.g. checkpoints/EMNLP/20250212_153306/checkpoint-600)')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Path or name of base model (e.g. meta-llama/Llama-2-7b-chat-hf)')
    parser.add_argument('--trainable_params', type=str, default="embed,norm",
                        help='Comma-separated list of trainable parameter names')
    parser.add_argument('--context_size', type=int, default=32768,
                        help='Context size used during training')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Cache directory for downloading models')
    args = parser.parse_args()
    return args

def convert_checkpoint(checkpoint_dir):
    """Convert DeepSpeed checkpoint to PyTorch format"""
    print(f"\nConverting DeepSpeed checkpoint at {checkpoint_dir}")
    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir=checkpoint_dir,
        output_dir=checkpoint_dir,
        max_shard_size=None  # Disable sharding
    )

def extract_trainable_weights(checkpoint_dir, trainable_params):
    """Extract trainable weights and LoRA adapters"""
    print(f"\nExtracting trainable weights with params: {trainable_params}")
    
    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    weights_all = torch.load(model_path)

    # Extract trainable weights
    weights_trainable = {}
    weights_lora = {}
    for k in weights_all:
        if "lora" in k:
            k_new = k.replace("default.", "") if "default." in k else k
            weights_lora[k_new] = weights_all[k]
        else:
            if any([n in k for n in trainable_params.split(",")]):
                weights_trainable[k[17:]] = weights_all[k]

    # Save extracted weights
    torch.save(weights_trainable, os.path.join(checkpoint_dir, "trainable_params.bin"))
    # torch.save(weights_lora, os.path.join(checkpoint_dir, "adapter_model_extracted.bin"))
    
    print(f"\nTrainable weights saved to {os.path.join(checkpoint_dir, 'trainable_params.bin')}")
    # print(f"\nExtracted adapter saved to {os.path.join(checkpoint_dir, 'adapter_model_extracted.bin')}")

def merge_and_save_model(base_model, checkpoint_dir, context_size, cache_dir):
    """Merge LoRA weights and save models"""
    print("\nMerging LoRA weights and saving models")
    
    # Ensure GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. This script requires a GPU to run.")

    # Load base model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        model_max_length=context_size,
        padding_side="right",
        use_fast=False,
    )

    # Add special tokens if needed
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # Load trainable parameters
    trainable_params = os.path.join(checkpoint_dir, "trainable_params.bin")
    if os.path.isfile(trainable_params):
        print(f"\nLoading trainable parameters from {trainable_params}")
        model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)

    # Create merged model with original adapter
    model_orig = PeftModel.from_pretrained(
        model,
        checkpoint_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model_orig = model_orig.merge_and_unload()
    
    # Save merged model with original adapter
    merged_path = os.path.join(os.path.dirname(checkpoint_dir), os.path.basename(checkpoint_dir) + "_merged")
    model_orig.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"\nMerged model saved to {merged_path}")

    # # Rename extracted adapter
    # extracted_adapter = os.path.join(checkpoint_dir, "adapter_model_extracted.bin")
    # if os.path.exists(extracted_adapter):
    #     # Temporarily move original adapter
    #     orig_adapter = os.path.join(checkpoint_dir, "adapter_model.bin")
    #     temp_adapter = os.path.join(checkpoint_dir, "adapter_model.bin.temp")
    #     if os.path.exists(orig_adapter):
    #         os.rename(orig_adapter, temp_adapter)
        
    #     # Move extracted adapter to standard name
    #     print(f"\nMoving extracted adapter to standard name")
    #     os.rename(extracted_adapter, orig_adapter)

    #     # Create merged model with extracted adapter

    #     model_extracted = PeftModel.from_pretrained(
    #         model,
    #         checkpoint_dir,
    #         device_map="auto",
    #         torch_dtype=torch.float16,
    #         local_files_only=True,
    #     )
    #     model_extracted = model_extracted.merge_and_unload()

    #     # Save merged model with extracted adapter
    #     merged_extracted_path = os.path.join(os.path.dirname(checkpoint_dir), 
    #                                        os.path.basename(checkpoint_dir) + "_merged_extracted")
    #     model_extracted.save_pretrained(merged_extracted_path)
    #     tokenizer.save_pretrained(merged_extracted_path)
    #     print(f"\nMerged model with extracted adapter saved to {merged_extracted_path}")
    #     # Restore original adapter
    #     os.rename(orig_adapter, extracted_adapter)
    #     os.rename(temp_adapter, orig_adapter)

def main():
    args = parse_config()
    
    # Convert DeepSpeed checkpoint
    convert_checkpoint(args.checkpoint_dir)
    
    # Extract trainable weights and adapters
    extract_trainable_weights(args.checkpoint_dir, args.trainable_params)
    
    # Merge and save models
    merge_and_save_model(args.base_model, args.checkpoint_dir, args.context_size, args.cache_dir)

if __name__ == "__main__":
    main() 