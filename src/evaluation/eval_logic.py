import torch
from tqdm import tqdm
from datasets import load_metric
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Example import path for your own code
from src.model import replace_llama_attn
from src.utils import (
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
)
from src.data.datasets import PeerSumDataset

# TODO: Use zero_to_fp32.py to concatenate all the checkpoints

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
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)
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
    # Replace attn
    replace_llama_attn(inference=True, use_flash_attn=False)

    # Load base model
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
    # Load LoRA/PEFT weights on top
    model = PeftModel.from_pretrained(model, args.peft_model)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

        # Decode prompt and target
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        valid_label_ids = [token_id for token_id in labels if token_id >= 0]
        reference = tokenizer.decode(valid_label_ids, skip_special_tokens=True)

        generated = generator(prompt)
        predictions.append(generated)
        references.append(reference)

    # Compute metrics
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    for key, value in results.items():
        print(f"{key}: {value.mid}")