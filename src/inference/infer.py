import sys
import math
import torch
import transformers
from peft import PeftModel
from transformers import TextStreamer

# Example import path for your own code
from src.model import replace_llama_attn
from src.utils import PROMPT_DICT


def build_generator(model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True):
    """
    Returns a function `response(prompt)` that uses the model to generate text.
    """
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextStreamer(tokenizer)

        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            repetition_penalty=1.1,
            streamer=streamer,
        )
        out = tokenizer.decode(output[0], skip_special_tokens=True)
        out = out.split(prompt.lstrip("<s>"))[1].strip() if prompt.lstrip("<s>") in out else out
        return out

    return response


def run_inference(args):
    """
    Main inference routine.
    1) Replace attention if needed,
    2) Load model/tokenizer,
    3) Possibly set RoPE scaling,
    4) Generate text for a sample prompt (or multiple).
    """
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

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    respond = build_generator(
        model, 
        tokenizer, 
        temperature=args.temperature, 
        top_p=args.top_p,
        max_gen_len=args.max_gen_len,
        use_cache=True
    )

    # Example usage with a peer-sum style prompt
    prompt_no_input = PROMPT_DICT["peersum_prompt"]
    prompt = prompt_no_input.format_map({
        "paper_abstract": (
            "Large Language Models (LLMs) have seen significant "
            "advancements in recent years..."
        ),
        "review_contents": "This paper is good!"
    })

    output = respond(prompt=prompt)
    print("\nGenerated Output:\n", output)