import math
import torch
import transformers
from transformers import BitsAndBytesConfig

# PEFT is used in lora.py, so no need to import it here unless needed for base logic
# from peft import LoraConfig, get_peft_model

from ..utils.constants import (
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
)
from ..utils.token_ops import smart_tokenizer_and_embedding_resize


def load_model_and_tokenizer(model_args, training_args):
    """
    Loads a model and tokenizer given the user-specified arguments.
    Applies rope scaling if needed, loads model in 4-bit, 
    and resizes tokenizer embeddings for special tokens.
    """

    # 1. Load config
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # 2. Possibly adjust rope scaling for larger context
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}
    orig_rope_scaling_factor = (
        orig_rope_scaling["factor"] if "factor" in orig_rope_scaling else 1
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len is not None:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # 3. Load model with appropriate quantization settings
    load_kwargs = {
        "config": config,
        "cache_dir": training_args.cache_dir,
        "torch_dtype": torch.bfloat16,
    }

    if training_args.quantization == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif training_args.quantization == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **load_kwargs,
    )

    # 4. Freeze the model parameters if using quantization
    if training_args.quantization != "none":
        for param in model.parameters():
            param.requires_grad = False
            # For embedding params of dimension 1, convert to fp32
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

    # 5. Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # 6. Handle missing special tokens
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # 7. Resize the tokenizer and the model's embeddings
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    return model, tokenizer