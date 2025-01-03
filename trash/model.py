import math
import torch
import transformers
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from .utils import (
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
    smart_tokenizer_and_embedding_resize,
)

def load_model_and_tokenizer(model_args, training_args):
    # Load config
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # RoPE scaling
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}
    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

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

    return model, tokenizer

def apply_lora_if_needed(model, model_args, training_args):
    if training_args.low_rank_training:
        if model_args.model_type == "gpt-neox":
            targets = ["query_key_value", "dense"]
        else:
            targets=["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

        # enable trainable params if specified
        for n, p in model.named_parameters():
            if any(k in n for k in training_args.trainable_params.split(",")):
                p.requires_grad = True

    # Cast output to float32
    class CastOutputToFloat(torch.nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)

    return model