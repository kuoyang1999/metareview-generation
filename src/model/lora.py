import torch
from peft import LoraConfig, get_peft_model
from ..utils.token_ops import CastOutputToFloat

def apply_lora_if_needed(model, model_args, training_args):
    """
    Applies LoRA (PEFT) to the given model if training_args.low_rank_training is enabled.
    Allows selectively unfreezing parameters specified in training_args.trainable_params.
    """
    if training_args.low_rank_training:
        # Identify LoRA target modules
        # if model_args.model_type == "gpt-neox":
        #     targets = ["query_key_value", "dense"]
        # else:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Enable trainable params for Embedding and LayerNorm
        for n, p in model.named_parameters():
            if any(k in n for k in training_args.trainable_params.split(",")):
                p.requires_grad = True

    # Cast output to float32 only when using quantization
    if training_args.quantization != "none":
        model.lm_head = CastOutputToFloat(model.lm_head)

    return model