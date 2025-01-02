import torch
import transformers
from peft import PeftModel
from src.attn.llama_attn_replace_sft import replace_llama_attn

# Path to your checkpoint directory from the fine-tuning
# This directory should contain adapter_model.safetensors and adapter_config.json
checkpoint_dir = "./checkpoints/sft_final/checkpoint-1000"

# If you have a base model name or path (the same one you used during training)
base_model_name_or_path = "meta-llama/Llama-2-7b-hf"

# Set this to True if you want to use flash attention during inference
use_flash_attn = False

# Replace the LLaMA attention forward function for inference if desired
replace_llama_attn(use_flash_attn=use_flash_attn, inference=True)

# Load the model configuration and tokenizer
config = transformers.AutoConfig.from_pretrained(base_model_name_or_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    base_model_name_or_path,
    padding_side="right",
    use_fast=True,
)

# Ensure PAD, BOS, EOS, and UNK tokens are correctly configured if needed
# This depends on how you handled special tokens during training
# For instance, if needed:
# tokenizer.add_special_tokens({
#     "pad_token": "<pad>",
#     "bos_token": "<s>",
#     "eos_token": "</s>",
#     "unk_token": "<unk>"
# })

# Load the base model in half precision if your GPU supports it.
# You can also load in float16 if you prefer.
model = transformers.AutoModelForCausalLM.from_pretrained(
    base_model_name_or_path,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load the LoRA adapter weights
model = PeftModel.from_pretrained(model, checkpoint_dir, device_map="auto")

# If you want to merge the LoRA weights into the model for inference speed:
# model = model.merge_and_unload()

model.eval()

# Prepare a prompt to test generation
prompt = "Summarize the following reviews about a paper: \n\Abstract: LongLoRA is good. \n\nReviews: \nReview 1: The paper is interesting.\nReview 2: There is room for improvement. \n\nMeta Review:"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate text
with torch.no_grad():
    # Adjust generation settings as needed
    generation_config = transformers.GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        do_sample=True
    )
    outputs = model.generate(**inputs, generation_config=generation_config)

# Decode the output tokens
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:", decoded)