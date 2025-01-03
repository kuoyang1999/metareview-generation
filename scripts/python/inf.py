import argparse
from src.inference import run_inference

def parse_config():
    parser = argparse.ArgumentParser(description='Inference arguments')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='Context size')
    parser.add_argument('--flash_attn', action='store_true', help='Enable flash attention')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='Max generation length')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()
    run_inference(args)