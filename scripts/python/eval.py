import argparse
from src.evaluation import evaluate

def parse_config():
    parser = argparse.ArgumentParser(description='Evaluation arguments')
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Base model path')
    parser.add_argument('--data_name_or_path', type=str, default="MetaGen", help='Dataset name or path')
    parser.add_argument('--split', type=str, default="test", help='Split to evaluate (e.g., test)')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='Cache directory')
    parser.add_argument('--context_size', type=int, default=8192 * 4, help='Context size for inference')
    parser.add_argument('--flash_attn', action='store_true', help='Enable flash attention for inference')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--max_gen_len', type=int, default=512, help='Maximum generation length')
    parser.add_argument('--max_samples', type=int, default=None, help='Max number of samples to evaluate')
    parser.add_argument('--conference', type=str, nargs='+', default=None, help='List of conferences to filter by')
    parser.add_argument('--tag', type=str, default=None, help='Output directory tag')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()
    evaluate(args)