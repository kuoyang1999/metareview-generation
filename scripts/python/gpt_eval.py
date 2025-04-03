import argparse
from src.evaluation.gpt_eval_logic import evaluate

def parse_config():
    parser = argparse.ArgumentParser(description='GPT Evaluation arguments')
    parser.add_argument('--base_model', type=str, default="gpt-4o-mini", help='GPT model to use')
    parser.add_argument('--data_name_or_path', type=str, default="MetaGen", help='Dataset name or path')
    parser.add_argument('--split', type=str, default="test", help='Split to evaluate (e.g., test)')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='Cache directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Max number of samples to evaluate')
    parser.add_argument('--conference', type=str, nargs='+', default=None, help='List of conferences to filter by')
    parser.add_argument('--tag', type=str, default=None, help='Output directory tag')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()
    evaluate(args) 