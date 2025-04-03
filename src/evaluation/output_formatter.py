import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

def format_submission(
    prompt_text: str,
    reference: str,
    predictions: Dict[str, str],
    metrics: Dict[str, Dict[str, float]],
    prompt_tokens: int,
    reference_tokens: int,
    metadata: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Format a single submission entry.
    
    Args:
        prompt_text: The input prompt text
        reference: The reference/target text
        predictions: Dictionary mapping tags to their predictions
        metrics: Dictionary mapping tags to their metric scores
        prompt_tokens: Number of tokens in the prompt
        reference_tokens: Number of tokens in the reference
        metadata: Optional metadata about the submission
    
    Returns:
        Formatted submission dictionary
    """
    return {
        "metadata": metadata or {},
        "prompt": prompt_text,
        "reference": reference,
        "predictions": predictions,
        "metrics": metrics,
        "tokens": {
            "prompt": prompt_tokens,
            "reference": reference_tokens,
            "total": prompt_tokens + reference_tokens
        }
    }

def format_aggregate_metrics(tag_metrics: Dict[str, List[Dict[str, float]]]) -> List[Dict[str, Dict[str, float]]]:
    """
    Compute and format aggregate metrics for each tag.
    
    Args:
        tag_metrics: Dictionary mapping tags to lists of metric dictionaries
    
    Returns:
        List of dictionaries containing averaged metrics per tag
    """
    metrics = []
    for tag, metric_list in tag_metrics.items():
        if metric_list:
            # Average metrics across all submissions for this tag
            avg_metrics = {
                metric: sum(m[metric] for m in metric_list) / len(metric_list)
                for metric in metric_list[0].keys()
            }
            metrics.append({tag: avg_metrics})
    return metrics

def save_evaluation_results(
    args: Any,
    metrics: List[Dict[str, Dict[str, float]]],
    submissions: List[Dict[str, Any]],
    results_dir: str,
    conference: str,
    timestamp: Optional[str] = None,
    tag: Optional[str] = None
) -> str:
    """
    Save evaluation results to a JSON file.
    
    Args:
        args: Command line arguments
        metrics: List of aggregate metrics per tag
        submissions: List of submission entries
        results_dir: Base directory for results
        conference: Conference name as string
        timestamp: Optional timestamp for the results directory
        tag: Optional tag for the results directory
    Returns:
        Path to the saved results file
    """
    # Create results directory if it doesn't exist
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory path with conference and tag
    if tag:
        results_dir = os.path.join(results_dir, conference, tag)
    else:
        results_dir = os.path.join(results_dir, conference)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results with timestamp as filename
    output_file = os.path.join(results_dir, f"{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump({
            "args": vars(args),
            "metrics": metrics,
            "submissions": submissions
        }, f, indent=2)
    
    return output_file 