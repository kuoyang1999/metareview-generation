import torch
import math
import numpy as np
from bert_score import score
from rouge_score import rouge_scorer
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

class MetricsCalculator:
    def __init__(self, cache_dir: Optional[str] = None):
        # Initialize ROUGE scorer with all ROUGE variants we want to compute
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], 
            use_stemmer=True
        )

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Convert to lowercase for better matching
        text = text.lower()
        return text
    
    def compute_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores."""
        try:
            # Preprocess texts
            prediction = self.preprocess_text(prediction)
            reference = self.preprocess_text(reference)
            
            # --- Modification for Maximizing ROUGE Score ---
            # Apply sentence tokenization and join with newlines.
            # This is particularly beneficial for ROUGE-Lsum.
            prediction = "\n".join(sent_tokenize(prediction))
            reference = "\n".join(sent_tokenize(reference))
            
            # Compute all ROUGE variants
            scores = self.rouge_scorer.score(reference, prediction)
            
            # Extract the F1 scores for each ROUGE variant
            results = {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure,
                'rougeLsum': scores['rougeLsum'].fmeasure
            }
            return {k: float(v) for k, v in results.items()}
        except Exception as e:
            print(f"Error computing ROUGE: {str(e)}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    
    def compute_bertscore(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute BERTScore metrics."""
        try:
            # BERTScore handles preprocessing internally
            P, R, F1 = score(
                [prediction], 
                [reference], 
                lang="en",
                model_type="microsoft/deberta-xlarge-mnli",
                batch_size=1,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            return {
                "bertscore_precision": float(P.mean()),
                "bertscore_recall": float(R.mean()),
                "bertscore_f1": float(F1.mean())
            }
        except Exception as e:
            print(f"Error computing BERTScore: {str(e)}")
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0
            }
    
    def compute_all_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute all metrics for a prediction-reference pair."""
        if not prediction or not reference:
            return {
                "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0,
                "bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0
            }
            
        metrics = {}
        
        # ROUGE scores
        metrics.update(self.compute_rouge(prediction, reference))
        
        # BERTScore
        metrics.update(self.compute_bertscore(prediction, reference))
        
        # Ensure all values are native Python types for JSON serialization
        metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                  for k, v in metrics.items()}
        
        return metrics 