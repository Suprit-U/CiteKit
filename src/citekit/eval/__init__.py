"""Evaluation harness for CiteKit."""

from .metrics import hit_at_k, mean, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from .runner import Case, EvalReport, load_cases, run

__all__ = [
    "Case",
    "EvalReport",
    "hit_at_k",
    "load_cases",
    "mean",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "run",
]