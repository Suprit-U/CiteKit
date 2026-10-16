"""Retrieval and answer-quality metrics."""

from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items retrieved in the top ``k``."""
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def hit_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """1.0 when at least one relevant item appears in the top ``k``."""
    return 1.0 if set(ranked_ids[:k]) & relevant else 0.0


def mrr_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant item, 0 if none in top ``k``."""
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG@k."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(ranked_ids[:k], start=1)
        if item_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return round(dcg / idcg, 4) if idcg else 0.0


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Share of the top ``k`` that is relevant."""
    if k <= 0:
        return 0.0
    hits = sum(1 for item_id in ranked_ids[:k] if item_id in relevant)
    return hits / k


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0