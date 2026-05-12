from __future__ import annotations

import math


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for d in top_k if d in relevant) / k


def dcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    score = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        rel = relevance.get(doc_id, 0)
        if rel > 0:
            score += rel / math.log2(rank + 1)
    return score


def ndcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    """DCG over the ideal DCG. Relevance grades must be non-negative ints."""
    actual = dcg_at_k(retrieved, relevance, k)
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum(
        rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_grades, start=1) if rel > 0
    )
    return actual / ideal if ideal > 0 else 0.0


def mean_reciprocal_rank(per_query: list[float]) -> float:
    return sum(per_query) / len(per_query) if per_query else 0.0
