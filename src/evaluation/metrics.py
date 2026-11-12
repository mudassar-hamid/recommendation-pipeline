"""Ranking metrics for top-K recommendation evaluation."""
from __future__ import annotations

from itertools import combinations


def precision_recall_ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> tuple[float, float, float]:
    top_k = recommended[:k]
    hits = [1 if item in relevant else 0 for item in top_k]
    hit_count = sum(hits)
    precision = hit_count / float(k)
    recall = hit_count / float(len(relevant)) if relevant else 0.0

    dcg = sum(hit / __import__("math").log2(rank + 2) for rank, hit in enumerate(hits))
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / __import__("math").log2(rank + 2) for rank in range(ideal_count))
    ndcg = dcg / idcg if idcg else 0.0
    return precision, recall, ndcg


def intra_list_diversity(recommended: list[int], item_genres: dict[int, set[str]]) -> float:
    """Mean pairwise Jaccard distance of genre sets; zero for lists with <2 items."""
    if len(recommended) < 2:
        return 0.0
    distances = []
    for left, right in combinations(recommended, 2):
        a = item_genres.get(int(left), set())
        b = item_genres.get(int(right), set())
        union = a | b
        similarity = (len(a & b) / len(union)) if union else 0.0
        distances.append(1.0 - similarity)
    return float(sum(distances) / len(distances))
