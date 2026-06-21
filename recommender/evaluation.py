"""Offline top-K ranking metrics for recommender evaluation."""

from __future__ import annotations

import math
from typing import Protocol

import pandas as pd


class Recommender(Protocol):
    catalog: set[int]

    def recommend(self, user_id: int, k: int = 10) -> list[int]: ...

    def recommend_many(
        self,
        user_ids: list[int],
        k: int = 10,
    ) -> dict[int, list[int]]: ...


def recall_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]).intersection(relevant)) / len(relevant)


def ndcg_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, movie_id in enumerate(recommended[:k])
        if movie_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_top_k(
    model: Recommender,
    test_interactions: pd.DataFrame,
    k: int = 10,
) -> dict[str, float | int]:
    """Evaluate recommendations against held-out positive interactions."""
    if k <= 0:
        raise ValueError("k must be positive")
    if test_interactions.empty:
        return {
            "users": 0,
            "k": k,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "catalog_coverage": 0.0,
        }

    relevant_by_user = {
        int(user_id): set(group["movie_id"].astype(int))
        for user_id, group in test_interactions.groupby("user_id")
    }
    recalls: list[float] = []
    ndcgs: list[float] = []
    recommended_items: set[int] = set()
    recommendations = model.recommend_many(list(relevant_by_user), k)

    for user_id, relevant in relevant_by_user.items():
        recommended = recommendations.get(user_id, [])
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        recommended_items.update(recommended)

    catalog_size = len(model.catalog)
    return {
        "users": len(relevant_by_user),
        "k": k,
        "recall_at_k": sum(recalls) / len(recalls),
        "ndcg_at_k": sum(ndcgs) / len(ndcgs),
        "catalog_coverage": (
            len(recommended_items) / catalog_size if catalog_size else 0.0
        ),
    }
