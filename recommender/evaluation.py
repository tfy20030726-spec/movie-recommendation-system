"""Offline top-K ranking metrics for recommender evaluation."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
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

    recommendations = model.recommend_many(
        test_interactions["user_id"].astype(int).unique().tolist(),
        k,
    )
    return evaluate_recommendation_lists(
        recommendations,
        test_interactions,
        model.catalog,
        k,
    )


def evaluate_recommendation_lists(
    recommendations: dict[int, list[int]],
    test_interactions: pd.DataFrame,
    catalog: set[int],
    k: int = 10,
) -> dict[str, float | int]:
    """Evaluate precomputed recommendation lists against held-out positives."""
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
    for user_id, relevant in relevant_by_user.items():
        recommended = recommendations.get(user_id, [])
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        recommended_items.update(recommended)

    catalog_size = len(catalog)
    return {
        "users": len(relevant_by_user),
        "k": k,
        "recall_at_k": sum(recalls) / len(recalls),
        "ndcg_at_k": sum(ndcgs) / len(ndcgs),
        "catalog_coverage": (
            len(recommended_items) / catalog_size if catalog_size else 0.0
        ),
    }


def paired_bootstrap_metric_differences(
    baseline_recommendations: dict[int, list[int]],
    candidate_recommendations: dict[int, list[int]],
    test_interactions: pd.DataFrame,
    k: int = 10,
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, object]:
    """Estimate user-level confidence intervals for candidate minus baseline."""
    if k <= 0:
        raise ValueError("k must be positive")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if test_interactions.empty:
        raise ValueError("test_interactions cannot be empty")

    relevant_by_user = {
        int(user_id): set(group["movie_id"].astype(int))
        for user_id, group in test_interactions.groupby("user_id", sort=True)
    }
    user_ids = sorted(relevant_by_user)
    recall_differences = np.array(
        [
            recall_at_k(
                candidate_recommendations.get(user_id, []),
                relevant_by_user[user_id],
                k,
            )
            - recall_at_k(
                baseline_recommendations.get(user_id, []),
                relevant_by_user[user_id],
                k,
            )
            for user_id in user_ids
        ],
        dtype=np.float64,
    )
    ndcg_differences = np.array(
        [
            ndcg_at_k(
                candidate_recommendations.get(user_id, []),
                relevant_by_user[user_id],
                k,
            )
            - ndcg_at_k(
                baseline_recommendations.get(user_id, []),
                relevant_by_user[user_id],
                k,
            )
            for user_id in user_ids
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(random_state)
    recall_samples = np.empty(n_resamples, dtype=np.float64)
    ndcg_samples = np.empty(n_resamples, dtype=np.float64)
    user_count = len(user_ids)
    for index in range(n_resamples):
        sampled_indices = rng.integers(0, user_count, size=user_count)
        recall_samples[index] = recall_differences[sampled_indices].mean()
        ndcg_samples[index] = ndcg_differences[sampled_indices].mean()

    tail_probability = (1.0 - confidence_level) / 2.0

    def summarize(point_values: np.ndarray, samples: np.ndarray) -> dict[str, float]:
        return {
            "estimate": float(point_values.mean()),
            "lower": float(np.quantile(samples, tail_probability)),
            "upper": float(np.quantile(samples, 1.0 - tail_probability)),
        }

    return {
        "method": "paired user bootstrap",
        "users": user_count,
        "resamples": n_resamples,
        "confidence_level": confidence_level,
        "random_state": random_state,
        "difference": "candidate minus baseline",
        "recall_at_k": summarize(recall_differences, recall_samples),
        "ndcg_at_k": summarize(ndcg_differences, ndcg_samples),
    }
