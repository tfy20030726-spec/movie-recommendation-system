"""Recommendation baselines used as honest comparison points."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


class PopularityRecommender:
    """Recommend globally popular unseen items with deterministic tie-breaking."""

    def __init__(self) -> None:
        self.ranked_items: list[int] = []
        self.seen_by_user: dict[int, set[int]] = {}
        self.catalog: set[int] = set()

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        required = {"user_id", "movie_id"}
        missing = required.difference(interactions.columns)
        if missing:
            raise ValueError(
                f"Interactions are missing required columns: {sorted(missing)}"
            )

        popularity = (
            interactions.groupby("movie_id", as_index=False)
            .size()
            .sort_values(["size", "movie_id"], ascending=[False, True])
        )
        self.ranked_items = popularity["movie_id"].astype(int).tolist()
        self.catalog = set(self.ranked_items)
        self.seen_by_user = {
            int(user_id): set(group["movie_id"].astype(int))
            for user_id, group in interactions.groupby("user_id")
        }
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        if k <= 0:
            raise ValueError("k must be positive")
        seen = self.seen_by_user.get(int(user_id), set())
        recommendations: list[int] = []
        for movie_id in self.ranked_items:
            if movie_id in seen:
                continue
            recommendations.append(movie_id)
            if len(recommendations) == k:
                break
        return recommendations

    def recommend_many(
        self,
        user_ids: Iterable[int],
        k: int = 10,
    ) -> dict[int, list[int]]:
        return {int(user_id): self.recommend(int(user_id), k) for user_id in user_ids}
