"""Implicit-feedback alternating least squares recommender."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits


class ALSRecommender:
    """Map external IDs to an implicit ALS model and filter seen items."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        alpha: float = 20.0,
        iterations: int = 20,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state
        self.catalog: set[int] = set()
        self._user_to_index: dict[int, int] = {}
        self._index_to_item: np.ndarray = np.array([], dtype=np.int64)
        self._user_items: csr_matrix | None = None
        self._model: object | None = None

    def fit(self, interactions: pd.DataFrame) -> "ALSRecommender":
        required = {"user_id", "movie_id"}
        missing = required.difference(interactions.columns)
        if missing:
            raise ValueError(
                f"Interactions are missing required columns: {sorted(missing)}"
            )
        if interactions.empty:
            raise ValueError("ALS requires at least one interaction")

        user_ids = np.sort(interactions["user_id"].unique()).astype(np.int64)
        item_ids = np.sort(interactions["movie_id"].unique()).astype(np.int64)
        self._user_to_index = {
            int(user_id): index for index, user_id in enumerate(user_ids)
        }
        item_to_index = {
            int(movie_id): index for index, movie_id in enumerate(item_ids)
        }
        rows = interactions["user_id"].map(self._user_to_index).to_numpy()
        columns = interactions["movie_id"].map(item_to_index).to_numpy()
        values = np.ones(len(interactions), dtype=np.float32)
        self._user_items = csr_matrix(
            (values, (rows, columns)),
            shape=(len(user_ids), len(item_ids)),
            dtype=np.float32,
        )
        self._index_to_item = item_ids
        self.catalog = set(item_ids.astype(int))

        with threadpool_limits(limits=1, user_api="blas"):
            self._model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                alpha=self.alpha,
                iterations=self.iterations,
                random_state=self.random_state,
                num_threads=1,
            )
            self._model.fit(self._user_items, show_progress=False)
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        return self.recommend_many([user_id], k).get(int(user_id), [])

    def recommend_many(
        self,
        user_ids: Iterable[int],
        k: int = 10,
    ) -> dict[int, list[int]]:
        if k <= 0:
            raise ValueError("k must be positive")
        if self._model is None or self._user_items is None:
            raise RuntimeError("ALSRecommender must be fitted before recommendation")

        requested = [int(user_id) for user_id in user_ids]
        results = {user_id: [] for user_id in requested}
        known_users = [
            user_id for user_id in requested if user_id in self._user_to_index
        ]
        if not known_users:
            return results

        internal_users = np.array(
            [self._user_to_index[user_id] for user_id in known_users],
            dtype=np.int32,
        )
        item_indices, _ = self._model.recommend(
            internal_users,
            self._user_items[internal_users],
            N=k,
            filter_already_liked_items=True,
        )
        for user_id, indices in zip(known_users, item_indices):
            results[user_id] = self._index_to_item[indices].astype(int).tolist()
        return results
