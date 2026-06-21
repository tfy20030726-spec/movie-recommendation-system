"""Candidate feature generation and LightGBM learning-to-rank."""

from __future__ import annotations

import math

import pandas as pd
from lightgbm import LGBMRanker, early_stopping

from .als import ALSRecommender


FEATURE_COLUMNS = [
    "als_score",
    "reciprocal_als_rank",
    "log_item_popularity",
    "log_user_activity",
    "genre_affinity",
    "genre_count",
    "release_year_distance",
]


def build_candidate_frame(
    model: ALSRecommender,
    train_interactions: pd.DataFrame,
    target_interactions: pd.DataFrame,
    movies: pd.DataFrame | None = None,
    candidate_k: int = 100,
    force_target: bool = False,
) -> pd.DataFrame:
    """Build candidate features without using target labels as model inputs."""
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")

    target_by_user = {
        int(user_id): set(group["movie_id"].astype(int))
        for user_id, group in target_interactions.groupby("user_id")
    }
    user_ids = list(target_by_user)
    scored_candidates = model.recommend_scored_many(user_ids, candidate_k)
    item_popularity = train_interactions.groupby("movie_id").size().to_dict()
    user_activity = train_interactions.groupby("user_id").size().to_dict()
    genres_by_movie: dict[int, tuple[str, ...]] = {}
    year_by_movie: dict[int, int] = {}
    user_genre_counts: dict[tuple[int, str], int] = {}
    user_mean_year: dict[int, float] = {}
    if movies is not None and not movies.empty:
        genres_by_movie = {
            int(row.movie_id): tuple(str(row.genres).split("|"))
            for row in movies.itertuples()
        }
        year_by_movie = {
            int(row.movie_id): int(row.release_year)
            for row in movies.dropna(subset=["release_year"]).itertuples()
        }
        movie_genres = movies[["movie_id", "genres"]].copy()
        movie_genres["genre"] = movie_genres["genres"].str.split("|")
        movie_genres = movie_genres.explode("genre")
        train_genres = train_interactions[["user_id", "movie_id"]].merge(
            movie_genres[["movie_id", "genre"]],
            on="movie_id",
            how="inner",
        )
        user_genre_counts = (
            train_genres.groupby(["user_id", "genre"])
            .size()
            .astype(int)
            .to_dict()
        )
        train_years = train_interactions[["user_id", "movie_id"]].copy()
        train_years["release_year"] = train_years["movie_id"].map(year_by_movie)
        user_mean_year = (
            train_years.dropna(subset=["release_year"])
            .groupby("user_id")["release_year"]
            .mean()
            .to_dict()
        )

    forced_pairs: list[tuple[int, int]] = []
    if force_target:
        for user_id, targets in target_by_user.items():
            existing = {
                movie_id for movie_id, _ in scored_candidates.get(user_id, [])
            }
            forced_pairs.extend(
                (user_id, movie_id)
                for movie_id in targets
                if movie_id in model.catalog and movie_id not in existing
            )
    forced_scores = model.score_pairs(forced_pairs)

    rows: list[dict[str, float | int]] = []
    for user_id, targets in target_by_user.items():
        candidates = list(scored_candidates.get(user_id, []))
        if force_target:
            candidates.extend(
                (movie_id, forced_scores[(user_id, movie_id)])
                for movie_id in targets
                if (user_id, movie_id) in forced_scores
            )
        candidates.sort(key=lambda item: (-item[1], item[0]))

        for rank, (movie_id, als_score) in enumerate(candidates, start=1):
            genres = genres_by_movie.get(movie_id, ())
            activity = user_activity.get(user_id, 0)
            genre_affinity = 0.0
            if genres and activity:
                genre_affinity = sum(
                    user_genre_counts.get((user_id, genre), 0)
                    for genre in genres
                ) / (activity * len(genres))
            movie_year = year_by_movie.get(movie_id)
            preferred_year = user_mean_year.get(user_id)
            release_year_distance = (
                abs(movie_year - preferred_year) / 10.0
                if movie_year is not None and preferred_year is not None
                else 0.0
            )
            rows.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "label": int(movie_id in targets),
                    "als_rank": rank,
                    "als_score": als_score,
                    "reciprocal_als_rank": 1.0 / rank,
                    "log_item_popularity": math.log1p(
                        item_popularity.get(movie_id, 0)
                    ),
                    "log_user_activity": math.log1p(
                        activity
                    ),
                    "genre_affinity": genre_affinity,
                    "genre_count": len(genres),
                    "release_year_distance": release_year_distance,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "movie_id",
            "label",
            "als_rank",
            *FEATURE_COLUMNS,
        ],
    )


def recommendation_lists_from_candidates(
    candidates: pd.DataFrame,
    score_column: str,
    k: int,
) -> dict[int, list[int]]:
    if k <= 0:
        raise ValueError("k must be positive")
    if candidates.empty:
        return {}

    ranked = candidates.sort_values(
        ["user_id", score_column, "movie_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    top_k = ranked.groupby("user_id", sort=True).head(k)
    return {
        int(user_id): group["movie_id"].astype(int).tolist()
        for user_id, group in top_k.groupby("user_id", sort=True)
    }


class LightGBMReranker:
    """Rank ALS candidates using validation interactions as relevance labels."""

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        num_leaves: int = 15,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.training_users = 0
        self.validation_users = 0
        self.training_rows = 0
        self.validation_rows = 0
        self._model: LGBMRanker | None = None

    def fit(self, candidates: pd.DataFrame) -> "LightGBMReranker":
        if candidates.empty:
            raise ValueError("Ranking candidates cannot be empty")

        positive_counts = candidates.groupby("user_id")["label"].transform("sum")
        eligible = candidates.loc[positive_counts > 0].copy()
        eligible = eligible.sort_values(
            ["user_id", "als_rank", "movie_id"],
            kind="stable",
        )
        if eligible.empty:
            raise ValueError("No ranking group contains a positive target")

        validation_user_mask = eligible["user_id"].astype(int) % 5 == 0
        training = eligible.loc[~validation_user_mask]
        validation = eligible.loc[validation_user_mask]
        if training.empty or validation.empty:
            training = eligible
            validation = eligible.iloc[0:0]

        group_sizes = training.groupby("user_id", sort=True).size().to_numpy()
        self.training_users = len(group_sizes)
        validation_group_sizes = (
            validation.groupby("user_id", sort=True).size().to_numpy()
        )
        self.validation_users = len(validation_group_sizes)
        self.training_rows = len(training)
        self.validation_rows = len(validation)
        self._model = LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=1,
            min_child_samples=50,
            reg_lambda=1.0,
            verbosity=-1,
        )
        fit_arguments: dict[str, object] = {}
        if not validation.empty:
            fit_arguments = {
                "eval_set": [(validation[FEATURE_COLUMNS], validation["label"])],
                "eval_group": [validation_group_sizes],
                "eval_metric": "ndcg",
                "eval_at": [10],
                "callbacks": [early_stopping(20, verbose=False)],
            }
        self._model.fit(
            training[FEATURE_COLUMNS],
            training["label"],
            group=group_sizes,
            **fit_arguments,
        )
        return self

    def rerank(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("LightGBMReranker must be fitted before prediction")
        ranked = candidates.copy()
        ranked["ranking_score"] = self._model.predict(ranked[FEATURE_COLUMNS])
        return ranked.sort_values(
            ["user_id", "ranking_score", "movie_id"],
            ascending=[True, False, True],
            kind="stable",
        ).reset_index(drop=True)

    def feature_importance(self) -> dict[str, int]:
        if self._model is None:
            raise RuntimeError("LightGBMReranker must be fitted first")
        return {
            feature: int(importance)
            for feature, importance in zip(
                FEATURE_COLUMNS,
                self._model.feature_importances_,
            )
        }

    @property
    def best_iteration(self) -> int:
        if self._model is None:
            raise RuntimeError("LightGBMReranker must be fitted first")
        return int(self._model.best_iteration_ or self.n_estimators)
