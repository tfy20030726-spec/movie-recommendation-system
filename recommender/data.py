"""Load MovieLens ratings and build leakage-resistant offline splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["movie_id", "title", "genres"]


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Load the MovieLens 1M ratings.dat file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ratings file does not exist: {path}")

    ratings = pd.read_csv(
        path,
        sep="::",
        names=RATING_COLUMNS,
        engine="python",
        encoding="latin-1",
    )
    ratings = ratings.astype(
        {
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "float64",
            "timestamp": "int64",
        }
    )
    return ratings


def load_movies(path: str | Path) -> pd.DataFrame:
    """Load MovieLens 1M movie titles, release years, and genres."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Movies file does not exist: {path}")

    movies = pd.read_csv(
        path,
        sep="::",
        names=MOVIE_COLUMNS,
        engine="python",
        encoding="latin-1",
    )
    movies["movie_id"] = movies["movie_id"].astype("int64")
    movies["release_year"] = pd.to_numeric(
        movies["title"].str.extract(r"\((\d{4})\)$", expand=False),
        errors="coerce",
    ).astype("Int64")
    return movies


def make_positive_interactions(
    ratings: pd.DataFrame,
    min_rating: float = 4.0,
) -> pd.DataFrame:
    """Treat ratings at or above min_rating as implicit positive feedback."""
    missing = set(RATING_COLUMNS).difference(ratings.columns)
    if missing:
        raise ValueError(f"Ratings are missing required columns: {sorted(missing)}")

    positives = ratings.loc[
        ratings["rating"] >= min_rating,
        ["user_id", "movie_id", "rating", "timestamp"],
    ].copy()
    positives = positives.sort_values(
        ["user_id", "timestamp", "movie_id"],
        kind="stable",
    )
    positives = positives.drop_duplicates(
        ["user_id", "movie_id"],
        keep="last",
    )
    return positives.reset_index(drop=True)


def temporal_leave_one_out(
    interactions: pd.DataFrame,
    min_interactions: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each eligible user's latest positive interaction for testing."""
    if min_interactions < 2:
        raise ValueError("min_interactions must be at least 2")
    if interactions.empty:
        return interactions.copy(), interactions.copy()

    counts = interactions.groupby("user_id")["movie_id"].transform("size")
    eligible = interactions.loc[counts >= min_interactions].copy()
    eligible = eligible.sort_values(
        ["user_id", "timestamp", "movie_id"],
        kind="stable",
    )
    test_indices = eligible.groupby("user_id", sort=True).tail(1).index

    test = eligible.loc[test_indices].sort_values("user_id")
    train = eligible.drop(index=test_indices).sort_values(
        ["user_id", "timestamp", "movie_id"],
        kind="stable",
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def temporal_train_validation_test_split(
    interactions: pd.DataFrame,
    min_interactions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out each eligible user's last two interactions for validation/test."""
    if min_interactions < 3:
        raise ValueError("min_interactions must be at least 3")
    if interactions.empty:
        empty = interactions.copy()
        return empty, empty.copy(), empty.copy()

    counts = interactions.groupby("user_id")["movie_id"].transform("size")
    eligible = interactions.loc[counts >= min_interactions].copy()
    eligible = eligible.sort_values(
        ["user_id", "timestamp", "movie_id"],
        kind="stable",
    )
    held_out = eligible.groupby("user_id", sort=True).tail(2)
    validation_indices = held_out.groupby("user_id", sort=True).head(1).index
    test_indices = held_out.groupby("user_id", sort=True).tail(1).index

    validation = eligible.loc[validation_indices].sort_values("user_id")
    test = eligible.loc[test_indices].sort_values("user_id")
    train = eligible.drop(index=held_out.index).sort_values(
        ["user_id", "timestamp", "movie_id"],
        kind="stable",
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
