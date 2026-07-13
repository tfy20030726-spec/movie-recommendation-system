"""Train implicit ALS and compare it with the popularity baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommender import (  # noqa: E402
    ALSRecommender,
    PopularityRecommender,
    evaluate_recommendation_lists,
    load_ratings,
    make_positive_interactions,
    paired_bootstrap_metric_differences,
    temporal_leave_one_out,
)
from scripts.download_movielens import download_dataset  # noqa: E402


def run_comparison(
    ratings_path: Path,
    min_rating: float,
    k: int,
    factors: int,
    regularization: float,
    alpha: float,
    iterations: int,
    bootstrap_resamples: int,
) -> dict[str, object]:
    ratings = load_ratings(ratings_path)
    positives = make_positive_interactions(ratings, min_rating=min_rating)
    train, test = temporal_leave_one_out(positives)
    test_user_ids = test["user_id"].astype(int).unique().tolist()

    popularity = PopularityRecommender().fit(train)
    popularity_recommendations = popularity.recommend_many(test_user_ids, k=k)
    popularity_metrics = evaluate_recommendation_lists(
        popularity_recommendations,
        test,
        popularity.catalog,
        k=k,
    )
    als = ALSRecommender(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
    ).fit(train)
    als_recommendations = als.recommend_many(test_user_ids, k=k)
    als_metrics = evaluate_recommendation_lists(
        als_recommendations,
        test,
        als.catalog,
        k=k,
    )
    paired_intervals = paired_bootstrap_metric_differences(
        popularity_recommendations,
        als_recommendations,
        test,
        k=k,
        n_resamples=bootstrap_resamples,
    )

    return {
        "dataset": "MovieLens 1M",
        "split": "latest positive interaction per eligible user",
        "positive_rating_threshold": min_rating,
        "train_interactions": len(train),
        "test_interactions": len(test),
        "k": k,
        "popularity": popularity_metrics,
        "als": {
            "hyperparameters": {
                "factors": factors,
                "regularization": regularization,
                "alpha": alpha,
                "iterations": iterations,
                "random_state": 42,
            },
            "metrics": als_metrics,
        },
        "absolute_change": {
            metric: als_metrics[metric] - popularity_metrics[metric]
            for metric in ("recall_at_k", "ndcg_at_k", "catalog_coverage")
        },
        "paired_bootstrap_change": paired_intervals,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ALS with popularity.")
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "als_metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    ratings_path = arguments.ratings
    if ratings_path is None:
        ratings_path = download_dataset(PROJECT_ROOT / "data" / "raw")

    report = run_comparison(
        ratings_path=ratings_path,
        min_rating=arguments.min_rating,
        k=arguments.k,
        factors=arguments.factors,
        regularization=arguments.regularization,
        alpha=arguments.alpha,
        iterations=arguments.iterations,
        bootstrap_resamples=arguments.bootstrap_resamples,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
