"""Train and evaluate the ALS plus LightGBM two-stage recommender."""

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
    evaluate_recommendation_lists,
    load_movies,
    load_ratings,
    make_positive_interactions,
    temporal_train_validation_test_split,
)
from recommender.ranking import (  # noqa: E402
    LightGBMReranker,
    build_candidate_frame,
    recommendation_lists_from_candidates,
)
from scripts.download_movielens import download_dataset  # noqa: E402


def run_two_stage(
    ratings_path: Path,
    min_rating: float,
    k: int,
    candidate_k: int,
) -> dict[str, object]:
    ratings = load_ratings(ratings_path)
    movies = load_movies(ratings_path.with_name("movies.dat"))
    positives = make_positive_interactions(ratings, min_rating=min_rating)
    train, validation, test = temporal_train_validation_test_split(positives)

    als = ALSRecommender().fit(train)
    validation_candidates = build_candidate_frame(
        als,
        train,
        validation,
        movies=movies,
        candidate_k=candidate_k,
        force_target=True,
    )
    reranker = LightGBMReranker().fit(validation_candidates)
    test_candidates = build_candidate_frame(
        als,
        train,
        test,
        movies=movies,
        candidate_k=candidate_k,
        force_target=False,
    )

    als_lists = recommendation_lists_from_candidates(
        test_candidates,
        "als_score",
        k,
    )
    candidate_lists = recommendation_lists_from_candidates(
        test_candidates,
        "als_score",
        candidate_k,
    )
    reranked = reranker.rerank(test_candidates)
    reranked_lists = recommendation_lists_from_candidates(
        reranked,
        "ranking_score",
        k,
    )
    als_metrics = evaluate_recommendation_lists(als_lists, test, als.catalog, k)
    reranker_metrics = evaluate_recommendation_lists(
        reranked_lists,
        test,
        als.catalog,
        k,
    )
    candidate_metrics = evaluate_recommendation_lists(
        candidate_lists,
        test,
        als.catalog,
        candidate_k,
    )

    return {
        "dataset": "MovieLens 1M",
        "split": "per-user chronological train/validation/test",
        "positive_rating_threshold": min_rating,
        "train_interactions": len(train),
        "validation_interactions": len(validation),
        "test_interactions": len(test),
        "candidate_k": candidate_k,
        "ranking_k": k,
        "ranking_training_users": reranker.training_users,
        "ranking_early_stopping_users": reranker.validation_users,
        "ranking_training_rows": reranker.training_rows,
        "ranking_early_stopping_rows": reranker.validation_rows,
        "best_iteration": reranker.best_iteration,
        "als_candidate_recall": candidate_metrics["recall_at_k"],
        "als_at_k": als_metrics,
        "lightgbm_at_k": reranker_metrics,
        "absolute_change": {
            metric: reranker_metrics[metric] - als_metrics[metric]
            for metric in ("recall_at_k", "ndcg_at_k", "catalog_coverage")
        },
        "feature_importance": reranker.feature_importance(),
        "methodology_note": (
            "Validation targets train the reranker; test targets are used only "
            "for final offline evaluation."
        ),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two-stage ranking.")
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "reranker_metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    ratings_path = arguments.ratings
    if ratings_path is None:
        ratings_path = download_dataset(PROJECT_ROOT / "data" / "raw")

    report = run_two_stage(
        ratings_path=ratings_path,
        min_rating=arguments.min_rating,
        k=arguments.k,
        candidate_k=arguments.candidate_k,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
