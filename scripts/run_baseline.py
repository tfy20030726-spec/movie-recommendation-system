"""Run the popularity baseline on a chronological MovieLens split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommender import (  # noqa: E402
    PopularityRecommender,
    evaluate_top_k,
    load_ratings,
    make_positive_interactions,
    temporal_leave_one_out,
)
from scripts.download_movielens import download_dataset  # noqa: E402


def run_baseline(
    ratings_path: Path,
    min_rating: float,
    k: int,
) -> dict[str, object]:
    ratings = load_ratings(ratings_path)
    positives = make_positive_interactions(ratings, min_rating=min_rating)
    train, test = temporal_leave_one_out(positives)
    model = PopularityRecommender().fit(train)
    metrics = evaluate_top_k(model, test, k=k)

    return {
        "dataset": "MovieLens 1M",
        "split": "latest positive interaction per eligible user",
        "positive_rating_threshold": min_rating,
        "raw_ratings": len(ratings),
        "users": int(ratings["user_id"].nunique()),
        "movies": int(ratings["movie_id"].nunique()),
        "positive_interactions": len(positives),
        "train_interactions": len(train),
        "test_interactions": len(test),
        "model": "global popularity excluding seen items",
        "metrics": metrics,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the popularity baseline.")
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "baseline_metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    ratings_path = arguments.ratings
    if ratings_path is None:
        ratings_path = download_dataset(PROJECT_ROOT / "data" / "raw")

    report = run_baseline(ratings_path, arguments.min_rating, arguments.k)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
