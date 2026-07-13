import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recommender import (
    ALSRecommender,
    LightGBMReranker,
    PopularityRecommender,
    build_candidate_frame,
    evaluate_top_k,
    load_movies,
    make_positive_interactions,
    paired_bootstrap_metric_differences,
    recommendation_lists_from_candidates,
    temporal_leave_one_out,
    temporal_train_validation_test_split,
)


class RecommenderTest(unittest.TestCase):
    def test_movie_metadata_extracts_release_year_and_genres(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            movies_path = Path(temporary_directory) / "movies.dat"
            movies_path.write_text(
                "1::Toy Story (1995)::Animation|Children's\n"
                "2::Unknown Year::Drama\n",
                encoding="latin-1",
            )

            movies = load_movies(movies_path)

            self.assertEqual(int(movies.loc[0, "release_year"]), 1995)
            self.assertEqual(movies.loc[0, "genres"], "Animation|Children's")
            self.assertTrue(pd.isna(movies.loc[1, "release_year"]))

    def test_temporal_split_holds_out_latest_positive_without_leakage(self):
        ratings = pd.DataFrame(
            [
                (1, 10, 5.0, 100),
                (1, 20, 2.0, 200),
                (1, 30, 4.0, 300),
                (2, 10, 4.0, 100),
                (2, 40, 5.0, 400),
                (3, 50, 5.0, 500),
            ],
            columns=["user_id", "movie_id", "rating", "timestamp"],
        )

        positives = make_positive_interactions(ratings, min_rating=4.0)
        train, test = temporal_leave_one_out(positives)

        self.assertEqual(set(train["user_id"]), {1, 2})
        self.assertEqual(set(test["user_id"]), {1, 2})
        self.assertEqual(
            dict(zip(test["user_id"], test["movie_id"])),
            {1: 30, 2: 40},
        )
        self.assertFalse(
            set(zip(train["user_id"], train["movie_id"])).intersection(
                zip(test["user_id"], test["movie_id"])
            )
        )

    def test_popularity_baseline_excludes_seen_items_and_scores_hits(self):
        train = pd.DataFrame(
            [
                (1, 10),
                (1, 20),
                (2, 10),
                (2, 30),
            ],
            columns=["user_id", "movie_id"],
        )
        test = pd.DataFrame(
            [(1, 30), (2, 20)],
            columns=["user_id", "movie_id"],
        )

        model = PopularityRecommender().fit(train)
        metrics = evaluate_top_k(model, test, k=1)

        self.assertEqual(model.recommend(1, k=1), [30])
        self.assertEqual(model.recommend(2, k=1), [20])
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["ndcg_at_k"], 1.0)
        self.assertAlmostEqual(metrics["catalog_coverage"], 2 / 3)

    def test_paired_bootstrap_detects_consistent_candidate_improvement(self):
        test = pd.DataFrame(
            [(1, 10), (2, 20), (3, 30), (4, 40)],
            columns=["user_id", "movie_id"],
        )
        baseline = {1: [99], 2: [99], 3: [99], 4: [99]}
        candidate = {1: [10], 2: [20], 3: [30], 4: [40]}

        intervals = paired_bootstrap_metric_differences(
            baseline,
            candidate,
            test,
            k=1,
            n_resamples=200,
            random_state=7,
        )

        self.assertEqual(intervals["recall_at_k"]["estimate"], 1.0)
        self.assertGreater(intervals["recall_at_k"]["lower"], 0.0)
        self.assertGreater(intervals["ndcg_at_k"]["lower"], 0.0)

    def test_paired_bootstrap_is_reproducible_and_validates_arguments(self):
        test = pd.DataFrame(
            [(1, 10), (2, 20)],
            columns=["user_id", "movie_id"],
        )
        baseline = {1: [10], 2: [99]}
        candidate = {1: [99], 2: [20]}

        first = paired_bootstrap_metric_differences(
            baseline,
            candidate,
            test,
            k=1,
            n_resamples=50,
            random_state=11,
        )
        second = paired_bootstrap_metric_differences(
            baseline,
            candidate,
            test,
            k=1,
            n_resamples=50,
            random_state=11,
        )
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            paired_bootstrap_metric_differences(
                baseline,
                candidate,
                test,
                n_resamples=0,
            )

    def test_als_recommendations_are_catalog_items_not_seen_in_training(self):
        train = pd.DataFrame(
            [
                (1, 10),
                (1, 20),
                (2, 10),
                (2, 30),
                (3, 20),
                (3, 40),
                (4, 30),
                (4, 40),
            ],
            columns=["user_id", "movie_id"],
        )

        model = ALSRecommender(
            factors=2,
            iterations=3,
            random_state=42,
        ).fit(train)
        recommendations = model.recommend_many([1, 2], k=2)

        seen_by_user = {
            user_id: set(group["movie_id"])
            for user_id, group in train.groupby("user_id")
        }
        for user_id, movie_ids in recommendations.items():
            self.assertTrue(set(movie_ids).issubset(model.catalog))
            self.assertFalse(set(movie_ids).intersection(seen_by_user[user_id]))

    def test_three_way_split_uses_second_last_for_validation_and_last_for_test(self):
        interactions = pd.DataFrame(
            [
                (1, 10, 5.0, 100),
                (1, 20, 5.0, 200),
                (1, 30, 5.0, 300),
                (1, 40, 5.0, 400),
                (2, 10, 5.0, 100),
                (2, 20, 5.0, 200),
                (2, 30, 5.0, 300),
                (3, 10, 5.0, 100),
                (3, 20, 5.0, 200),
            ],
            columns=["user_id", "movie_id", "rating", "timestamp"],
        )

        train, validation, test = temporal_train_validation_test_split(
            interactions
        )

        self.assertEqual(set(train["user_id"]), {1, 2})
        self.assertEqual(
            dict(zip(validation["user_id"], validation["movie_id"])),
            {1: 30, 2: 20},
        )
        self.assertEqual(
            dict(zip(test["user_id"], test["movie_id"])),
            {1: 40, 2: 30},
        )

    def test_reranker_trains_on_forced_validation_targets(self):
        train = pd.DataFrame(
            [
                (1, 10),
                (1, 20),
                (2, 10),
                (2, 30),
                (3, 20),
                (3, 40),
                (4, 30),
                (4, 40),
                (5, 10),
                (5, 40),
                (6, 20),
                (6, 30),
            ],
            columns=["user_id", "movie_id"],
        )
        validation = pd.DataFrame(
            [(1, 30), (2, 20), (3, 10), (4, 20), (5, 30), (6, 40)],
            columns=["user_id", "movie_id"],
        )
        model = ALSRecommender(
            factors=2,
            iterations=3,
            random_state=42,
        ).fit(train)

        candidates = build_candidate_frame(
            model,
            train,
            validation,
            candidate_k=2,
            force_target=True,
        )
        reranker = LightGBMReranker(n_estimators=5, num_leaves=7).fit(candidates)
        reranked = reranker.rerank(candidates)
        recommendations = recommendation_lists_from_candidates(
            reranked,
            "ranking_score",
            k=2,
        )

        self.assertTrue((candidates.groupby("user_id")["label"].sum() == 1).all())
        self.assertEqual(reranker.training_users, 5)
        self.assertEqual(reranker.validation_users, 1)
        self.assertEqual(set(recommendations), {1, 2, 3, 4, 5, 6})


if __name__ == "__main__":
    unittest.main()
