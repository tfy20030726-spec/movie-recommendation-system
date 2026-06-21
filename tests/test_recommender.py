import unittest

import pandas as pd

from recommender import (
    PopularityRecommender,
    evaluate_top_k,
    make_positive_interactions,
    temporal_leave_one_out,
)


class RecommenderTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
