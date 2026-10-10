import unittest

import numpy as np

from cosine_filter_audit.multi_signal import score_updates


class MultiSignalTests(unittest.TestCase):
    def test_outlying_direction_and_norm_receive_larger_score(self):
        updates = {
            1: np.array([1.0, 0.0]),
            2: np.array([1.1, 0.0]),
            3: np.array([0.9, 0.0]),
            4: np.array([1.0, 0.1]),
            5: np.array([10.0, 10.0]),
        }

        results = score_updates(updates, threshold=3.5)

        self.assertGreater(results[5]["score"], results[1]["score"])
        self.assertTrue(results[5]["flagged"])
        self.assertFalse(results[1]["flagged"])
        self.assertIsNone(results[1]["temporal_z"])
        self.assertIsNone(results[1]["history_z"])

    def test_temporal_and_history_signals_are_reported_when_available(self):
        updates = {
            1: np.array([1.0, 0.0]),
            2: np.array([1.1, 0.0]),
            3: np.array([0.9, 0.0]),
        }
        previous = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0]),
            3: np.array([0.9, 0.0]),
        }
        history = {1: [0.0, 0.1], 2: [0.0, 0.1], 3: [0.0, 0.1]}

        results = score_updates(
            updates, previous_updates=previous, score_history=history
        )

        self.assertIsNotNone(results[1]["temporal_z"])
        self.assertIsNotNone(results[1]["history_z"])
        self.assertEqual(set(results), {1, 2, 3})

    def test_rejects_mismatched_or_nonfinite_vectors(self):
        with self.assertRaisesRegex(ValueError, "same-shaped"):
            score_updates({1: np.array([1.0]), 2: np.array([1.0, 2.0])})
        with self.assertRaisesRegex(ValueError, "finite"):
            score_updates({1: np.array([np.nan]), 2: np.array([1.0])})

    def test_constant_cohort_has_zero_scores(self):
        updates = {
            1: np.array([1.0, 2.0]),
            2: np.array([1.0, 2.0]),
            3: np.array([1.0, 2.0]),
        }

        results = score_updates(updates)

        self.assertTrue(all(result["score"] == 0.0 for result in results.values()))
        self.assertTrue(all(not result["flagged"] for result in results.values()))


if __name__ == "__main__":
    unittest.main()
