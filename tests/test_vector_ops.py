import unittest

import numpy as np
import torch

from cosine_filter_audit.vector_ops import (
    apply_delta_attack,
    cosine_similarity,
    delta_state_dict,
    flatten_state,
    legacy_scale_state_dict,
    normalized_sample_weights,
)


class VectorRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.global_state = {
            "weight": torch.tensor([1.0, -2.0]),
            "counter": torch.tensor([3], dtype=torch.int64),
        }
        self.local_state = {
            "weight": torch.tensor([4.0, 2.0]),
            "counter": torch.tensor([5], dtype=torch.int64),
        }

    def test_legacy_full_model_scaling_is_unchanged(self):
        scale = -3.0
        actual = legacy_scale_state_dict(self.local_state, scale)
        expected = {
            "weight": torch.tensor([-12.0, -6.0]),
            "counter": torch.tensor([-15.0]),
        }
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])

    def test_delta_attack_invariants_and_nonfloating_policy(self):
        for scale, expected_weight in (
            (1.0, self.local_state["weight"]),
            (0.0, self.global_state["weight"]),
            (-1.0, torch.tensor([-2.0, -6.0])),
        ):
            attacked = apply_delta_attack(
                self.local_state, self.global_state, scale
            )
            torch.testing.assert_close(attacked["weight"], expected_weight)
            torch.testing.assert_close(attacked["counter"], self.global_state["counter"])

    def test_delta_vector_is_local_minus_global(self):
        delta = delta_state_dict(self.local_state, self.global_state)
        self.assertEqual(tuple(delta), ("weight",))
        torch.testing.assert_close(delta["weight"], torch.tensor([3.0, 4.0]))
        np.testing.assert_array_equal(
            flatten_state(delta, floating_only=True), np.array([3.0, 4.0])
        )

    def test_mean_reference_includes_every_submitted_vector(self):
        vectors = [np.array([1.0, 0.0]), np.array([-9.0, 0.0])]
        reference = np.mean(vectors, axis=0)
        np.testing.assert_array_equal(reference, np.array([-4.0, 0.0]))

    def test_cosine_threshold_direction_and_boundary(self):
        score = cosine_similarity(np.array([-1.0, 0.0]), np.array([1.0, 0.0]))
        self.assertLess(score, 0.7)
        self.assertFalse(0.7 < 0.7)
        self.assertFalse(cosine_similarity(
            np.array([1.0, 0.0]), np.array([1.0, 0.0])
        ) < 0.7)

    def test_cosine_rejects_nonfinite_and_near_zero_inputs(self):
        for vector, reference in (
            (np.array([np.nan, 1.0]), np.array([1.0, 0.0])),
            (np.array([np.inf, 1.0]), np.array([1.0, 0.0])),
            (np.array([0.0, 0.0]), np.array([1.0, 0.0])),
            (np.array([1.0, 0.0]), np.array([1e-15, 0.0])),
        ):
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    cosine_similarity(vector, reference)

    def test_state_validation_rejects_key_order_mismatch(self):
        reordered = {
            "counter": self.local_state["counter"],
            "weight": self.local_state["weight"],
        }
        with self.assertRaises(ValueError):
            delta_state_dict(reordered, self.global_state)

    def test_state_validation_rejects_shape_mismatch(self):
        wrong_shape = {
            "weight": torch.tensor([1.0]),
            "counter": torch.tensor([3], dtype=torch.int64),
        }
        with self.assertRaises(ValueError):
            delta_state_dict(wrong_shape, self.global_state)

    def test_fedavg_sample_weights_renormalize_after_filtering(self):
        weights = normalized_sample_weights([2, 6])
        self.assertEqual(weights, [0.25, 0.75])
        aggregate = weights[0] * 4.0 + weights[1] * 8.0
        self.assertEqual(aggregate, 7.0)

    def test_invalid_sample_counts_are_rejected(self):
        for counts in ([], [0, 1], [-1, 2], [1, 2.5]):
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    normalized_sample_weights(counts)


if __name__ == "__main__":
    unittest.main()
