import unittest

import numpy as np
import torch
from torch import nn
from flwr.app import RecordDict

from cosine_filter_audit.adaptive_attack import (
    ATTACK_STATUS_CODES,
    approximate_score,
    cohort_score,
    constrained_local_loss_ascent,
)
from cosine_filter_audit.client_app import (
    _read_client_attack_history,
    _write_client_attack_history,
)
from cosine_filter_audit.multi_signal import score_updates


class AdaptiveAttackTests(unittest.TestCase):
    def test_client_history_persists_bounded_updates_and_scores(self):
        class ContextStub:
            state = RecordDict()

        context = ContextStub()
        for index in range(10):
            _write_client_attack_history(
                context,
                np.array([index, index + 1], dtype=np.float32),
                float(index),
                history_limit=3,
            )

        history = _read_client_attack_history(context)

        self.assertEqual(len(history["updates"]), 3)
        self.assertEqual(history["scores"], [7.0, 8.0, 9.0])
        np.testing.assert_array_equal(history["updates"][0], [7.0, 8.0])

    def test_phase1_surrogate_uses_only_client_update_history(self):
        clean = np.array([1.0, 0.0])
        candidate = np.array([1.0, 0.1])
        previous = [np.array([1.0, 0.0])]
        score = approximate_score(candidate, clean, previous, [0.0, 0.1])

        self.assertGreater(score, 0.0)
        self.assertEqual(
            approximate_score(clean, clean, previous, []),
            0.0,
        )

    def test_phase2_score_matches_server_multi_signal_implementation(self):
        peers = {
            1: np.array([1.0, 0.0]),
            2: np.array([1.1, 0.0]),
            3: np.array([0.9, 0.0]),
        }
        candidate = np.array([1.0, 0.1])
        history = {1: [0.1], 2: [0.0], 3: [0.0]}
        previous = {
            1: np.array([0.9, 0.0]),
            2: np.array([1.0, 0.0]),
            3: np.array([0.8, 0.0]),
        }
        expected = score_updates(
            {**peers, 4: candidate},
            previous_updates=previous,
            score_history=history,
        )[4]["score"]

        actual = cohort_score(
            candidate,
            client_id=4,
            peer_updates=peers,
            previous_updates=previous,
            score_history=history,
            threshold=3.5,
        )

        self.assertEqual(actual, expected)

    def test_local_loss_ascent_respects_score_constraint(self):
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.1, 0.0], [0.0, -0.1]]))
        clean = {name: value.detach().clone() for name, value in model.state_dict().items()}
        global_state = {"weight": torch.zeros_like(clean["weight"])}
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 0])
        batches = [{"img": images, "label": labels}]

        result = constrained_local_loss_ascent(
            model,
            global_state,
            clean,
            batches,
            torch.device("cpu"),
            threshold=0.2,
            score_candidate=lambda update: float(
                np.linalg.norm(update - np.array([0.1, 0.0, 0.0, -0.1]))
            ),
            steps=1,
            step_size=1.0,
        )

        self.assertEqual(result["status_code"], ATTACK_STATUS_CODES["CONSTRAINED"])
        self.assertLessEqual(result["final_score"], 0.2)
        self.assertGreater(result["final_loss"], result["initial_loss"])

    def test_no_feasible_clean_update_is_reported_without_modification(self):
        model = nn.Linear(1, 2, bias=False)
        clean = {name: value.detach().clone() for name, value in model.state_dict().items()}
        global_state = {"weight": torch.zeros_like(clean["weight"])}
        original = {name: value.clone() for name, value in model.state_dict().items()}

        result = constrained_local_loss_ascent(
            model,
            global_state,
            clean,
            [],
            torch.device("cpu"),
            threshold=0.0,
            score_candidate=lambda update: 1.0,
            steps=1,
            step_size=0.1,
        )

        self.assertEqual(
            result["status_code"], ATTACK_STATUS_CODES["NO_FEASIBLE_BASE"]
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name])


if __name__ == "__main__":
    unittest.main()
