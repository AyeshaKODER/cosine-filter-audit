import unittest

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from cosine_filter_audit.experiment_strategy import CosineAuditStrategy


def _reply(node_id, state):
    request = Message(
        content=RecordDict({"config": ConfigRecord()}),
        dst_node_id=node_id,
        message_type="train",
    )
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(state),
                "metrics": MetricRecord(
                    {
                        "train_loss": 1.0,
                        "num-examples": 1,
                        "partition-id": node_id,
                        "attack-status-code": 0,
                        "attack-initial-score": 0.0,
                        "attack-final-score": 0.0,
                        "attack-initial-loss": 0.0,
                        "attack-final-loss": 0.0,
                        "attack-steps-applied": 0,
                    }
                ),
            }
        ),
        reply_to=request,
    )

from cosine_filter_audit.robust_aggregators import aggregate_updates


class RobustAggregatorTests(unittest.TestCase):
    def setUp(self):
        self.updates = {
            1: np.array([0.0, 0.0]),
            2: np.array([0.1, 0.0]),
            3: np.array([0.0, 0.1]),
            4: np.array([-0.1, 0.0]),
            5: np.array([100.0, 100.0]),
        }

    def test_krum_avoids_extreme_update(self):
        aggregate, details = aggregate_updates(
            self.updates, method="krum", byzantine_count=1
        )

        self.assertNotEqual(details["selected_node_ids"], [5])
        self.assertLess(np.linalg.norm(aggregate), 1.0)

    def test_multi_krum_averages_configured_top_candidates(self):
        aggregate, details = aggregate_updates(
            self.updates,
            method="multi_krum",
            byzantine_count=1,
            multi_krum_count=3,
        )

        self.assertEqual(len(details["selected_node_ids"]), 3)
        self.assertNotIn(5, details["selected_node_ids"])
        self.assertLess(np.linalg.norm(aggregate), 1.0)

    def test_coordinate_median_and_trimmed_mean_resist_extreme_value(self):
        median, _ = aggregate_updates(
            self.updates, method="coordinate_median", byzantine_count=1
        )
        trimmed, _ = aggregate_updates(
            self.updates, method="trimmed_mean", byzantine_count=1
        )

        self.assertLess(np.linalg.norm(median), 1.0)
        self.assertLess(np.linalg.norm(trimmed), 1.0)

    def test_krum_rejects_insufficient_cohort(self):
        with self.assertRaisesRegex(ValueError, "more clients"):
            aggregate_updates(
                {1: np.array([1.0]), 2: np.array([2.0]), 3: np.array([3.0])},
                method="krum",
                byzantine_count=1,
            )

    def test_strategy_applies_robust_aggregate_as_global_plus_delta(self):
        instance = CosineAuditStrategy(
            representation_mode="delta",
            selection_mode="none",
            aggregation_mode="coordinate_median",
            attacker_node_id=10,
        )
        instance._round_global_state = {"weight": torch.tensor([5.0, -5.0])}
        replies = [
            _reply(10, {"weight": torch.tensor([5.0, -5.0])}),
            _reply(11, {"weight": torch.tensor([5.1, -5.0])}),
            _reply(12, {"weight": torch.tensor([5.0, -4.9])}),
            _reply(13, {"weight": torch.tensor([4.9, -5.0])}),
            _reply(14, {"weight": torch.tensor([105.0, 95.0])}),
        ]

        aggregate, _ = instance.aggregate_train(1, replies)

        self.assertIsInstance(aggregate, ArrayRecord)
        state = aggregate.to_torch_state_dict()
        torch.testing.assert_close(state["weight"], torch.tensor([5.0, -5.0]))
        self.assertEqual(
            instance.round_log[1]["aggregation"]["aggregation_mode"],
            "coordinate_median",
        )
        self.assertTrue(
            all(
                row["aggregation_weight"] is None
                for row in instance.round_log[1]["clients"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
