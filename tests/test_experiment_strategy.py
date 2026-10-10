import unittest

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from cosine_filter_audit.adaptive_attack import PHASE2
from cosine_filter_audit.experiment_strategy import (
    ROUND_STATUS_CODES,
    CosineAuditStrategy,
    _vector_sha256,
)
from cosine_filter_audit.vector_ops import DELTA


def reply(node_id, state, sample_count=1, client_label=None):
    request = Message(
        content=RecordDict({"config": ConfigRecord()}),
        dst_node_id=node_id,
        message_type="train",
    )
    metrics = {
        "train_loss": 1.0,
        "num-examples": sample_count,
        "partition-id": node_id,
        "attack-status-code": 0,
        "attack-initial-score": 0.0,
        "attack-final-score": 0.0,
        "attack-initial-loss": 0.0,
        "attack-final-loss": 0.0,
        "attack-steps-applied": 0,
    }
    if client_label is not None:
        metrics["is_malicious"] = client_label
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(state),
                "metrics": MetricRecord(metrics),
            }
        ),
        reply_to=request,
    )


def strategy(
    *,
    selection_mode="none",
    threshold=0.7,
    multi_signal_threshold=3.5,
    attacker_node_id=10,
):
    result = CosineAuditStrategy(
        representation_mode=DELTA,
        selection_mode=selection_mode,
        threshold=threshold,
        multi_signal_threshold=multi_signal_threshold,
        attacker_node_id=attacker_node_id,
    )
    result._round_global_state = {"weight": torch.tensor([0.0, 0.0])}
    return result


class ExperimentStrategyTests(unittest.TestCase):
    def test_oracle_uses_server_node_id_not_client_metric(self):
        instance = strategy(selection_mode="oracle")
        replies = [
            reply(10, {"weight": torch.tensor([2.0, 0.0])}, client_label=0),
            reply(11, {"weight": torch.tensor([1.0, 0.0])}, client_label=1),
        ]
        aggregate, metrics = instance.aggregate_train(1, replies)
        self.assertIsNotNone(aggregate)
        self.assertEqual(
            metrics["round_status_code"], ROUND_STATUS_CODES["AGGREGATE_FINITE"]
        )
        self.assertEqual(instance.round_log[1]["selected_node_ids"], [11])
        self.assertTrue(instance.round_log[1]["clients"][10]["is_attacker"])
        self.assertFalse(instance.round_log[1]["clients"][11]["is_attacker"])

    def test_detector_all_filtered_is_explicit_and_skips_aggregation(self):
        instance = strategy(selection_mode="detector", threshold=0.9)
        replies = [
            reply(10, {"weight": torch.tensor([1.0, 0.0])}),
            reply(11, {"weight": torch.tensor([0.0, 1.0])}),
        ]
        aggregate, metrics = instance.aggregate_train(1, replies)
        self.assertIsNone(aggregate)
        self.assertEqual(
            metrics["round_status_code"], ROUND_STATUS_CODES["ALL_CLIENTS_FILTERED"]
        )
        self.assertEqual(instance.round_log[1]["selected_node_ids"], [])

    def test_multi_signal_uses_robust_current_reference_and_logs_components(self):
        instance = strategy(
            selection_mode="multi_signal",
            multi_signal_threshold=3.5,
            attacker_node_id=13,
        )
        replies = [
            reply(10, {"weight": torch.tensor([1.0, 0.0])}),
            reply(11, {"weight": torch.tensor([1.1, 0.0])}),
            reply(12, {"weight": torch.tensor([0.9, 0.0])}),
            reply(13, {"weight": torch.tensor([10.0, 10.0])}),
        ]

        aggregate, metrics = instance.aggregate_train(1, replies)

        self.assertIsNotNone(aggregate)
        self.assertEqual(
            metrics["round_status_code"], ROUND_STATUS_CODES["AGGREGATE_FINITE"]
        )
        round_log = instance.round_log[1]
        self.assertEqual(
            round_log["effective_reference"], "current_geometric_median"
        )
        self.assertTrue(round_log["clients"][13]["flagged"])
        self.assertFalse(round_log["clients"][13]["selected"])
        self.assertIn("direction_z", round_log["clients"][13]["score_components"])
        self.assertEqual(round_log["selected_node_ids"], [10, 11, 12])

    def test_multi_signal_rejects_full_model_representation(self):
        with self.assertRaisesRegex(ValueError, "requires delta"):
            CosineAuditStrategy(
                representation_mode="legacy_full_model",
                selection_mode="multi_signal",
            )

    def test_phase2_sends_peer_updates_and_logs_refinement_outcome(self):
        class RefinementGrid:
            request = None

            def send_and_receive(self, messages):
                self.request = list(messages)[0]
                refined = reply(
                    10,
                    {"weight": torch.tensor([1.0, 0.0])},
                    sample_count=1,
                )
                refined.content["metrics"]["train_loss"] = 1.2
                refined.content["metrics"]["attack-status-code"] = 1
                refined.content["metrics"]["attack-initial-score"] = 0.2
                refined.content["metrics"]["attack-final-score"] = 0.1
                refined.content["metrics"]["attack-initial-loss"] = 1.0
                refined.content["metrics"]["attack-final-loss"] = 1.2
                refined.content["metrics"]["attack-steps-applied"] = 1
                return [refined]

        instance = CosineAuditStrategy(
            representation_mode=DELTA,
            selection_mode="multi_signal",
            adaptive_attack_mode=PHASE2,
            attack_start_round=1,
            attacker_node_id=10,
        )
        instance._round_global_state = {"weight": torch.tensor([0.0, 0.0])}
        fake_grid = RefinementGrid()
        instance._active_grid = fake_grid
        attack_reply = reply(10, {"weight": torch.tensor([1.0, 0.0])})
        aggregate, _ = instance.aggregate_train(
            1,
            [
                attack_reply,
                reply(11, {"weight": torch.tensor([1.1, 0.0])}),
            ],
        )

        self.assertIsNotNone(aggregate)
        packed = fake_grid.request.content["arrays"].to_torch_state_dict()
        self.assertIn("peer__11", packed)
        config = fake_grid.request.content["config"]
        self.assertEqual(config["adaptive_attack_phase"], "whitebox_refine")
        logged = instance.round_log[1]["clients"][10]["adaptive_attack"]
        self.assertEqual(logged["status_code"], 1)
        self.assertEqual(logged["final_score"], 0.1)

    def test_nonfinite_client_state_is_not_scored_or_selected(self):
        instance = strategy(selection_mode="none")
        replies = [
            reply(10, {"weight": torch.tensor([float("nan"), 0.0])}),
            reply(11, {"weight": torch.tensor([1.0, 0.0])}),
        ]
        aggregate, metrics = instance.aggregate_train(1, replies)
        self.assertIsNotNone(aggregate)
        self.assertEqual(
            metrics["round_status_code"], ROUND_STATUS_CODES["AGGREGATE_FINITE"]
        )
        self.assertIn(10, instance.round_log[1]["invalid_clients"])
        self.assertFalse(instance.round_log[1]["clients"][10]["selected"])

    def test_sample_weights_are_renormalized_after_selection(self):
        instance = strategy(selection_mode="oracle")
        replies = [
            reply(10, {"weight": torch.tensor([1.0, 0.0])}, sample_count=2),
            reply(11, {"weight": torch.tensor([3.0, 0.0])}, sample_count=2),
            reply(12, {"weight": torch.tensor([5.0, 0.0])}, sample_count=6),
        ]
        aggregate, _ = instance.aggregate_train(1, replies)
        state = aggregate.to_torch_state_dict()
        torch.testing.assert_close(state["weight"], torch.tensor([4.5, 0.0]))
        self.assertEqual(instance.round_log[1]["reference_dimension"], 2)
        self.assertEqual(
            instance.round_log[1]["reference_sha256"],
            _vector_sha256(np.array([3.0, 0.0], dtype=np.float32)),
        )
        self.assertNotIn(10, instance.round_log[1]["selected_node_ids"])
        self.assertEqual(
            instance.round_log[1]["clients"][11]["aggregation_weight"], 0.25
        )
        self.assertEqual(
            instance.round_log[1]["clients"][12]["aggregation_weight"], 0.75
        )

    def test_no_replies_has_explicit_status(self):
        instance = strategy()
        aggregate, metrics = instance.aggregate_train(1, [])
        self.assertIsNone(aggregate)
        self.assertEqual(
            metrics["round_status_code"], ROUND_STATUS_CODES["NO_REPLIES"]
        )

    def test_previous_delta_reference_uses_current_mean_first_round(self):
        instance = CosineAuditStrategy(
            representation_mode=DELTA,
            reference_mode="previous_accepted_delta",
            selection_mode="none",
            attacker_node_id=10,
        )
        instance._round_global_state = {"weight": torch.tensor([0.0, 0.0])}
        instance.aggregate_train(
            1,
            [reply(10, {"weight": torch.tensor([1.0, 0.0])})],
        )
        self.assertEqual(
            instance.round_log[1]["effective_reference"],
            "current_mean_first_round_fallback",
        )
        instance._round_global_state = {"weight": torch.tensor([1.0, 0.0])}
        instance.aggregate_train(
            2,
            [reply(10, {"weight": torch.tensor([2.0, 0.0])})],
        )
        self.assertEqual(
            instance.round_log[2]["effective_reference"], "previous_accepted_delta"
        )

    def test_previous_model_reference_is_distinct_and_has_first_round_fallback(self):
        instance = CosineAuditStrategy(
            representation_mode="legacy_full_model",
            reference_mode="previous_global_model",
            selection_mode="none",
        )
        instance._round_global_state = {"weight": torch.tensor([2.0, 0.0])}
        instance.aggregate_train(1, [reply(10, {"weight": torch.tensor([4.0, 0.0])})])
        self.assertEqual(
            instance.round_log[1]["effective_reference"],
            "current_mean_first_round_fallback",
        )
        instance._round_global_state = {"weight": torch.tensor([4.0, 0.0])}
        instance.aggregate_train(2, [reply(10, {"weight": torch.tensor([6.0, 0.0])})])
        self.assertEqual(
            instance.round_log[2]["effective_reference"], "previous_global_model"
        )
        self.assertEqual(
            instance.round_log[2]["reference_sha256"],
            _vector_sha256(np.array([4.0, 0.0], dtype=np.float32)),
        )

    def test_invalid_only_replies_do_not_become_ordinary_clients(self):
        instance = strategy(selection_mode="none")
        aggregate, metrics = instance.aggregate_train(
            1,
            [reply(10, {"weight": torch.tensor([float("inf"), 0.0])})],
        )
        self.assertIsNone(aggregate)
        self.assertEqual(
            metrics["round_status_code"],
            ROUND_STATUS_CODES["INVALID_INPUTS_NO_AGGREGATION"],
        )
        self.assertEqual(
            instance.round_log[1]["round_status"],
            "INVALID_INPUTS_NO_AGGREGATION",
        )

    def test_invalid_sample_count_is_excluded(self):
        instance = strategy(selection_mode="none")
        aggregate, _ = instance.aggregate_train(
            1,
            [
                reply(10, {"weight": torch.tensor([1.0, 0.0])}, sample_count=0),
                reply(11, {"weight": torch.tensor([2.0, 0.0])}, sample_count=1),
            ],
        )
        self.assertIsNotNone(aggregate)
        self.assertIn(10, instance.round_log[1]["invalid_clients"])
        self.assertNotIn(10, instance.round_log[1]["selected_node_ids"])

    def test_nonfinite_aggregate_is_explicit_and_not_accepted(self):
        class NonfiniteAggregateStrategy(CosineAuditStrategy):
            def _aggregate_selected(self, server_round, selected_replies):
                return (
                    ArrayRecord({"weight": torch.tensor([float("nan"), 0.0])}),
                    MetricRecord(),
                )

        instance = NonfiniteAggregateStrategy(
            representation_mode=DELTA,
            selection_mode="none",
            attacker_node_id=10,
        )
        instance._round_global_state = {"weight": torch.tensor([0.0, 0.0])}
        aggregate, metrics = instance.aggregate_train(
            1,
            [reply(10, {"weight": torch.tensor([1.0, 0.0])})],
        )
        self.assertIsNone(aggregate)
        self.assertEqual(
            metrics["round_status_code"], ROUND_STATUS_CODES["NONFINITE_AGGREGATE"]
        )
        diagnostic = instance.round_log[1]["model_diagnostics"]
        self.assertFalse(diagnostic["is_finite"])
        self.assertEqual(
            diagnostic["nonfinite_tensors"]["weight"]["nonfinite_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
