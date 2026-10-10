"""Explicit full-model and delta-space cosine-filter experiment strategy."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from cosine_filter_audit.adaptive_attack import ATTACK_STATUS_CODES, PHASE2
from cosine_filter_audit.multi_signal import _geometric_median, score_updates
from cosine_filter_audit.robust_aggregators import (
    ROBUST_AGGREGATORS,
    aggregate_updates,
)
from cosine_filter_audit.vector_ops import (
    DELTA,
    FULL_MODEL,
    clone_state_dict,
    cosine_similarity,
    delta_state_dict,
    flatten_state,
    normalized_sample_weights,
    validate_state_pair,
)

COSINE_THRESHOLD = 0.7
SELECTION_MODES = {
    "none",
    "detector",
    "multi_signal",
    "oracle",
    "random_same_size",
    "benign_only",
}
REFERENCE_MODES = {
    "current_mean",
    "previous_global_model",
    "previous_accepted_delta",
}
ROUND_STATUS_CODES = {
    "NO_REPLIES": 0,
    "INVALID_INPUTS_NO_AGGREGATION": 1,
    "ALL_CLIENTS_FILTERED": 2,
    "NO_SCOREABLE_CLIENTS": 3,
    "NO_AGGREGATE_RETURNED": 4,
    "NONFINITE_AGGREGATE": 5,
    "AGGREGATE_FINITE": 6,
    "INVALID_AGGREGATION_COHORT": 7,
}


def _vector_sha256(vector: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(vector)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _metric_record(status: str, **values: int | float) -> MetricRecord:
    return MetricRecord({"round_status_code": ROUND_STATUS_CODES[status], **values})


def _tensor_diagnostics(
    state: dict[str, torch.Tensor] | None,
) -> dict[str, object] | None:
    if state is None:
        return None
    details: dict[str, object] = {}
    for name, tensor in state.items():
        value = tensor.detach().cpu()
        finite_mask = torch.isfinite(value) if (
            torch.is_floating_point(value) or torch.is_complex(value)
        ) else torch.ones_like(value, dtype=torch.bool)
        if not bool(finite_mask.all()):
            bad = (~finite_mask).nonzero(as_tuple=False)
            details[name] = {
                "shape": list(value.shape),
                "nonfinite_count": int((~finite_mask).sum().item()),
                "first_nonfinite_index": bad[0].tolist() if len(bad) else None,
            }
    try:
        flat = flatten_state(state)
        return {
            "norm": float(np.linalg.norm(flat)),
            "max_abs": float(np.max(np.abs(flat))),
            "is_finite": True,
            "nonfinite_tensors": details,
        }
    except ValueError:
        return {
            "norm": None,
            "max_abs": None,
            "is_finite": False,
            "nonfinite_tensors": details,
        }


class CosineAuditStrategy(FedAvg):
    """Research implementation; it makes no upstream-defense identity claim."""

    def __init__(
        self,
        *args,
        representation_mode: str = DELTA,
        reference_mode: str = "current_mean",
        selection_mode: str = "detector",
        aggregation_mode: str = "fedavg",
        byzantine_count: int = 1,
        multi_krum_count: int | None = None,
        threshold: float = COSINE_THRESHOLD,
        multi_signal_threshold: float = 3.5,
        adaptive_attack_mode: str = "none",
        adaptive_attack_steps: int = 3,
        adaptive_attack_step_size: float = 0.01,
        partitioning_mode: str = "iid",
        dirichlet_alpha: float = 0.3,
        partition_seed: int = 42,
        batch_size: int = 32,
        experiment_arm_id: str = "default",
        attack_start_round: int = 999,
        attack_scale: float = 1.0,
        attacker_node_id: int | None = None,
        malicious_partition_id: int = 0,
        partition_mapping_probe: bool = False,
        seed: int = 0,
        deterministic: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if representation_mode not in {FULL_MODEL, DELTA}:
            raise ValueError(f"Unknown representation mode: {representation_mode}")
        if reference_mode not in REFERENCE_MODES:
            raise ValueError(f"Unknown reference mode: {reference_mode}")
        if selection_mode not in SELECTION_MODES:
            raise ValueError(f"Unknown selection mode: {selection_mode}")
        if aggregation_mode != "fedavg" and aggregation_mode not in ROBUST_AGGREGATORS:
            raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")
        if byzantine_count < 0:
            raise ValueError("byzantine_count must be non-negative")
        if multi_krum_count is not None and multi_krum_count < 1:
            raise ValueError("multi_krum_count must be positive")
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("Cosine threshold must be finite and within [-1, 1]")
        if not math.isfinite(multi_signal_threshold) or multi_signal_threshold < 0.0:
            raise ValueError("Multi-signal threshold must be finite and non-negative")
        if not math.isfinite(attack_scale):
            raise ValueError("Attack scale must be finite")
        if representation_mode == FULL_MODEL and reference_mode == "previous_accepted_delta":
            raise ValueError("previous_accepted_delta requires delta representation")
        if representation_mode == DELTA and reference_mode == "previous_global_model":
            raise ValueError("previous_global_model requires legacy full-model representation")
        if selection_mode == "multi_signal" and representation_mode != DELTA:
            raise ValueError("multi_signal selection requires delta representation")
        if adaptive_attack_mode not in {"none", "phase1_own_history", PHASE2}:
            raise ValueError(f"Unknown adaptive attack mode: {adaptive_attack_mode}")
        if adaptive_attack_mode != "none" and representation_mode != DELTA:
            raise ValueError("Adaptive attacks require delta representation")
        if (
            adaptive_attack_steps < 1
            or not math.isfinite(adaptive_attack_step_size)
            or adaptive_attack_step_size <= 0
        ):
            raise ValueError("Adaptive attack steps and step size must be positive")
        if partitioning_mode not in {"iid", "dirichlet"}:
            raise ValueError(f"Unknown partitioning mode: {partitioning_mode}")
        if not math.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0:
            raise ValueError("Dirichlet alpha must be finite and positive")
        if partition_seed < 0 or batch_size < 1:
            raise ValueError("Partition seed and batch size are invalid")

        self.representation_mode = representation_mode
        self.reference_mode = reference_mode
        self.selection_mode = selection_mode
        self.aggregation_mode = aggregation_mode
        self.byzantine_count = int(byzantine_count)
        self.multi_krum_count = multi_krum_count
        self.threshold = float(threshold)
        self.multi_signal_threshold = float(multi_signal_threshold)
        self.adaptive_attack_mode = adaptive_attack_mode
        self.adaptive_attack_steps = int(adaptive_attack_steps)
        self.adaptive_attack_step_size = float(adaptive_attack_step_size)
        self.attack_start_round = int(attack_start_round)
        self.attack_scale = float(attack_scale)
        self.attacker_node_id = attacker_node_id
        self.malicious_partition_id = int(malicious_partition_id)
        self.partition_mapping_probe = bool(partition_mapping_probe)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.round_log: dict[int, dict[str, object]] = {}
        self.registered_node_ids: list[int] = []
        self._round_global_state: dict[str, torch.Tensor] | None = None
        self._previous_global_state: dict[str, torch.Tensor] | None = None
        self._previous_accepted_delta: dict[str, torch.Tensor] | None = None
        self._client_update_history: dict[int, np.ndarray] = {}
        self._client_score_history: dict[int, list[float]] = {}
        self._active_grid: Grid | None = None
        self._partitioning_mode = partitioning_mode
        self._dirichlet_alpha = float(dirichlet_alpha)
        self._partition_seed = int(partition_seed)
        self._batch_size = int(batch_size)
        self.experiment_arm_id = experiment_arm_id

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        self._round_global_state = clone_state_dict(arrays.to_torch_state_dict())
        self._active_grid = grid
        node_ids = sorted(grid.get_node_ids())
        self.registered_node_ids = node_ids
        if self.representation_mode == DELTA:
            if not node_ids:
                raise RuntimeError("Cannot designate attacker: Flower Grid has no nodes")
            if self.attacker_node_id is None:
                self.attacker_node_id = node_ids[0]
            if self.attacker_node_id not in node_ids:
                raise ValueError(
                    f"Configured attacker node {self.attacker_node_id} is not registered"
                )
        config["representation_mode"] = self.representation_mode
        config["reference_mode"] = self.reference_mode
        config["attack_start_round"] = self.attack_start_round
        config["attack_scale"] = self.attack_scale
        config["attacker_node_id"] = (
            int(self.attacker_node_id) if self.attacker_node_id is not None else -1
        )
        config["malicious_partition_id"] = self.malicious_partition_id
        config["server_round"] = server_round
        config["seed"] = self.seed
        config["deterministic"] = self.deterministic
        config["adaptive_attack_mode"] = self.adaptive_attack_mode
        config["aggregation_mode"] = self.aggregation_mode
        config["experiment_arm_id"] = self.experiment_arm_id
        config["adaptive_attack_steps"] = self.adaptive_attack_steps
        config["adaptive_attack_step_size"] = self.adaptive_attack_step_size
        config["multi_signal_threshold"] = self.multi_signal_threshold
        config["partitioning_mode"] = self._partitioning_mode
        config["dirichlet_alpha"] = self._dirichlet_alpha
        config["partition_seed"] = self._partition_seed
        config["batch_size"] = self._batch_size
        config["partition_mapping_probe"] = self.partition_mapping_probe
        return super().configure_train(server_round, arrays, config, grid)

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        config["partitioning_mode"] = self._partitioning_mode
        config["dirichlet_alpha"] = self._dirichlet_alpha
        config["partition_seed"] = self._partition_seed
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        if self._round_global_state is None:
            raise RuntimeError("configure_train must set the round's global model first")
        if not replies:
            self.round_log[server_round] = {"round_status": "NO_REPLIES"}
            return None, _metric_record("NO_REPLIES")

        valid: dict[int, tuple[Message, np.ndarray, int]] = {}
        client_log: dict[int, dict[str, object]] = {}
        invalid_clients: dict[int, str] = {}
        for reply in replies:
            node_id = int(reply.metadata.src_node_id)
            if not reply.has_content():
                invalid_clients[node_id] = "NO_CONTENT"
                client_log[node_id] = {
                    "is_attacker": (
                        node_id == self.attacker_node_id
                        if self.representation_mode == DELTA
                        else None
                    ),
                    "score": None,
                    "vector_sha256": None,
                    "flagged": None,
                    "selected": False,
                    "sample_count": None,
                    "partition_id": None,
                    "status": "NO_CONTENT",
                }
                continue
            try:
                state = reply.content["arrays"].to_torch_state_dict()
                validate_state_pair(state, self._round_global_state)
                metrics = reply.content["metrics"]
                count = metrics["num-examples"]
                if (
                    isinstance(count, (bool, np.bool_))
                    or not isinstance(count, (int, np.integer))
                    or count <= 0
                ):
                    raise ValueError("Invalid num-examples")
                partition_id = metrics.get("partition-id")
                if partition_id is not None and (
                    isinstance(partition_id, (bool, np.bool_))
                    or not isinstance(partition_id, (int, np.integer))
                    or partition_id < 0
                ):
                    raise ValueError("Invalid partition-id")
                if self.representation_mode == DELTA:
                    vector_state = delta_state_dict(state, self._round_global_state)
                else:
                    vector_state = state
                vector = flatten_state(
                    vector_state, floating_only=self.representation_mode == DELTA
                )
                valid[node_id] = (reply, vector, int(count))
                client_log[node_id] = {
                    "is_attacker": (
                        node_id == self.attacker_node_id
                        if self.representation_mode == DELTA
                        else None
                    ),
                    "score": None,
                    "vector_sha256": _vector_sha256(vector),
                    "flagged": None,
                    "selected": False,
                    "sample_count": int(count),
                    "partition_id": (
                        int(partition_id) if partition_id is not None else None
                    ),
                    "status": "PENDING_SCORE",
                }
                if node_id == self.attacker_node_id:
                    client_log[node_id]["adaptive_attack"] = {
                        "status_code": int(
                            metrics.get("attack-status-code", 0)
                        ),
                        "initial_score": float(
                            metrics.get("attack-initial-score", 0.0)
                        ),
                        "final_score": float(metrics.get("attack-final-score", 0.0)),
                        "initial_loss": float(
                            metrics.get("attack-initial-loss", 0.0)
                        ),
                        "final_loss": float(metrics.get("attack-final-loss", 0.0)),
                        "steps_applied": int(
                            metrics.get("attack-steps-applied", 0)
                        ),
                    }
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                invalid_clients[node_id] = f"INVALID_INPUT: {exc}"
                client_log[node_id] = {
                    "is_attacker": (
                        node_id == self.attacker_node_id
                        if self.representation_mode == DELTA
                        else None
                    ),
                    "score": None,
                    "flagged": None,
                    "selected": False,
                    "sample_count": None,
                    "partition_id": None,
                    "status": invalid_clients[node_id],
                }

        if not valid:
            status = "INVALID_INPUTS_NO_AGGREGATION"
            self.round_log[server_round] = {
                "round_status": status,
                "clients": client_log,
                "invalid_clients": invalid_clients,
                "selected_node_ids": [],
            }
            return None, _metric_record(status)

        vectors = {
            node_id: valid[node_id][1] for node_id in sorted(valid)
        }
        if (
            self.adaptive_attack_mode == PHASE2
            and self.attacker_node_id in valid
            and server_round >= self.attack_start_round
        ):
            refined = self._refine_whitebox_update(server_round, valid, vectors)
            if refined is not None:
                refined_reply, refined_vector, attack_metrics = refined
                count = valid[self.attacker_node_id][2]
                valid[self.attacker_node_id] = (
                    refined_reply,
                    refined_vector,
                    count,
                )
                vectors[self.attacker_node_id] = refined_vector
                client_log[self.attacker_node_id]["adaptive_attack"] = attack_metrics
                client_log[self.attacker_node_id]["vector_sha256"] = _vector_sha256(
                    refined_vector
                )
            else:
                client_log[self.attacker_node_id]["adaptive_attack"] = {
                    "status_code": ATTACK_STATUS_CODES["REFINEMENT_NO_REPLY"],
                    "initial_score": None,
                    "final_score": None,
                    "initial_loss": None,
                    "final_loss": None,
                    "steps_applied": 0,
                }

        multi_signal_scores = None
        if self.selection_mode == "multi_signal":
            multi_signal_scores = score_updates(
                vectors,
                previous_updates=self._client_update_history,
                score_history=self._client_score_history,
                threshold=self.multi_signal_threshold,
            )
            reference = _geometric_median(list(vectors.values()))
            effective_reference = "current_geometric_median"
        elif self.reference_mode == "current_mean":
            reference = np.mean(list(vectors.values()), axis=0)
            effective_reference = "current_mean"
        elif self.reference_mode == "previous_global_model":
            if self._previous_global_state is None:
                reference = np.mean(list(vectors.values()), axis=0)
                effective_reference = "current_mean_first_round_fallback"
            else:
                reference = flatten_state(self._previous_global_state)
                effective_reference = "previous_global_model"
        else:
            if self._previous_accepted_delta is None:
                reference = np.mean(list(vectors.values()), axis=0)
                effective_reference = "current_mean_first_round_fallback"
            else:
                reference = flatten_state(
                    self._previous_accepted_delta, floating_only=True
                )
                effective_reference = "previous_accepted_delta"

        invalid_scores: set[int] = set()
        flagged: set[int] = set()
        for node_id, vector in vectors.items():
            if multi_signal_scores is not None:
                signal_result = multi_signal_scores[node_id]
                score = float(signal_result["score"])
                is_flagged = bool(signal_result["flagged"])
                client_log[node_id]["score_components"] = {
                    key: signal_result[key]
                    for key in (
                        "direction_z",
                        "norm_abs_z",
                        "temporal_z",
                        "history_z",
                    )
                }
            else:
                try:
                    score = cosine_similarity(vector, reference)
                except ValueError as exc:
                    invalid_scores.add(node_id)
                    invalid_clients[node_id] = f"INVALID_SCORE: {exc}"
                    client_log[node_id].update(status=invalid_clients[node_id])
                    continue
                is_flagged = score < self.threshold
            if is_flagged:
                flagged.add(node_id)
            client_log[node_id].update(
                score=score,
                vector_sha256=_vector_sha256(vector),
                flagged=is_flagged,
                status="SCORED",
            )

        if multi_signal_scores is not None:
            for node_id, vector in vectors.items():
                self._client_update_history[node_id] = vector.copy()
                self._client_score_history.setdefault(node_id, []).append(
                    float(multi_signal_scores[node_id]["score"])
                )

        scoreable = set(vectors) - invalid_scores
        if self.selection_mode == "none":
            selected = set(vectors)
        elif self.selection_mode in {"detector", "multi_signal"}:
            selected = scoreable - flagged
        elif self.selection_mode in {"oracle", "benign_only"}:
            if self.representation_mode != DELTA or self.attacker_node_id is None:
                raise ValueError(
                    "Oracle selection requires delta mode and a server-designated attacker"
                )
            selected = set(vectors) - {self.attacker_node_id}
        else:
            k = min(len(flagged), len(vectors))
            rng = random.Random(self.seed + server_round)
            excluded = set(rng.sample(sorted(vectors), k))
            selected = set(vectors) - excluded

        for node_id in selected:
            client_log[node_id]["selected"] = True
        if not selected:
            status = "ALL_CLIENTS_FILTERED" if scoreable else "NO_SCOREABLE_CLIENTS"
            self.round_log[server_round] = {
                "round_status": status,
                "representation_mode": self.representation_mode,
                "reference_mode": self.reference_mode,
                "effective_reference": effective_reference,
                "reference_dimension": int(reference.size),
                "reference_sha256": _vector_sha256(reference),
                "reference_norm": float(np.linalg.norm(reference)),
                "selection_mode": self.selection_mode,
                "aggregation_mode": self.aggregation_mode,
                "multi_signal_threshold": self.multi_signal_threshold,
                "adaptive_attack_mode": self.adaptive_attack_mode,
                "clients": client_log,
                "invalid_clients": invalid_clients,
                "selected_node_ids": [],
            }
            return None, _metric_record(status, n_flagged=len(flagged))

        selected_ids = sorted(selected)
        selected_replies = [valid[node_id][0] for node_id in selected_ids]
        counts = [valid[node_id][2] for node_id in selected_ids]
        weights = normalized_sample_weights(counts)
        if self.aggregation_mode == "fedavg":
            for node_id, weight in zip(selected_ids, weights, strict=True):
                client_log[node_id]["aggregation_weight"] = weight
        else:
            for node_id in selected_ids:
                client_log[node_id]["aggregation_weight"] = None

        if self.aggregation_mode == "fedavg":
            aggregated, aggregated_metrics = self._aggregate_selected(
                server_round, selected_replies
            )
            aggregation_details = {"aggregation_mode": "fedavg"}
        else:
            try:
                aggregate_vector, aggregation_details = aggregate_updates(
                    {node_id: vectors[node_id] for node_id in selected_ids},
                    method=self.aggregation_mode,
                    byzantine_count=self.byzantine_count,
                    multi_krum_count=self.multi_krum_count,
                )
            except ValueError as exc:
                status = "INVALID_AGGREGATION_COHORT"
                self._record_round(
                    server_round, status, effective_reference, reference,
                    client_log, invalid_clients, selected_ids, None, None
                )
                self.round_log[server_round]["aggregation_error"] = str(exc)
                return None, _metric_record(status)
            aggregated = ArrayRecord(
                self._state_from_delta_vector(aggregate_vector)
            )
            aggregated_metrics = self._average_train_metrics(selected_replies)
            aggregation_details["input_node_ids"] = selected_ids

        if aggregated is None:
            status = "NO_AGGREGATE_RETURNED"
            self._record_round(
                server_round, status, effective_reference, reference, client_log,
                invalid_clients, selected_ids, None, None
            )
            self.round_log[server_round]["aggregation"] = aggregation_details
            return None, _metric_record(status, n_flagged=len(flagged))

        aggregated_state = aggregated.to_torch_state_dict()
        model_diagnostics = _tensor_diagnostics(aggregated_state)
        update_state = {
            name: aggregated_state[name] - self._round_global_state[name]
            for name in self._round_global_state
            if torch.is_floating_point(self._round_global_state[name])
        }
        update_diagnostics = _tensor_diagnostics(update_state)
        if model_diagnostics is None or not model_diagnostics["is_finite"]:
            status = "NONFINITE_AGGREGATE"
            accepted_arrays = None
        else:
            status = "AGGREGATE_FINITE"
            accepted_arrays = aggregated
            self._previous_global_state = clone_state_dict(aggregated_state)
            if self.representation_mode == DELTA:
                self._previous_accepted_delta = delta_state_dict(
                    aggregated_state, self._round_global_state
                )

        self._record_round(
            server_round, status, effective_reference, reference, client_log,
            invalid_clients, selected_ids, model_diagnostics, update_diagnostics
        )
        self.round_log[server_round]["aggregation"] = aggregation_details
        if aggregated_metrics is None:
            aggregated_metrics = MetricRecord()
        aggregated_metrics["round_status_code"] = ROUND_STATUS_CODES[status]
        aggregated_metrics["n_flagged"] = len(flagged)
        aggregated_metrics["n_selected"] = len(selected_ids)
        return accepted_arrays, aggregated_metrics

    def _state_from_delta_vector(self, vector: np.ndarray) -> dict[str, torch.Tensor]:
        if self._round_global_state is None:
            raise RuntimeError("Round global model is unavailable")
        state = clone_state_dict(self._round_global_state)
        offset = 0
        for name, tensor in state.items():
            if not torch.is_floating_point(tensor):
                continue
            count = tensor.numel()
            delta = torch.from_numpy(
                np.asarray(vector[offset : offset + count]).copy()
            ).reshape(tensor.shape).to(dtype=tensor.dtype, device=tensor.device)
            state[name] = tensor + delta
            offset += count
        if offset != vector.size:
            raise ValueError("Robust aggregate vector has the wrong dimension")
        return state

    @staticmethod
    def _average_train_metrics(replies: list[Message]) -> MetricRecord:
        losses = [
            float(reply.content["metrics"]["train_loss"])
            for reply in replies
            if reply.has_content() and "train_loss" in reply.content["metrics"]
        ]
        return MetricRecord(
            {"train_loss": float(np.mean(losses)) if losses else 0.0}
        )

    def _refine_whitebox_update(
        self,
        server_round: int,
        valid: dict[int, tuple[Message, np.ndarray, int]],
        vectors: dict[int, np.ndarray],
    ) -> tuple[Message, np.ndarray, dict[str, float | int]] | None:
        """Give the designated attacker the cohort updates for exact refinement."""
        if self._active_grid is None or self.attacker_node_id is None:
            raise RuntimeError("White-box refinement requires an active Flower Grid")
        attacker = valid[self.attacker_node_id][0]
        clean_state = attacker.content["arrays"].to_torch_state_dict()
        payload: dict[str, torch.Tensor] = {}
        for name, tensor in self._round_global_state.items():
            payload[f"global__{name}"] = tensor.detach().cpu()
        for name, tensor in clean_state.items():
            payload[f"candidate__{name}"] = tensor.detach().cpu()
        for node_id, vector in vectors.items():
            if node_id != self.attacker_node_id:
                payload[f"peer__{node_id}"] = torch.from_numpy(
                    np.asarray(vector, dtype=np.float64).copy()
                )
        for node_id, vector in self._client_update_history.items():
            payload[f"prior__{node_id}"] = torch.from_numpy(vector.copy())
        for node_id, scores in self._client_score_history.items():
            if scores:
                payload[f"scores__{node_id}"] = torch.tensor(
                    scores, dtype=torch.float64
                )

        config = ConfigRecord(
            {
                "adaptive_attack_phase": "whitebox_refine",
                "server_round": server_round,
                "seed": self.seed,
                "deterministic": self.deterministic,
                "multi_signal_threshold": self.multi_signal_threshold,
                "adaptive_attack_steps": self.adaptive_attack_steps,
                "adaptive_attack_step_size": self.adaptive_attack_step_size,
                "partitioning_mode": self._partitioning_mode,
                "dirichlet_alpha": self._dirichlet_alpha,
                "partition_seed": self._partition_seed,
                "batch_size": self._batch_size,
                "experiment_arm_id": self.experiment_arm_id,
                "train_loss": float(
                    attacker.content["metrics"].get("train_loss", 0.0)
                ),
            }
        )
        request = Message(
            content=RecordDict(
                {
                    "arrays": ArrayRecord(payload),
                    "config": config,
                }
            ),
            message_type="train",
            dst_node_id=self.attacker_node_id,
            group_id=f"round-{server_round}-adaptive-refinement",
        )
        replies = list(self._active_grid.send_and_receive([request]))
        if not replies or not replies[0].has_content():
            return None
        reply = replies[0]
        if int(reply.metadata.src_node_id) != self.attacker_node_id:
            raise ValueError("White-box refinement reply came from the wrong client")
        state = reply.content["arrays"].to_torch_state_dict()
        validate_state_pair(state, self._round_global_state)
        vector = flatten_state(
            delta_state_dict(state, self._round_global_state),
            floating_only=True,
        )
        metrics = reply.content["metrics"]
        count = metrics["num-examples"]
        if (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or count != valid[self.attacker_node_id][2]
        ):
            raise ValueError("White-box refinement changed the client's sample count")
        original_partition = valid[self.attacker_node_id][0].content["metrics"].get(
            "partition-id"
        )
        if metrics.get("partition-id") != original_partition:
            raise ValueError("White-box refinement changed the client partition")
        attack_metrics = {
            "status_code": int(metrics["attack-status-code"]),
            "initial_score": float(metrics["attack-initial-score"]),
            "final_score": float(metrics["attack-final-score"]),
            "initial_loss": float(metrics["attack-initial-loss"]),
            "final_loss": float(metrics["attack-final-loss"]),
            "steps_applied": int(metrics["attack-steps-applied"]),
        }
        return reply, vector, attack_metrics

    def _aggregate_selected(
        self,
        server_round: int,
        selected_replies: list[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        return super().aggregate_train(server_round, selected_replies)

    def _record_round(
        self,
        server_round: int,
        status: str,
        effective_reference: str,
        reference: np.ndarray,
        client_log: dict[int, dict[str, object]],
        invalid_clients: dict[int, str],
        selected_ids: list[int],
        model_diagnostics: dict[str, object] | None,
        update_diagnostics: dict[str, object] | None,
    ) -> None:
        self.round_log[server_round] = {
            "round_status": status,
            "representation_mode": self.representation_mode,
            "reference_mode": self.reference_mode,
            "effective_reference": effective_reference,
            "reference_dimension": int(reference.size),
            "reference_sha256": _vector_sha256(reference),
            "reference_norm": float(np.linalg.norm(reference)),
            "selection_mode": self.selection_mode,
            "aggregation_mode": self.aggregation_mode,
            "multi_signal_threshold": self.multi_signal_threshold,
            "adaptive_attack_mode": self.adaptive_attack_mode,
            "attacker_node_id": self.attacker_node_id,
            "clients": client_log,
            "invalid_clients": invalid_clients,
            "selected_node_ids": selected_ids,
            "model_diagnostics": model_diagnostics,
            "update_diagnostics": update_diagnostics,
        }
