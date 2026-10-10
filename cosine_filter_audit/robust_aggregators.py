"""Update-space Byzantine-robust aggregation baselines."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


ROBUST_AGGREGATORS = {"krum", "multi_krum", "coordinate_median", "trimmed_mean"}


def aggregate_updates(
    updates: Mapping[int, np.ndarray],
    *,
    method: str,
    byzantine_count: int = 1,
    multi_krum_count: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Aggregate client deltas and return baseline-specific selection details."""
    if method not in ROBUST_AGGREGATORS:
        raise ValueError(f"Unknown robust aggregation method: {method}")
    if byzantine_count < 0:
        raise ValueError("byzantine_count must be non-negative")
    if not updates:
        raise ValueError("Cannot aggregate an empty update set")

    node_ids = sorted(updates)
    matrix = np.stack(
        [np.asarray(updates[node_id], dtype=np.float64) for node_id in node_ids]
    )
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("Updates must be finite same-shaped vectors")
    n_clients = len(node_ids)

    if method in {"krum", "multi_krum"}:
        if n_clients <= 2 * byzantine_count + 2:
            raise ValueError(
                "Krum requires more clients than 2 * byzantine_count + 2"
            )
        squared_norms = np.sum(matrix * matrix, axis=1)
        distances = (
            squared_norms[:, None]
            + squared_norms[None, :]
            - 2.0 * matrix @ matrix.T
        )
        np.maximum(distances, 0.0, out=distances)
        np.fill_diagonal(distances, np.inf)
        neighbor_count = n_clients - byzantine_count - 2
        scores = np.sum(
            np.partition(distances, neighbor_count - 1, axis=1)[
                :, :neighbor_count
            ],
            axis=1,
        )
        ranked = sorted(range(n_clients), key=lambda index: (scores[index], node_ids[index]))
        if method == "krum":
            selected_indices = ranked[:1]
        else:
            selected_count = (
                multi_krum_count
                if multi_krum_count is not None
                else n_clients - byzantine_count
            )
            if selected_count < 1 or selected_count > n_clients:
                raise ValueError("multi_krum_count must be between 1 and client count")
            selected_indices = ranked[:selected_count]
        aggregate = np.mean(matrix[selected_indices], axis=0)
        details = {
            "aggregation_mode": method,
            "selected_node_ids": [node_ids[index] for index in selected_indices],
            "krum_scores": {
                str(node_ids[index]): float(scores[index])
                for index in range(n_clients)
            },
        }
        return aggregate, details

    if method == "coordinate_median":
        aggregate = np.median(matrix, axis=0)
    else:
        if byzantine_count * 2 >= n_clients:
            raise ValueError("Trimmed mean requires 2 * byzantine_count < client count")
        ordered = np.sort(matrix, axis=0)
        aggregate = np.mean(
            ordered[byzantine_count : n_clients - byzantine_count], axis=0
        )
    return aggregate, {
        "aggregation_mode": method,
        "selected_node_ids": node_ids,
    }
