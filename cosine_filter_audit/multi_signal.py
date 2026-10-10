"""Rootless, multi-signal scoring candidate for client updates."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _geometric_median(
    vectors: list[np.ndarray], *, tolerance: float = 1e-6, max_iterations: int = 256
) -> np.ndarray:
    """Compute a geometric median with Weiszfeld iterations."""
    points = np.stack(vectors)
    estimate = np.median(points, axis=0)
    for _ in range(max_iterations):
        distances = np.linalg.norm(points - estimate, axis=1)
        if np.any(distances <= tolerance):
            estimate = points[int(np.argmin(distances))].copy()
            break
        weights = 1.0 / distances
        updated = np.average(points, axis=0, weights=weights)
        if np.linalg.norm(updated - estimate) <= tolerance:
            estimate = updated
            break
        estimate = updated
    return estimate


def _robust_z(values: Mapping[int, float]) -> dict[int, float]:
    if not values:
        return {}
    data = np.asarray(list(values.values()), dtype=np.float64)
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    if mad > 1e-12:
        scale = 1.4826 * mad
    else:
        q25, q75 = np.percentile(data, [25, 75])
        scale = float((q75 - q25) / 1.349)
    if scale <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - median) / scale for key, value in values.items()}


def score_updates(
    updates: Mapping[int, np.ndarray],
    *,
    previous_updates: Mapping[int, np.ndarray] | None = None,
    score_history: Mapping[int, list[float]] | None = None,
    threshold: float = 3.5,
) -> dict[int, dict[str, float | bool | None]]:
    """Score updates by direction, norm, temporal drift, and score history.

    The geometric-median reference is computed from the current submitted
    updates. Temporal and history signals are omitted until available and the
    remaining signals are equally weighted.
    """
    if not updates:
        return {}
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Multi-signal threshold must be finite and non-negative")
    ids = list(updates)
    vectors = [np.asarray(updates[node_id], dtype=np.float64) for node_id in ids]
    shape = vectors[0].shape
    if len(shape) != 1 or any(
        vector.shape != shape or not np.all(np.isfinite(vector))
        for vector in vectors
    ):
        raise ValueError("Updates must be finite, same-shaped one-dimensional vectors")
    vector_by_id = dict(zip(ids, vectors, strict=True))

    reference = _geometric_median(vectors)
    direction = {
        node_id: float(np.linalg.norm(vector_by_id[node_id] - reference))
        / max(float(np.sqrt(vector_by_id[node_id].size)), 1.0)
        for node_id in ids
    }
    log_norm = {
        node_id: float(
            np.log(max(float(np.linalg.norm(vector_by_id[node_id])), 1e-12))
        )
        for node_id in ids
    }
    direction_z = _robust_z(direction)
    norm_z = _robust_z(log_norm)

    temporal: dict[int, float] = {}
    for node_id in ids:
        prior = (previous_updates or {}).get(node_id)
        if prior is None:
            continue
        prior = np.asarray(prior, dtype=np.float64)
        if prior.shape != shape or not np.all(np.isfinite(prior)):
            raise ValueError(f"Invalid previous update for client {node_id}")
        current = vector_by_id[node_id]
        denominator = float(np.linalg.norm(current) * np.linalg.norm(prior))
        similarity = (
            float(np.dot(current, prior) / denominator)
            if denominator > 1e-12
            else 0.0
        )
        temporal[node_id] = 1.0 - float(np.clip(similarity, -1.0, 1.0))
    temporal_z = _robust_z(temporal)

    base_values: dict[int, float] = {}
    for node_id in ids:
        components = [max(direction_z[node_id], 0.0), abs(norm_z[node_id])]
        if node_id in temporal_z:
            components.append(max(temporal_z[node_id], 0.0))
        base_values[node_id] = float(np.mean(components))

    history_deviation = {}
    for node_id in ids:
        history = (score_history or {}).get(node_id)
        if not history:
            continue
        history_values = np.asarray(history, dtype=np.float64)
        if not np.all(np.isfinite(history_values)):
            raise ValueError(f"Invalid score history for client {node_id}")
        history_deviation[node_id] = abs(
            base_values[node_id] - float(np.median(history_values))
        )
    history_z = _robust_z(history_deviation)

    results: dict[int, dict[str, float | bool | None]] = {}
    for node_id in ids:
        components = [max(direction_z[node_id], 0.0), abs(norm_z[node_id])]
        temporal_score = None
        if node_id in temporal_z:
            temporal_score = max(temporal_z[node_id], 0.0)
            components.append(temporal_score)
        history_score = None
        if node_id in history_z:
            history_score = max(history_z[node_id], 0.0)
            components.append(history_score)
        score = float(np.mean(components))
        results[node_id] = {
            "direction_z": float(direction_z[node_id]),
            "norm_abs_z": float(abs(norm_z[node_id])),
            "temporal_z": (
                float(temporal_score) if temporal_score is not None else None
            ),
            "history_z": float(history_score) if history_score is not None else None,
            "score": score,
            "flagged": score > threshold,
        }
    return results
