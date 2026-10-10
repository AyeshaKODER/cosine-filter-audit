"""Tensor representations used by the experiment's detector and attacks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


FULL_MODEL = "legacy_full_model"
DELTA = "delta"


def clone_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def validate_state_pair(
    state: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> None:
    if tuple(state) != tuple(reference):
        raise ValueError("State dictionaries have different keys or key order")
    for name, tensor in state.items():
        other = reference[name]
        if tensor.shape != other.shape:
            raise ValueError(f"Tensor shape mismatch for {name}")
        if tensor.dtype != other.dtype:
            raise ValueError(f"Tensor dtype mismatch for {name}")


def delta_state_dict(
    local_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    validate_state_pair(local_state, global_state)
    deltas: dict[str, torch.Tensor] = {}
    for name, local_tensor in local_state.items():
        if torch.is_floating_point(local_tensor):
            deltas[name] = local_tensor - global_state[name]
    if not deltas:
        raise ValueError("No floating-point tensors available for delta representation")
    return deltas


def apply_delta_attack(
    local_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    scale: float,
) -> dict[str, torch.Tensor]:
    """Return global + scale * (local - global) for floating tensors.

    Non-floating tensors are copied from the received global state and are
    never scaled or locally advanced.
    """
    validate_state_pair(local_state, global_state)
    attacked: dict[str, torch.Tensor] = {}
    for name, local_tensor in local_state.items():
        if torch.is_floating_point(local_tensor):
            attacked[name] = global_state[name] + scale * (
                local_tensor - global_state[name]
            )
        else:
            attacked[name] = global_state[name].detach().clone()
    return attacked


def legacy_scale_state_dict(
    state: Mapping[str, torch.Tensor], scale: float
) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() * scale for name, tensor in state.items()}


def flatten_state(
    state: Mapping[str, torch.Tensor],
    *,
    floating_only: bool = False,
) -> np.ndarray:
    tensors = [
        tensor.detach().cpu().numpy().reshape(-1)
        for tensor in state.values()
        if not floating_only or torch.is_floating_point(tensor)
    ]
    if not tensors:
        raise ValueError("Cannot flatten an empty state")
    vector = np.concatenate(tensors)
    if not np.all(np.isfinite(vector)):
        raise ValueError("State contains non-finite values")
    return vector


def cosine_similarity(vector: np.ndarray, reference: np.ndarray) -> float:
    if vector.ndim != 1 or reference.ndim != 1 or vector.shape != reference.shape:
        raise ValueError("Cosine inputs must be same-shaped one-dimensional vectors")
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(reference)):
        raise ValueError("Cosine inputs contain non-finite values")
    denominator = np.linalg.norm(vector) * np.linalg.norm(reference)
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("Cosine input has a zero or near-zero norm")
    score = float(np.dot(vector, reference) / denominator)
    if not np.isfinite(score):
        raise ValueError("Cosine score is non-finite")
    return score


def normalized_sample_weights(sample_counts: list[int]) -> list[float]:
    if not sample_counts or any(
        not isinstance(count, (int, np.integer)) or count <= 0
        for count in sample_counts
    ):
        raise ValueError("Sample counts must be non-empty positive integers")
    total = sum(sample_counts)
    return [count / total for count in sample_counts]
