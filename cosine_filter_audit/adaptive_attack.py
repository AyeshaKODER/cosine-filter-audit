"""Local-loss poisoning with approximate and cohort-aware score constraints."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch
from torch import nn

from cosine_filter_audit.multi_signal import score_updates
from cosine_filter_audit.vector_ops import flatten_state


PHASE1 = "phase1_own_history"
PHASE2 = "phase2_cohort_aware"

ATTACK_STATUS_CODES = {
    "NOT_RUN": 0,
    "CONSTRAINED": 1,
    "NO_FEASIBLE_BASE": 2,
    "NO_BATCHES": 3,
    "NONFINITE_OBJECTIVE": 4,
    "REFINEMENT_NO_REPLY": 5,
}


def approximate_score(
    candidate: np.ndarray,
    clean_update: np.ndarray,
    previous_updates: list[np.ndarray],
    previous_scores: list[float],
) -> float:
    """Approximate the cohort score using only this client's update history."""
    candidate = np.asarray(candidate, dtype=np.float64)
    clean_update = np.asarray(clean_update, dtype=np.float64)
    if (
        candidate.ndim != 1
        or candidate.shape != clean_update.shape
        or not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(clean_update))
    ):
        raise ValueError("Candidate and clean update must be finite matching vectors")

    prior = (
        np.asarray(previous_updates[-1], dtype=np.float64)
        if previous_updates
        else clean_update
    )
    if prior.shape != candidate.shape or not np.all(np.isfinite(prior)):
        raise ValueError("Previous update must be finite and match candidate shape")

    candidate_norm = float(np.linalg.norm(candidate))
    clean_norm = float(np.linalg.norm(clean_update))
    relative_displacement = float(
        np.linalg.norm(candidate - clean_update) / max(clean_norm, 1e-12)
    )
    log_norm_deviation = abs(
        float(np.log(max(candidate_norm, 1e-12)))
        - float(np.log(max(clean_norm, 1e-12)))
    )
    prior_norm = float(np.linalg.norm(prior))
    denominator = candidate_norm * prior_norm
    temporal_drift = (
        1.0 - float(np.clip(np.dot(candidate, prior) / denominator, -1.0, 1.0))
        if denominator > 1e-12
        else 0.0
    )
    components = [relative_displacement, log_norm_deviation, temporal_drift]
    if previous_scores:
        past = np.asarray(previous_scores, dtype=np.float64)
        if not np.all(np.isfinite(past)):
            raise ValueError("Previous scores must be finite")
        median = float(np.median(past))
        mad = float(np.median(np.abs(past - median)))
        scale = max(1.4826 * mad, 0.1)
        components.append(abs(float(np.mean(components)) - median) / scale)
    return float(np.mean(components))


def cohort_score(
    candidate: np.ndarray,
    *,
    client_id: int,
    peer_updates: Mapping[int, np.ndarray],
    previous_updates: Mapping[int, np.ndarray],
    score_history: Mapping[int, list[float]],
    threshold: float,
) -> float:
    """Evaluate the detector's exact current-cohort score for one candidate."""
    updates = dict(peer_updates)
    updates[client_id] = np.asarray(candidate, dtype=np.float64)
    result = score_updates(
        updates,
        previous_updates=previous_updates,
        score_history=score_history,
        threshold=threshold,
    )
    return float(result[client_id]["score"])


def _blend_states(
    base: Mapping[str, torch.Tensor],
    proposed: Mapping[str, torch.Tensor],
    fraction: float,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, base_tensor in base.items():
        if torch.is_floating_point(base_tensor):
            result[name] = base_tensor + fraction * (
                proposed[name] - base_tensor
            )
        else:
            result[name] = base_tensor.detach().clone()
    return result


def _delta_vector(
    state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
) -> np.ndarray:
    return flatten_state(
        {
            name: state[name].detach().cpu() - global_state[name].detach().cpu()
            for name in state
            if torch.is_floating_point(state[name])
        },
        floating_only=True,
    )


def constrained_local_loss_ascent(
    model: nn.Module,
    global_state: Mapping[str, torch.Tensor],
    clean_state: Mapping[str, torch.Tensor],
    batches,
    device: torch.device,
    *,
    threshold: float,
    score_candidate: Callable[[np.ndarray], float],
    steps: int,
    step_size: float,
) -> dict[str, float | int]:
    """Maximize local cross-entropy while projecting into the score constraint."""
    if steps < 1 or step_size <= 0.0 or not np.isfinite(step_size):
        raise ValueError("Attack steps and step size must be positive")
    model.to(device)
    model.load_state_dict(clean_state)
    criterion = nn.CrossEntropyLoss()
    base_state = {
        name: tensor.detach().clone()
        for name, tensor in clean_state.items()
    }
    base_vector = _delta_vector(base_state, global_state)
    base_score = score_candidate(base_vector)
    if not np.isfinite(base_score):
        raise ValueError("Base update has a non-finite detector score")
    if base_score > threshold:
        return {
            "status_code": ATTACK_STATUS_CODES["NO_FEASIBLE_BASE"],
            "initial_score": base_score,
            "final_score": base_score,
            "initial_loss": float("nan"),
            "final_loss": float("nan"),
            "steps_applied": 0,
        }

    first_batch = None
    initial_loss = float("nan")
    last_score = base_score
    steps_applied = 0
    for batch_index, batch in enumerate(batches):
        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        if first_batch is None:
            first_batch = (images.detach().clone(), labels.detach().clone())
            model.eval()
            with torch.no_grad():
                initial_loss = float(criterion(model(images), labels).item())
            model.train()
            if not np.isfinite(initial_loss):
                return {
                    "status_code": ATTACK_STATUS_CODES["NONFINITE_OBJECTIVE"],
                    "initial_score": base_score,
                    "final_score": last_score,
                    "initial_loss": 0.0,
                    "final_loss": 0.0,
                    "steps_applied": steps_applied,
                }
        if steps_applied >= steps:
            break

        model.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        if not bool(torch.isfinite(loss)):
            return {
                "status_code": ATTACK_STATUS_CODES["NONFINITE_OBJECTIVE"],
                "initial_score": base_score,
                "final_score": last_score,
                "initial_loss": initial_loss,
                "final_loss": float("nan"),
                "steps_applied": steps_applied,
            }
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.add_(step_size * parameter.grad)
        proposed = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        proposed_vector = _delta_vector(proposed, global_state)
        proposed_score = score_candidate(proposed_vector)
        if not np.isfinite(proposed_score):
            raise ValueError("Proposed update has a non-finite detector score")

        if proposed_score <= threshold:
            accepted = proposed
            last_score = proposed_score
        else:
            accepted = base_state
            last_score = base_score
            fraction = 0.5
            for _ in range(32):
                trial = _blend_states(base_state, proposed, fraction)
                trial_vector = _delta_vector(trial, global_state)
                trial_score = score_candidate(trial_vector)
                if np.isfinite(trial_score) and trial_score <= threshold:
                    accepted = trial
                    last_score = trial_score
                    break
                fraction *= 0.5
        model.load_state_dict(accepted)
        steps_applied += 1

    final_loss = float("nan")
    if first_batch is not None:
        model.eval()
        with torch.no_grad():
            final_loss = float(criterion(model(first_batch[0]), first_batch[1]).item())
        if not np.isfinite(final_loss):
            return {
                "status_code": ATTACK_STATUS_CODES["NONFINITE_OBJECTIVE"],
                "initial_score": base_score,
                "final_score": last_score,
                "initial_loss": initial_loss,
                "final_loss": 0.0,
                "steps_applied": steps_applied,
            }
    else:
        return {
            "status_code": ATTACK_STATUS_CODES["NO_BATCHES"],
            "initial_score": base_score,
            "final_score": last_score,
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "steps_applied": steps_applied,
        }
    return {
        "status_code": ATTACK_STATUS_CODES["CONSTRAINED"],
        "initial_score": base_score,
        "final_score": last_score,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "steps_applied": steps_applied,
    }
