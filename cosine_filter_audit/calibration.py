"""Development-set cutoff selection utilities."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path

import numpy as np


def select_score_cutoff(
    benign_scores: Iterable[float],
    malicious_scores: Iterable[float],
    *,
    max_benign_false_rejection: float = 0.05,
) -> dict[str, float | int]:
    """Maximize malicious detection under a predeclared benign-FPR limit.

    Scores strictly greater than the returned cutoff are classified as
    malicious, matching the multi-signal detector's decision boundary.
    """
    benign = np.asarray(list(benign_scores), dtype=np.float64)
    malicious = np.asarray(list(malicious_scores), dtype=np.float64)
    if benign.ndim != 1 or malicious.ndim != 1 or not benign.size or not malicious.size:
        raise ValueError("Calibration requires non-empty benign and malicious scores")
    if not np.all(np.isfinite(benign)) or not np.all(np.isfinite(malicious)):
        raise ValueError("Calibration scores must be finite")
    if (
        not np.isfinite(max_benign_false_rejection)
        or not 0.0 <= max_benign_false_rejection <= 1.0
    ):
        raise ValueError("Maximum benign false-rejection rate must be within [0, 1]")

    candidates = np.unique(np.concatenate([benign, malicious]))
    eligible = []
    for cutoff in candidates:
        false_rejection = float(np.mean(benign > cutoff))
        detection = float(np.mean(malicious > cutoff))
        if false_rejection <= max_benign_false_rejection:
            eligible.append((detection, -false_rejection, -float(cutoff), cutoff))
    if not eligible:
        raise ValueError("No cutoff satisfies the benign false-rejection constraint")
    detection, negative_false_rejection, _, cutoff = max(eligible)
    return {
        "threshold": float(cutoff),
        "development_benign_false_rejection_rate": float(-negative_false_rejection),
        "development_malicious_detection_rate": float(detection),
        "max_benign_false_rejection_rate": float(max_benign_false_rejection),
        "benign_score_count": int(benign.size),
        "malicious_score_count": int(malicious.size),
    }


def select_benign_score_cutoff(
    benign_scores: Iterable[float],
    *,
    max_benign_false_rejection: float = 0.05,
) -> dict[str, float | int]:
    """Choose the least restrictive cutoff meeting a benign rejection cap."""
    benign = np.asarray(list(benign_scores), dtype=np.float64)
    if benign.ndim != 1 or not benign.size:
        raise ValueError("Calibration requires non-empty benign scores")
    if not np.all(np.isfinite(benign)):
        raise ValueError("Calibration scores must be finite")
    if (
        not np.isfinite(max_benign_false_rejection)
        or not 0.0 <= max_benign_false_rejection <= 1.0
    ):
        raise ValueError("Maximum benign false-rejection rate must be within [0, 1]")

    candidates = np.unique(benign)
    eligible = [
        (
            float(cutoff),
            float(np.mean(benign > cutoff)),
        )
        for cutoff in candidates
        if np.mean(benign > cutoff) <= max_benign_false_rejection
    ]
    if not eligible:
        raise ValueError("No cutoff satisfies the benign false-rejection constraint")
    cutoff, false_rejection = min(eligible, key=lambda item: item[0])
    return {
        "threshold": cutoff,
        "development_benign_false_rejection_rate": false_rejection,
        "max_benign_false_rejection_rate": float(max_benign_false_rejection),
        "benign_score_count": int(benign.size),
    }


def calibrate_run_records(
    paths: Iterable[str | Path],
    *,
    max_benign_false_rejection: float = 0.05,
) -> dict[str, float | int | str]:
    """Select a cutoff using explicitly labeled development-suite run records."""
    benign_scores: list[float] = []
    suite_ids: set[str] = set()
    development_seeds: set[int] = set()
    development_regimes: set[str] = set()
    included_records = 0
    for path in paths:
        run_path = Path(path)
        record = json.loads(run_path.read_text(encoding="utf-8"))
        if record.get("config", {}).get("suite_phase") != "development":
            raise ValueError(f"{run_path} is not labeled as a development run")
        if record.get("config", {}).get("selection_mode") != "multi_signal":
            continue
        config = record["config"]
        if config.get("attack_mode", "none") != "none":
            continue
        suite_id = record.get("suite_id")
        if not suite_id:
            raise ValueError(f"{run_path} has no suite_id")
        suite_ids.add(str(suite_id))
        if "seed" not in config:
            raise ValueError(f"{run_path} has no development seed")
        development_seeds.add(int(config["seed"]))
        partitioning_mode = config.get("partitioning_mode")
        if partitioning_mode == "iid":
            development_regimes.add("iid")
        elif partitioning_mode == "dirichlet" and config.get("dirichlet_alpha") is not None:
            development_regimes.add(
                f"dirichlet-{float(config['dirichlet_alpha']):.1f}"
            )
        included_records += 1
        for round_record in record.get("rounds", {}).values():
            for client in round_record.get("clients", {}).values():
                score = client.get("score")
                if score is None:
                    continue
                benign_scores.append(float(score))
    if not suite_ids:
        raise ValueError(
            "Calibration requires benign-only multi-signal records from "
            "development suites"
        )
    result = select_benign_score_cutoff(
        benign_scores,
        max_benign_false_rejection=max_benign_false_rejection,
    )
    enough_development_seeds = len(development_seeds) >= 5
    required_regimes = {"iid", "dirichlet-1.0", "dirichlet-0.3", "dirichlet-0.1"}
    complete_regime_coverage = required_regimes.issubset(development_regimes)
    calibration_status = (
        "ready_for_confirmatory"
        if enough_development_seeds and complete_regime_coverage
        else "requires_more_development_data"
    )
    return {
        **result,
        "development_suite_id": (
            next(iter(suite_ids)) if len(suite_ids) == 1 else "multiple"
        ),
        "development_suite_ids": sorted(suite_ids),
        "development_seed_count": len(development_seeds),
        "development_seeds": sorted(development_seeds),
        "development_regimes": sorted(development_regimes),
        "calibration_method": "benign_only_empirical_fpr",
        "included_run_count": included_records,
        "calibration_status": calibration_status,
        "calibration_note": (
            "Threshold was selected only from benign-only multi-signal client "
            "scores. Adaptive attacks must be evaluated on independent seeds "
            "after the threshold is frozen; their scores were not used to "
            "select this threshold. Confirmatory use requires all four "
            "development regimes and at least five development seeds."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a multi-signal cutoff from development run records."
    )
    parser.add_argument("run_records", nargs="+", type=Path)
    parser.add_argument("--max-benign-fpr", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = calibrate_run_records(
        args.run_records,
        max_benign_false_rejection=args.max_benign_fpr,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
