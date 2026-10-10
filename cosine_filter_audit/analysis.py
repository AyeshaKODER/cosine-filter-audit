"""Summarize paired-suite metrics and seed-paired effect estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _bootstrap_interval(values: list[float], seed: int = 0) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = np.asarray(values, dtype=np.float64)
    draws = rng.choice(samples, size=(10_000, len(samples)), replace=True)
    means = draws.mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return [float(lower), float(upper)]


def _paired_effect(differences: list[float]) -> dict[str, float | int | list[float] | None]:
    mean = float(np.mean(differences))
    deviation = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
    return {
        "paired_seed_count": len(differences),
        "mean_difference": mean,
        "bootstrap_95pct_ci": _bootstrap_interval(differences),
        "cohen_dz": (
            mean / deviation if deviation > 0.0 else (0.0 if mean == 0.0 else None)
        ),
    }


def summarize_suite(manifest_path: str | Path) -> dict[str, object]:
    """Summarize per-run quality and paired differences from a suite manifest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("suite_status", "completed") != "completed":
        raise ValueError(
            "Cannot summarize an incomplete suite; resume it or analyze "
            "individual run records explicitly"
        )
    rows: list[dict[str, object]] = []
    for item in manifest["run_files"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        record = json.loads(path.read_text(encoding="utf-8"))
        rounds = record.get("rounds", {})
        final_round = str(record["config"]["num-server-rounds"])
        evaluations = record.get("server_evaluation_metrics", {})
        final_metrics = evaluations.get(final_round) or evaluations.get(
            str(max((int(key) for key in evaluations), default=0))
        ) or {}

        benign_clients = []
        attacker_rows = []
        for round_record in rounds.values():
            clients = round_record.get("clients", {})
            for client in clients.values():
                if client.get("is_attacker") is True:
                    attacker_rows.append(client)
                elif client.get("is_attacker") is False:
                    benign_clients.append(client)
        scored_benign = [client for client in benign_clients if client.get("score") is not None]
        false_rejections = [
            client for client in scored_benign if not client.get("selected", False)
        ]
        attacker_scored = [
            client for client in attacker_rows if client.get("score") is not None
        ]
        attacker_retained = [
            client for client in attacker_scored if client.get("selected", False)
        ]
        config = record["config"]
        selection_is_client_filter = (
            config.get("aggregation_mode", "fedavg") == "fedavg"
        )
        rows.append(
            {
                "seed": int(config["seed"]),
                "regime": item["regime"],
                "attack_mode": item["arm_id"].split("__", 1)[0],
                "arm_id": item["arm_id"],
                "selection_mode": config.get("selection_mode"),
                "aggregation_mode": config.get("aggregation_mode", "fedavg"),
                "benign_false_rejection_rate": (
                    len(false_rejections) / len(scored_benign)
                    if scored_benign and selection_is_client_filter
                    else None
                ),
                "benign_scored_count": (
                    len(scored_benign) if selection_is_client_filter else 0
                ),
                "attacker_detection_rate": (
                    1.0 - len(attacker_retained) / len(attacker_scored)
                    if attacker_scored and selection_is_client_filter
                    else None
                ),
                "attacker_retained_count": (
                    len(attacker_retained) if selection_is_client_filter else 0
                ),
                "attacker_scored_count": (
                    len(attacker_scored) if selection_is_client_filter else 0
                ),
                "final_accuracy": final_metrics.get("accuracy"),
                "final_loss": final_metrics.get("loss"),
                "round_statuses": {
                    key: value.get("round_status") for key, value in rounds.items()
                },
            }
        )

    by_key = {
        (
            row["seed"],
            row["regime"],
            row["attack_mode"],
            row["arm_id"].split("__", 1)[1],
        ): row
        for row in rows
    }
    comparisons = {}
    selection_targets = ("cosine", "multi_signal", "none", "oracle", "random_same_size", "benign_only")
    for regime in manifest["regimes"]:
        for attack_mode in {row["attack_mode"] for row in rows}:
            for target in selection_targets:
                for metric in (
                    "benign_false_rejection_rate",
                    "attacker_detection_rate",
                    "final_accuracy",
                    "final_loss",
                ):
                    differences = []
                    for seed in manifest["seeds"]:
                        candidate = by_key.get((seed, regime, attack_mode, "multi_signal"))
                        baseline = by_key.get((seed, regime, attack_mode, target))
                        if candidate is None or baseline is None:
                            continue
                        left = candidate.get(metric)
                        right = baseline.get(metric)
                        if left is not None and right is not None:
                            differences.append(float(left) - float(right))
                    if differences:
                        comparisons[
                            f"{regime}/{attack_mode}/multi_signal-minus-{target}/{metric}"
                        ] = _paired_effect(differences)

    return {
        "suite_id": manifest["suite_id"],
        "suite_phase": manifest.get("suite_phase"),
        "pairing_scope": manifest.get("pairing_scope"),
        "run_count": len(rows),
        "per_run_metrics": rows,
        "paired_effects": comparisons,
        "interpretation_note": (
            "Bootstrap intervals resample paired seed-level differences. "
            "One-seed suites are descriptive only; these estimates do not "
            "correct for development-set tuning or multiple comparisons."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a paired Flower experiment suite."
    )
    parser.add_argument("suite_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize_suite(args.suite_manifest)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
