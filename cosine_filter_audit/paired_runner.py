"""Within-run paired comparison suite for selection and aggregation arms."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid

from cosine_filter_audit.adaptive_attack import PHASE1, PHASE2
from cosine_filter_audit.experiment_strategy import CosineAuditStrategy
from cosine_filter_audit.robust_aggregators import ROBUST_AGGREGATORS
from cosine_filter_audit.reproducibility import seed_everything
from cosine_filter_audit.vector_ops import DELTA


SELECTION_ARMS = {
    "none": "none",
    "cosine": "detector",
    "multi_signal": "multi_signal",
    "oracle": "oracle",
    "random_same_size": "random_same_size",
    "benign_only": "benign_only",
}


def parse_csv_values(value: str, *, name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one value")
    if len(items) != len(set(items)):
        raise ValueError(f"{name} contains duplicate values")
    return items


def build_suite_arms(
    selection_arms: list[str],
    aggregation_arms: list[str],
    attack_modes: list[str],
) -> list[dict[str, str]]:
    """Validate and expand the requested suite into explicit paired arms."""
    unknown_selections = set(selection_arms) - set(SELECTION_ARMS)
    unknown_aggregators = set(aggregation_arms) - set(ROBUST_AGGREGATORS)
    unknown_attacks = set(attack_modes) - {"none", PHASE1, PHASE2}
    if unknown_selections or unknown_aggregators or unknown_attacks:
        raise ValueError(
            "Unknown suite arm(s): "
            f"selection={sorted(unknown_selections)}, "
            f"aggregation={sorted(unknown_aggregators)}, "
            f"attack={sorted(unknown_attacks)}"
        )
    if not selection_arms and not aggregation_arms:
        raise ValueError("A suite must include at least one selection or aggregation arm")
    arms = []
    for attack_mode in attack_modes:
        for name in selection_arms:
            arms.append(
                {
                    "arm_id": f"{attack_mode}__{name}",
                    "selection_mode": SELECTION_ARMS[name],
                    "aggregation_mode": "fedavg",
                    "attack_mode": attack_mode,
                }
            )
        for name in aggregation_arms:
            arms.append(
                {
                    "arm_id": f"{attack_mode}__{name}",
                    "selection_mode": "none",
                    "aggregation_mode": name,
                    "attack_mode": attack_mode,
                }
            )
    return arms


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _git_provenance(project_root: Path) -> tuple[str | None, str, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, "Git revision unavailable", None
    clean = not status.strip()
    note = "Clean committed Git revision" if clean else "Working tree has uncommitted changes"
    return revision, note, clean


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(payload, output, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _suite_signature(
    run_config,
    *,
    seeds: list[int],
    regimes: list[str],
    arms: list[dict[str, str]],
    calibration_sha256: str | None,
) -> str:
    configuration = {
        str(key): _json_safe(value) for key, value in run_config.items()
    }
    for transient_key in (
        "run-output-dir",
        "suite-calibration-source",
        "suite-resume-manifest",
    ):
        configuration.pop(transient_key, None)
    payload = {
        "config": configuration,
        "seeds": seeds,
        "regimes": regimes,
        "arms": arms,
        "calibration_sha256": calibration_sha256,
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _source_fingerprint(project_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        [*project_root.joinpath("cosine_filter_audit").rglob("*.py"),
         project_root / "pyproject.toml"]
    )
    for path in paths:
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_key(item: dict[str, object]) -> tuple[int, str, str]:
    return int(item["seed"]), str(item["regime"]), str(item["arm_id"])


def _validate_completed_run_artifact(
    path: Path,
    item: dict[str, object],
    *,
    suite_id: str,
    num_rounds: int,
) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    seed, regime, arm_id = _run_key(item)
    config = record.get("config", {})
    if (
        record.get("suite_id") != suite_id
        or record.get("suite_regime") != regime
        or record.get("suite_arm_id") != arm_id
        or int(config.get("seed", -1)) != seed
    ):
        raise ValueError(f"Run artifact does not match its manifest entry: {path}")
    rounds = record.get("rounds", {})
    expected_rounds = {
        str(round_number) for round_number in range(1, num_rounds + 1)
    }
    if not isinstance(rounds, dict) or set(rounds) != expected_rounds or any(
        not isinstance(round_record, dict)
        or not isinstance(round_record.get("round_status"), str)
        for round_record in rounds.values()
    ):
        raise ValueError(
            f"Run artifact is missing completed round outcomes: {path}"
        )


def _runtime_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "flower": _package_version("flwr"),
    }


def _validate_resume_checkpoint(
    manifest: dict[str, object],
    *,
    suite_signature: str,
    source_fingerprint: str,
    calibration_provenance: dict[str, object] | None,
    runtime_versions: dict[str, str | None],
) -> None:
    if manifest.get("suite_status") == "completed":
        raise ValueError("Cannot resume a suite that is already completed")
    if manifest.get("suite_signature") != suite_signature:
        raise ValueError(
            "Resume configuration does not match the saved suite manifest"
        )
    if manifest.get("source_fingerprint_sha256") != source_fingerprint:
        raise ValueError(
            "Source code changed since the suite checkpoint; start a new suite"
        )
    if manifest.get("runtime_versions") != runtime_versions:
        raise ValueError(
            "Python/Flower/numerical package versions changed since the "
            "checkpoint; resume in the original environment"
        )
    saved_calibration = manifest.get("threshold_calibration") or {}
    current_calibration = calibration_provenance or {}
    if (
        saved_calibration.get("sha256") != current_calibration.get("sha256")
        or saved_calibration.get("threshold")
        != current_calibration.get("threshold")
    ):
        raise ValueError(
            "Calibration provenance changed since the suite checkpoint"
        )


def run_paired_suite(
    grid: Grid,
    context: Context,
    *,
    model_factory,
    evaluate_fn,
) -> Path:
    """Run all requested arms sequentially against one persistent Flower Grid."""
    run_config = context.run_config
    seeds = [int(item) for item in parse_csv_values(run_config["suite-seeds"], name="suite-seeds")]
    if any(seed < 0 for seed in seeds):
        raise ValueError("suite-seeds must be non-negative integers")
    regimes = parse_csv_values(
        run_config["suite-regimes"], name="suite-regimes"
    )
    invalid_regimes = set(regimes) - {"iid", "dirichlet-1.0", "dirichlet-0.3", "dirichlet-0.1"}
    if invalid_regimes:
        raise ValueError(f"Unsupported suite regime(s): {sorted(invalid_regimes)}")
    attack_modes = parse_csv_values(
        run_config["suite-attack-modes"], name="suite-attack-modes"
    )
    suite_phase = str(run_config.get("suite-phase", "development"))
    if suite_phase not in {"development", "confirmatory"}:
        raise ValueError("suite-phase must be development or confirmatory")
    calibrated_threshold = float(run_config["multi-signal-threshold"])
    calibration_source = str(run_config.get("suite-calibration-source", "")).strip()
    calibration_provenance = None
    if suite_phase == "confirmatory":
        if not calibration_source:
            raise ValueError(
                "Confirmatory suites require suite-calibration-source from a development suite"
            )
        calibration_path = Path(calibration_source).expanduser()
        calibration_record = json.loads(
            calibration_path.read_text(encoding="utf-8")
        )
        if calibration_record.get("calibration_status") != "ready_for_confirmatory":
            raise ValueError(
                "Calibration record is not ready for confirmatory use; "
                "expand the benign-only development data and recalibrate"
            )
        if calibration_record.get("calibration_method") != "benign_only_empirical_fpr":
            raise ValueError(
                "Confirmatory suites require a benign-only empirical-FPR calibration"
            )
        recorded_threshold = float(calibration_record["threshold"])
        if not np.isclose(recorded_threshold, calibrated_threshold):
            raise ValueError(
                "multi-signal-threshold must match the supplied calibration record"
            )
        development_seeds = {
            int(seed) for seed in calibration_record.get("development_seeds", [])
        }
        requested_seeds = {
            int(seed)
            for seed in parse_csv_values(
                run_config["suite-seeds"], name="suite-seeds"
            )
        }
        if not development_seeds or development_seeds.intersection(requested_seeds):
            raise ValueError(
                "Calibration must record development seeds, and confirmatory "
                "seeds must be disjoint from them"
            )
        calibration_provenance = {
            "filename": calibration_path.name,
            "sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
            "development_suite_id": calibration_record["development_suite_id"],
            "development_suite_ids": calibration_record.get(
                "development_suite_ids",
                [calibration_record["development_suite_id"]],
            ),
            "threshold": recorded_threshold,
        }
    selection_arms = parse_csv_values(
        run_config["suite-selection-arms"], name="suite-selection-arms"
    )
    aggregation_arms = parse_csv_values(
        run_config["suite-aggregation-arms"], name="suite-aggregation-arms"
    ) if str(run_config.get("suite-aggregation-arms", "")).strip() else []
    arms = build_suite_arms(selection_arms, aggregation_arms, attack_modes)

    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(run_config.get("run-output-dir", "runs")).expanduser()
    if not output_root.is_absolute():
        output_root = project_root / output_root
    code_revision, code_revision_note, working_tree_clean = _git_provenance(
        project_root
    )
    source_fingerprint = _source_fingerprint(project_root)
    runtime_versions = _runtime_versions()
    suite_signature = _suite_signature(
        run_config,
        seeds=seeds,
        regimes=regimes,
        arms=arms,
        calibration_sha256=(
            calibration_provenance["sha256"] if calibration_provenance else None
        ),
    )
    resume_source = str(run_config.get("suite-resume-manifest", "")).strip()
    if resume_source:
        manifest_path = Path(resume_source).expanduser()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_resume_checkpoint(
            manifest,
            suite_signature=suite_signature,
            source_fingerprint=source_fingerprint,
            calibration_provenance=calibration_provenance,
            runtime_versions=runtime_versions,
        )
        suite_dir = manifest_path.parent
        suite_id = str(manifest["suite_id"])
        run_files = list(manifest.get("run_files", []))
        target_attacker_partition = int(manifest["attacker_partition_id"])
        previous_partition_ids = sorted(
            int(item) for item in manifest["partition_ids"]
        )
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    else:
        suite_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:8]
        )
        suite_dir = output_root / "paired_suites" / suite_id
        suite_dir.mkdir(parents=True, exist_ok=False)
        run_files = []
        previous_partition_ids = []
        target_attacker_partition = int(
            run_config.get("malicious-partition-id", 0)
        )
        manifest_path = suite_dir / "suite_manifest.json"
        manifest = {
            "schema_version": 2,
            "suite_id": suite_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seeds": seeds,
            "regimes": regimes,
            "arms": arms,
            "suite_phase": suite_phase,
            "suite_signature": suite_signature,
            "suite_status": "in_progress",
            "source_fingerprint_sha256": source_fingerprint,
            "runtime_versions": runtime_versions,
            "threshold_calibration": calibration_provenance,
            "code_revision": code_revision,
            "code_revision_note": code_revision_note,
            "working_tree_clean": working_tree_clean,
            "attacker_partition_id": target_attacker_partition,
            "run_files": run_files,
            "completed_run_count": 0,
            "resume_count": 0,
        }

    node_ids = sorted(grid.get_node_ids())
    if len(node_ids) < 3:
        raise ValueError("Suite requires at least three registered Flower nodes")

    probe_seed = seeds[0]
    if regimes[0] == "iid":
        probe_partitioning_mode, probe_alpha = "iid", 0.3
    else:
        probe_partitioning_mode = "dirichlet"
        probe_alpha = float(regimes[0].removeprefix("dirichlet-"))
    seed_everything(probe_seed, bool(run_config.get("deterministic", False)))
    probe_model = model_factory()
    probe_strategy = CosineAuditStrategy(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=2,
        min_evaluate_nodes=0,
        min_available_nodes=3,
        representation_mode=DELTA,
        selection_mode="none",
        aggregation_mode="fedavg",
        partitioning_mode=probe_partitioning_mode,
        dirichlet_alpha=probe_alpha,
        partition_seed=int(run_config.get("partition-seed", 42)),
        batch_size=int(run_config["batch-size"]),
        attacker_node_id=node_ids[0],
        partition_mapping_probe=True,
        seed=probe_seed,
        deterministic=bool(run_config.get("deterministic", False)),
    )
    probe_strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(probe_model.state_dict()),
        train_config=ConfigRecord({"lr": float(run_config["learning-rate"])}),
        num_rounds=1,
        evaluate_fn=None,
    )
    probe_round = probe_strategy.round_log.get(1, {})
    current_mapping = {
        str(node_id): int(client["partition_id"])
        for node_id, client in probe_round.get("clients", {}).items()
        if client.get("partition_id") is not None
    }
    partition_ids = sorted(current_mapping.values())
    if (
        len(current_mapping) != len(node_ids)
        or len(partition_ids) != len(set(partition_ids))
        or target_attacker_partition not in partition_ids
    ):
        raise RuntimeError(
            "Partition mapping probe did not return a unique partition ID "
            "for every registered Flower node"
        )
    if previous_partition_ids and partition_ids != previous_partition_ids:
        raise RuntimeError(
            "Partition IDs differ from the saved checkpoint; refusing to resume "
            "a non-paired suite"
        )
    attacker_node_id = next(
        int(node_id)
        for node_id, partition_id in current_mapping.items()
        if partition_id == target_attacker_partition
    )
    manifest.update(
        suite_status="in_progress",
        updated_at=datetime.now(timezone.utc).isoformat(),
        partition_ids=partition_ids,
        partition_mapping_for_current_grid=current_mapping,
        shared_registered_node_ids=node_ids,
        shared_attacker_node_id=attacker_node_id,
        attacker_partition_id=target_attacker_partition,
        pairing_scope=(
            "same seed, model initialization, partition IDs, and designated "
            "attacker partition; Flower node IDs may change after resume"
        ),
        partition_mapping_note=(
            "The server probes and designates the Flower node assigned to the "
            "fixed attacker partition. Client-reported partition IDs are used "
            "only to verify simulation assignment; attacker labels in each run "
            "derive from the server-designated Flower node ID."
        ),
        run_files=run_files,
        completed_run_count=len(run_files),
    )
    _atomic_write_json(manifest_path, manifest)

    expected_mapping: dict[str, int] | None = None
    completed_keys = set()
    for item in run_files:
        run_path = Path(str(item["path"]))
        if not run_path.is_absolute():
            run_path = suite_dir / run_path
        if not run_path.is_file():
            raise FileNotFoundError(
                f"Completed run artifact is missing: {run_path}"
            )
        _validate_completed_run_artifact(
            run_path,
            item,
            suite_id=suite_id,
            num_rounds=int(run_config["num-server-rounds"]),
        )
        run_key = _run_key(item)
        if run_key in completed_keys:
            raise ValueError(f"Duplicate completed arm in suite manifest: {run_key}")
        completed_keys.add(run_key)

    for current_seed in seeds:
        deterministic = bool(run_config.get("deterministic", False))
        seed_everything(current_seed, deterministic)
        seed_model = model_factory()
        seed_model_hash = _state_dict_sha256(seed_model.state_dict())
        for regime in regimes:
            if regime == "iid":
                partitioning_mode, alpha = "iid", 0.3
            else:
                partitioning_mode = "dirichlet"
                alpha = float(regime.removeprefix("dirichlet-"))
            for arm in arms:
                run_key = (current_seed, regime, arm["arm_id"])
                if run_key in completed_keys:
                    continue
                seed_everything(current_seed, deterministic)
                model = model_factory()
                arrays = ArrayRecord(model.state_dict())
                if _state_dict_sha256(model.state_dict()) != seed_model_hash:
                    raise RuntimeError("Initial model initialization changed across seeds")

                attack_mode = arm["attack_mode"]
                experiment_arm_id = f"seed{current_seed}_{regime}_{arm['arm_id']}"
                attack_start = int(run_config.get("suite-attack-start-round", 1))
                strategy = CosineAuditStrategy(
                    fraction_evaluate=float(run_config["fraction-evaluate"]),
                    representation_mode=DELTA,
                    reference_mode="current_mean",
                    selection_mode=arm["selection_mode"],
                    aggregation_mode=arm["aggregation_mode"],
                    byzantine_count=int(run_config.get("byzantine-count", 1)),
                    multi_krum_count=(
                        int(run_config["multi-krum-count"])
                        if int(run_config.get("multi-krum-count", -1)) > 0
                        else None
                    ),
                    multi_signal_threshold=float(
                        run_config["multi-signal-threshold"]
                    ),
                    adaptive_attack_mode=(
                        attack_mode if attack_mode != "none" else "none"
                    ),
                    adaptive_attack_steps=int(
                        run_config.get("adaptive-attack-steps", 3)
                    ),
                    adaptive_attack_step_size=float(
                        run_config.get("adaptive-attack-step-size", 0.01)
                    ),
                    partitioning_mode=partitioning_mode,
                    dirichlet_alpha=alpha,
                    partition_seed=int(run_config.get("partition-seed", 42)),
                    batch_size=int(run_config["batch-size"]),
                    experiment_arm_id=experiment_arm_id,
                    attack_start_round=(
                        attack_start
                        if attack_mode != "none"
                        else int(run_config["num-server-rounds"]) + 1
                    ),
                    attacker_node_id=attacker_node_id,
                    malicious_partition_id=target_attacker_partition,
                    seed=current_seed,
                    deterministic=deterministic,
                )
                result = strategy.start(
                    grid=grid,
                    initial_arrays=arrays,
                    train_config=ConfigRecord(
                        {"lr": float(run_config["learning-rate"])}
                    ),
                    num_rounds=int(run_config["num-server-rounds"]),
                    evaluate_fn=evaluate_fn,
                )
                first_round = strategy.round_log.get(1, {})
                observed_mapping = {
                    str(node_id): int(client["partition_id"])
                    for node_id, client in first_round.get("clients", {}).items()
                    if client.get("partition_id") is not None
                }
                observed_partition_ids = sorted(observed_mapping.values())
                if (
                    len(observed_mapping) != len(node_ids)
                    or observed_partition_ids != partition_ids
                    or len(observed_partition_ids) != len(set(observed_partition_ids))
                    or observed_mapping.get(str(attacker_node_id))
                    != target_attacker_partition
                ):
                    raise RuntimeError(
                        "Arm partition mapping or designated attacker partition "
                        "differs from the preflight mapping"
                    )
                if expected_mapping is None:
                    expected_mapping = observed_mapping
                elif observed_mapping != expected_mapping:
                    raise RuntimeError(
                        "Flower node-to-partition mapping changed within this Grid"
                    )
                run_id = (
                    f"seed{current_seed}_{regime}_{arm['arm_id']}_"
                    + uuid.uuid4().hex[:8]
                )
                arm_dir = suite_dir / run_id
                arm_dir.mkdir(parents=True, exist_ok=False)
                record_config = {
                    str(key): _json_safe(value)
                    for key, value in run_config.items()
                    if key
                    not in {
                        "run-output-dir",
                        "suite-calibration-source",
                        "suite-resume-manifest",
                    }
                }
                record = {
                    "schema_version": 2,
                    "suite_id": suite_id,
                    "suite_regime": regime,
                    "suite_arm_id": arm["arm_id"],
                    "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "provenance": {
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                        "torch": torch.__version__,
                        "numpy": np.__version__,
                        "flower": _package_version("flwr"),
                        "code_revision": code_revision,
                        "code_revision_note": code_revision_note,
                        "working_tree_clean": working_tree_clean,
                        "source_fingerprint_sha256": source_fingerprint,
                        "dataset": "uoft-cs/cifar10",
                        "dataset_revision": None,
                        "partitioning": partitioning_mode,
                        "dirichlet_alpha": (
                            alpha if partitioning_mode == "dirichlet" else None
                        ),
                        "partition_seed": int(run_config.get("partition-seed", 42)),
                        "partition_split_seed": 42,
                        "registered_node_ids": node_ids,
                        "attacker_node_id": attacker_node_id,
                        "attacker_partition_id": target_attacker_partition,
                        "client_rng": "seed + 1000003 * partition_id + 9176 * round modulo 2^32",
                        "reported_client_partition_mapping": observed_mapping,
                        "initial_model_sha256": _state_dict_sha256(
                            model.state_dict()
                        ),
                        "threshold_calibration": calibration_provenance,
                    },
                    "config": {
                        **record_config,
                        "seed": current_seed,
                        "partitioning_mode": partitioning_mode,
                        "suite_phase": suite_phase,
                        "threshold_calibration": calibration_provenance,
                        "suite_attack_start_round": attack_start,
                        "dirichlet_alpha": (
                            alpha if partitioning_mode == "dirichlet" else None
                        ),
                        **arm,
                        "attacker_node_id": attacker_node_id,
                        "attacker_partition_id": target_attacker_partition,
                    },
                    "rounds": _json_safe(strategy.round_log),
                    "server_evaluation_metrics": {
                        str(round_number): _json_safe(dict(metrics))
                        for round_number, metrics
                        in result.evaluate_metrics_serverapp.items()
                    },
                }
                run_path = arm_dir / "run.json"
                with run_path.open("x", encoding="utf-8") as output:
                    json.dump(record, output, indent=2, allow_nan=False)
                _validate_completed_run_artifact(
                    run_path,
                    {
                        "seed": current_seed,
                        "regime": regime,
                        "arm_id": arm["arm_id"],
                    },
                    suite_id=suite_id,
                    num_rounds=int(run_config["num-server-rounds"]),
                )
                run_files.append(
                    {
                        "seed": current_seed,
                        "regime": regime,
                        "arm_id": arm["arm_id"],
                        "path": run_path.relative_to(suite_dir).as_posix(),
                    }
                )
                completed_keys.add(run_key)
                manifest.update(
                    run_files=run_files,
                    completed_run_count=len(run_files),
                    last_completed_run={
                        "seed": current_seed,
                        "regime": regime,
                        "arm_id": arm["arm_id"],
                    },
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    partition_mapping_for_current_grid=expected_mapping,
                )
                _atomic_write_json(manifest_path, manifest)

    manifest.update(
        suite_status="completed",
        completed_at=datetime.now(timezone.utc).isoformat(),
        completed_run_count=len(run_files),
        updated_at=datetime.now(timezone.utc).isoformat(),
        run_files=run_files,
    )
    _atomic_write_json(manifest_path, manifest)
    print(f"[Experiment suite] Manifest saved to {manifest_path}")
    return manifest_path
