"""Flower/PyTorch experiment for explicit cosine-filter representations."""

import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from cosine_filter_audit.experiment_strategy import CosineAuditStrategy
from cosine_filter_audit.paired_runner import run_paired_suite
from cosine_filter_audit.task import Net, load_centralized_dataset, test
from cosine_filter_audit.reproducibility import seed_everything
from cosine_filter_audit.vector_ops import DELTA

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    representation_mode: str = context.run_config.get(
        "representation-mode", DELTA
    )
    reference_mode: str = context.run_config.get("reference-mode", "current_mean")
    selection_mode: str = context.run_config.get("selection-mode", "detector")
    aggregation_mode: str = context.run_config.get("aggregation-mode", "fedavg")
    byzantine_count: int = context.run_config.get("byzantine-count", 1)
    multi_krum_count: int = context.run_config.get("multi-krum-count", -1)
    threshold: float = context.run_config.get("cosine-threshold", 0.7)
    multi_signal_threshold: float = context.run_config.get(
        "multi-signal-threshold", 3.5
    )
    adaptive_attack_mode: str = context.run_config.get(
        "adaptive-attack-mode", "none"
    )
    adaptive_attack_steps: int = context.run_config.get("adaptive-attack-steps", 3)
    adaptive_attack_step_size: float = context.run_config.get(
        "adaptive-attack-step-size", 0.01
    )
    attack_start_round: int = context.run_config.get("attack-start-round", 999)
    attack_scale: float = context.run_config.get("attack-scale", 1.0)
    configured_attacker_node_id: int = context.run_config.get("attacker-node-id", -1)
    malicious_partition_id: int = context.run_config.get("malicious-partition-id", 0)
    seed: int = context.run_config.get("seed", 42)
    deterministic: bool = context.run_config.get("deterministic", False)
    partitioning_mode: str = context.run_config.get("partitioning-mode", "iid")
    dirichlet_alpha: float = context.run_config.get("dirichlet-alpha", 0.3)
    partition_seed: int = context.run_config.get("partition-seed", 42)

    seed_everything(seed, deterministic)
    if context.run_config.get("run-suite", False):
        run_paired_suite(
            grid,
            context,
            model_factory=Net,
            evaluate_fn=global_evaluate,
        )
        return

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = CosineAuditStrategy(
        fraction_evaluate=fraction_evaluate,
        representation_mode=representation_mode,
        reference_mode=reference_mode,
        selection_mode=selection_mode,
        aggregation_mode=aggregation_mode,
        byzantine_count=byzantine_count,
        multi_krum_count=multi_krum_count if multi_krum_count > 0 else None,
        threshold=threshold,
        multi_signal_threshold=multi_signal_threshold,
        adaptive_attack_mode=adaptive_attack_mode,
        adaptive_attack_steps=adaptive_attack_steps,
        adaptive_attack_step_size=adaptive_attack_step_size,
        partitioning_mode=partitioning_mode,
        dirichlet_alpha=dirichlet_alpha,
        partition_seed=partition_seed,
        batch_size=int(context.run_config["batch-size"]),
        attack_start_round=attack_start_round,
        attack_scale=attack_scale,
        attacker_node_id=(
            configured_attacker_node_id
            if configured_attacker_node_id >= 0
            else None
        ),
        malicious_partition_id=malicious_partition_id,
        seed=seed,
        deterministic=deterministic,
    )

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:12]
    )
    configured_output_root = Path(
        context.run_config.get("run-output-dir", "runs")
    ).expanduser()
    if not configured_output_root.is_absolute():
        configured_output_root = (
            Path(__file__).resolve().parent.parent / configured_output_root
        )
    output_dir = configured_output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    print(
        f"\n[Experiment] representation={representation_mode} "
        f"reference={reference_mode} selection={selection_mode} "
        f"aggregation={aggregation_mode} "
        f"cosine_threshold={threshold} "
        f"multi_signal_threshold={multi_signal_threshold} "
        f"adaptive_attack={adaptive_attack_mode} "
        f"partitioning={partitioning_mode} "
        f"alpha={dirichlet_alpha if partitioning_mode == 'dirichlet' else 'n/a'} "
        f"attack_start_round={attack_start_round} "
        f"attack_scale={attack_scale} seed={seed}\n"
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        print(f"\nSaving final model to {output_dir / 'final_model.pt'}...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, output_dir / "final_model.pt")

    def metrics_by_round(metric_history):
        return {
            str(round_number): _json_safe(dict(metrics))
            for round_number, metrics in metric_history.items()
        }

    project_config = {
        str(key): _json_safe(value) for key, value in context.run_config.items()
    }
    code_revision, code_revision_note, working_tree_clean = _git_provenance()
    run_record = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "code_revision": code_revision,
            "code_revision_note": code_revision_note,
            "working_tree_clean": working_tree_clean,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "flower": _package_version("flwr"),
            "dataset": "uoft-cs/cifar10",
            "dataset_revision": None,
            "dataset_revision_note": "Dataset revision is not pinned by this project",
            "partitioning": partitioning_mode,
            "dirichlet_alpha": (
                dirichlet_alpha if partitioning_mode == "dirichlet" else None
            ),
            "partition_split_seed": 42,
            "partition_assignment_seed": (
                partition_seed if partitioning_mode == "dirichlet" else None
            ),
            "registered_node_ids": strategy.registered_node_ids,
            "initial_model_sha256": _state_dict_sha256(
                global_model.state_dict()
            ),
            "run_artifact_path": str(output_dir.resolve()),
        },
        "config": {
            **project_config,
            "representation_mode": representation_mode,
            "reference_mode": reference_mode,
            "selection_mode": selection_mode,
            "aggregation_mode": aggregation_mode,
            "byzantine_count": byzantine_count,
            "multi_krum_count": (
                multi_krum_count if multi_krum_count > 0 else None
            ),
            "threshold": threshold,
            "multi_signal_threshold": multi_signal_threshold,
            "adaptive_attack_mode": adaptive_attack_mode,
            "adaptive_attack_steps": adaptive_attack_steps,
            "adaptive_attack_step_size": adaptive_attack_step_size,
            "attack_start_round": attack_start_round,
            "attack_scale": attack_scale,
            "attacker_node_id": strategy.attacker_node_id,
            "malicious_partition_id": malicious_partition_id,
            "num_rounds": num_rounds,
            "seed": seed,
            "deterministic": deterministic,
        },
        "rounds": _json_safe(strategy.round_log),
        "server_evaluation_metrics": metrics_by_round(
            result.evaluate_metrics_serverapp
        ),
    }
    with (output_dir / "run.json").open("x", encoding="utf-8") as output:
        json.dump(run_record, output, indent=2, allow_nan=False)
    print(f"\n[Experiment] Run record saved to {output_dir / 'run.json'}")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _git_provenance() -> tuple[str | None, str, bool | None]:
    project_root = Path(__file__).resolve().parent.parent
    try:
        repository_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, "Git metadata unavailable", None
    if Path(repository_root).resolve() != project_root:
        return None, "Project directory is not the Git repository root", None

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
        return None, "Git revision could not be read", None
    clean = not status.strip()
    note = "Clean committed Git revision" if clean else "Working tree has uncommitted changes"
    return revision, note, clean