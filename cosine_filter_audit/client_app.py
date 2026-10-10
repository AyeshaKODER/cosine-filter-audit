"""Flower/PyTorch client for full-model and delta-space attack experiments."""

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from cosine_filter_audit.adaptive_attack import (
    ATTACK_STATUS_CODES,
    PHASE1,
    PHASE2,
    approximate_score,
    cohort_score,
    constrained_local_loss_ascent,
)
from cosine_filter_audit.task import Net, load_data
from cosine_filter_audit.task import test as test_fn
from cosine_filter_audit.task import train as train_fn
from cosine_filter_audit.reproducibility import client_seed, seed_everything
from cosine_filter_audit.vector_ops import (
    DELTA,
    apply_delta_attack,
    clone_state_dict,
    delta_state_dict,
    flatten_state,
    legacy_scale_state_dict,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data. If this client is the designated
    attacker AND the current round is >= attack_start_round, poison the
    resulting update before sending it back."""

    config = msg.content["config"]
    if config.get("adaptive_attack_phase") == "whitebox_refine":
        return _refine_whitebox(msg, context)

    server_round = config.get("server_round", 0)
    run_seed = int(config.get("seed", 0))
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    if config.get("partition_mapping_probe", False):
        content = RecordDict(
            {
                "arrays": msg.content["arrays"],
                "metrics": MetricRecord(
                    {
                        "train_loss": 0.0,
                        "num-examples": 1,
                        "partition-id": partition_id,
                    }
                ),
            }
        )
        return Message(content=content, reply_to=msg)

    local_seed = client_seed(run_seed, partition_id, server_round)
    seed_everything(local_seed, bool(config.get("deterministic", False)))

    # Keep the server state so delta-space attacks are defined relative to it.
    model = Net()
    global_state = clone_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.load_state_dict(global_state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    global_state = {
        name: tensor.to(device)
        for name, tensor in global_state.items()
    }

    # Load the data
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(
        partition_id,
        num_partitions,
        batch_size,
        seed=local_seed,
        partitioning_mode=config.get("partitioning_mode", "iid"),
        dirichlet_alpha=config.get("dirichlet_alpha", 0.3),
        partition_seed=config.get("partition_seed", 42),
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    clean_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    attack_start_round = config.get("attack_start_round", 999)
    attack_scale = config.get("attack_scale", 1.0)
    malicious_partition_id = config.get("malicious_partition_id", -1)
    attacker_node_id = config.get("attacker_node_id", -1)
    representation_mode = config.get("representation_mode", "legacy_full_model")

    is_attacker = (
        context.node_id == attacker_node_id
        if representation_mode == DELTA
        else partition_id == malicious_partition_id
    )
    attack_active = is_attacker and server_round >= attack_start_round
    adaptive_attack_mode = config.get("adaptive_attack_mode", "none")
    arm_id = str(config.get("experiment_arm_id", "default"))
    attack_result = {
        "status_code": ATTACK_STATUS_CODES["NOT_RUN"],
        "initial_score": 0.0,
        "final_score": 0.0,
        "initial_loss": train_loss,
        "final_loss": train_loss,
        "steps_applied": 0,
    }

    state_dict = model.state_dict()
    if (
        attack_active
        and representation_mode == DELTA
        and adaptive_attack_mode == PHASE1
    ):
        history = _read_client_attack_history(context, arm_id)
        clean_delta = flatten_state(
            delta_state_dict(clean_state, global_state), floating_only=True
        )
        score_candidate = lambda candidate: approximate_score(
            candidate,
            clean_delta,
            history["updates"],
            history["scores"],
        )
        attack_result = constrained_local_loss_ascent(
            model,
            global_state,
            clean_state,
            trainloader,
            device,
            threshold=float(config.get("multi_signal_threshold", 3.5)),
            score_candidate=score_candidate,
            steps=int(config.get("adaptive_attack_steps", 3)),
            step_size=float(config.get("adaptive_attack_step_size", 0.01)),
        )
        state_dict = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        print(
            f"[CLIENT node={context.node_id}] Phase 1 adaptive attack "
            f"(score={attack_result['final_score']:.4f}, "
            f"status={attack_result['status_code']})"
        )
    elif attack_active and adaptive_attack_mode == PHASE2:
        print(
            f"[CLIENT node={context.node_id}] Sending clean update for "
            "Phase 2 cohort-aware refinement"
        )
    elif attack_active:
        if representation_mode == DELTA:
            state_dict = apply_delta_attack(state_dict, global_state, attack_scale)
        elif representation_mode == "legacy_full_model":
            state_dict = legacy_scale_state_dict(state_dict, attack_scale)
        else:
            raise ValueError(f"Unknown representation mode: {representation_mode}")
        print(
            f"[CLIENT node={context.node_id}] ATTACKING this round "
            f"(round={server_round}, mode={representation_mode}, scale={attack_scale})"
        )

    if representation_mode == DELTA and not (
        attack_active and adaptive_attack_mode == PHASE2
    ):
        _write_client_attack_history(
            context,
            flatten_state(
                delta_state_dict(state_dict, global_state), floating_only=True
            ),
            float(attack_result["final_score"]),
            arm_id,
        )

    # Construct and return reply Message
    model_record = ArrayRecord(state_dict)
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "partition-id": partition_id,
        "attack-status-code": int(attack_result["status_code"]),
        "attack-initial-score": float(attack_result["initial_score"]),
        "attack-final-score": float(attack_result["final_score"]),
        "attack-initial-loss": _finite_metric(attack_result["initial_loss"]),
        "attack-final-loss": _finite_metric(attack_result["final_loss"]),
        "attack-steps-applied": int(attack_result["steps_applied"]),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


def _history_record_key(arm_id: str) -> str:
    safe_arm_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in arm_id
    )
    return f"adaptive_attack_history_{safe_arm_id}"


def _read_client_attack_history(
    context: Context, arm_id: str = "default"
) -> dict[str, list]:
    record = context.state.get(_history_record_key(arm_id))
    if record is None:
        return {"updates": [], "scores": []}
    values = record.to_torch_state_dict()
    updates = values.get("updates", torch.empty((0, 0))).detach().cpu()
    scores = values.get("scores", torch.empty(0)).detach().cpu()
    return {
        "updates": [
            np.asarray(update, dtype=np.float64) for update in updates.numpy()
        ],
        "scores": [float(value) for value in scores.numpy()],
    }


def _write_client_attack_history(
    context: Context,
    update: np.ndarray,
    score: float,
    arm_id: str = "default",
    *,
    history_limit: int = 8,
) -> None:
    history = _read_client_attack_history(context, arm_id)
    updates = [*history["updates"], update][-history_limit:]
    scores = [*history["scores"], score][-history_limit:]
    context.state[_history_record_key(arm_id)] = ArrayRecord(
        {
            "updates": torch.as_tensor(np.stack(updates), dtype=torch.float64),
            "scores": torch.as_tensor(scores, dtype=torch.float64),
        }
    )


def _finite_metric(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _refine_whitebox(msg: Message, context: Context) -> Message:
    config = msg.content["config"]
    packed = msg.content["arrays"].to_torch_state_dict()
    global_state = {
        name.removeprefix("global__"): tensor
        for name, tensor in packed.items()
        if name.startswith("global__")
    }
    clean_state = {
        name.removeprefix("candidate__"): tensor
        for name, tensor in packed.items()
        if name.startswith("candidate__")
    }
    peer_updates = {
        int(name.removeprefix("peer__")): tensor.detach().cpu().numpy().astype(
            np.float64
        )
        for name, tensor in packed.items()
        if name.startswith("peer__")
    }
    previous_updates = {
        int(name.removeprefix("prior__")): tensor.detach().cpu().numpy().astype(
            np.float64
        )
        for name, tensor in packed.items()
        if name.startswith("prior__")
    }
    score_history = {
        int(name.removeprefix("scores__")): [
            float(value)
            for value in tensor.detach().cpu().numpy().tolist()
        ]
        for name, tensor in packed.items()
        if name.startswith("scores__")
    }
    if not global_state or not clean_state:
        raise ValueError("White-box request is missing model state")

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    round_number = int(config["server_round"])
    local_seed = client_seed(int(config["seed"]), partition_id, round_number)
    seed_everything(local_seed, bool(config.get("deterministic", False)))
    trainloader, _ = load_data(
        partition_id,
        num_partitions,
        int(config["batch_size"]),
        seed=local_seed + 1,
        partitioning_mode=config.get("partitioning_mode", "iid"),
        dirichlet_alpha=float(config.get("dirichlet_alpha", 0.3)),
        partition_seed=int(config.get("partition_seed", 42)),
    )
    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    score_candidate = lambda candidate: cohort_score(
        candidate,
        client_id=int(context.node_id),
        peer_updates=peer_updates,
        previous_updates=previous_updates,
        score_history=score_history,
        threshold=float(config["multi_signal_threshold"]),
    )
    attack_result = constrained_local_loss_ascent(
        model,
        global_state,
        clean_state,
        trainloader,
        device,
        threshold=float(config["multi_signal_threshold"]),
        score_candidate=score_candidate,
        steps=int(config["adaptive_attack_steps"]),
        step_size=float(config["adaptive_attack_step_size"]),
    )
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    _write_client_attack_history(
        context,
        flatten_state(
            delta_state_dict(state_dict, global_state), floating_only=True
        ),
        float(attack_result["final_score"]),
        str(config.get("experiment_arm_id", "default")),
    )
    metrics = MetricRecord(
        {
            "train_loss": float(config["train_loss"]),
            "num-examples": len(trainloader.dataset),
            "partition-id": partition_id,
            "attack-status-code": int(attack_result["status_code"]),
            "attack-initial-score": float(attack_result["initial_score"]),
            "attack-final-score": float(attack_result["final_score"]),
            "attack-initial-loss": _finite_metric(attack_result["initial_loss"]),
            "attack-final-loss": _finite_metric(attack_result["final_loss"]),
            "attack-steps-applied": int(attack_result["steps_applied"]),
        }
    )
    return Message(
        content=RecordDict(
            {"arrays": ArrayRecord(state_dict), "metrics": metrics}
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data. Unmodified -- attacks target
    training updates, not evaluation, so this stays exactly as-is."""

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    config = msg.content["config"]
    _, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size,
        partitioning_mode=config.get("partitioning_mode", "iid"),
        dirichlet_alpha=config.get("dirichlet_alpha", 0.3),
        partition_seed=config.get("partition_seed", 42),
    )

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)