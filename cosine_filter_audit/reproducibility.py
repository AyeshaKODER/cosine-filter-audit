"""Seed helpers for repeatable simulation runs."""

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False


def client_seed(seed: int, client_identity: int, server_round: int) -> int:
    """Derive local RNG streams from a stable simulation client identity."""
    return (seed + 1_000_003 * client_identity + 9_176 * server_round) % (2**32)
