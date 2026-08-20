"""Unified seeding for deterministic RNG across all Route C scripts.

Covers:
  - Python random
  - NumPy global RNG
  - PyTorch CPU RNG
  - PyTorch CUDA RNG (all devices)
  - CUDA deterministic flags
  - environment seed (Libero init_state_id)
  - policy noise seed
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, cudnn_deterministic: bool = True) -> dict[str, Any]:
    """Set all RNG states to a known seed and return state snapshot.

    Returns a dict that can be passed to ``restore_rng_state()`` to restore
    the state before seeding (useful for ensuring paired runners start from
    identical RNG states).
    """
    prev_state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        prev_state["torch_cuda"] = {}
        for i in range(torch.cuda.device_count()):
            prev_state["torch_cuda"][i] = torch.cuda.get_rng_state(i)
        torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        prev_state["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        prev_state["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    return prev_state


def restore_rng_state(prev_state: dict[str, Any]) -> None:
    """Restore RNG states from a snapshot returned by ``seed_everything()``."""
    if "python_random" in prev_state:
        random.setstate(prev_state["python_random"])
    if "numpy_random" in prev_state:
        np.random.set_state(prev_state["numpy_random"])
    if "torch_cpu" in prev_state:
        torch.set_rng_state(prev_state["torch_cpu"])
    if "torch_cuda" in prev_state and torch.cuda.is_available():
        for i, state in prev_state["torch_cuda"].items():
            torch.cuda.set_rng_state(state, int(i))
    if "cudnn_deterministic" in prev_state:
        torch.backends.cudnn.deterministic = prev_state["cudnn_deterministic"]
    if "cudnn_benchmark" in prev_state:
        torch.backends.cudnn.benchmark = prev_state["cudnn_benchmark"]


def record_policy_rng(policy_bundle: dict[str, Any]) -> dict[str, Any]:
    """Record policy-level RNG state for parity audit.

    Captures PyTorch generator state and CUDA RNG so we can verify that
    two runners consume the same random numbers in the same order.
    """
    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            state[f"torch_cuda_{i}"] = torch.cuda.get_rng_state(i)

    # Record per-module generator state if policy exposes one.
    policy = policy_bundle.get("policy")
    if policy is not None and hasattr(policy, "_action_queue"):
        state["action_cache_length"] = len(policy._action_queue)
    return state


def verify_rng_identical(state_a: dict[str, Any], state_b: dict[str, Any]) -> dict[str, bool]:
    """Compare two RNG state snapshots, returning per-component equality."""
    checks: dict[str, bool] = {}
    for key in state_a:
        if key not in state_b:
            checks[key] = False
            continue
        val_a = state_a[key]
        val_b = state_b[key]
        if isinstance(val_a, torch.Tensor):
            checks[key] = bool(torch.equal(val_a.cpu(), val_b.cpu()))
        elif isinstance(val_a, np.ndarray):
            checks[key] = bool(np.array_equal(val_a, val_b))
        else:
            checks[key] = val_a == val_b
    for key in state_b:
        if key not in checks:
            checks[key] = False
    return checks
