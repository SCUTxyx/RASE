"""Shared LeRobot observation / action / success helpers for collect and W3."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def success_from_info(info: Mapping[str, Any]) -> bool:
    """Read SyncVectorEnv final_info.is_success (batch dim allowed)."""
    final_info = info.get("final_info")
    if not isinstance(final_info, Mapping) or "is_success" not in final_info:
        return False
    value = np.asarray(final_info["is_success"]).reshape(-1)
    return bool(value[0]) if value.size else False


def as_batched_action(action: np.ndarray) -> np.ndarray:
    """Ensure env.step receives shape ``[1, 7]``."""
    array = np.asarray(action, dtype=np.float32)
    if array.shape == (7,):
        return array[None, ...]
    if array.ndim == 2 and array.shape[0] == 1 and array.shape[1] == 7:
        return array
    raise ValueError(f"expected action shape (7,) or (1, 7), got {array.shape}")


def select_env_action(
    policy_bundle: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    task: str,
    temperature: float | None = None,
) -> np.ndarray:
    """One LeRobot ``select_action`` step → env-space ``[1, 7]`` numpy action.

    When ``temperature`` is set, flow-matching initial noise is scaled as
    ``N(0, temperature^2)`` (same convention as W2 candidate sampling). Noise is
    only consumed when the policy action queue is empty.
    """
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    from rase.collect.smolvla_candidate_policy import flow_matching_noise

    policy_observation = preprocess_observation(
        {key: value for key, value in observation.items() if key != "task"}
    )
    policy_observation["task"] = [task]
    env_observation = policy_bundle["env_preprocessor"](policy_observation)
    processed = policy_bundle["preprocessor"](env_observation)
    policy = policy_bundle["policy"]
    if temperature is None:
        action = policy.select_action(processed)
    else:
        device = next(policy.parameters()).device
        noise = flow_matching_noise(
            (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
            device=device,
            temperature=float(temperature),
        )
        action = policy.select_action(processed, noise=noise)
    action = policy_bundle["postprocessor"](action)
    transition = policy_bundle["env_postprocessor"]({ACTION: action})
    return np.asarray(transition[ACTION].detach().cpu().numpy(), dtype=np.float32)


def current_timestep(control_env: Any) -> int:
    """Best-effort episode timestep from the inner LIBERO task env."""
    task_env = getattr(control_env, "env", control_env)
    for attr in ("timestep", "cur_time"):
        if hasattr(task_env, attr):
            try:
                return int(getattr(task_env, attr))
            except (TypeError, ValueError):
                continue
    return 0
