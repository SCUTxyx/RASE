"""Shared LeRobot observation / action / success helpers for collect and W3."""

from __future__ import annotations

import copy
import hashlib
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InferenceEvent:
    """Immutable record of one native ``predict_action_chunk`` forward.

    Created at inference time (inside the same ``select_action`` call that
    consumes the first chunk step), never reconstructed from LeRobot's mutable
    action queue. ``env_chunk`` holds the first ``horizon`` steps converted to
    env-space ``[H, 7]``; ``chunk_size`` is the native chunk length ``T``.
    """

    inference_event_id: str
    native_chunk: np.ndarray  # [1, T, D] float32 (detached, CPU)
    env_chunk: np.ndarray  # [H, 7] float32
    chunk_size: int  # T
    candidate_generation_seed: int | None
    boundary_step: int
    policy_state_hash: str
    model_forward_calls: int
    wall_s: float


@dataclass
class _CaptureContext:
    """Mutable per-continuation bookkeeping for inference-event provenance."""

    events: list[InferenceEvent] = field(default_factory=list)
    current_event_index: int | None = None
    consumed_in_current: int = 0
    boundary_step: int = 0
    capture_horizon: int = 10
    capture_enabled: bool = True


def _rng_state_dict() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {"numpy": np.random.get_state()}
    state["torch_cpu"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def _policy_queue_containers(policy: Any) -> list[tuple[str, Any]]:
    """Return (attribute, container) pairs for all action-queue containers.

    LeRobot stores ``_queues`` (dict of deques); the pi0-fast port stores a
    single ``_action_queue`` deque that is pre-filled at load time.  Both must
    be snapshot/restored/cleared for correct K3-E0 provenance.
    """
    found: list[tuple[str, Any]] = []
    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict):
        found.append(("_queues", queues))
    action_queue = getattr(policy, "_action_queue", None)
    if action_queue is not None:
        found.append(("_action_queue", action_queue))
    return found


def clear_policy_queues(policy: Any) -> None:
    """Empty every action-queue container (both LeRobot and pi0-fast forms)."""
    for _name, container in _policy_queue_containers(policy):
        if isinstance(container, dict):
            for value in container.values():
                if hasattr(value, "clear"):
                    value.clear()
        elif hasattr(container, "clear"):
            container.clear()


def policy_state_snapshot(policy_bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot queue + RNG state so a forced requery cannot pollute continue."""
    policy = policy_bundle["policy"]
    containers: dict[str, Any] = {}
    for name, container in _policy_queue_containers(policy):
        if isinstance(container, dict):
            containers[name] = {
                key: copy.copy(value) if hasattr(value, "__copy__") else list(value)
                for key, value in container.items()
            }
        else:
            containers[name] = copy.copy(container)
    return {"queues": containers, "rng": _rng_state_dict()}


def policy_state_restore(policy_bundle: Mapping[str, Any], snapshot: dict[str, Any]) -> None:
    """Restore queue + RNG state captured by :func:`policy_state_snapshot`."""
    import torch

    policy = policy_bundle["policy"]
    containers = snapshot.get("queues") or {}
    for name, saved in containers.items():
        if isinstance(saved, dict):
            setattr(
                policy, name,
                {key: copy.copy(value) if hasattr(value, "__copy__") else list(value)
                 for key, value in saved.items()},
            )
        else:
            setattr(policy, name, copy.copy(saved))
    rng = snapshot.get("rng") or {}
    np_state = rng.get("numpy")
    if np_state is not None:
        np.random.set_state(np_state)
    torch_cpu = rng.get("torch_cpu")
    if torch_cpu is not None:
        torch.set_rng_state(torch_cpu)
    torch_cuda = rng.get("torch_cuda")
    if torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(torch_cuda)


def policy_state_fingerprint(policy_bundle: Mapping[str, Any]) -> str:
    """Stable content hash of the policy queue + RNG state (cheap, small state)."""
    parts: list[str] = []
    policy = policy_bundle["policy"]
    for name, container in _policy_queue_containers(policy):
        items = container.values() if isinstance(container, dict) else container
        payload = b""
        for item in items:
            # LeRobot _queues is dict[str, deque[Tensor]], whereas pi0-fast's
            # _action_queue is a deque[Tensor].  Hash deque elements rather
            # than asking NumPy to coerce the container (which fails for CUDA
            # tensors and obscures queue ordering).
            nested = item if isinstance(item, (list, tuple, deque)) else (item,)
            for value_item in nested:
                value = (
                    value_item.detach().cpu().numpy()
                    if hasattr(value_item, "detach")
                    else np.asarray(value_item)
                )
                payload += np.ascontiguousarray(value).tobytes()
        parts.append(f"{name}:{hashlib.sha256(payload).hexdigest()}")
    rng = _rng_state_dict()
    for name in ("numpy", "torch_cpu", "torch_cuda"):
        value = rng.get(name)
        if value is None:
            continue
        if name == "numpy":
            token = repr(value).encode()
        else:
            token = value.cpu().numpy().tobytes() if hasattr(value, "cpu") else bytes(value)
        parts.append(f"{name}:{hashlib.sha256(bytes(token)).hexdigest()}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _event_id(boundary_step: int, generation_seed: int | None, policy_hash: str) -> str:
    counter = int(time.perf_counter_ns())
    token = f"{boundary_step}:{generation_seed}:{policy_hash}:{counter}".encode()
    return hashlib.sha256(token).hexdigest()[:24]


def capture_inference_event(
    policy_bundle: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    task: str,
    boundary_step: int,
    generation_seed: int | None = None,
    horizon: int = 10,
    temperature: float | None = None,
) -> tuple[np.ndarray, InferenceEvent]:
    """Run one inference and return ``(executed_first, InferenceEvent)``.

    The native chunk is intercepted at ``predict_action_chunk`` during the same
    ``select_action`` call that supplies the executed first action. Raises when
    ``select_action`` consumes a cached queue instead of invoking
    ``predict_action_chunk`` (callers must reset the queue first for requery).
    """
    from lerobot.utils.constants import ACTION

    policy = policy_bundle["policy"]
    original = getattr(policy, "predict_action_chunk", None)
    if not callable(original):
        raise RuntimeError("policy does not expose predict_action_chunk for native capture")
    # Current LeRobot SmolVLA's select_action calls the private
    # _get_action_chunk directly, while Pi policies call predict_action_chunk.
    # Intercept both paths so native capture remains an execution-aligned
    # observation rather than a second reconstructed inference.
    original_get = getattr(policy, "_get_action_chunk", None)
    captured: dict[str, Any] = {}

    def capture(*args: Any, **kwargs: Any) -> Any:
        value = original(*args, **kwargs)
        captured["value"] = value.detach().clone()
        return value

    def capture_get(*args: Any, **kwargs: Any) -> Any:
        value = original_get(*args, **kwargs)
        captured["value"] = value.detach().clone()
        return value

    started = time.perf_counter()
    policy.predict_action_chunk = capture
    if callable(original_get):
        policy._get_action_chunk = capture_get
    try:
        # Capture must use the *same* sampling path as execution.  In
        # particular, SmolVLA's flow-matching temperature controls the initial
        # noise draw.  Omitting it here made an ostensibly matched-seed
        # re-query silently compare a deterministic/default forward to a
        # temperature-sampled source chunk.
        first = select_env_action(
            policy_bundle,
            observation,
            task=task,
            temperature=temperature,
        )
    finally:
        policy.predict_action_chunk = original
        if callable(original_get):
            policy._get_action_chunk = original_get
    wall_s = time.perf_counter() - started
    native = captured.get("value")
    if native is None:
        raise RuntimeError("select_action did not invoke predict_action_chunk (queue not empty?)")
    native_np = native.detach().cpu().numpy().astype(np.float32, copy=False)
    if native_np.ndim != 3 or native_np.shape[0] != 1 or native_np.shape[1] < 1:
        raise ValueError(f"native chunk must be [1,T,D] with T>=1, got {tuple(native_np.shape)}")
    steps: list[np.ndarray] = []
    for index in range(min(horizon, native_np.shape[1])):
        action = policy_bundle["postprocessor"](native[:, index, :])
        transition = policy_bundle["env_postprocessor"]({ACTION: action})
        value = np.asarray(transition[ACTION].detach().cpu().numpy(), dtype=np.float32).reshape(-1, 7)
        if value.shape != (1, 7) or not np.isfinite(value).all():
            raise ValueError(f"invalid native env action at step {index}: {value.shape}")
        steps.append(value[0])
    env_chunk = np.stack(steps).astype(np.float32, copy=False)
    if not np.array_equal(first.reshape(1, 7), env_chunk[:1]):
        raise RuntimeError("executed first action differs from captured native chunk")
    event = InferenceEvent(
        inference_event_id=_event_id(boundary_step, generation_seed, policy_state_fingerprint(policy_bundle)),
        native_chunk=native_np,
        env_chunk=env_chunk,
        chunk_size=int(native_np.shape[1]),
        candidate_generation_seed=generation_seed,
        boundary_step=int(boundary_step),
        policy_state_hash=policy_state_fingerprint(policy_bundle),
        model_forward_calls=0,
        wall_s=wall_s,
    )
    return first, event


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


def select_env_action_with_native_chunk(
    policy_bundle: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    task: str,
    horizon: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one policy inference and atomically retain its native action chunk.

    The chunk is intercepted at ``predict_action_chunk`` during the same
    ``select_action`` call that supplies the executed first action.  This is
    intentionally not reconstructed from LeRobot's mutable action queue.
    """
    import torch
    from lerobot.utils.constants import ACTION

    if horizon < 1:
        raise ValueError("horizon must be positive")
    policy = policy_bundle["policy"]
    original = getattr(policy, "predict_action_chunk", None)
    if not callable(original):
        raise RuntimeError("policy does not expose predict_action_chunk for native capture")
    captured: dict[str, Any] = {}

    def capture(*args: Any, **kwargs: Any) -> Any:
        value = original(*args, **kwargs)
        captured["value"] = value.detach().clone()
        return value

    policy.predict_action_chunk = capture
    try:
        first = select_env_action(policy_bundle, observation, task=task)
    finally:
        policy.predict_action_chunk = original
    native = captured.get("value")
    if native is None:
        raise RuntimeError("select_action did not invoke predict_action_chunk")
    if native.ndim != 3 or native.shape[0] != 1 or native.shape[1] < horizon:
        raise ValueError(f"native chunk must be [1,T,D] with T >= {horizon}, got {tuple(native.shape)}")
    steps: list[np.ndarray] = []
    for index in range(horizon):
        action = policy_bundle["postprocessor"](native[:, index, :])
        transition = policy_bundle["env_postprocessor"]({ACTION: action})
        value = transition[ACTION].detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32).reshape(-1, 7)
        if value.shape != (1, 7) or not np.isfinite(value).all():
            raise ValueError(f"invalid native env action at step {index}: {value.shape}")
        steps.append(value[0])
    chunk = np.stack(steps).astype(np.float32, copy=False)
    if not np.array_equal(first.reshape(1, 7), chunk[:1]):
        raise RuntimeError("executed first action differs from captured native chunk")
    return first, chunk


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
