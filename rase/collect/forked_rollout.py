"""Pool restore → execute candidate chunk → continuation policy → success."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rase.collect.candidates import seed_everything
from rase.collect.libero_env_factory import LiberoEnvHandle, make_libero_env_for_task
from rase.collect.policy_step import (
    InferenceEvent,
    _CaptureContext,
    as_batched_action,
    capture_inference_event,
    current_timestep,
    select_env_action,
    success_from_info,
)
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.state_pool import LoadedState, StatePool, bundle_to_env_snapshot
from rase.envs.forkable_env import ForkableEnv
from rase.envs.snapshot import EnvSnapshot


class ContinuationPolicy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: Mapping[str, Any], *, task: str) -> np.ndarray: ...


class RolloutTraceCallback(Protocol):
    """Optional observation sink used by video/QC tooling."""

    def __call__(
        self,
        observation: Mapping[str, Any],
        *,
        phase: str,
        timestep: int,
    ) -> None: ...


@dataclass(frozen=True)
class RolloutConfig:
    n_action_steps: int = 10
    num_steps: int = 10
    observation_height: int = 360
    observation_width: int = 360
    strict_fingerprint: bool = False
    continuation_temperature: float = 0.5


@dataclass(frozen=True)
class RolloutResult:
    success: bool
    terminated: bool
    truncated: bool
    env_steps: int
    candidate_steps: int
    continuation_steps: int
    final_timestep: int
    stop_reason: str
    elapsed_s: float
    restore_s: float
    candidate_s: float
    continuation_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RestoredPoolState:
    handle: LiberoEnvHandle
    forkable: ForkableEnv
    snapshot: EnvSnapshot
    loaded: LoadedState
    check_task_fingerprint: bool

    def close(self) -> None:
        self.handle.close()


def rollout_seed(state_key: str, candidate: int, rollout: int, *, salt: int = 0) -> int:
    """Deterministic per-rollout seed in ``[0, 2**32 - 1]``."""
    token = f"rase-rollout/v1:{salt}:{state_key}:{candidate}:{rollout}".encode()
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:4], "big")


def restore_pool_state(
    pool: StatePool,
    state_key: str,
    *,
    libero_plus_root: str | None = None,
    observation_height: int = 360,
    observation_width: int = 360,
    strict_fingerprint: bool = False,
) -> RestoredPoolState:
    """Read, bind, and restore one pool bundle into a fresh LiberoEnv."""
    loaded = pool.read_state(state_key)
    snapshot = bundle_to_env_snapshot(loaded)
    meta = loaded.metadata
    flavor = (
        str(getattr(meta, "libero_flavor", None) or "")
        or ("clean" if meta.perturb_dim == "clean" else "plus")
    )
    handle = make_libero_env_for_task(
        meta.task_id,
        init_state_id=meta.init_state_id if meta.init_state_id is not None else 0,
        seed=int(meta.seed),
        observation_height=observation_height,
        observation_width=observation_width,
        libero_plus_root=libero_plus_root,
        libero_flavor=flavor,  # type: ignore[arg-type]
    )
    try:
        live_desc = str(getattr(handle.vector_env.envs[0], "task_description", ""))
        if live_desc and live_desc != meta.instruction:
            raise AssertionError(
                f"task_description mismatch: live={live_desc!r} meta={meta.instruction!r}"
            )
        forkable = ForkableEnv(handle.control_env)
        live_fp = forkable._compute_task_fingerprint()
        check_fp = strict_fingerprint or live_fp == snapshot.task_fingerprint
        forkable.restore(snapshot, check_task_fingerprint=check_fp)
        return RestoredPoolState(
            handle=handle,
            forkable=forkable,
            snapshot=snapshot,
            loaded=loaded,
            check_task_fingerprint=check_fp,
        )
    except Exception:
        handle.close()
        raise


class InProcessSmolVLAContinuation:
    """Stochastic SmolVLA continuation via LeRobot ``select_action``."""

    def __init__(
        self,
        policy_bundle: Mapping[str, Any],
        *,
        temperature: float | None = 0.5,
        seed: int | None = None,
    ) -> None:
        self.policy_bundle = policy_bundle
        self.temperature = float(temperature) if temperature is not None else None
        self.seed = seed
        self._amp = None
        self._action_select_calls = 0
        self._action_select_elapsed_s = 0.0
        self._model_forward_calls = 0
        self._action_queue_resets = 0
        # Native inference-event provenance (immutable events; queue cursor
        # bookkeeping lives on the continuation, never on the event).
        self._capture = _CaptureContext(capture_enabled=False)

    @property
    def capture_enabled(self) -> bool:
        return self._capture.capture_enabled

    def enable_capture(self, *, horizon: int = 10) -> None:
        self._capture.capture_enabled = True
        self._capture.capture_horizon = int(horizon)

    def note_boundary_step(self, step: int) -> None:
        self._capture.boundary_step = int(step)

    def current_inference_event(self) -> InferenceEvent | None:
        index = self._capture.current_event_index
        if index is None:
            return None
        return self._capture.events[index]

    def consumed_in_current_event(self) -> int:
        return self._capture.consumed_in_current

    def reset_metrics(self) -> None:
        """Zero cumulative counters (call once per rollout, not at receding boundaries)."""
        self._action_select_calls = 0
        self._action_select_elapsed_s = 0.0
        self._model_forward_calls = 0
        self._action_queue_resets = 0

    def reset(self) -> None:
        """Clear policy action queue / RNG for a fresh forward on next act().

        Does not zero cumulative forward counters so receding-horizon boundaries
        can still report total ``model_forward_calls`` for a rollout.
        """
        self._action_queue_resets += 1
        if self.seed is not None:
            seed_everything(int(self.seed))
        self.policy_bundle["policy"].reset()

    def _action_queue_empty(self) -> bool:
        policy = self.policy_bundle["policy"]
        # pi0-fast maintains ``_action_queue`` and leaves ``_queues`` empty, so
        # it must be checked first; LeRobot policies use ``_queues`` only.
        action_queue = getattr(policy, "_action_queue", None)
        if action_queue is not None:
            return len(action_queue) == 0
        queues = getattr(policy, "_queues", None)
        if isinstance(queues, dict):
            action_q = queues.get("action")
            if action_q is None:
                # LeRobot uses ACTION constant; fall back to any single queue.
                if len(queues) == 1:
                    action_q = next(iter(queues.values()))
                else:
                    action_q = None
            if action_q is not None:
                return len(action_q) == 0
        return True

    def act(self, observation: Mapping[str, Any], *, task: str) -> np.ndarray:
        import torch

        policy = self.policy_bundle["policy"]
        if self._amp is None:
            device_type = next(policy.parameters()).device.type
            self._amp = (
                torch.autocast(device_type=device_type)
                if bool(getattr(policy.config, "use_amp", False))
                else nullcontext()
            )
        will_forward = self._action_queue_empty()
        started = time.perf_counter()
        try:
            with torch.no_grad(), self._amp:
                if will_forward and self._capture.capture_enabled:
                    first, event = capture_inference_event(
                        self.policy_bundle,
                        observation,
                        task=task,
                        boundary_step=self._capture.boundary_step,
                        generation_seed=self.seed,
                        horizon=self._capture.capture_horizon,
                        temperature=self.temperature,
                    )
                    self._capture.events.append(event)
                    self._capture.current_event_index = len(self._capture.events) - 1
                    self._capture.consumed_in_current = 1
                    return first
                return select_env_action(
                    self.policy_bundle,
                    observation,
                    task=task,
                    temperature=self.temperature,
                )
        finally:
            self._action_select_calls += 1
            self._action_select_elapsed_s += time.perf_counter() - started
            if will_forward:
                self._model_forward_calls += 1
            elif self._capture.capture_enabled and self._capture.current_event_index is not None:
                self._capture.consumed_in_current += 1

    def metrics(self) -> dict[str, float | int | str]:
        return {
            "measurement_scope": (
                "wall time inside LeRobot select_env_action; includes cached action "
                "queue access and model forward passes, excludes environment stepping"
            ),
            "action_select_calls": self._action_select_calls,
            "action_select_elapsed_s": self._action_select_elapsed_s,
            "model_forward_calls": self._model_forward_calls,
            "action_queue_resets": self._action_queue_resets,
        }


class InProcessLeRobotContinuation(InProcessSmolVLAContinuation):
    """Generic frozen LeRobot VLA continuation using its native action sampler.

    Native inference events are captured by default (``capture_horizon`` steps
    of env-space actions per event); pass ``capture=False`` to keep the legacy
    queue-only behaviour.
    """

    def __init__(
        self,
        policy_bundle: Mapping[str, Any],
        *,
        seed: int | None = None,
        capture: bool = True,
        capture_horizon: int = 10,
    ) -> None:
        # None is important: only SmolVLA accepts the explicit flow-noise kwarg
        # used by the legacy temperature path. Pi0Fast/Pi0.5 sample natively.
        super().__init__(policy_bundle, temperature=None, seed=seed)
        self._capture.capture_enabled = bool(capture)
        self._capture.capture_horizon = int(capture_horizon)


class FixedActionContinuation:
    """Replay a fixed action sequence (tests / deterministic stubs)."""

    def __init__(self, actions: Sequence[np.ndarray] | np.ndarray) -> None:
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, ...]
        if array.ndim != 2 or array.shape[1] != 7:
            raise ValueError(f"expected [T, 7] actions, got {array.shape}")
        self._actions = array
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def act(self, observation: Mapping[str, Any], *, task: str) -> np.ndarray:
        del observation, task
        if self._index >= len(self._actions):
            return np.zeros(7, dtype=np.float32)
        action = self._actions[self._index]
        self._index += 1
        return action


def evaluate_candidate(
    restored: RestoredPoolState,
    candidate_chunk: np.ndarray,
    continuation: ContinuationPolicy,
    *,
    max_episode_steps: int | None = None,
    trace_callback: RolloutTraceCallback | None = None,
) -> RolloutResult:
    """Execute env-space candidate then continuation until done/horizon."""
    t0 = time.perf_counter()
    chunk = np.asarray(candidate_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise ValueError(f"candidate chunk must be [T, 7], got {chunk.shape}")

    handle = restored.handle
    vector_env = handle.vector_env
    single = vector_env.envs[0]
    task = str(getattr(single, "task_description", "") or restored.loaded.metadata.instruction)
    horizon = (
        int(max_episode_steps)
        if max_episode_steps is not None
        else int(getattr(single, "_max_episode_steps", 600))
    )

    restore_t0 = time.perf_counter()
    restored.forkable.restore(
        restored.snapshot, check_task_fingerprint=restored.check_task_fingerprint
    )
    # Remote OFT continuations need the live ControlEnv for raw camera/proprio.
    if hasattr(continuation, "bind_control_env"):
        continuation.bind_control_env(handle.control_env)
    continuation.reset()
    restore_s = time.perf_counter() - restore_t0

    observation = observation_from_libero_env(single)
    if trace_callback is not None:
        trace_callback(
            observation,
            phase="initial",
            timestep=current_timestep(handle.control_env),
        )
    candidate_steps = 0
    continuation_steps = 0
    success = False
    terminated = False
    truncated = False
    stop_reason = "horizon"
    candidate_s = 0.0
    continuation_s = 0.0

    cand_t0 = time.perf_counter()
    for action in chunk:
        timestep = current_timestep(handle.control_env)
        if timestep >= horizon:
            stop_reason = "horizon"
            truncated = True
            break
        observation, _, term, trunc, info = vector_env.step(as_batched_action(action))
        candidate_steps += 1
        if trace_callback is not None:
            trace_callback(
                observation,
                phase="candidate",
                timestep=current_timestep(handle.control_env),
            )
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            success = success_from_info(info)
            stop_reason = "success" if success else ("terminated" if terminated else "truncated")
            break
    candidate_s = time.perf_counter() - cand_t0

    if not (terminated or truncated):
        cont_t0 = time.perf_counter()
        while True:
            timestep = current_timestep(handle.control_env)
            if timestep >= horizon:
                stop_reason = "horizon"
                truncated = True
                break
            action = continuation.act(observation, task=task)
            observation, _, term, trunc, info = vector_env.step(as_batched_action(action))
            continuation_steps += 1
            if trace_callback is not None:
                trace_callback(
                    observation,
                    phase="continuation",
                    timestep=current_timestep(handle.control_env),
                )
            terminated = bool(np.asarray(term).reshape(-1)[0])
            truncated = bool(np.asarray(trunc).reshape(-1)[0])
            if terminated or truncated:
                success = success_from_info(info)
                stop_reason = (
                    "success" if success else ("terminated" if terminated else "truncated")
                )
                break
        continuation_s = time.perf_counter() - cont_t0

    return RolloutResult(
        success=bool(success),
        terminated=bool(terminated),
        truncated=bool(truncated),
        env_steps=candidate_steps + continuation_steps,
        candidate_steps=candidate_steps,
        continuation_steps=continuation_steps,
        final_timestep=current_timestep(handle.control_env),
        stop_reason=stop_reason,
        elapsed_s=round(time.perf_counter() - t0, 6),
        restore_s=round(restore_s, 6),
        candidate_s=round(candidate_s, 6),
        continuation_s=round(continuation_s, 6),
    )


def run_one_forked_rollout(
    pool: StatePool,
    state_key: str,
    candidate_actions: np.ndarray,
    continuation: ContinuationPolicy,
    *,
    libero_plus_root: str | None = None,
    config: RolloutConfig | None = None,
    trace_callback: RolloutTraceCallback | None = None,
) -> RolloutResult:
    """Fresh env + restore + evaluate one candidate chunk, then always close it.

    A restored environment must not be shared across candidates.  Successful or
    terminal rollouts may mutate task/model state that participates in the
    snapshot fingerprint, making a later restore into that environment unsafe.
    """
    cfg = config or RolloutConfig()
    restored = restore_pool_state(
        pool,
        state_key,
        libero_plus_root=libero_plus_root,
        observation_height=cfg.observation_height,
        observation_width=cfg.observation_width,
        strict_fingerprint=cfg.strict_fingerprint,
    )
    try:
        return evaluate_candidate(
            restored,
            candidate_actions,
            continuation,
            trace_callback=trace_callback,
        )
    finally:
        restored.close()


def load_lerobot_policy_bundle(
    policy_path: str | Path,
    *,
    device: str = "cuda",
    num_steps: int = 10,
    n_action_steps: int = 10,
    tokenizer_path: str | Path | None = None,
    action_tokenizer_path: str | Path | None = None,
    observation_height: int = 360,
    observation_width: int = 360,
) -> dict[str, Any]:
    """Load a cached frozen LeRobot policy bundle used by collection."""
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    from rase.backends.lerobot_libero_plus import _get_or_load_policy

    path = Path(policy_path).expanduser().resolve()
    tok = Path(tokenizer_path).expanduser().resolve() if tokenizer_path else None
    action_tok = (
        Path(action_tokenizer_path).expanduser().resolve()
        if action_tokenizer_path else None
    )
    env_cfg = LiberoEnvConfig(
        task="libero_spatial",
        task_ids=[0],
        obs_type="pixels_agent_pos",
        init_states=True,
        observation_height=observation_height,
        observation_width=observation_width,
    )
    return _get_or_load_policy(
        path,
        device=device,
        num_steps=num_steps,
        n_action_steps=n_action_steps,
        env_cfg=env_cfg,
        tokenizer_path=tok,
        action_tokenizer_path=action_tok,
    )


def load_smolvla_policy_bundle(
    policy_path: str | Path,
    *,
    device: str = "cuda",
    num_steps: int = 10,
    n_action_steps: int = 10,
    tokenizer_path: str | Path | None = None,
    action_tokenizer_path: str | Path | None = None,
    observation_height: int = 360,
    observation_width: int = 360,
) -> dict[str, Any]:
    """Backward-compatible SmolVLA name for the generic LeRobot loader."""
    return load_lerobot_policy_bundle(
        policy_path,
        device=device,
        num_steps=num_steps,
        n_action_steps=n_action_steps,
        tokenizer_path=tokenizer_path,
        action_tokenizer_path=action_tokenizer_path,
        observation_height=observation_height,
        observation_width=observation_width,
    )
