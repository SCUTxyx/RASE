"""Contract tests for candidate execution without GPU / MuJoCo."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rase.collect.forked_rollout import (
    FixedActionContinuation,
    InProcessSmolVLAContinuation,
    RolloutResult,
    evaluate_candidate,
    rollout_seed,
    run_one_forked_rollout,
)
from rase.collect.policy_step import as_batched_action, success_from_info
from rase.envs.snapshot import EnvSnapshot


class _FakeVectorEnv:
    def __init__(self, *, succeed_at: int | None = None, horizon: int = 30):
        self.succeed_at = succeed_at
        self.horizon = horizon
        self.steps = 0
        self.actions = []
        self.envs = [SimpleNamespace(task_description="do the task", _max_episode_steps=horizon)]

    def step(self, action):
        self.actions.append(np.asarray(action))
        self.steps += 1
        terminated = self.succeed_at is not None and self.steps >= self.succeed_at
        truncated = self.steps >= self.horizon and not terminated
        info = {}
        if terminated or truncated:
            info["final_info"] = {"is_success": np.array([bool(terminated)])}
        obs = {"pixels": {"image": np.zeros((1, 8, 8, 3), dtype=np.uint8)}}
        return obs, np.array([0.0]), np.array([terminated]), np.array([truncated]), info


class _FakeForkable:
    def __init__(self):
        self.restores = 0

    def restore(self, snapshot, *, check_task_fingerprint=True):
        del snapshot, check_task_fingerprint
        self.restores += 1


class _FakeRestored:
    def __init__(self, vector_env, forkable):
        self.handle = SimpleNamespace(
            vector_env=vector_env,
            control_env=SimpleNamespace(env=SimpleNamespace(timestep=0)),
        )
        self.forkable = forkable
        self.snapshot = EnvSnapshot(task_fingerprint="fp", payload={
            "sim_state": np.zeros(3),
            "env_counters": {},
            "robots": [],
            "observables": {},
            "obs_cache": {},
            "rng": {},
        })
        self.loaded = SimpleNamespace(
            metadata=SimpleNamespace(instruction="do the task")
        )
        self.check_task_fingerprint = False


def test_rollout_seed_stable_and_in_range():
    a = rollout_seed("sp1_abc", 2, 7)
    b = rollout_seed("sp1_abc", 2, 7)
    c = rollout_seed("sp1_abc", 2, 8)
    assert a == b
    assert a != c
    assert 0 <= a <= 2**32 - 1


def test_inprocess_continuation_resets_action_selection_metrics():
    policy = SimpleNamespace(reset=lambda: None)
    continuation = InProcessSmolVLAContinuation(
        {"policy": policy}, temperature=0.5, seed=None
    )
    continuation._action_select_calls = 7
    continuation._action_select_elapsed_s = 1.25
    continuation._model_forward_calls = 3
    continuation._action_queue_resets = 2
    # reset() is the receding-horizon boundary hook; it must NOT zero cumulative
    # counters (they report totals per rollout). reset_metrics() does.
    continuation.reset()
    assert continuation.metrics()["action_select_calls"] == 7
    continuation.reset_metrics()
    assert continuation.metrics()["action_select_calls"] == 0
    assert continuation.metrics()["action_select_elapsed_s"] == 0.0
    assert continuation.metrics()["model_forward_calls"] == 0
    assert continuation.metrics()["action_queue_resets"] == 0
    assert "excludes environment stepping" in continuation.metrics()["measurement_scope"]


def test_as_batched_action_and_success_helpers():
    assert as_batched_action(np.zeros(7)).shape == (1, 7)
    assert success_from_info({"final_info": {"is_success": np.array([True])}})
    assert not success_from_info({})


def test_evaluate_candidate_early_success(monkeypatch):
    from rase.collect import forked_rollout as mod

    vector = _FakeVectorEnv(succeed_at=3, horizon=50)
    forkable = _FakeForkable()
    restored = _FakeRestored(vector, forkable)

    def fake_obs(_env):
        return {"pixels": {"image": np.zeros((1, 8, 8, 3), dtype=np.uint8)}}

    monkeypatch.setattr(mod, "observation_from_libero_env", fake_obs)
    monkeypatch.setattr(mod, "current_timestep", lambda _env: vector.steps)

    chunk = np.ones((10, 7), dtype=np.float32)
    continuation = FixedActionContinuation(np.zeros((20, 7), dtype=np.float32))
    result = evaluate_candidate(restored, chunk, continuation)
    assert isinstance(result, RolloutResult)
    assert result.success is True
    assert result.candidate_steps == 3
    assert result.continuation_steps == 0
    assert forkable.restores == 1
    assert len(vector.actions) == 3


def test_evaluate_candidate_runs_continuation(monkeypatch):
    from rase.collect import forked_rollout as mod

    vector = _FakeVectorEnv(succeed_at=15, horizon=50)
    forkable = _FakeForkable()
    restored = _FakeRestored(vector, forkable)
    monkeypatch.setattr(
        mod,
        "observation_from_libero_env",
        lambda _env: {"pixels": {"image": np.zeros((1, 8, 8, 3), dtype=np.uint8)}},
    )
    monkeypatch.setattr(mod, "current_timestep", lambda _env: vector.steps)

    chunk = np.ones((10, 7), dtype=np.float32)
    continuation = FixedActionContinuation(np.zeros((20, 7), dtype=np.float32))
    result = evaluate_candidate(restored, chunk, continuation)
    assert result.success is True
    assert result.candidate_steps == 10
    assert result.continuation_steps == 5


def test_evaluate_candidate_supports_direct_continuation_control(monkeypatch):
    from rase.collect import forked_rollout as mod

    vector = _FakeVectorEnv(succeed_at=2, horizon=50)
    restored = _FakeRestored(vector, _FakeForkable())
    monkeypatch.setattr(
        mod,
        "observation_from_libero_env",
        lambda _env: {"pixels": {"image": np.zeros((1, 8, 8, 3), dtype=np.uint8)}},
    )
    monkeypatch.setattr(mod, "current_timestep", lambda _env: vector.steps)

    result = evaluate_candidate(
        restored,
        np.empty((0, 7), dtype=np.float32),
        FixedActionContinuation(np.zeros((5, 7), dtype=np.float32)),
    )
    assert result.success is True
    assert result.candidate_steps == 0
    assert result.continuation_steps == 2


def test_evaluate_candidate_emits_trace_phases(monkeypatch):
    from rase.collect import forked_rollout as mod

    vector = _FakeVectorEnv(succeed_at=3, horizon=50)
    restored = _FakeRestored(vector, _FakeForkable())
    monkeypatch.setattr(
        mod,
        "observation_from_libero_env",
        lambda _env: {"pixels": {"image": np.zeros((1, 8, 8, 3), dtype=np.uint8)}},
    )
    monkeypatch.setattr(mod, "current_timestep", lambda _env: vector.steps)
    seen = []

    def capture(_observation, *, phase, timestep):
        seen.append((phase, timestep))

    result = evaluate_candidate(
        restored,
        np.ones((10, 7), dtype=np.float32),
        FixedActionContinuation(np.zeros((20, 7), dtype=np.float32)),
        trace_callback=capture,
    )
    assert result.success is True
    assert seen == [
        ("initial", 0),
        ("candidate", 1),
        ("candidate", 2),
        ("candidate", 3),
    ]


def test_rejects_double_normalized_shapes():
    restored = _FakeRestored(_FakeVectorEnv(), _FakeForkable())
    with pytest.raises(ValueError, match="\\[T, 7\\]"):
        evaluate_candidate(
            restored,
            np.ones((10, 1, 7), dtype=np.float32),
            FixedActionContinuation(np.zeros((1, 7))),
        )


def test_run_one_forked_rollout_uses_fresh_env_and_always_closes(monkeypatch):
    from rase.collect import forked_rollout as mod

    restored_states = []
    trace_callback = object()
    expected = object()

    class FakeRestored:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def fake_restore(*_args, **_kwargs):
        restored = FakeRestored()
        restored_states.append(restored)
        return restored

    def fake_evaluate(restored, actions, continuation, *, trace_callback=None):
        assert restored is restored_states[-1]
        assert actions.shape == (2, 7)
        assert continuation == "continuation"
        assert trace_callback is expected_trace
        return expected

    expected_trace = trace_callback
    monkeypatch.setattr(mod, "restore_pool_state", fake_restore)
    monkeypatch.setattr(mod, "evaluate_candidate", fake_evaluate)

    for _ in range(2):
        result = run_one_forked_rollout(
            object(),
            "state-key",
            np.zeros((2, 7), dtype=np.float32),
            "continuation",
            trace_callback=trace_callback,
        )
        assert result is expected

    assert len(restored_states) == 2
    assert all(restored.closed for restored in restored_states)
