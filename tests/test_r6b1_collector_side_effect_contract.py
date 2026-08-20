"""R6-B1 regression: boundary bookkeeping must be side-effect-free on source.

Reproduces the R6-B1.0 bisection finding at the environment level, without a
VLA policy:

1. ``forkable.snapshot()`` interleaved with a rollout is side-effect-free: the
   observable sampling state after the rollout is identical to a plain rollout.
2. A force-updated observable read (the exact call the legacy collector made via
   ``raw_libero_to_oracle_arrays`` / ``_get_observations(force_update=True)``)
   is *not* side-effect-free: it consumes observable delay RNG and perturbs the
   sampling schedule.
3. Two-stage feature parity: reading observations from a snapshot restored into
   a separate env reproduces the live observation at that boundary exactly.

Opt-in: set ``RASE_TEST_BDDL`` to a LIBERO-plus task BDDL path.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _env():
    bddl = os.environ.get("RASE_TEST_BDDL")
    if not bddl:
        pytest.skip("set RASE_TEST_BDDL to run the LIBERO side-effect regression test")
    from scripts.smoke_test import create_env

    return create_env(
        bddl,
        seed=int(os.environ.get("RASE_TEST_SEED", "19")),
        render_gpu_device_id=int(os.environ.get("RASE_TEST_GPU_ID", "-1")),
    )


def _action(env):
    dim = getattr(env, "action_dim", None) or env.env.action_dim
    return np.zeros(dim, dtype=np.float64)


def _observable_state(env):
    task_env = getattr(env, "env", None)
    observables = getattr(task_env, "_observables", {})
    return {
        name: {
            "sampled": bool(getattr(obs, "_sampled", None)),
            "delay": float(getattr(obs, "_current_delay", float("nan"))),
            "time_since_last_sample": float(getattr(obs, "_time_since_last_sample", float("nan"))),
        }
        for name, obs in observables.items()
    }


def _observable_rng_states(env):
    task_env = getattr(env, "env", None)
    observables = getattr(task_env, "_observables", {})
    states = {}
    for name, obs in observables.items():
        rng = getattr(obs, "np_random", None) or getattr(obs, "_np_random", None)
        if rng is not None:
            states[name] = rng.get_state()
    return states


def _run_rollout(env, forkable, snapshot, *, steps, bookkeeping):
    """Roll out a fixed-action trajectory, interleaving one bookkeeping op at step 1."""
    from rase.collect.oracle_continuation import raw_libero_to_oracle_arrays

    forkable.restore(snapshot)
    action = _action(env)
    observation = None
    for index in range(steps):
        if index == 1:
            if bookkeeping == "snapshot":
                forkable.snapshot()
            elif bookkeeping == "force":
                raw_libero_to_oracle_arrays(env, force_update=True)
        observation = forkable.step(action)
    return observation, _observable_state(env), _observable_rng_states(env)


def _assert_obs_equal(left, right, *, context=""):
    """Compare robosuite observation dicts (arrays inside OrderedDict)."""
    assert set(left) == set(right), f"{context}: observation keys differ"
    for key in left:
        lv, rv = left[key], right[key]
        if isinstance(lv, np.ndarray):
            np.testing.assert_array_equal(lv, rv, err_msg=f"{context}:{key}")
        else:
            assert lv == rv, f"{context}:{key}"


def test_snapshot_interleaved_is_side_effect_free():
    from rase.envs.forkable_env import ForkableEnv

    env = _env()
    try:
        forkable = ForkableEnv(env)
        snapshot = forkable.snapshot()
        baseline_obs, baseline_state, baseline_rng = _run_rollout(
            env, forkable, snapshot, steps=6, bookkeeping="none")
        snap_obs, snap_state, snap_rng = _run_rollout(
            env, forkable, snapshot, steps=6, bookkeeping="snapshot")
        # A snapshot must leave observable sampling state and observable RNG
        # identical, otherwise boundary recording would perturb the source
        # trajectory exactly as observed in R6-B1.0.
        assert snap_state == baseline_state
        for name in baseline_rng:
            np.testing.assert_array_equal(snap_rng[name][0], baseline_rng[name][0])
        _assert_obs_equal(snap_obs[0], baseline_obs[0], context="final obs")
    finally:
        env.close()


def test_force_updated_obs_read_is_not_side_effect_free():
    from rase.envs.forkable_env import ForkableEnv

    env = _env()
    try:
        forkable = ForkableEnv(env)
        snapshot = forkable.snapshot()
        _, baseline_state, baseline_rng = _run_rollout(
            env, forkable, snapshot, steps=6, bookkeeping="none")
        _, force_state, force_rng = _run_rollout(
            env, forkable, snapshot, steps=6, bookkeeping="force")
        # The legacy in-loop force-updated read (what the R6-B1.0 collector did
        # through raw_libero_to_oracle_arrays) must not be free of side effects:
        # either the observable sampling schedule or its RNG state changed.
        perturbed = force_state != baseline_state or any(
            force_rng.get(name, [None])[0] is not None
            and not np.array_equal(force_rng[name][0], baseline_rng[name][0])
            for name in baseline_rng
        )
        assert perturbed, "expected the force-updated read to perturb observable state"
    finally:
        env.close()


def test_two_stage_restored_features_match_live_boundary():
    """Post-hoc feature extraction from a restored snapshot env reproduces the
    live observation at the boundary (the two-stage collector design)."""
    from rase.envs.forkable_env import ForkableEnv
    from rase.collect.oracle_continuation import raw_libero_to_oracle_arrays

    env = _env()
    try:
        forkable = ForkableEnv(env)
        snapshot = forkable.snapshot()
        action = _action(env)
        forkable.restore(snapshot)
        for index in range(3):
            boundary_snapshot = forkable.snapshot()
            if index == 2:
                live = raw_libero_to_oracle_arrays(env, force_update=True)
            forkable.step(action)
        branch = ForkableEnv(env)
        branch.restore(boundary_snapshot)
        staged = raw_libero_to_oracle_arrays(branch.env, force_update=True)
        for left, right, name in zip(live, staged, ("agentview", "wrist", "proprio")):
            np.testing.assert_allclose(
                np.asarray(left), np.asarray(right), rtol=0.0, atol=0.0, err_msg=name)
    finally:
        # ``branch`` wraps the same env as ``env``; do not close it twice
        # (ControlEnv.close() deletes its inner env).
        env.close()
