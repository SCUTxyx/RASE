"""Deterministic fork round-trip assertions shared by smoke tests and pool gates."""

from __future__ import annotations

from typing import Any


def observation_from_step(step_result: Any) -> Any:
    return step_result[0] if isinstance(step_result, tuple) else step_result


def assert_tree_equal(left: Any, right: Any, path: str = "obs") -> None:
    import numpy as np

    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(f"{path}: observation keys differ")
        for key in left:
            assert_tree_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"{path}: sequence shape differs")
        for index, (l_item, r_item) in enumerate(zip(left, right)):
            assert_tree_equal(l_item, r_item, f"{path}[{index}]")
        return
    if isinstance(left, np.ndarray):
        if left.dtype.kind in "fc":
            np.testing.assert_allclose(
                left, right, rtol=0.0, atol=1e-12, err_msg=path
            )
        else:
            np.testing.assert_array_equal(left, right, err_msg=path)
        return
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def fork_roundtrip_from_snapshot(
    forkable: Any,
    snapshot: Any,
    *,
    steps: int,
    action: Any | None = None,
    sim_env: Any | None = None,
    check_task_fingerprint: bool = True,
) -> None:
    """Restore ``snapshot`` twice and assert identical rollouts.

    ``sim_env`` defaults to ``forkable.env`` (the ControlEnv / OffScreenRenderEnv).
    """
    import numpy as np

    env = sim_env if sim_env is not None else forkable.env
    if action is None:
        action_dim = getattr(env, "action_dim", None)
        if action_dim is None:
            action_dim = env.env.action_dim
        action = np.zeros(action_dim, dtype=np.float64)
    else:
        action = np.asarray(action, dtype=np.float64)
    actions = [action.copy() for _ in range(steps)]

    rollouts = []
    final_states = []
    for _ in range(2):
        forkable.restore(
            snapshot, check_task_fingerprint=check_task_fingerprint
        )
        rollouts.append(
            [observation_from_step(forkable.step(item)) for item in actions]
        )
        final_states.append(np.asarray(env.sim.get_state().flatten()).copy())

    for index, (left, right) in enumerate(zip(*rollouts)):
        assert_tree_equal(left, right, path=f"step[{index}]")
    np.testing.assert_allclose(
        final_states[0], final_states[1], rtol=0.0, atol=1e-9
    )


def fork_roundtrip(env: Any, *, steps: int, action: Any | None = None) -> None:
    """Capture a live snapshot then run the double-restore gate."""
    from rase.envs.forkable_env import ForkableEnv

    forkable = ForkableEnv(env)
    snapshot = forkable.snapshot()
    fork_roundtrip_from_snapshot(
        forkable, snapshot, steps=steps, action=action, sim_env=env
    )
