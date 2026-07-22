import os

import pytest


def _task_paths():
    first = os.environ.get("RASE_TEST_BDDL")
    second = os.environ.get("RASE_TEST_OTHER_BDDL")
    if not first or not second:
        pytest.skip(
            "set RASE_TEST_BDDL and RASE_TEST_OTHER_BDDL to run cross-task restore"
        )
    return first, second


def test_snapshot_rejects_a_different_task_before_mutation():
    from rase.envs.forkable_env import ForkableEnv, TaskMismatchError
    from scripts.smoke_test import create_env

    first_path, second_path = _task_paths()
    gpu_id = int(os.environ.get("RASE_TEST_GPU_ID", "-1"))
    first = create_env(first_path, seed=3, render_gpu_device_id=gpu_id)
    second = create_env(second_path, seed=3, render_gpu_device_id=gpu_id)
    try:
        source = ForkableEnv(first)
        target = ForkableEnv(second)
        snapshot = source.snapshot()
        state_before = second.sim.get_state().flatten().copy()

        with pytest.raises(TaskMismatchError, match="fingerprint"):
            target.restore(snapshot)

        state_after = second.sim.get_state().flatten().copy()
        assert (state_before == state_after).all()
    finally:
        first.close()
        second.close()
