import os

import numpy as np
import pytest


def _configured_noisy_env():
    bddl = os.environ.get("RASE_TEST_BDDL")
    if not bddl:
        pytest.skip("set RASE_TEST_BDDL to run the LIBERO noise/RNG integration test")

    from scripts.smoke_test import create_env

    return create_env(
        bddl,
        seed=19,
        noise=int(os.environ.get("RASE_TEST_NOISE", "31")),
        render_gpu_device_id=int(os.environ.get("RASE_TEST_GPU_ID", "-1")),
    )


def test_restore_rewinds_libero_process_global_rng():
    from rase.envs.forkable_env import ForkableEnv

    env = _configured_noisy_env()
    try:
        forkable = ForkableEnv(env)
        snapshot = forkable.snapshot()
        first = np.random.standard_normal(32)
        forkable.restore(snapshot)
        second = np.random.standard_normal(32)
        np.testing.assert_array_equal(first, second)
    finally:
        env.close()


def test_noisy_observation_replays_identically():
    from scripts.smoke_test import fork_roundtrip

    env = _configured_noisy_env()
    try:
        fork_roundtrip(env, steps=2)
    finally:
        env.close()
