import os

import pytest


def _configured_env():
    bddl = os.environ.get("RASE_TEST_BDDL")
    if not bddl:
        pytest.skip("set RASE_TEST_BDDL to run the LIBERO fork integration test")

    from scripts.smoke_test import create_env

    return create_env(
        bddl,
        seed=7,
        render_gpu_device_id=int(os.environ.get("RASE_TEST_GPU_ID", "-1")),
    )


def test_same_snapshot_replays_identically():
    from scripts.smoke_test import fork_roundtrip

    env = _configured_env()
    try:
        fork_roundtrip(env, steps=50)
    finally:
        env.close()
