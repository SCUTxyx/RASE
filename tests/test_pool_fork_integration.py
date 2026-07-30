"""Opt-in pool → ForkableEnv gate (requires real pool + EGL)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_pool_fork_roundtrip_sample():
    root = os.environ.get("RASE_POOL_ROOT")
    if not root:
        pytest.skip("set RASE_POOL_ROOT to run the pool fork integration test")
    pool_root = Path(root)
    if not (pool_root / "manifest.json").is_file():
        pytest.skip(f"no manifest under {pool_root}")

    from rase.collect.libero_env_factory import make_libero_env_for_task
    from rase.collect.state_pool import StatePool, bundle_to_env_snapshot
    from rase.envs.fork_checks import fork_roundtrip_from_snapshot
    from rase.envs.forkable_env import ForkableEnv

    pool = StatePool(pool_root)
    keys = list(pool.manifest()["states"])
    if not keys:
        pytest.skip("empty state pool")
    key = sorted(keys)[0]
    loaded = pool.read_state(key)
    snapshot = bundle_to_env_snapshot(loaded)
    handle = make_libero_env_for_task(
        loaded.metadata.task_id,
        init_state_id=(
            loaded.metadata.init_state_id
            if loaded.metadata.init_state_id is not None
            else 0
        ),
        seed=int(loaded.metadata.seed),
        libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
    )
    try:
        forkable = ForkableEnv(handle.control_env)
        live_fp = forkable._compute_task_fingerprint()
        check_fp = live_fp == snapshot.task_fingerprint
        fork_roundtrip_from_snapshot(
            forkable,
            snapshot,
            steps=int(os.environ.get("RASE_POOL_FORK_STEPS", "5")),
            sim_env=handle.control_env,
            check_task_fingerprint=check_fp,
        )
    finally:
        handle.close()
