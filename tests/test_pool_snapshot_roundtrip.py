"""Lightweight pool decode / EnvSnapshot reassembly tests (no GPU)."""

from __future__ import annotations

import numpy as np

from rase.collect.schema import StateMetadata
from rase.collect.state_pool import (
    StatePool,
    _json_safe,
    _json_unsafe,
    bundle_to_env_snapshot,
)


def test_json_safe_roundtrip_preserves_arrays_and_tuples():
    value = {
        "arr": np.arange(6, dtype=np.float32).reshape(2, 3),
        "tup": (1, np.array([2, 3], dtype=np.uint32)),
        "nested": {"ok": True, "none": None},
    }
    restored = _json_unsafe(_json_safe(value))
    np.testing.assert_array_equal(restored["arr"], value["arr"])
    assert restored["tup"][0] == 1
    np.testing.assert_array_equal(restored["tup"][1], value["tup"][1])
    assert restored["nested"] == value["nested"]


def test_read_state_and_bundle_to_env_snapshot(tmp_path):
    metadata = StateMetadata(
        task_id="libero_goal_000001",
        instruction="test",
        suite="Goal",
        episode_id="ep-00000000-00000000",
        step=0,
        perturb_dim="robot",
        perturb_sub="initial_state",
        level=3,
        episode_outcome="failure",
        seed=7,
    )
    controller = {
        "snapshot_format": "rase.forkable_env/v1",
        "task_fingerprint": "abc123",
        "env_counters": {"cur_time": 0.0, "timestep": 0, "done": False},
        "robots": [{"name": "robot0"}],
        "observables": {},
        "obs_cache": {},
    }
    rng = {
        "python": (1, (2, 3), None),
        "numpy_global": ("MT19937", np.arange(4, dtype=np.uint32), 0, 0, 0.0),
        "environment": {},
    }
    pool = StatePool(tmp_path)
    pool.write_state(
        metadata,
        sim_state=np.linspace(0, 1, 8, dtype=np.float64),
        controller_state=controller,
        rng_state=rng,
        observations={"agentview": b"png-bytes", "wrist": b"png2"},
        proprio=np.zeros(3, dtype=np.float32),
    )
    loaded = pool.read_state(metadata.state_key)
    assert loaded.state_key == metadata.state_key
    assert loaded.observations["agentview"] == b"png-bytes"
    np.testing.assert_allclose(loaded.sim_state, np.linspace(0, 1, 8))
    snapshot = bundle_to_env_snapshot(loaded)
    assert snapshot.task_fingerprint == "abc123"
    assert set(snapshot.payload) == {
        "sim_state",
        "env_counters",
        "robots",
        "observables",
        "obs_cache",
        "rng",
    }
    np.testing.assert_array_equal(snapshot.payload["sim_state"], loaded.sim_state)
    assert snapshot.payload["robots"] == controller["robots"]


def test_parse_pool_task_id():
    from rase.collect.libero_env_factory import parse_pool_task_id

    parsed = parse_pool_task_id("libero_10_000758")
    assert parsed.suite == "libero_10"
    assert parsed.catalog_task_id == 758


def test_clean_control_metadata_uses_explicit_l0_semantics():
    metadata = StateMetadata(
        task_id="libero_goal_000001",
        instruction="test",
        suite="Goal",
        episode_id="ep-clean",
        step=0,
        perturb_dim="clean",
        perturb_sub="none",
        level=0,
        episode_outcome="success",
        seed=7,
    )
    metadata.validate()


def test_resolve_plus_task_index_hints_clean_libero():
    from types import SimpleNamespace

    import pytest

    from rase.collect.libero_env_factory import _resolve_plus_task_index

    suite = SimpleNamespace(name="libero_spatial", tasks=[object()] * 10)
    with pytest.raises(ValueError, match="LIBERO-Plus"):
        _resolve_plus_task_index(suite, 649)
    assert _resolve_plus_task_index(
        SimpleNamespace(name="libero_spatial", tasks=[object()] * 2402), 649
    ) == 648
