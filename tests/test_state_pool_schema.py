import json

import numpy as np
import pytest

from rase.collect.schema import SCHEMA_VERSION, StateMetadata
from rase.collect.state_pool import StatePool


def metadata(**changes):
    values = {
        "task_id": "long_001",
        "instruction": "put the cup away",
        "suite": "Long",
        "episode_id": "ep-0001",
        "step": 4,
        "perturb_dim": "camera",
        "perturb_sub": "viewpoint",
        "level": 4,
        "episode_outcome": "failure",
        "seed": 7,
    }
    values.update(changes)
    return StateMetadata(**values)


def test_state_key_is_stable_versioned_and_annotation_independent():
    first = metadata()
    same_state = metadata(
        instruction="wording can be corrected",
        episode_outcome="success",
    )
    assert first.state_key == same_state.state_key
    assert first.state_key.startswith("sp1_")
    assert first.snapshot_version == SCHEMA_VERSION
    assert metadata(step=6).state_key != first.state_key


def test_metadata_rejects_mismatched_key():
    value = metadata().to_dict()
    value["state_key"] = "sp1_" + "0" * 32
    with pytest.raises(ValueError, match="state_key"):
        StateMetadata.from_dict(value)


def test_state_bundle_has_verified_checksums(tmp_path):
    pool = StatePool(tmp_path / "pool")
    result = pool.write_state(
        metadata(),
        sim_state=np.arange(4),
        controller_state={"integrator": np.asarray([0.0], dtype=np.float32)},
        rng_state={"legacy": ("MT19937", np.arange(4, dtype=np.uint32))},
        observations={"agentview": b"png", "wrist": b"png2"},
        proprio=np.asarray([1.0, 2.0]),
    )
    checksums = pool.verify_state(result.state_key)
    assert result.created
    assert set(checksums["files"]) == {
        "meta.json",
        "obs_agentview.png",
        "obs_wrist.png",
        "proprio.npy",
        "sim_state.npz",
    }
    on_disk = json.loads((result.path / "meta.json").read_text())
    assert on_disk["state_key"] == result.state_key
    assert result.path.relative_to(pool.root).parts == ("long_001", "ep-0001", "000004")
    assert pool.manifest()["states"][result.state_key]["bundle_sha256"]


def test_corruption_is_detected(tmp_path):
    pool = StatePool(tmp_path)
    result = pool.write_state(
        metadata(),
        sim_state=np.arange(2),
        controller_state={},
        rng_state={},
        observations={"agentview": b"original"},
        proprio=np.zeros(2),
    )
    (result.path / "obs_agentview.png").write_bytes(b"corrupt")
    with pytest.raises(OSError, match="checksum"):
        pool.verify_state(result.state_key)
