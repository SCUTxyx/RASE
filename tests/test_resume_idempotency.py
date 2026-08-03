import numpy as np
import pytest

from rase.collect.pipeline import collect
from rase.collect.schema import StateMetadata
from rase.collect.state_pool import StatePool, retain_snapshot, snapshot_steps


def test_dry_run_resume_is_byte_stable(tmp_path):
    config = {
        "adapter": None,
        "collection": {
            "output_dir": str(tmp_path / "pool"),
            "episodes": 10,
            "seed": 17,
            "action_chunks_per_episode": 6,
            "snapshot_cadence_action_chunks": 2,
            "successful_snapshot_retention": 0.20,
            "dry_run": True,
        },
    }
    first = collect(config)
    manifest_path = tmp_path / "pool" / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    second = collect(config)
    assert first["states_created"] == first["snapshots_retained"]
    assert second["states_created"] == 0
    # Resume skips whole episodes already in the pool (real sims are not bit-stable).
    assert second["episodes_skipped_already_in_pool"] == 10
    assert second["states_idempotently_skipped"] == 0
    assert manifest_path.read_bytes() == manifest_before


def test_collect_passes_configured_levels_to_sampler(tmp_path):
    config = {
        "adapter": None,
        "collection": {
            "output_dir": str(tmp_path / "pool"),
            "episodes": 5,
            "seed": 7,
            "action_chunks_per_episode": 2,
            "snapshot_cadence_action_chunks": 2,
            "successful_snapshot_retention": 0.20,
            "dry_run": True,
        },
        "sampling": {
            "dimension_quotas": {"camera": 1},
            "levels_by_dimension": {"camera": [1, 2]},
        },
    }
    collect(config)
    pool = StatePool(tmp_path / "pool")
    manifest = pool.manifest()
    levels = {pool.read_state(key).metadata.level for key in manifest["states"]}
    assert levels
    assert levels <= {1, 2}


def test_failure_retention_and_cadence_are_complete():
    assert list(snapshot_steps(7, cadence=2)) == [0, 2, 4, 6]
    assert all(retain_snapshot("failure", "ep", step, 3) for step in range(100))
    success_a = [
        step for step in range(1000) if retain_snapshot("success", "ep", step, 3)
    ]
    success_b = [
        step for step in range(1000) if retain_snapshot("success", "ep", step, 3)
    ]
    assert success_a == success_b
    assert 150 <= len(success_a) <= 250


def test_same_key_with_changed_payload_is_rejected(tmp_path):
    metadata = StateMetadata(
        task_id="task",
        instruction="instruction",
        suite="Goal",
        episode_id="episode",
        step=0,
        perturb_dim="robot",
        perturb_sub="initial_state",
        level=3,
        episode_outcome="failure",
        seed=1,
    )
    pool = StatePool(tmp_path)
    arguments = {
        "sim_state": np.zeros(2),
        "controller_state": {},
        "rng_state": {},
        "observations": {"agentview": b"png"},
        "proprio": np.zeros(2),
    }
    pool.write_state(metadata, **arguments)
    arguments["sim_state"] = np.ones(2)
    with pytest.raises(FileExistsError, match="non-idempotent"):
        pool.write_state(metadata, **arguments)
