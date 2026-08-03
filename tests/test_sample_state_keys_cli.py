import json

from scripts.audit_state_key_split import audit_split
from scripts.collect_state_pool import _apply_collection_overrides
from scripts.rollout_pool_candidates import _state_keys_checksum
from scripts.sample_state_keys import _checksum, _coverage_status, _sample_kwargs


def test_sample_cli_maps_frontier_config_and_exclusion_artifacts(tmp_path):
    excluded = tmp_path / "prior.json"
    excluded.write_text(
        json.dumps({"state_keys": ["prior-a", "prior-b"]}), encoding="utf-8"
    )
    kwargs = _sample_kwargs(
        {
            "per_cell": 3,
            "sample_seed": 9,
            "strata": ["suite", "dim", "level", "t0_bin"],
            "t0_bins": {"early": [0, 40], "late": [40, None]},
            "selection": "random",
            "episode_outcome": "failure",
            "excluded_keys": ["inline"],
            "excluded_keys_json": str(excluded),
        }
    )
    assert kwargs["per_cell"] == 3
    assert kwargs["seed"] == 9
    assert kwargs["strata"] == ("suite", "dim", "level", "t0_bin")
    assert kwargs["selection"] == "random"
    assert kwargs["episode_outcomes"] == ("failure",)
    assert kwargs["excluded_keys"] == {"inline", "prior-a", "prior-b"}


def test_sample_cli_maps_episode_level_exclusion_artifact(tmp_path):
    excluded = tmp_path / "pilot.json"
    excluded.write_text(json.dumps({"state_keys": ["pilot-a"]}), encoding="utf-8")
    kwargs = _sample_kwargs({"excluded_episode_keys_json": str(excluded)})
    assert kwargs["excluded_episode_keys"] == {"pilot-a"}


def test_sample_artifact_checksum_matches_rollout_consumer():
    keys = ["sp1_a", "sp1_b"]
    assert _checksum(keys) == _state_keys_checksum(keys)


def test_coverage_status_lists_exact_deficits():
    inventory = {
        "cells": [
            {"outcome": "success", "dim": "camera", "level": 1, "n": 0},
            {"outcome": "failure", "dim": "camera", "level": 1, "n": 3},
        ]
    }
    complete, deficits = _coverage_status(inventory, per_cell=2)
    assert not complete
    assert deficits == [
        {
            "outcome": "success",
            "dim": "camera",
            "level": 1,
            "n": 0,
            "missing": 2,
        }
    ]


def test_collection_batch_overrides_are_copied_and_validated():
    config = {"collection": {"episodes": 40, "seed": 1}}
    updated = _apply_collection_overrides(config, episodes=24, seed=9)
    assert updated["collection"] == {"episodes": 24, "seed": 9}
    assert config["collection"] == {"episodes": 40, "seed": 1}


def test_heldout_audit_rejects_episode_group_leakage():
    metadata = {
        "pilot": {
            "task_id": "task",
            "episode_id": "shared",
            "suite": "Spatial",
            "perturb_dim": "camera",
            "level": 1,
            "step": 0,
        },
        "leak": {
            "task_id": "task",
            "episode_id": "shared",
            "suite": "Spatial",
            "perturb_dim": "camera",
            "level": 1,
            "step": 20,
        },
    }
    result = audit_split(["pilot"], ["leak"], metadata)
    assert not result["valid"]
    assert result["state_overlap"] == []
    assert result["episode_group_overlap"] == [["task", "shared"]]
    assert result["heldout_snapshot_steps"] == {
        "n": 1,
        "n_missing": 0,
        "min": 20,
        "median": 20,
        "max": 20,
        "unique": [20],
    }
    assert result["heldout_snapshot_steps_by_suite"]["Spatial"]["median"] == 20
