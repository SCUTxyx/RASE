from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from rase.collect.lerobot_libero_plus_adapter import (
    _lerobot_env_kwargs,
    _validated_init_state_id,
)
from rase.collect.pipeline import collect, load_config
from rase.collect.schema import StateMetadata
from rase.collect.state_pool import StatePool, retain_snapshot
from rase.collect.w9b_schedule import (
    BATCH_SIZES,
    PROTOCOL_VERSION,
    generate_w9b_schedule,
    load_w9b_schedule,
    requests_for_batch,
    schedule_bytes,
    schedule_sha256,
    validate_w9b_schedule,
    write_w9b_schedule,
)


def test_schedule_is_byte_deterministic_and_seed_sensitive():
    first = generate_w9b_schedule(20260730)
    second = generate_w9b_schedule(20260730)
    changed = generate_w9b_schedule(20260731)
    assert schedule_bytes(first) == schedule_bytes(second)
    assert schedule_sha256(first) == schedule_sha256(second)
    assert schedule_bytes(first) != schedule_bytes(changed)


def test_schedule_batches_are_balanced_and_do_not_reset_init_allocation():
    payload = generate_w9b_schedule(20260730)
    rows = validate_w9b_schedule(payload)
    assert [len(requests_for_batch(payload, batch)) for batch in (1, 2, 3)] == list(
        BATCH_SIZES
    )
    for batch_id, expected_size in enumerate(BATCH_SIZES, 1):
        batch = [row for row in rows if row.batch_id == batch_id]
        assert Counter(row.suite for row in batch) == {
            suite: expected_size // 4
            for suite in ("Goal", "Long", "Object", "Spatial")
        }

    per_task: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        per_task[(row.suite, row.task_id)].append(row.init_state_id)
    assert all(len(values) == len(set(values)) for values in per_task.values())

    # A task spanning batches continues its allocation rather than restarting at 0.
    spanning = [
        key
        for key in per_task
        if len({row.batch_id for row in rows if (row.suite, row.task_id) == key}) > 1
    ]
    assert spanning
    for key in spanning:
        task_rows = [
            row for row in rows if (row.suite, row.task_id) == key
        ]
        assert len({row.init_state_id for row in task_rows}) == len(task_rows)


def test_task_and_init_use_independent_salted_mappings():
    rows = validate_w9b_schedule(generate_w9b_schedule(20260730))
    assert any((row.task_id - 1) != row.init_state_id % 10 for row in rows)
    by_task: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        by_task[(row.suite, row.task_id)].add(row.init_state_id % 10)
    assert any(len(residues) > 1 for residues in by_task.values())
    assert len({row.policy_seed for row in rows}) == len(rows)


def test_schedule_validation_rejects_missing_duplicate_and_out_of_range_rows():
    payload = generate_w9b_schedule(20260730)

    missing = copy.deepcopy(payload)
    missing["rows"].pop()
    with pytest.raises(ValueError, match="140 rows"):
        validate_w9b_schedule(missing)

    duplicate = copy.deepcopy(payload)
    duplicate["rows"][1]["episode_id"] = duplicate["rows"][0]["episode_id"]
    with pytest.raises(ValueError, match="duplicate episode_id"):
        validate_w9b_schedule(duplicate)

    invalid_init = copy.deepcopy(payload)
    invalid_init["rows"][0]["init_state_id"] = 50
    with pytest.raises(ValueError, match="out of range"):
        validate_w9b_schedule(invalid_init)


def test_schedule_sha_mismatch_fails_closed(tmp_path):
    path = tmp_path / "schedule.json"
    payload = generate_w9b_schedule(20260730)
    digest = write_w9b_schedule(path, payload)
    assert load_w9b_schedule(path, expected_sha256=digest) == payload
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_w9b_schedule(path, expected_sha256="0" * 64)


def test_adapter_kwargs_pass_explicit_init_as_episode_index():
    kwargs = _lerobot_env_kwargs(
        suite=object(),
        task_index=4,
        suite_name="libero_spatial",
        camera_name="agentview_image",
        init_state_id=17,
        obs_type="pixels_agent_pos",
        observation_height=360,
        observation_width=360,
        control_mode="relative",
    )
    assert kwargs["episode_index"] == 17
    assert kwargs["task_id"] == 4
    assert kwargs["n_envs"] == 1


def test_scheduled_init_is_required_and_bounds_checked():
    request = requests_for_batch(generate_w9b_schedule(20260730), 1)[0]
    assert _validated_init_state_id(request, n_init_states=50, required=True) == (
        request.init_state_id
    )
    missing = copy.copy(request)
    object.__setattr__(missing, "init_state_id", None)
    with pytest.raises(ValueError, match="requires init_state_id"):
        _validated_init_state_id(missing, n_init_states=50, required=True)
    invalid = copy.copy(request)
    object.__setattr__(invalid, "init_state_id", 50)
    with pytest.raises(ValueError, match="out of range"):
        _validated_init_state_id(invalid, n_init_states=50, required=True)


def _metadata(**changes) -> StateMetadata:
    values = {
        "task_id": "libero_goal_000001",
        "instruction": "test",
        "suite": "Goal",
        "episode_id": "ep-w9b-00000000",
        "step": 0,
        "perturb_dim": "clean",
        "perturb_sub": "none",
        "level": 0,
        "episode_outcome": "success",
        "seed": 7,
        "init_state_id": 13,
    }
    values.update(changes)
    return StateMetadata(**values)


def test_init_state_metadata_roundtrip_and_legacy_compatibility(tmp_path):
    current = _metadata()
    current_dict = current.to_dict()
    restored = StateMetadata.from_dict(current_dict)
    assert restored.init_state_id == 13

    legacy_dict = current_dict.copy()
    legacy_dict.pop("init_state_id")
    legacy = StateMetadata.from_dict(legacy_dict)
    assert legacy.init_state_id is None
    assert legacy.state_key == current.state_key

    pool = StatePool(tmp_path)
    written = pool.write_state(
        current,
        sim_state=np.zeros(2),
        controller_state={},
        rng_state={},
        observations={"agentview": b"png"},
        proprio=np.zeros(2),
    )
    assert pool.read_state(written.state_key).metadata.init_state_id == 13
    assert pool.manifest()["states"][written.state_key]["init_state_id"] == 13


def test_success_retention_one_and_legacy_point_two():
    assert all(
        retain_snapshot("success", "ep", step, 3, 1.0) for step in range(100)
    )
    legacy = [
        step
        for step in range(1000)
        if retain_snapshot("success", "ep", step, 3, 0.20)
    ]
    assert legacy == [
        step
        for step in range(1000)
        if retain_snapshot("success", "ep", step, 3, 0.20)
    ]
    assert 150 <= len(legacy) <= 250


def _w9b_test_config(tmp_path: Path) -> tuple[dict, Path]:
    schedule_path = tmp_path / "schedule.json"
    payload = generate_w9b_schedule(20260730)
    digest = write_w9b_schedule(schedule_path, payload)
    output = tmp_path / "runs" / "ngc_w9b_resume_smoke"
    config = {
        "run_name": "ngc-w9b-test-smoke",
        "protocol": {
            "version": PROTOCOL_VERSION,
            "schedule_path": str(schedule_path),
            "schedule_sha256": digest,
            "maximum_episodes": 140,
        },
        "adapter": None,
        "collection": {
            "output_dir": str(output),
            "episodes": 60,
            "seed": 20260730,
            "schedule_batch_id": 1,
            "action_chunks_per_episode": 1,
            "snapshot_cadence_action_chunks": 2,
            "successful_snapshot_retention": 1.0,
            "dry_run": True,
            "smoke_mode": True,
        },
    }
    return config, output


def test_resume_uses_same_episode_task_and_init_without_duplicates(tmp_path):
    config, output = _w9b_test_config(tmp_path)
    first = collect(config)
    manifest_before = (output / "manifest.json").read_bytes()
    second = collect(config)
    assert first["states_created"] == 60
    assert len(first["scheduled_episodes"]) == 60
    assert first["scheduled_episodes"][0]["init_state_id"] is not None
    assert first["provenance"]["schedule_sha256"]
    assert first["provenance"]["protocol_version"] == PROTOCOL_VERSION
    assert second["states_created"] == 0
    assert second["episodes_skipped_already_in_pool"] == 60
    assert (output / "manifest.json").read_bytes() == manifest_before
    entries = StatePool(output).manifest()["states"].values()
    rows = requests_for_batch(generate_w9b_schedule(20260730), 1)
    expected = {
        row.episode_id: (row.suite, row.task_id, row.init_state_id) for row in rows
    }
    assert len(entries) == 60
    for entry in entries:
        suite, task_id, init_state_id = expected[entry["episode_id"]]
        suite_name = {
            "Spatial": "libero_spatial",
            "Object": "libero_object",
            "Goal": "libero_goal",
            "Long": "libero_10",
        }[suite]
        assert entry["task_id"] == f"{suite_name}_{task_id:06d}"
        assert entry["init_state_id"] == init_state_id


def test_w9b_config_cannot_write_legacy_w9_pool(tmp_path):
    config, _ = _w9b_test_config(tmp_path)
    config["collection"]["output_dir"] = "pool/ngc_w9_clean_controls"
    config["collection"]["smoke_mode"] = False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="W9B output_dir|legacy W9"):
        load_config(path)
