import json

import pytest

from rase.collect.state_pool import StatePool
from rase.collect.stratified_sample import (
    inventory_cell_counts,
    remaining_steps,
    sample_stratified_keys,
)


def _write_state(
    root,
    key,
    *,
    suite,
    dim,
    level,
    step=10,
    outcome="failure",
    episode_id=None,
):
    path = root / "states" / key
    path.mkdir(parents=True)
    (path / "meta.json").write_text(
        json.dumps(
            {
                "suite": suite,
                "perturb_dim": dim,
                "level": level,
                "task_id": "task-0",
                "episode_id": episode_id or f"episode-{key}",
                "seed": 0,
                "instruction": "x",
                "step": step,
                "episode_outcome": outcome,
            }
        ),
        encoding="utf-8",
    )
    return f"states/{key}"


def test_sample_stratified_keys_one_per_cell(tmp_path):
    root = tmp_path / "pool"
    states = {}
    suites = ("Spatial", "Object", "Goal", "Long")
    dims = ("camera", "robot")
    n = 0
    for suite in suites:
        for dim in dims:
            for i in range(3):
                key = f"sp1_{n:032d}"
                n += 1
                rel = _write_state(root, key, suite=suite, dim=dim, level=3 + (i % 3))
                states[key] = {"path": rel}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "ngc-state-pool-manifest/v1",
                "schema_version": "ngc-state-pool/v1",
                "states": states,
            }
        ),
        encoding="utf-8",
    )
    pool = StatePool(root)
    keys = sample_stratified_keys(pool, per_cell=1, seed=1)
    assert len(keys) == 8
    # Stable under same seed.
    assert keys == sample_stratified_keys(pool, per_cell=1, seed=1)


def test_remaining_steps_uses_suite_horizon():
    assert remaining_steps({"suite": "Spatial", "step": 10}) == 270
    assert remaining_steps({"suite": "libero_10", "timestep": 410}) == 110
    assert remaining_steps({"suite": "Goal", "step": 290}) == 10


def test_sample_stratified_keys_min_remaining_filters(tmp_path):
    root = tmp_path / "pool"
    states = {}
    # One early and one late state per Spatial×camera cell; other cells early.
    n = 0
    for suite in ("Spatial", "Object", "Goal", "Long"):
        for dim in ("camera", "robot"):
            for step in (10, 270):
                key = f"sp1_{n:032d}"
                n += 1
                # Long horizon 520 → step 270 still adequate; Spatial step 270 → rem=10.
                rel = _write_state(
                    root, key, suite=suite, dim=dim, level=3, step=step
                )
                states[key] = {"path": rel}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "ngc-state-pool-manifest/v1",
                "schema_version": "ngc-state-pool/v1",
                "states": states,
            }
        ),
        encoding="utf-8",
    )
    pool = StatePool(root)
    keys = sample_stratified_keys(pool, per_cell=1, seed=0, min_remaining_steps=100)
    assert len(keys) == 8
    for key in keys:
        meta = json.loads((root / "states" / key / "meta.json").read_text())
        assert remaining_steps(meta) >= 100

    with pytest.raises(ValueError, match="min_remaining_steps"):
        sample_stratified_keys(
            pool,
            per_cell=2,
            seed=0,
            suites=("Spatial",),
            dims=("camera",),
            min_remaining_steps=100,
        )


def test_sample_stratified_keys_max_t0_prefers_early(tmp_path):
    root = tmp_path / "pool"
    states = {}
    n = 0
    for suite in ("Spatial", "Object", "Goal", "Long"):
        for dim in ("camera", "robot"):
            for step in (5, 15, 50):
                key = f"sp1_{n:032d}"
                n += 1
                rel = _write_state(
                    root, key, suite=suite, dim=dim, level=3, step=step
                )
                states[key] = {"path": rel}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "ngc-state-pool-manifest/v1",
                "schema_version": "ngc-state-pool/v1",
                "states": states,
            }
        ),
        encoding="utf-8",
    )
    pool = StatePool(root)
    keys = sample_stratified_keys(
        pool, per_cell=2, seed=0, min_remaining_steps=100, max_t0=40
    )
    assert len(keys) == 16
    for key in keys:
        meta = json.loads((root / "states" / key / "meta.json").read_text())
        assert meta["step"] <= 40
        assert remaining_steps(meta) >= 100
    # Earliest forks preferred within each cell (steps 5 then 15, never 50).
    spatial_cam = [
        json.loads((root / "states" / k / "meta.json").read_text())["step"]
        for k in keys[:2]
    ]
    assert spatial_cam == [5, 15]


def test_inventory_cell_counts(tmp_path):
    root = tmp_path / "pool"
    states = {}
    n = 0
    for suite in ("Spatial", "Object", "Goal", "Long"):
        for dim in ("camera", "robot"):
            for _ in range(2):
                key = f"sp1_{n:032d}"
                n += 1
                rel = _write_state(root, key, suite=suite, dim=dim, level=3, step=10)
                states[key] = {"path": rel}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "ngc-state-pool-manifest/v1",
                "schema_version": "ngc-state-pool/v1",
                "states": states,
            }
        ),
        encoding="utf-8",
    )
    inv = inventory_cell_counts(StatePool(root), min_remaining_steps=100)
    assert inv["total"] == 16
    assert inv["max_per_cell"] == 2
    assert inv["n_cells"] == 8


def test_frontier_strata_outcome_exclusion_and_exact_inventory(tmp_path):
    root = tmp_path / "pool"
    states = {}
    specs = [
        ("keep-early", 3, 5, "failure"),
        ("excluded", 3, 10, "failure"),
        ("success", 3, 15, "success"),
        ("keep-3-late", 3, 25, "failure"),
        ("keep-4-early", 4, 5, "failure"),
        ("keep-4-late", 4, 25, "failure"),
        ("outside", 4, 50, "failure"),
    ]
    for index, (label, level, step, outcome) in enumerate(specs):
        key = f"sp1_{index:032d}"
        rel = _write_state(
            root,
            key,
            suite="Spatial",
            dim="camera",
            level=level,
            step=step,
            outcome=outcome,
        )
        states[key] = {"path": rel, "label": label}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "ngc-state-pool-manifest/v1",
                "schema_version": "ngc-state-pool/v1",
                "states": states,
            }
        ),
        encoding="utf-8",
    )
    pool = StatePool(root)
    bins = {"early": [0, 20], "late": [20, 40]}
    excluded = {"sp1_00000000000000000000000000000001"}
    kwargs = {
        "dims": ("camera",),
        "suites": ("Spatial",),
        "levels": (3, 4),
        "strata": ("suite", "dim", "level", "t0_bin"),
        "t0_bins": bins,
        "episode_outcomes": ("failure",),
        "excluded_keys": excluded,
    }
    inventory = inventory_cell_counts(pool, **kwargs)
    assert inventory["total"] == 4
    assert inventory["n_cells"] == 4
    assert inventory["max_per_cell"] == 1
    assert inventory["cells"] == [
        {"suite": "Spatial", "dim": "camera", "level": 3, "t0_bin": "early", "n": 1},
        {"suite": "Spatial", "dim": "camera", "level": 3, "t0_bin": "late", "n": 1},
        {"suite": "Spatial", "dim": "camera", "level": 4, "t0_bin": "early", "n": 1},
        {"suite": "Spatial", "dim": "camera", "level": 4, "t0_bin": "late", "n": 1},
    ]
    assert inventory["audit"] == {
        "excluded": 1,
        "excluded_episode_group": 0,
        "missing_metadata": 0,
        "outside_t0_bins": 1,
        "missing_episode_group": 0,
    }
    assert sample_stratified_keys(pool, per_cell=1, seed=2, **kwargs) == [
        "sp1_00000000000000000000000000000000",
        "sp1_00000000000000000000000000000003",
        "sp1_00000000000000000000000000000004",
        "sp1_00000000000000000000000000000005",
    ]


def test_random_selection_is_seeded_and_not_earliest(tmp_path):
    root = tmp_path / "pool"
    states = {}
    for index, step in enumerate(range(8)):
        key = f"sp1_{index:032d}"
        states[key] = {
            "path": _write_state(
                root, key, suite="Spatial", dim="camera", level=3, step=step
            )
        }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "ngc-state-pool-manifest/v1",
                "schema_version": "ngc-state-pool/v1",
                "states": states,
            }
        ),
        encoding="utf-8",
    )
    pool = StatePool(root)
    common = {
        "per_cell": 2,
        "seed": 5,
        "suites": ("Spatial",),
        "dims": ("camera",),
        "levels": (3,),
    }
    earliest = sample_stratified_keys(pool, selection="earliest", **common)
    random_keys = sample_stratified_keys(pool, selection="random", **common)
    assert earliest == [
        "sp1_00000000000000000000000000000000",
        "sp1_00000000000000000000000000000001",
    ]
    assert random_keys == sample_stratified_keys(pool, selection="random", **common)
    assert random_keys != earliest


def test_episode_level_exclusion_removes_all_snapshots_from_source_episode(tmp_path):
    root = tmp_path / "pool"
    states = {}
    for index, (episode, step) in enumerate(
        (("pilot", 5), ("pilot", 15), ("heldout-a", 5), ("heldout-b", 5))
    ):
        key = f"sp1_{index:032d}"
        states[key] = {
            "path": _write_state(
                root,
                key,
                suite="Spatial",
                dim="camera",
                level=1,
                step=step,
                episode_id=episode,
            )
        }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "ngc-state-pool-manifest/v1",
                "schema_version": "ngc-state-pool/v1",
                "states": states,
            }
        ),
        encoding="utf-8",
    )
    pool = StatePool(root)
    kwargs = {
        "dims": ("camera",),
        "suites": ("Spatial",),
        "levels": (1,),
        "strata": ("dim", "level"),
        "distinct_episodes": True,
        "excluded_episode_keys": {"sp1_00000000000000000000000000000000"},
    }
    inventory = inventory_cell_counts(pool, **kwargs)
    assert inventory["total"] == 2
    assert inventory["audit"]["excluded_episode_group"] == 2
    keys = sample_stratified_keys(pool, per_cell=2, seed=0, **kwargs)
    assert set(keys) == {
        "sp1_00000000000000000000000000000002",
        "sp1_00000000000000000000000000000003",
    }


def test_frontier_sampling_rejects_invalid_configuration(tmp_path):
    pool = StatePool(tmp_path / "pool")
    with pytest.raises(ValueError, match="t0_bins are required"):
        sample_stratified_keys(pool, strata=("t0_bin",))
    with pytest.raises(ValueError, match="overlap"):
        sample_stratified_keys(
            pool,
            strata=("t0_bin",),
            t0_bins={"first": [0, 20], "second": [10, 30]},
        )
    with pytest.raises(ValueError, match="selection"):
        sample_stratified_keys(pool, selection="newest")
    with pytest.raises(ValueError, match="episode_outcomes"):
        sample_stratified_keys(pool, strata=("outcome",))


def test_outcome_strata_and_distinct_episode_sampling(tmp_path):
    root = tmp_path / "pool"
    states = {}
    index = 0
    for outcome in ("success", "failure"):
        for episode in range(3):
            for step in (5, 15):
                key = f"sp1_{index:032d}"
                index += 1
                states[key] = {
                    "path": _write_state(
                        root,
                        key,
                        suite="Spatial",
                        dim="camera",
                        level=1,
                        step=step,
                        outcome=outcome,
                        episode_id=f"{outcome}-{episode}",
                    )
                }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "ngc-state-pool-manifest/v1",
                "schema_version": "ngc-state-pool/v1",
                "states": states,
            }
        ),
        encoding="utf-8",
    )
    kwargs = {
        "dims": ("camera",),
        "suites": ("Spatial",),
        "levels": (1,),
        "strata": ("outcome",),
        "episode_outcomes": ("success", "failure"),
        "distinct_episodes": True,
    }
    pool = StatePool(root)
    inventory = inventory_cell_counts(pool, **kwargs)
    assert inventory["cells"] == [
        {"outcome": "success", "n": 3},
        {"outcome": "failure", "n": 3},
    ]
    keys = sample_stratified_keys(pool, per_cell=2, seed=3, **kwargs)
    assert len(keys) == 4
    episodes = []
    outcomes = []
    for key in keys:
        meta = json.loads((root / "states" / key / "meta.json").read_text())
        episodes.append(meta["episode_id"])
        outcomes.append(meta["episode_outcome"])
    assert len(set(episodes)) == 4
    assert outcomes.count("success") == 2
    assert outcomes.count("failure") == 2
