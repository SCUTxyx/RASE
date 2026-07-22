import json

import pytest

from rase.collect.state_pool import StatePool
from rase.collect.stratified_sample import remaining_steps, sample_stratified_keys


def _write_state(root, key, *, suite, dim, level, step=10):
    path = root / "states" / key
    path.mkdir(parents=True)
    (path / "meta.json").write_text(
        json.dumps(
            {
                "suite": suite,
                "perturb_dim": dim,
                "level": level,
                "task_id": 0,
                "seed": 0,
                "instruction": "x",
                "step": step,
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
