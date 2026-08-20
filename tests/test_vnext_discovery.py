from __future__ import annotations

import json
from pathlib import Path

import pytest

from rase.vnext.discovery import build_discovery_manifest, validate_root_catalog


CONFIG = Path(__file__).parents[1] / "configs" / "rase_vnext_protocol_v1.json"


def _catalog() -> list[dict]:
    rows = []
    for suite in ("Goal", "Spatial"):
        for task_index in range(3):
            for root_index in range(2):
                rows.append({
                    "root_id": f"{suite}-{task_index}-{root_index}",
                    "state_key": f"state-{suite}-{task_index}-{root_index}",
                    "task_id": f"{suite}-task-{task_index}", "suite": suite,
                    "init_state_id": root_index, "environment_seed": 100 + root_index,
                    "restore_state_ref": f"pool/{suite}/{task_index}/{root_index}",
                })
    return rows


def test_catalog_rejects_any_nested_outcome_field() -> None:
    rows = _catalog()
    rows[0]["metadata"] = {"success": True}
    with pytest.raises(ValueError, match="outcome-derived"):
        validate_root_catalog(rows)


def test_manifest_is_fixed_k_and_task_folds_do_not_leak() -> None:
    protocol = json.loads(CONFIG.read_text())
    manifest = build_discovery_manifest(_catalog(), protocol, salt="test")
    assert manifest["expected_roots"] == 8
    assert manifest["expected_jobs"] == 8 * 2 * 2 * 5 * 3
    assert {job["seed_ledger"]["exact_repeat_replica"] for job in manifest["jobs"]} == {0, 1, 2}
    task_to_folds = {}
    for job in manifest["jobs"]:
        task_to_folds.setdefault(job["task_id"], set()).add(job["outer_fold"])
    assert all(len(folds) == 1 for folds in task_to_folds.values())


def test_selection_is_deterministic_and_outcome_independent() -> None:
    protocol = json.loads(CONFIG.read_text())
    first = build_discovery_manifest(_catalog(), protocol, salt="same")
    second = build_discovery_manifest(list(reversed(_catalog())), protocol, salt="same")
    assert [row["root_id"] for row in first["roots"]] == [row["root_id"] for row in second["roots"]]
    assert first["jobs"] == second["jobs"]
