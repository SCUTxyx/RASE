from __future__ import annotations

import json
from pathlib import Path

from rase.vnext.confirmation import build_confirmation_manifest


CONFIG = Path(__file__).parents[1] / "configs" / "rase_vnext_protocol_v1.json"


def _catalog() -> list[dict]:
    return [
        {
            "root_id": f"root-{suite}-{task}-{root}",
            "state_key": f"state-{suite}-{task}-{root}",
            "task_id": f"task-{suite}-{task}", "suite": suite,
            "init_state_id": root, "environment_seed": 1000 + root,
            "restore_state_ref": f"pool/{suite}/{task}/{root}",
        }
        for suite in ("Goal", "Long")
        for task in range(2)
        for root in range(3)
    ]


def test_confirmation_is_disjoint_all_task_fixed_k() -> None:
    protocol = json.loads(CONFIG.read_text())
    excluded = {"root-Goal-0-0", "root-Long-1-0"}
    manifest = build_confirmation_manifest(
        _catalog(), protocol, discovery_root_ids=excluded,
        roots_per_task=1, salt="test",
        operator_masks={("pi0fast.libero", "resample.source"): "no_diversity"},
    )
    assert manifest["status"] == "frozen_confirmation"
    assert manifest["expected_roots"] == 4
    assert not ({row["root_id"] for row in manifest["roots"]} & excluded)
    assert manifest["expected_jobs"] == 4 * 2 * 2 * 5 * 5
    assert manifest["expected_available_jobs"] == 4 * 2 * 2 * 5 * 5 - 4 * 2 * 5
    assert {job["seed_ledger"]["exact_repeat_replica"] for job in manifest["jobs"]} == set(range(5))
    masked = [job for job in manifest["jobs"] if not job["available_by_contract"]]
    assert masked
    assert {job["operator_id"] for job in masked} == {"resample.source"}
    assert {job["policy_id"] for job in masked} == {"pi0fast.libero"}


def test_confirmation_selection_is_order_independent() -> None:
    protocol = json.loads(CONFIG.read_text())
    first = build_confirmation_manifest(
        _catalog(), protocol, discovery_root_ids=set(), roots_per_task=1, salt="same",
    )
    second = build_confirmation_manifest(
        list(reversed(_catalog())), protocol, discovery_root_ids=set(), roots_per_task=1,
        salt="same",
    )
    assert first["roots"] == second["roots"]
    assert first["jobs"] == second["jobs"]

