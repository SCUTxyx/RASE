from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _keys():
    records = [
        {
            "state_key": f"s{i}",
            "episode_id": f"e{i}",
            "task_id": f"t{i}",
            "suite": "Spatial",
            "perturbation_dimension": "clean",
            "perturbation_level": 0,
            "step": 2,
            "suffix_steps": 5,
        }
        for i in range(3)
    ]
    checksum = _module("audit_independent_keys")._checksum(["s0", "s1", "s2"])
    return {"state_keys": ["s0", "s1", "s2"], "state_keys_sha256": checksum, "records": records}


def test_independent_key_audit_matches_frozen_design():
    keys = _keys()
    design = {
        "status": "ready",
        "records": [
            {
                "episode_id": row["episode_id"],
                "task_id": row["task_id"],
                "suite": row["suite"],
                "dimension": row["perturbation_dimension"],
                "level": row["perturbation_level"],
            }
            for row in keys["records"]
        ],
    }
    config = {
        "protocol": {"expected_states": 3, "decision_step": 2, "expected_suffix_steps": 5}
    }
    result = _module("audit_independent_keys").audit(keys, design, config)
    assert result["status"] == "ready"
    assert result["design_identity_match"] is True


def test_independent_key_audit_rejects_task_reuse():
    keys = _keys()
    keys["records"][1]["task_id"] = "t0"
    design = {"status": "ready", "records": []}
    config = {
        "protocol": {"expected_states": 3, "decision_step": 2, "expected_suffix_steps": 5}
    }
    result = _module("audit_independent_keys").audit(keys, design, config)
    assert result["status"] == "not_ready"
    assert "task ids are not unique" in result["reasons"]


def _analysis(patterns: list[tuple[bool, bool]]):
    rows = []
    for i, (direct, deferred) in enumerate(patterns):
        classification = {
            (False, False): "neither",
            (True, False): "direct_only",
            (False, True): "deferred_only",
            (True, True): "both",
        }[(direct, deferred)]
        rows.append(
            {
                "state_key": f"s{i}",
                "task_id": f"t{i}",
                "episode_id": f"e{i}",
                "classification": classification,
                "direct_oft_success": direct,
                "decision_suffix_oft_success": deferred,
            }
        )
    return {"three_operator": {"per_state": rows}}


def test_timing_gate_requires_bidirectional_task_support_and_gap():
    module = _module("audit_timing_opportunity")
    ready = module.audit(
        _analysis([(True, False), (True, False), (False, True), (False, True)]),
        min_gap=0.05,
        min_tasks_per_timing=2,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )
    assert ready["status"] == "ready"
    assert ready["oracle_minus_best_fixed"] == pytest.approx(0.5)
    blocked = module.audit(
        _analysis([(True, False), (True, True), (True, True), (False, False)]),
        min_gap=0.05,
        min_tasks_per_timing=2,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )
    assert blocked["status"] == "not_ready"
    assert "insufficient deferred-only task support" in blocked["reasons"]
