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


def _row(key: str, classification: str, direct: bool, deferred: bool):
    return {
        "state_key": key,
        "classification": classification,
        "direct_oft_success": direct,
        "decision_suffix_oft_success": deferred,
        "arms": [
            {"arm_label": "direct_oft", "prefix_sha256": "0" * 64},
            {"arm_label": "decision_suffix_oft", "prefix_sha256": "a" * 64},
        ],
    }


def test_select_disagreements_preserves_records_and_valid_checksum():
    keys = {
        "artifact_version": "source/v1",
        "records": [
            {"state_key": "a", "task_id": "task-a"},
            {"state_key": "b", "task_id": "task-b"},
            {"state_key": "c", "task_id": "task-c"},
        ],
    }
    analysis = {
        "schema_version": "rase-deferred-switch-analysis/v1",
        "per_state": [
            _row("a", "direct_only", True, False),
            _row("b", "both", True, True),
            _row("c", "deferred_only", False, True),
        ],
    }
    result = _module("select_deferred_disagreement_keys").select(keys, analysis)
    assert result["state_keys"] == ["a", "c"]
    assert [row["task_id"] for row in result["records"]] == ["task-a", "task-c"]
    rollout = _module("rollout_oft_prefix_ablation")
    assert result["state_keys_sha256"] == rollout._checksum(["a", "c"])


def test_replay_audit_requires_exact_disagreement_coverage_and_outcomes():
    reference = {
        "per_state": [
            _row("a", "direct_only", True, False),
            _row("b", "both", True, True),
            _row("c", "deferred_only", False, True),
        ]
    }
    replay = {
        "per_state": [
            _row("a", "direct_only", True, False),
            _row("c", "deferred_only", False, True),
        ]
    }
    result = _module("audit_deferred_replay").audit(reference, replay)
    assert result["status"] == "pass"
    replay["per_state"][1]["decision_suffix_oft_success"] = False
    assert _module("audit_deferred_replay").audit(reference, replay)["status"] == "mismatch"


def test_replay_audit_rejects_partial_disagreement_subset():
    reference = {
        "per_state": [
            _row("a", "direct_only", True, False),
            _row("c", "deferred_only", False, True),
        ]
    }
    replay = {"per_state": [_row("a", "direct_only", True, False)]}
    with pytest.raises(ValueError, match="disagreement set"):
        _module("audit_deferred_replay").audit(reference, replay)
