from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_deferred_switch.py"
    spec = importlib.util.spec_from_file_location("analyze_deferred_switch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(key: str, direct: bool, deferred: bool):
    classification = {
        (False, False): "neither",
        (True, False): "direct_only",
        (False, True): "deferred_only",
        (True, True): "both",
    }[(direct, deferred)]
    common = {
        "env_steps": 8,
        "elapsed_s": 1.0,
        "oracle_predict_calls": 1,
        "oracle_predict_elapsed_s": 0.1,
    }
    return {
        "state_key": key,
        "suite": "Spatial",
        "dim": "clean",
        "level": 0,
        "episode_id": f"e-{key}",
        "classification": classification,
        "direct_oft_success": direct,
        "decision_suffix_oft_success": deferred,
        "arms": [
            {
                **common,
                "arm_label": "direct_oft",
                "success": direct,
                "prefix_steps": 0,
            },
            {
                **common,
                "arm_label": "decision_suffix_oft",
                "success": deferred,
                "prefix_source": "decision_context.active_action_suffix",
                "prefix_steps": 5,
                "candidate_steps": 5,
                "prefix_completed": True,
                "terminal_during_prefix": False,
                "prefix_sha256": "a" * 64,
            },
        ],
    }


def test_deferred_analysis_requires_parity_and_reports_oracle_gap():
    rows = [_row("a", True, False), _row("b", False, True), _row("c", True, True)]
    summary = {
        "schema_version": "rase-oft-decision-suffix/v1",
        "status": "complete",
        "per_state": rows,
    }
    result = _module().analyze({"state_keys": ["a", "b", "c"]}, [summary])
    assert result["prefix_parity"]["status"] == "pass"
    assert result["overall"]["oracle_minus_best_fixed"] == pytest.approx(1 / 3)
    assert result["overall"]["classification_counts"]["deferred_only"] == 1
    assert result["mcnemar_exact_p"] == 1.0


def test_deferred_analysis_rejects_wrong_prefix_source():
    row = _row("a", False, True)
    row["arms"][1]["prefix_source"] = "candidate"
    summary = {
        "schema_version": "rase-oft-decision-suffix/v1",
        "status": "complete",
        "per_state": [row],
    }
    with pytest.raises(ValueError, match="prefix parity"):
        _module().analyze({"state_keys": ["a"]}, [summary])


def test_three_operator_analysis_reports_patterns_and_unique_tasks():
    rows = [_row("a", True, False), _row("b", False, True), _row("c", True, True)]
    summary = {
        "schema_version": "rase-oft-decision-suffix/v1",
        "status": "complete",
        "per_state": rows,
    }
    continue_summary = {
        "schema_version": "rase-smol-intervention-summary/v1",
        "status": "complete",
        "per_pair": [
            {"state_key": "a", "continue_smol_active_chunk": False},
            {"state_key": "b", "continue_smol_active_chunk": False},
            {"state_key": "c", "continue_smol_active_chunk": True},
        ],
    }
    keys = {
        "state_keys": ["a", "b", "c"],
        "records": [
            {"state_key": "a", "task_id": "task-a"},
            {"state_key": "b", "task_id": "task-b"},
            {"state_key": "c", "task_id": "task-c"},
        ],
    }
    result = _module().analyze(keys, [summary], continue_summary)
    three = result["three_operator"]["overall"]
    assert three["successes"] == {
        "continue_smol_active_chunk": 1,
        "direct_oft": 2,
        "decision_suffix_oft": 2,
    }
    assert three["same_state_oracle_successes"] == 3
    assert three["oracle_minus_best_fixed"] == pytest.approx(1 / 3)
    assert three["success_pattern_counts"] == {"C0D0S1": 1, "C0D1S0": 1, "C1D1S1": 1}
    assert three["unique_success_task_counts"]["direct_oft"] == 1


def test_three_operator_analysis_rejects_coverage_mismatch():
    row = _row("a", True, False)
    summary = {
        "schema_version": "rase-oft-decision-suffix/v1",
        "status": "complete",
        "per_state": [row],
    }
    continue_summary = {
        "schema_version": "rase-smol-intervention-summary/v1",
        "status": "complete",
        "per_pair": [],
    }
    keys = {"state_keys": ["a"], "records": [{"state_key": "a", "task_id": "x"}]}
    with pytest.raises(ValueError, match="coverage"):
        _module().analyze(keys, [summary], continue_summary)
