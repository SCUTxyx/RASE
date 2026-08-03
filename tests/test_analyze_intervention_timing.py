from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_intervention_timing.py"
    spec = importlib.util.spec_from_file_location("analyze_intervention_timing", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_analysis_requires_exact_coverage_and_separates_scopes():
    keys = {
        "state_keys": ["s1", "s2"],
        "records": [
            {"state_key": "s1", "task_id": "a", "episode_id": "e1"},
            {"state_key": "s2", "task_id": "b", "episode_id": "e2"},
        ],
    }
    smol_rows = []
    for index, key in enumerate(keys["state_keys"]):
        smol_rows.append(
            {
                "state_key": key,
                "continue_smol_active_chunk": index == 0,
                "continue_smol_active_chunk_action_select_calls": 2,
                "continue_smol_active_chunk_action_select_elapsed_s": 0.2,
                "continue_smol_active_chunk_latency_seconds": 1.0,
                "continue_smol_active_chunk_env_steps": 4,
                "replan_smol": False,
                "replan_smol_action_select_calls": 4,
                "replan_smol_action_select_elapsed_s": 0.4,
                "replan_smol_latency_seconds": 1.5,
                "replan_smol_env_steps": 5,
            }
        )
    oft = {
        "per_state": [
            {
                "state_key": key,
                "direct_oft_success": True,
                "result": {
                    "oracle_predict_calls": 1,
                    "oracle_predict_elapsed_s": 0.1,
                    "elapsed_s": 0.8,
                    "env_steps": 3,
                },
            }
            for key in keys["state_keys"]
        ]
    }
    result = _module().analyze(keys, {"per_pair": smol_rows}, [oft])
    assert result["n_states"] == 2
    assert result["n_episodes"] == 2
    assert result["operators"]["continue_smol_active_chunk"]["successes"] == 1
    assert result["operators"]["continue_smol_active_chunk"][
        "mean_ms_per_policy_call"
    ] == pytest.approx(100)
    assert result["operators"]["switch_oft"]["total_policy_calls"] == 2
    assert result["relative_to_continue"]["switch_oft"][
        "policy_ms_per_env_step"
    ] == pytest.approx(2 / 3)
    assert "do not equate per-call" in result["comparability_warning"]
    assert "action-chunk amortization" in result["normalization_note"]


def test_analysis_rejects_missing_oft_state():
    with pytest.raises(ValueError, match="OFT timing coverage"):
        _module().analyze(
            {
                "state_keys": ["s1"],
                "records": [{"task_id": "a", "episode_id": "e1"}],
            },
            {
                "per_pair": [
                    {
                        "state_key": "s1",
                        "continue_smol_active_chunk_action_select_calls": 1,
                        "continue_smol_active_chunk_action_select_elapsed_s": 0.1,
                        "replan_smol_action_select_calls": 1,
                        "replan_smol_action_select_elapsed_s": 0.1,
                    }
                ]
            },
            [],
        )
