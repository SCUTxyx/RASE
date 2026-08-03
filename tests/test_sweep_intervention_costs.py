from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sweep_intervention_costs.py"
    spec = importlib.util.spec_from_file_location("sweep_intervention_costs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cost_point_exposes_three_unique_utility_winners():
    operators = ["continue", "replan", "switch"]
    rows = [
        {"task_id": "a", "success": {"continue": 1, "replan": 1, "switch": 1}},
        {"task_id": "b", "success": {"continue": 0, "replan": 1, "switch": 1}},
        {"task_id": "c", "success": {"continue": 0, "replan": 0, "switch": 1}},
        {"task_id": "d", "success": {"continue": 0, "replan": 0, "switch": 0}},
    ]

    result = _module().evaluate_cost_point(
        rows,
        operators,
        {"continue": 0.0, "replan": 0.01, "switch": 0.1},
    )

    assert result["best_fixed_operator"] == "switch"
    assert result["best_fixed_utility"] == pytest.approx(0.65)
    assert result["same_state_oracle_utility"] == pytest.approx(0.7225)
    assert result["unique_winner_state_counts"] == {
        "continue": 2,
        "replan": 1,
        "switch": 1,
    }
    assert result["success_supported_unique_winner_state_counts"] == {
        "continue": 1,
        "replan": 1,
        "switch": 1,
    }
    assert result["failure_only_cost_winner_state_counts"] == {
        "continue": 1,
        "replan": 0,
        "switch": 0,
    }
    assert result["n_no_success_states"] == 1
