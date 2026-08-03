from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_factorial_pool.py"
    spec = importlib.util.spec_from_file_location("audit_factorial_pool", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_records_checks_factorial_cells_and_episode_consistency():
    expected = [
        {"suite": "Spatial", "dimension": "clean", "level": 0},
        {"suite": "Spatial", "dimension": "camera", "level": 1},
    ]
    records = [
        {
            "episode_id": episode,
            "suite": "Spatial",
            "dimension": dimension,
            "level": level,
            "task_id": f"task-{index}",
            "outcome": outcome,
            "step": step,
        }
        for index, (episode, dimension, level, outcome) in enumerate(
            (("a", "clean", 0, "success"), ("b", "camera", 1, "failure"))
        )
        for step in (0, 2)
    ]

    result = _module().audit_records(records, expected)

    assert result["status"] == "ready"
    assert result["n_observed_episodes"] == 2
    assert result["n_states"] == 4
    assert result["n_unique_tasks"] == 2
    assert result["by_cell"]["Spatial|clean:L0"]["source_success_rate"] == 1.0
