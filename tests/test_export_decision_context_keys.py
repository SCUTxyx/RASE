from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_decision_context_keys.py"
    spec = importlib.util.spec_from_file_location("export_decision_context_keys", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_records_filters_step_and_episode_before_task_round_robin():
    records = [
        {"state_key": "a0", "task_id": "a", "episode_id": "ea", "step": 0},
        {"state_key": "a2", "task_id": "a", "episode_id": "ea", "step": 2},
        {"state_key": "b2", "task_id": "b", "episode_id": "eb", "step": 2},
        {"state_key": "b2x", "task_id": "b", "episode_id": "eb", "step": 2},
        {"state_key": "c2", "task_id": "c", "episode_id": "ec", "step": 2},
    ]
    selected = _module().select_records(
        records, step=2, one_per_episode=True, max_states=2
    )
    assert [row["state_key"] for row in selected] == ["a2", "b2"]
