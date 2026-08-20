from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts/analyze_multiscale_temperature_probe.py"
    spec = importlib.util.spec_from_file_location("multiscale_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_gate_requires_two_first_fail_rescues() -> None:
    mod = _load()
    state_keys = [f"s{i}" for i in range(4)]
    records = [
        {"state_key": key, "task_id": f"t{i // 2}", "suite": "Goal", "step": i}
        for i, key in enumerate(state_keys)
    ]
    keys = {
        "selection_uses_outcomes": False,
        "state_keys": state_keys,
        "state_keys_sha256": "sum",
        "records": records,
    }
    outcomes = [
        [False, False, True, False, False, False, False, False],
        [False, False, False, False, True, False, False, False],
        [True, True, True, True, True, True, True, True],
        [False, False, False, False, False, False, False, False],
    ]
    rollout = {
        "continuation_seed_mode": "common_root_rollout",
        "state_keys_provenance": {"selected_state_keys_sha256": "sum"},
        "per_state": [
            {
                "state_key": key,
                "candidates": [
                    {"trials": 1, "successes": int(success)} for success in values
                ],
            }
            for key, values in zip(state_keys, outcomes, strict=True)
        ],
    }
    schedule = [0.5, 0.5, 0.3, 0.3, 0.7, 0.7, 0.9, 0.9]
    generation = {"k": 8, "temperatures": schedule, "state_keys": state_keys}
    result = mod.analyze(keys, rollout, generation)
    assert result["gate"]["passed"] is True
    assert result["metrics"]["first_fail_later_successes"] == 2
    assert result["metrics"]["oracle_minus_first"] == 0.5
