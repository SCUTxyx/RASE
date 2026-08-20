from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_g2a_pi0fast_clean.py"
SPEC = importlib.util.spec_from_file_location("eval_g2a", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_gate_boundaries_are_inclusive() -> None:
    gate = {
        "long_pair_eligible_interval": [0.3, 0.7],
        "below_interval_decision": "low",
        "inside_interval_decision": "inside",
        "above_interval_decision": "high",
    }
    assert MODULE.gate_decision(0.299, gate) == "low"
    assert MODULE.gate_decision(0.3, gate) == "inside"
    assert MODULE.gate_decision(0.7, gate) == "inside"
    assert MODULE.gate_decision(0.701, gate) == "high"


def test_wilson_and_cluster_bootstrap() -> None:
    assert MODULE.wilson_interval(0, 8)[0] == 0.0
    rows = [
        {"task_id": "a", "success": True},
        {"task_id": "a", "success": True},
        {"task_id": "b", "success": False},
        {"task_id": "b", "success": False},
    ]
    low, high = MODULE.task_cluster_bootstrap(rows, seed=7, replicates=2000)
    assert low == 0.0
    assert high == 1.0


def test_frozen_protocol_validates() -> None:
    freeze_path = Path(__file__).resolve().parents[1] / "scripts" / "freeze_g2a_protocol.py"
    freeze_spec = importlib.util.spec_from_file_location("freeze_g2a", freeze_path)
    assert freeze_spec and freeze_spec.loader
    freeze_module = importlib.util.module_from_spec(freeze_spec)
    freeze_spec.loader.exec_module(freeze_module)
    protocol = freeze_module.build_protocol(2026082002)
    rows = MODULE.validate_protocol(protocol)
    assert len(rows) == 80
    assert len({row["episode_id"] for row in rows}) == 80
