from __future__ import annotations

import pytest

from rase.vnext.opportunity import aggregate_confirmation, audit_opportunity


WEIGHTS = {
    "success_reward": 1.0, "harm_weight": 1.0, "query_weight": 0.0,
    "fallback_weight": 0.0, "latency_weight": 0.0,
}
GATE = {
    "minimum_oracle_minus_best_fixed": 0.1, "minimum_root_winner_fraction": 0.1,
    "minimum_winner_operators": 2, "minimum_tasks": 2, "minimum_suites": 1,
    "required_policies": ["pi05.libero", "pi0fast.libero"],
}


def _rows(repeats: int = 5) -> list[dict]:
    rows = []
    for policy in GATE["required_policies"]:
        for task_index in range(2):
            for root_index in range(2):
                winner = "continue.source" if (task_index + root_index) % 2 == 0 else "fallback.persistent"
                for point in ("p1", "p2"):
                    for operator in ("continue.source", "fallback.persistent"):
                        for replica in range(repeats):
                            rows.append({
                                "root_id": f"{policy}-{task_index}-{root_index}",
                                "task_id": f"task-{task_index}", "suite": "Goal",
                                "policy_id": policy, "decision_point_id": point,
                                "operator_id": operator, "exact_repeat_replica": replica,
                                "success": 1.0 if operator == winner else 0.0,
                                "harm": 0.0, "query_cost": 0.0, "fallback_cost": 0.0,
                                "latency_cost": 0.0,
                            })
    return rows


def test_exact_k_is_contractual() -> None:
    with pytest.raises(ValueError, match="exact replicas"):
        aggregate_confirmation(_rows()[:-1], repeats=5, weights=WEIGHTS)


def test_root_level_diverse_opportunity_passes() -> None:
    result = audit_opportunity(
        _rows(), repeats=5, weights=WEIGHTS, gate=GATE,
        bootstrap_samples=200, bootstrap_seed=1,
    )
    assert result["status"] == "PASS"
    assert set(result["qualifying_winner_operators"]) == {
        "continue.source", "fallback.persistent",
    }
    assert result["roots"] == 8
