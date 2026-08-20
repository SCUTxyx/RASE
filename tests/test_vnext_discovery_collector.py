from __future__ import annotations

from scripts.collect_rase_vnext_discovery import finalize_group_rows


def test_frozen_utility_is_derived_from_raw_paired_costs() -> None:
    utility = {
        "success_reward": 1.0, "harm_weight": 1.0, "query_weight": 0.02,
        "fallback_weight": 0.1, "latency_weight": 0.01,
        "normalization": {"max_episode_steps": 250, "control_hz": 10.0},
    }
    rows = [
        {
            "operator_id": "continue.source", "available": True, "success": True,
            "branch_wall_s": 2.0, "intervention_query_count": 0, "fallback_steps": 0,
        },
        {
            "operator_id": "fallback.persistent", "available": True, "success": False,
            "branch_wall_s": 3.0, "intervention_query_count": 2, "fallback_steps": 20,
        },
    ]
    finalize_group_rows(rows, utility=utility)
    fallback = rows[1]
    assert fallback["harm"] == 1.0
    assert fallback["query_cost"] == 2 / 250
    assert fallback["fallback_cost"] == 20 / 250
    assert fallback["latency_cost"] == 1 / 25
    assert fallback["utility"] < -1.0
