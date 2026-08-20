from __future__ import annotations

from rase.vnext.shadow import audit_discovery_shadow


WEIGHTS = {
    "success_reward": 1.0, "harm_weight": 1.0, "query_weight": 0.02,
    "fallback_weight": 0.1, "latency_weight": 0.01,
}
GATE = {
    "minimum_oracle_minus_best_fixed": 0.03,
    "minimum_root_winner_fraction": 0.1,
    "minimum_winner_operators": 2,
    "minimum_tasks": 2,
    "minimum_suites": 1,
    "required_policies": ["pi05.libero", "pi0fast.libero"],
}


def _rows(*, abort_only_difference: bool) -> list[dict]:
    rows = []
    for policy in GATE["required_policies"]:
        for task in range(2):
            for root in range(2):
                winner = "continue.source" if (task + root) % 2 == 0 else "fallback.persistent"
                for point in ("p8", "p16"):
                    for replica in range(3):
                        for operator in (
                            "continue.source", "requery.source", "resample.source",
                            "fallback.persistent", "abort.safe",
                        ):
                            success = operator != "abort.safe"
                            if not abort_only_difference:
                                success = operator == winner
                                if operator == "abort.safe":
                                    success = False
                            rows.append({
                                "available": True,
                                "root_id": f"root-{task}-{root}",
                                "task_id": f"task-{task}", "suite": "Goal",
                                "policy_id": policy, "decision_point_id": point,
                                "operator_id": operator, "exact_repeat_replica": replica,
                                "success": success, "harm": 0.0, "query_cost": 0.0,
                                "fallback_cost": 0.0, "latency_cost": 0.0,
                            })
    return rows


def test_shadow_stops_when_abort_is_only_difference() -> None:
    result = audit_discovery_shadow(
        _rows(abort_only_difference=True), repeats=3, weights=WEIGHTS,
        opportunity_gate=GATE, minimum_nondegenerate_fraction=0.05,
        bootstrap_samples=100, bootstrap_seed=1,
    )
    assert result["all_operator_nondegeneracy"]["fraction"] == 1.0
    assert result["non_abort_nondegeneracy"]["fraction"] == 0.0
    assert result["status"] == "STOP_REVISE_OPERATOR"


def test_shadow_goes_when_non_abort_winners_vary() -> None:
    result = audit_discovery_shadow(
        _rows(abort_only_difference=False), repeats=3, weights=WEIGHTS,
        opportunity_gate=GATE, minimum_nondegenerate_fraction=0.05,
        bootstrap_samples=100, bootstrap_seed=1,
    )
    assert result["non_abort_nondegeneracy"]["fraction"] == 1.0
    assert result["status"] == "GO_CONFIRMATION"

