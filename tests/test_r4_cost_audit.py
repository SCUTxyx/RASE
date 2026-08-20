from scripts.audit_pre_a3_operator_opportunity import audit


def _row(state, task, operator, success, steps):
    return {
        "state_key": state,
        "task_id": task,
        "suite": "Spatial",
        "operator": operator,
        "success": success,
        "executed_oft_steps": steps,
        "env_steps": steps,
        "split": "train",
        "concrete_task_id": task,
        "episode_id": state,
    }


def test_deterministic_prefix_violation_closes_safe_handback_gate():
    records = [
        _row("bad", "task0", "CONTINUE", False, 0),
        _row("bad", "task0", "OFT_H96", False, 96),
        _row("bad", "task0", "OFT_H128", False, 128),
        _row("bad", "task0", "OFT_PERSISTENT", True, 93),
        _row("good", "task1", "CONTINUE", False, 0),
        _row("good", "task1", "OFT_H96", True, 80),
        _row("good", "task1", "OFT_H128", True, 80),
        _row("good", "task1", "OFT_PERSISTENT", True, 80),
    ]
    report = audit(
        records,
        min_complete=1,
        min_gap=0.0,
        min_winners=0,
        min_tasks_per_winner=0,
        max_fixed_harm=1.0,
    )
    assert report["deterministic_prefix_consistency_status"] == "not_ready"
    assert report["deterministic_prefix_violation_states"] == 1
    assert report["deterministic_prefix_violations"][0]["state_key"] == "bad"
    assert "OFT_H96" in report["deterministic_prefix_violations"][0][
        "violating_finite_operators"
    ]
    assert any(
        "deterministic OFT-prefix consistency" in reason
        for reason in report["safe_handback_reasons"]
    )
