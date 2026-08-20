from __future__ import annotations

from pathlib import Path

import torch

from rase.adapt.pre_c1_2 import (
    capacity_ladder_step,
    check_receding_invariants,
    choose_batch_kind,
    dagger_qc_report,
    interface_mismatch_decision,
    load_protocol_lock,
    piecewise_horizon_weights,
    select_recovery_horizon,
    validate_protocol_lock,
    weighted_flow_loss_from_unreduced,
)


def test_protocol_lock_c1_2_schema():
    path = Path("artifacts/pre_c1/pre_c1_2_protocol_lock.yaml")
    payload = load_protocol_lock(path)
    assert validate_protocol_lock(payload) == []
    assert payload["phase"] == "PRE-C1.2"
    assert payload["evaluation"]["recovery"]["comparator"] == "adapted_minus_base_same_horizon"
    assert payload["batch_schedule"]["recovery_batches"] == 9
    assert payload["loss"]["auxiliary_sampled_action_mse"] is False
    assert payload["dagger"]["beta_unit"] == "replan_boundary"


def test_batch_schedule_is_nine_plus_one():
    kinds = [choose_batch_kind(i) for i in range(20)]
    assert kinds.count("clean") == 2
    assert kinds.count("recovery") == 18
    assert kinds[9] == "clean"
    assert kinds[19] == "clean"


def test_select_recovery_horizon_requires_positive_adapter_delta():
    rows = []
    for h in (1, 2, 4):
        for i in range(9):
            rows.append(
                {
                    "horizon": h,
                    "base_success": h == 2 and i < 3,
                    "adapted_success": h == 4 and i < 2,
                    "best_progress": 0.5 if h == 4 else 0.1,
                    "first_divergence_step": 10,
                }
            )
    # H=2: base 3 adapted 0 — fail
    # H=4: base 0 adapted 2 — pass
    result = select_recovery_horizon(rows)
    assert result["selected_horizon"] == 4
    assert result["selection_mode"] == "rule"


def test_select_recovery_horizon_fallback():
    rows = [
        {"horizon": 2, "base_success": True, "adapted_success": False}
        for _ in range(9)
    ]
    result = select_recovery_horizon(rows, fallback_horizon=2)
    assert result["selected_horizon"] == 2
    assert result["selection_mode"] == "fallback"


def test_receding_invariants():
    ok = check_receding_invariants(
        env_steps=10,
        execution_horizon=2,
        model_forward_calls=5,
        cache_resets=5,
    )
    assert ok["passed"] is True
    bad = check_receding_invariants(
        env_steps=10,
        execution_horizon=2,
        model_forward_calls=1,
        cache_resets=1,
    )
    assert bad["passed"] is False


def test_piecewise_weights_mean_one():
    w = piecewise_horizon_weights(50)
    assert torch.isclose(w.mean(), torch.tensor(1.0), atol=1e-5)


def test_weighted_flow_no_aux_mse():
    losses = torch.ones(2, 10, 7)
    loss, metrics = weighted_flow_loss_from_unreduced(losses, enable_weighting=True)
    assert loss.ndim == 0
    assert "loss_prefix_4" in metrics
    assert "prefix_gripper_error" in metrics


def test_interface_mismatch_blocks_when_cross_far_above_floor():
    decision = interface_mismatch_decision(
        env_action_mae=0.01,
        cross_successor_error=1.0,
        sim_floor_error=0.01,
        cross_over_sim_floor_ratio=5.0,
    )
    assert decision["block_training"] is True


def test_interface_mismatch_ignores_near_zero_sim_floor_false_positive():
    # Ordinary cross-policy gap with tiny restore floor should not block.
    decision = interface_mismatch_decision(
        env_action_mae=0.034,
        cross_successor_error=0.085,
        sim_floor_error=0.0,
        student_repeat_error=0.0,
    )
    assert decision["block_training"] is False


def test_dagger_qc_distinguishes_sources():
    rows = [
        {
            "anchor_id": "a0",
            "source": "student_query_state",
            "query_state_id": "q0",
            "query_id": "Q0",
            "teacher_rollout_success": True,
            "teacher_recovery_length": 40,
            "offset_from_student_state": 0,
        },
        {
            "anchor_id": "a0",
            "source": "teacher_suffix_after_student_query",
            "query_state_id": "q0",
            "query_id": "Q0",
            "teacher_rollout_success": True,
            "teacher_recovery_length": 40,
            "offset_from_student_state": 1,
        },
    ]
    qc = dagger_qc_report(rows)
    assert qc["query_state_chunks"] == 1
    assert qc["teacher_suffix_chunks"] == 1
    assert qc["unique_student_query_states"] == 1


def test_capacity_ladder_single_variable():
    lock = load_protocol_lock("artifacts/pre_c1/pre_c1_2_protocol_lock.yaml")
    a = capacity_ladder_step("expand_lora_targets", lock)
    b = capacity_ladder_step("rank_32", lock)
    assert a["lora_rank"] == 16
    assert b["lora_rank"] == 32
    assert a["target_modules"] == b["target_modules"]
    d = capacity_ladder_step("full_action_expert", lock)
    assert d["train_mode"] == "full_action_expert"
    assert d["optimizer"]["preregistered"] is True
