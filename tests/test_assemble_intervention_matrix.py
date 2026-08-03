from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rase.interventions.schema import (
    CostVector,
    Feasibility,
    InterventionOutcome,
    InterventionSnapshot,
)


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "assemble_intervention_matrix.py"
    spec = importlib.util.spec_from_file_location("assemble_intervention_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _outcome(snapshot_id: str, operator_id: str, success: bool, steps: int = 0):
    return InterventionOutcome(
        snapshot_id=snapshot_id,
        operator_id=operator_id,
        continuation_seed=0,
        feasibility=Feasibility(feasible=True),
        observed=True,
        success=success,
        operator_completed=True,
        stop_reason="success" if success else "horizon",
        utility_cost=0.0,
        cost_source="test_cost",
        costs=CostVector(
            env_steps=steps,
            compute_seconds=steps / 10,
            latency_seconds=steps / 10,
        ),
    )


def test_summary_finds_oracle_gap_and_unique_operator_winners():
    operators = ["continue_smol_active_chunk", "replan_smol", "switch_oft"]
    snapshots = [
        InterventionSnapshot(
            snapshot_id=f"sp1_{index:032x}",
            state_key=f"sp1_{index:032x}",
            task_id=f"task-{index}",
            episode_id=f"episode-{index}",
            step=index,
            source_policy="smolvla",
            restore_state_ref=f"state_pool:{index}",
            active_action_suffix_ref=f"actions:{index}",
        )
        for index in range(3)
    ]
    patterns = [(True, False, False), (False, True, False), (False, False, True)]
    outcomes = [
        _outcome(snapshot.snapshot_id, operator_id, success)
        for snapshot, pattern in zip(snapshots, patterns)
        for operator_id, success in zip(operators, pattern)
    ]
    summary = _module().summarize_success_matrix(snapshots, outcomes, operators)
    assert summary["n_complete_snapshots"] == 3
    assert summary["n_episodes"] == 3
    assert summary["n_tasks"] == 3
    assert summary["best_fixed_success_rate"] == 1 / 3
    assert summary["same_state_oracle_success_rate"] == 1.0
    assert summary["oracle_minus_best_fixed"] == pytest.approx(2 / 3)
    assert summary["unique_winner_counts"] == {
        "continue_smol_active_chunk": 1,
        "replan_smol": 1,
        "switch_oft": 1,
    }


def test_oft_rpc_metrics_keep_inference_separate_from_rollout_time():
    rows = [
        {
            "result": {
                "elapsed_s": 12.0,
                "oracle_predict_calls": 4,
                "oracle_predict_elapsed_s": 0.5,
            }
        },
        {
            "result": {
                "elapsed_s": 20.0,
                "oracle_predict_calls": 6,
                "oracle_predict_elapsed_s": 0.8,
            }
        },
        {"result": {"elapsed_s": 8.0}},
    ]
    metrics = _module().summarize_oft_rpc_metrics(rows)
    assert metrics["n_states"] == 3
    assert metrics["n_measured_states"] == 2
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["predict_calls"] == 10
    assert metrics["predict_elapsed_s"] == pytest.approx(1.3)
    assert metrics["mean_predict_calls_per_measured_state"] == 5
    assert metrics["mean_ms_per_predict_call"] == pytest.approx(130)


def test_summary_marks_incomplete_state_without_inventing_outcome():
    operators = ["continue_smol_active_chunk", "replan_smol", "switch_oft"]
    snapshot = InterventionSnapshot(
        snapshot_id="sp1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        state_key="sp1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        task_id="task-a",
        episode_id="episode-a",
        step=1,
        source_policy="smolvla",
        restore_state_ref="state_pool:a",
        active_action_suffix_ref="actions:a",
    )
    outcomes = [_outcome(snapshot.snapshot_id, operators[0], True)]
    summary = _module().summarize_success_matrix([snapshot], outcomes, operators)
    assert summary["n_complete_snapshots"] == 0
    assert summary["oracle_minus_best_fixed"] is None


def test_summary_reports_cost_routing_signal_when_success_vectors_tie():
    operators = ["continue_smol_active_chunk", "replan_smol", "switch_oft"]
    snapshots = [
        InterventionSnapshot(
            snapshot_id=f"sp1_{index + 10:032x}",
            state_key=f"sp1_{index + 10:032x}",
            task_id="task-a",
            episode_id=f"episode-{index}",
            step=index,
            source_policy="smolvla",
            restore_state_ref=f"state_pool:{index}",
            active_action_suffix_ref=f"actions:{index}",
        )
        for index in range(3)
    ]
    patterns = [
        ((True, 10), (True, 8), (True, 6)),
        ((True, 4), (False, 20), (True, 20)),
        ((False, 20), (False, 20), (False, 20)),
    ]
    outcomes = [
        _outcome(snapshot.snapshot_id, operator_id, success, steps)
        for snapshot, pattern in zip(snapshots, patterns)
        for operator_id, (success, steps) in zip(operators, pattern)
    ]
    summary = _module().summarize_success_matrix(snapshots, outcomes, operators)
    assert summary["per_operator_success_rate"] == {
        "continue_smol_active_chunk": 2 / 3,
        "replan_smol": 1 / 3,
        "switch_oft": 2 / 3,
    }
    assert summary["pairwise_vs_continue"]["switch_oft"] == {
        "higher_success_states": 0,
        "lower_success_states": 0,
        "tied_success_states": 3,
    }
    assert summary["n_no_operator_support"] == 1
    assert summary["n_all_operator_success"] == 1
    assert summary["success_pattern_counts"] == {"000": 1, "101": 1, "111": 1}
    assert summary["success_then_env_steps"]["best_fixed_operator"] == (
        "continue_smol_active_chunk"
    )
    assert summary["success_then_env_steps"]["n_oracle_supported_states"] == 2
    assert summary["success_then_env_steps"][
        "same_state_oracle_mean_env_steps_on_supported_states"
    ] == 5
    assert summary["success_then_env_steps"][
        "oracle_steps_saved_vs_best_fixed_on_supported_states"
    ] == 2
