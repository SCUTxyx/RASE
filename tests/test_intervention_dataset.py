from rase.interventions.dataset import (
    OpportunityGate,
    migrate_legacy_escalation_rows,
    opportunity_audit,
)
from rase.interventions.schema import (
    CostVector,
    Feasibility,
    InterventionOutcome,
    InterventionSnapshot,
    OperatorFamily,
    OperatorSpec,
)


def _legacy_row():
    return {
        "state_key": "sp1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "task_id": "libero_goal_000001",
        "episode_id": "ep-00000001",
        "suite": "Goal",
        "perturb_dim": "camera",
        "perturb_sub": "viewpoint",
        "level": 1,
        "t0": 4,
        "cohort": "failure_challenge",
        "arms": {
            "continue_smol": {
                "observed": True,
                "success": False,
                "cost": 0.02,
                "proxy": False,
                "outcome_semantics": "direct_smol_from_snapshot",
            },
            "escalate_oft": {
                "observed": True,
                "success": True,
                "cost": 0.1,
                "proxy": False,
                "outcome_semantics": "direct_oft_from_snapshot",
            },
            "abstain": {
                "observed": True,
                "success": False,
                "cost": 0.0,
                "proxy": False,
            },
        },
    }


def test_legacy_direct_smol_defaults_to_replan_not_continue():
    specs, snapshots, outcomes = migrate_legacy_escalation_rows([_legacy_row()])
    assert [spec.operator_id for spec in specs] == ["replan_smol", "switch_oft", "abstain"]
    assert specs[0].family is OperatorFamily.REPLAN
    assert not snapshots[0].supports_true_continue
    assert outcomes[0].outcome_semantics == "direct_smol_from_snapshot"


def test_legacy_three_arm_data_cannot_clear_strict_continue_gate():
    specs, snapshots, outcomes = migrate_legacy_escalation_rows([_legacy_row()])
    result = opportunity_audit(
        snapshots,
        outcomes,
        specs,
        gate=OpportunityGate(
            min_complete_snapshots=1,
            min_oracle_gap=0.0,
            min_winning_operators=2,
            min_tasks_per_winning_operator=1,
            require_harm=False,
            require_futility=False,
        ),
    )
    assert result["status"] == "not_ready"
    assert any("strict CONTINUE" in reason for reason in result["reasons"])


def _make_opportunity_fixture():
    specs = [
        OperatorSpec("continue_source", OperatorFamily.CONTINUE, "source", "active_suffix"),
        OperatorSpec("replan_source", OperatorFamily.REPLAN, "source", "current_observation"),
        OperatorSpec(
            "switch_target",
            OperatorFamily.SWITCH_POLICY,
            "target",
            "current_observation",
        ),
    ]
    snapshots = []
    outcomes = []
    successes = [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ]
    for index, pattern in enumerate(successes):
        snapshot_id = f"sp1_{index:032x}"
        snapshots.append(
            InterventionSnapshot(
                snapshot_id=snapshot_id,
                state_key=snapshot_id,
                task_id=f"task-{index % 3}",
                episode_id=f"episode-{index}",
                step=2,
                source_policy="source",
                restore_state_ref=f"state_pool:{snapshot_id}",
                active_action_suffix_ref=f"actions:{snapshot_id}",
            )
        )
        for spec, success in zip(specs, pattern):
            outcomes.append(
                InterventionOutcome(
                    snapshot_id=snapshot_id,
                    operator_id=spec.operator_id,
                    continuation_seed=0,
                    feasibility=Feasibility(feasible=True),
                    observed=True,
                    success=success,
                    operator_completed=True,
                    stop_reason="success" if success else "horizon",
                    utility_cost=0.1 if spec.family is not OperatorFamily.CONTINUE else 0.0,
                    cost_source="test_cost",
                    costs=CostVector(env_steps=10),
                )
            )
    return specs, snapshots, outcomes


def test_opportunity_gate_measures_oracle_gap_and_complementary_winners():
    specs, snapshots, outcomes = _make_opportunity_fixture()
    result = opportunity_audit(
        snapshots,
        outcomes,
        specs,
        gate=OpportunityGate(
            min_complete_snapshots=6,
            min_oracle_gap=0.5,
            min_winning_operators=3,
            min_tasks_per_winning_operator=1,
            require_harm=True,
            require_futility=True,
        ),
        continue_operator_id="continue_source",
    )
    assert result["status"] == "ready_for_method"
    assert result["same_state"]["oracle_minus_best_fixed"] > 0.5
    assert result["same_state"]["task_supported_winning_operators"] == [
        "continue_source",
        "replan_source",
        "switch_target",
    ]
    assert result["harm"]["n_harmful_operator_state_pairs"] > 0


def test_opportunity_gate_rejects_missing_operator_coverage():
    specs, snapshots, outcomes = _make_opportunity_fixture()
    result = opportunity_audit(
        snapshots,
        outcomes[:-1],
        specs,
        gate=OpportunityGate(
            min_complete_snapshots=6,
            min_winning_operators=3,
            min_tasks_per_winning_operator=1,
        ),
        continue_operator_id="continue_source",
    )
    assert result["status"] == "not_ready"
    assert result["coverage"]["n_incomplete_snapshots"] == 1


def test_opportunity_gate_does_not_count_ties_as_operator_advantage():
    specs, snapshots, outcomes = _make_opportunity_fixture()
    tied_outcomes = [
        InterventionOutcome(
            snapshot_id=row.snapshot_id,
            operator_id=row.operator_id,
            continuation_seed=row.continuation_seed,
            feasibility=row.feasibility,
            observed=True,
            success=False,
            operator_completed=True,
            stop_reason="horizon",
            utility_cost=0.0,
            cost_source="zero_cost_tie_test",
            costs=row.costs,
        )
        for row in outcomes
    ]
    result = opportunity_audit(
        snapshots,
        tied_outcomes,
        specs,
        gate=OpportunityGate(
            min_complete_snapshots=6,
            min_oracle_gap=0.0,
            min_winning_operators=3,
            min_tasks_per_winning_operator=1,
            require_harm=False,
            require_futility=False,
        ),
        continue_operator_id="continue_source",
    )
    assert result["status"] == "not_ready"
    assert result["same_state"]["winner_state_counts"] == {
        "continue_source": 6,
        "replan_source": 6,
        "switch_target": 6,
    }
    assert result["same_state"]["unique_winner_state_counts"] == {
        "continue_source": 0,
        "replan_source": 0,
        "switch_target": 0,
    }
    assert result["same_state"]["task_supported_winning_operators"] == []
