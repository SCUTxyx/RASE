import pytest

from rase.interventions.schema import (
    CostVector,
    Feasibility,
    InterventionOutcome,
    InterventionSnapshot,
    OperatorFamily,
    OperatorSpec,
)


def _snapshot(identifier="sp1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"):
    return InterventionSnapshot(
        snapshot_id=identifier,
        state_key=identifier,
        task_id="libero_goal_000001",
        episode_id="ep-00000001",
        step=4,
        source_policy="smolvla",
        restore_state_ref=f"state_pool:{identifier}",
    )


def test_operator_and_outcome_roundtrip():
    spec = OperatorSpec(
        operator_id="switch_oft",
        family=OperatorFamily.SWITCH_POLICY,
        executor="openvla_oft",
        recovery_target="current_observation",
        parameters={"handoff": "public_history"},
    )
    assert OperatorSpec.from_dict(spec.to_dict()) == spec
    outcome = InterventionOutcome(
        snapshot_id=_snapshot().snapshot_id,
        operator_id=spec.operator_id,
        continuation_seed=7,
        feasibility=Feasibility(feasible=True),
        observed=True,
        success=True,
        operator_completed=True,
        stop_reason="success",
        utility_cost=0.1,
        cost_source="measured_v1",
        costs=CostVector(latency_seconds=0.3, env_steps=12),
        outcome_semantics="direct_oft_from_snapshot",
    )
    restored = InterventionOutcome.from_dict(outcome.to_dict())
    assert restored == outcome
    assert restored.utility() == pytest.approx(0.9)


def test_true_continue_requires_explicit_active_suffix_provenance():
    snapshot = _snapshot()
    assert not snapshot.supports_true_continue
    with_suffix = InterventionSnapshot(
        **{**snapshot.to_dict(), "active_action_suffix_ref": "actions:chunk-1"}
    )
    assert with_suffix.supports_true_continue


def test_unobserved_outcome_cannot_claim_success():
    outcome = InterventionOutcome(
        snapshot_id=_snapshot().snapshot_id,
        operator_id="switch_oft",
        continuation_seed=0,
        feasibility=Feasibility(feasible=False, reason_codes=("budget_exceeded",)),
        observed=False,
        success=True,
        operator_completed=False,
        stop_reason="infeasible",
        utility_cost=0.0,
        cost_source="measured_v1",
    )
    with pytest.raises(ValueError, match="success must be present"):
        outcome.validate()
