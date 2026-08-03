"""Migration and opportunity-audit helpers for same-state intervention data."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .schema import (
    CostVector,
    Feasibility,
    InterventionOutcome,
    InterventionSnapshot,
    OperatorFamily,
    OperatorSpec,
)

OPPORTUNITY_AUDIT_SCHEMA_VERSION = "rase-intervention-opportunity-audit/v1"


@dataclass(frozen=True)
class OpportunityGate:
    min_complete_snapshots: int = 20
    min_oracle_gap: float = 0.05
    min_winning_operators: int = 3
    min_tasks_per_winning_operator: int = 2
    min_repeats_per_arm: int = 1
    require_continue: bool = True
    require_harm: bool = True
    require_futility: bool = True

    def validate(self) -> None:
        if self.min_complete_snapshots < 1:
            raise ValueError("min_complete_snapshots must be positive")
        if self.min_winning_operators < 2:
            raise ValueError("min_winning_operators must be at least two")
        if self.min_tasks_per_winning_operator < 1:
            raise ValueError("min_tasks_per_winning_operator must be positive")
        if self.min_repeats_per_arm < 1:
            raise ValueError("min_repeats_per_arm must be positive")
        if not np.isfinite(self.min_oracle_gap) or self.min_oracle_gap < 0:
            raise ValueError("min_oracle_gap must be finite and non-negative")


def registry_payload(specs: Sequence[OperatorSpec]) -> dict[str, Any]:
    seen: set[str] = set()
    for spec in specs:
        spec.validate()
        if spec.operator_id in seen:
            raise ValueError(f"duplicate operator_id: {spec.operator_id}")
        seen.add(spec.operator_id)
    return {
        "schema_version": "rase-intervention-registry/v1",
        "operators": [spec.to_dict() for spec in specs],
    }


def parse_registry(payload: Mapping[str, Any]) -> list[OperatorSpec]:
    if payload.get("schema_version") != "rase-intervention-registry/v1":
        raise ValueError("unsupported intervention registry schema")
    specs = [OperatorSpec.from_dict(row) for row in payload.get("operators") or []]
    registry_payload(specs)
    return specs


def migrate_legacy_escalation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    direct_smol_family: OperatorFamily = OperatorFamily.REPLAN,
    allow_proxy: bool = False,
) -> tuple[list[OperatorSpec], list[InterventionSnapshot], list[InterventionOutcome]]:
    """Map the old three-arm dataset without inflating its scientific semantics.

    Restored direct Smol calls reset the policy. They default to ``REPLAN``;
    callers must explicitly override this only for a snapshot protocol that
    proves an active source action suffix is preserved.
    """
    if direct_smol_family not in {OperatorFamily.CONTINUE, OperatorFamily.REPLAN}:
        raise ValueError("direct_smol_family must be CONTINUE or REPLAN")
    smol_id = "continue_smol" if direct_smol_family is OperatorFamily.CONTINUE else "replan_smol"
    specs = [
        OperatorSpec(
            operator_id=smol_id,
            family=direct_smol_family,
            executor="smolvla",
            recovery_target=(
                "active_action_suffix"
                if direct_smol_family is OperatorFamily.CONTINUE
                else "current_observation"
            ),
            parameters={"legacy_arm": "continue_smol", "policy_reset": True},
            requires=(
                ("active_action_suffix",)
                if direct_smol_family is OperatorFamily.CONTINUE
                else ("public_observation",)
            ),
        ),
        OperatorSpec(
            operator_id="switch_oft",
            family=OperatorFamily.SWITCH_POLICY,
            executor="openvla_oft",
            recovery_target="current_observation",
            parameters={"legacy_arm": "escalate_oft", "handoff": "public_history"},
            requires=("public_observation",),
        ),
        OperatorSpec(
            operator_id="abstain",
            family=OperatorFamily.ABSTAIN,
            executor="system",
            recovery_target="safe_stop",
            parameters={"legacy_arm": "abstain"},
        ),
    ]
    snapshots: list[InterventionSnapshot] = []
    outcomes: list[InterventionOutcome] = []
    seen: set[str] = set()
    arm_map = {
        "continue_smol": smol_id,
        "escalate_oft": "switch_oft",
        "abstain": "abstain",
    }
    for raw in rows:
        snapshot_id = str(raw["state_key"])
        if snapshot_id in seen:
            raise ValueError(f"duplicate legacy state row: {snapshot_id}")
        seen.add(snapshot_id)
        snapshots.append(
            InterventionSnapshot(
                snapshot_id=snapshot_id,
                state_key=snapshot_id,
                task_id=str(raw["task_id"]),
                episode_id=str(raw["episode_id"]),
                step=int(raw.get("t0", 0)),
                source_policy="smolvla",
                restore_state_ref=f"state_pool:{snapshot_id}",
                public_history_ref=None,
                active_action_suffix_ref=(
                    f"legacy_unverified:{snapshot_id}"
                    if direct_smol_family is OperatorFamily.CONTINUE
                    else None
                ),
                suite=str(raw.get("suite") or "unknown"),
                cohort=str(raw.get("cohort") or "unknown"),
                perturbation={
                    "dimension": raw.get("perturb_dim"),
                    "subdimension": raw.get("perturb_sub"),
                    "level": raw.get("level"),
                },
                split=str(raw["split"]) if raw.get("split") is not None else None,
            )
        )
        arms = dict(raw.get("arms") or {})
        missing = sorted(set(arm_map) - set(arms))
        if missing:
            raise ValueError(f"legacy state {snapshot_id} missing arms: {missing}")
        for legacy_arm, operator_id in arm_map.items():
            arm = dict(arms[legacy_arm])
            proxy = bool(arm.get("proxy", False))
            if proxy and not allow_proxy:
                raise ValueError(
                    f"legacy state {snapshot_id}/{legacy_arm} is a proxy outcome"
                )
            observed = bool(arm.get("observed", False))
            semantics = str(arm.get("outcome_semantics") or f"legacy_{legacy_arm}")
            outcomes.append(
                InterventionOutcome(
                    snapshot_id=snapshot_id,
                    operator_id=operator_id,
                    continuation_seed=0,
                    feasibility=Feasibility(feasible=True),
                    observed=observed,
                    success=bool(arm.get("success", False)) if observed else None,
                    operator_completed=observed,
                    stop_reason="legacy_terminal_outcome" if observed else "not_observed",
                    utility_cost=float(arm.get("cost", 0.0)),
                    cost_source="legacy_preregistered_scalar",
                    costs=CostVector(),
                    outcome_semantics=semantics,
                    proxy=proxy,
                )
            )
    for snapshot in snapshots:
        snapshot.validate()
    for outcome in outcomes:
        outcome.validate()
    return specs, snapshots, outcomes


def _mean_utility(
    outcomes: Sequence[InterventionOutcome], *, success_reward: float
) -> float | None:
    values = [
        value
        for row in outcomes
        if (value := row.utility(success_reward=success_reward)) is not None
    ]
    return float(np.mean(values)) if values else None


def opportunity_audit(
    snapshots: Sequence[InterventionSnapshot],
    outcomes: Sequence[InterventionOutcome],
    specs: Sequence[OperatorSpec],
    *,
    gate: OpportunityGate | None = None,
    success_reward: float = 1.0,
    continue_operator_id: str | None = None,
) -> dict[str, Any]:
    """Audit whether state-dependent allocation is empirically worth learning."""
    chosen_gate = gate or OpportunityGate()
    chosen_gate.validate()
    if not np.isfinite(success_reward) or success_reward <= 0:
        raise ValueError("success_reward must be finite and positive")
    snapshot_by_id: dict[str, InterventionSnapshot] = {}
    for snapshot in snapshots:
        snapshot.validate()
        if snapshot.snapshot_id in snapshot_by_id:
            raise ValueError(f"duplicate snapshot: {snapshot.snapshot_id}")
        snapshot_by_id[snapshot.snapshot_id] = snapshot
    spec_by_id: dict[str, OperatorSpec] = {}
    for spec in specs:
        spec.validate()
        if spec.operator_id in spec_by_id:
            raise ValueError(f"duplicate operator: {spec.operator_id}")
        spec_by_id[spec.operator_id] = spec
    expected = [spec.operator_id for spec in specs if spec.enabled_for_pilot]
    if len(expected) < 2:
        raise ValueError("opportunity audit requires at least two enabled operators")
    by_arm: dict[tuple[str, str], list[InterventionOutcome]] = defaultdict(list)
    seen_arm_keys: set[tuple[str, str, int]] = set()
    proxy_arms = 0
    for outcome in outcomes:
        outcome.validate()
        if outcome.snapshot_id not in snapshot_by_id:
            raise ValueError(f"outcome references unknown snapshot {outcome.snapshot_id}")
        if outcome.operator_id not in spec_by_id:
            raise ValueError(f"outcome references unknown operator {outcome.operator_id}")
        if outcome.arm_key in seen_arm_keys:
            raise ValueError(f"duplicate outcome arm: {outcome.arm_key}")
        seen_arm_keys.add(outcome.arm_key)
        by_arm[(outcome.snapshot_id, outcome.operator_id)].append(outcome)
        proxy_arms += int(outcome.proxy)

    per_operator: dict[str, Any] = {}
    for operator_id in expected:
        rows = [
            outcome
            for (snapshot_id, candidate), values in by_arm.items()
            if candidate == operator_id
            for outcome in values
        ]
        observed = [row for row in rows if row.observed and not row.proxy]
        state_values = [
            value
            for snapshot_id in snapshot_by_id
            if (
                value := _mean_utility(
                    by_arm[(snapshot_id, operator_id)], success_reward=success_reward
                )
            )
            is not None
        ]
        per_operator[operator_id] = {
            "family": spec_by_id[operator_id].family.value,
            "n_outcomes": len(rows),
            "n_observed_non_proxy": len(observed),
            "n_states_observed": len(state_values),
            "n_success": sum(bool(row.success) for row in observed),
            "success_rate": (
                sum(bool(row.success) for row in observed) / len(observed) if observed else None
            ),
            "mean_state_utility": float(np.mean(state_values)) if state_values else None,
        }

    complete: dict[str, dict[str, float]] = {}
    missing_by_snapshot: dict[str, list[str]] = {}
    repeat_deficits: dict[str, dict[str, int]] = {}
    for snapshot_id in snapshot_by_id:
        values: dict[str, float] = {}
        missing: list[str] = []
        deficits: dict[str, int] = {}
        for operator_id in expected:
            usable = [
                row
                for row in by_arm[(snapshot_id, operator_id)]
                if row.observed and not row.proxy
            ]
            if len(usable) < chosen_gate.min_repeats_per_arm:
                missing.append(operator_id)
                deficits[operator_id] = chosen_gate.min_repeats_per_arm - len(usable)
                continue
            value = _mean_utility(usable, success_reward=success_reward)
            if value is None:
                missing.append(operator_id)
                continue
            values[operator_id] = value
        if missing:
            missing_by_snapshot[snapshot_id] = missing
            if deficits:
                repeat_deficits[snapshot_id] = deficits
        else:
            complete[snapshot_id] = values

    winner_states: Counter[str] = Counter()
    winner_tasks: dict[str, set[str]] = defaultdict(set)
    unique_winner_states: Counter[str] = Counter()
    unique_winner_tasks: dict[str, set[str]] = defaultdict(set)
    oracle_values: list[float] = []
    for snapshot_id, values in complete.items():
        best = max(values.values())
        winners = [operator_id for operator_id in expected if np.isclose(values[operator_id], best)]
        oracle_values.append(best)
        for operator_id in winners:
            winner_states[operator_id] += 1
            winner_tasks[operator_id].add(snapshot_by_id[snapshot_id].task_id)
        if len(winners) == 1:
            unique_winner_states[winners[0]] += 1
            unique_winner_tasks[winners[0]].add(
                snapshot_by_id[snapshot_id].task_id
            )
    fixed_values = {
        operator_id: float(np.mean([values[operator_id] for values in complete.values()]))
        for operator_id in expected
    } if complete else {}
    best_fixed_id = (
        max(expected, key=lambda name: (fixed_values[name], -expected.index(name)))
        if fixed_values
        else None
    )
    best_fixed_utility = fixed_values.get(best_fixed_id) if best_fixed_id else None
    oracle_utility = float(np.mean(oracle_values)) if oracle_values else None
    oracle_gap = (
        oracle_utility - best_fixed_utility
        if oracle_utility is not None and best_fixed_utility is not None
        else None
    )

    if continue_operator_id is None:
        candidates = [
            spec.operator_id
            for spec in specs
            if spec.enabled_for_pilot and spec.family is OperatorFamily.CONTINUE
        ]
        continue_operator_id = candidates[0] if len(candidates) == 1 else None
    harmful_pairs = 0
    comparable_pairs = 0
    if continue_operator_id in expected:
        for values in complete.values():
            baseline = values[continue_operator_id]
            for operator_id in expected:
                if operator_id == continue_operator_id:
                    continue
                comparable_pairs += 1
                harmful_pairs += int(values[operator_id] < baseline)
    futile_outcomes = sum(
        row.observed
        and not row.proxy
        and not bool(row.success)
        and row.utility_cost > 0
        and spec_by_id[row.operator_id].family
        not in {OperatorFamily.CONTINUE, OperatorFamily.ABSTAIN}
        for row in outcomes
    )

    eligible_winners = [
        operator_id
        for operator_id in expected
        if unique_winner_states[operator_id] > 0
        and len(unique_winner_tasks[operator_id])
        >= chosen_gate.min_tasks_per_winning_operator
    ]
    reasons: list[str] = []
    if len(complete) < chosen_gate.min_complete_snapshots:
        reasons.append(
            f"complete snapshots {len(complete)} < {chosen_gate.min_complete_snapshots}"
        )
    if oracle_gap is None or oracle_gap < chosen_gate.min_oracle_gap:
        reasons.append(f"oracle gap {oracle_gap!r} < {chosen_gate.min_oracle_gap}")
    if len(eligible_winners) < chosen_gate.min_winning_operators:
        reasons.append(
            f"task-supported unique-winning operators {len(eligible_winners)} "
            f"< {chosen_gate.min_winning_operators}"
        )
    if chosen_gate.require_continue and continue_operator_id not in expected:
        reasons.append("strict CONTINUE operator is missing from the enabled profiles")
    if (
        chosen_gate.require_harm
        and continue_operator_id in expected
        and harmful_pairs == 0
    ):
        reasons.append("no harmful intervention relative to CONTINUE was observed")
    if chosen_gate.require_futility and futile_outcomes == 0:
        reasons.append("no costly failed intervention was observed")
    if proxy_arms:
        reasons.append(f"dataset contains {proxy_arms} proxy outcomes")

    return {
        "schema_version": OPPORTUNITY_AUDIT_SCHEMA_VERSION,
        "status": "ready_for_method" if not reasons else "not_ready",
        "reasons": reasons,
        "gate": asdict(chosen_gate),
        "n_snapshots": len(snapshot_by_id),
        "n_complete_snapshots": len(complete),
        "n_outcomes": len(outcomes),
        "n_proxy_outcomes": proxy_arms,
        "enabled_operator_ids": expected,
        "per_operator": per_operator,
        "same_state": {
            "best_fixed_operator": best_fixed_id,
            "best_fixed_utility": best_fixed_utility,
            "same_state_oracle_utility": oracle_utility,
            "oracle_minus_best_fixed": oracle_gap,
            "winner_state_counts": {name: winner_states[name] for name in expected},
            "winner_task_counts": {name: len(winner_tasks[name]) for name in expected},
            "unique_winner_state_counts": {
                name: unique_winner_states[name] for name in expected
            },
            "unique_winner_task_counts": {
                name: len(unique_winner_tasks[name]) for name in expected
            },
            "task_supported_winning_operators": eligible_winners,
        },
        "harm": {
            "continue_operator_id": continue_operator_id,
            "n_comparable_operator_state_pairs": comparable_pairs,
            "n_harmful_operator_state_pairs": harmful_pairs,
        },
        "futility": {"n_costly_failed_intervention_outcomes": futile_outcomes},
        "coverage": {
            "n_incomplete_snapshots": len(missing_by_snapshot),
            "missing_operators_by_snapshot": missing_by_snapshot,
            "repeat_deficits_by_snapshot": repeat_deficits,
        },
    }
