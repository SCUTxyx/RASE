from __future__ import annotations

import copy

from rase.vnext.phase_a_audit import (
    audit_confirmation_integrity,
    audit_phase_a,
    paired_trial_operator_evidence,
    strict_opportunity_audit,
)


WEIGHTS = {
    "success_reward": 1.0,
    "harm_weight": 1.0,
    "query_weight": 0.0,
    "fallback_weight": 0.0,
    "latency_weight": 0.0,
}
GATE = {
    "minimum_oracle_minus_best_fixed": 0.1,
    "minimum_root_winner_fraction": 0.1,
    "minimum_winner_operators": 2,
    "minimum_tasks": 2,
    "minimum_suites": 1,
    "required_policies": ["pi05.libero", "pi0fast.libero"],
}


def _manifest_and_rows(*, repeats: int = 3) -> tuple[dict, list[dict]]:
    jobs = []
    rows = []
    operators = ("continue.source", "fallback.persistent", "abort.safe")
    for task_index in range(8):
        root = f"root-{task_index}"
        winner = operators[task_index % 2]
        for policy in GATE["required_policies"]:
            for operator in operators:
                for replica in range(repeats):
                    job_id = f"{root}-{policy}-{operator}-{replica}"
                    job = {
                        "job_id": job_id,
                        "root_id": root,
                        "task_id": f"task-{task_index}",
                        "suite": "Goal" if task_index < 4 else "Spatial",
                        "policy_id": policy,
                        "decision_point": {"decision_point_id": "p1"},
                        "operator_id": operator,
                        "available_by_contract": True,
                        "seed_ledger": {"exact_repeat_replica": replica},
                    }
                    jobs.append(job)
                    rows.append({
                        "job_id": job_id,
                        "root_id": root,
                        "task_id": job["task_id"],
                        "suite": job["suite"],
                        "policy_id": policy,
                        "decision_point_id": "p1",
                        "operator_id": operator,
                        "exact_repeat_replica": replica,
                        "available": True,
                        "success": float(operator == winner),
                        "harm": 0.0,
                        "query_cost": 0.0,
                        "fallback_cost": 0.0,
                        "latency_cost": 0.0,
                    })
    return {
        "schema_version": "test-manifest/v1",
        "expected_jobs": len(jobs),
        "jobs": jobs,
    }, rows


def test_integrity_accepts_complete_frozen_schedule() -> None:
    manifest, rows = _manifest_and_rows()
    result = audit_confirmation_integrity(rows, manifest=manifest, repeats=3)
    assert result["status"] == "PASS"
    assert result["observed_available_jobs"] == len(rows)
    assert not result["missing_available_job_ids"]


def test_integrity_rejects_contract_mask_as_available() -> None:
    manifest, rows = _manifest_and_rows()
    manifest = copy.deepcopy(manifest)
    manifest["jobs"][0]["available_by_contract"] = False
    result = audit_confirmation_integrity(rows, manifest=manifest, repeats=3)
    assert result["status"] == "FAIL"
    assert any(
        "contract-masked job appeared available" in error
        for record in result["row_contract_errors"]
        for error in record["errors"]
    )


def test_strict_opportunity_uses_winner_root_task_and_suite_coverage() -> None:
    _, rows = _manifest_and_rows()
    result = strict_opportunity_audit(
        rows,
        repeats=3,
        weights=WEIGHTS,
        gate=GATE,
        excluded_operators={"abort.safe"},
        bootstrap_samples=400,
        bootstrap_seed=7,
    )
    assert result["status"] == "PASS"
    assert set(result["qualifying_winner_operators"]) == {
        "continue.source", "fallback.persistent",
    }
    assert result["qualifying_winner_task_coverage"] == 8
    assert result["qualifying_winner_suite_coverage"] == ["Goal", "Spatial"]


def test_pairwise_evidence_is_tie_aware() -> None:
    _, rows = _manifest_and_rows(repeats=3)
    for row in rows:
        if row["operator_id"] == "fallback.persistent":
            row["success"] = 0.95
        elif row["operator_id"] == "continue.source":
            row["success"] = 1.0
    evidence = paired_trial_operator_evidence(
        rows,
        weights=WEIGHTS,
        excluded_operators={"abort.safe"},
        tie_margin=0.1,
    )
    pair = evidence["continue.source__vs__fallback.persistent"]
    assert pair["ties"] == pair["pairs"]
    assert pair["soft_preference_left"] == 0.5


def test_phase_a_pass_requires_full_and_non_abort_pass() -> None:
    manifest, rows = _manifest_and_rows()
    result = audit_phase_a(
        rows,
        manifest=manifest,
        repeats=3,
        weights=WEIGHTS,
        gate=GATE,
        bootstrap_samples=400,
        bootstrap_seed=3,
    )
    assert result["status"] == "A_PASS"
    assert result["verdict"] == "UNLOCK_PHASE_B_C"
    assert result["unlocks"] == [
        "canonical_motion_parity", "low_cost_action_sensitivity",
    ]


def test_phase_a_reports_partial_when_only_one_policy_has_opportunity() -> None:
    manifest, rows = _manifest_and_rows()
    for row in rows:
        if row["policy_id"] == "pi05.libero":
            row["success"] = float(row["operator_id"] == "fallback.persistent")
    result = audit_phase_a(
        rows,
        manifest=manifest,
        repeats=3,
        weights=WEIGHTS,
        gate=GATE,
        bootstrap_samples=300,
        bootstrap_seed=11,
    )
    assert result["status"] == "A_PARTIAL"
    assert result["passing_non_abort_policies"] == ["pi0fast.libero"]
