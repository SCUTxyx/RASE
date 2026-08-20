from __future__ import annotations

from rase.vnext.feasibility import audit_discovery_feasibility


GATE = {
    "minimum_completed_branch_fraction": 1.0,
    "minimum_unmasked_operator_fraction": 0.8,
    "minimum_nondegenerate_outcome_fraction": 0.05,
}


def _fixture() -> tuple[dict, list[dict]]:
    jobs = []
    rows = []
    index = 0
    for replica in range(3):
        for operator, success in (("continue.source", 1), ("abort.safe", 0)):
            job_id = f"job-{index}"
            jobs.append({"job_id": job_id})
            rows.append({
                "job_id": job_id, "completed": True, "available": True,
                "root_id": "r", "policy_id": "p", "decision_point_id": "d",
                "operator_id": operator, "success": success,
            })
            index += 1
    return {"jobs": jobs}, rows


def test_complete_nondegenerate_discovery_passes() -> None:
    manifest, rows = _fixture()
    result = audit_discovery_feasibility(rows, manifest=manifest, gate=GATE)
    assert result["status"] == "PASS"


def test_missing_scheduled_branch_fails_closed() -> None:
    manifest, rows = _fixture()
    result = audit_discovery_feasibility(rows[:-1], manifest=manifest, gate=GATE)
    assert result["status"] == "FAIL"
    assert not result["checks"]["complete_schedule"]
