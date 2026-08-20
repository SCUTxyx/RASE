from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r8a_hazard_audit_passes_known_support_and_oracle(tmp_path: Path) -> None:
    rows = []
    suites = ("Spatial", "Object", "Goal", "Long")
    for group, suite in enumerate(suites):
        # Two groups lose recoverability by t8; two remain safe.  The first two
        # source arms succeed, ensuring the cost-aware oracle uses source+t0.
        sequence = (1, 0, 0) if group < 2 else (1, 1, 1)
        for elapsed, persistent in zip((0, 8, 16), sequence):
            rows.append((group, suite, elapsed, persistent, group < 2))
    count = len(rows)
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        group_id=np.asarray([f"group_{g}" for g, *_ in rows]),
        base_group_id=np.asarray([f"base_{g}" for g, *_ in rows]),
        cohort_role=np.asarray(["natural"] * count),
        policy_id=np.asarray(["pi0fast_libero"] * count),
        task_id=np.asarray([f"task_{g}" for g, *_ in rows]),
        suite=np.asarray([suite for _, suite, *_ in rows]),
        state_key=np.asarray([f"state_{g}" for g, *_ in rows]),
        elapsed_source_steps=np.asarray([elapsed for _, _, elapsed, *_ in rows]),
        source_successes=np.asarray([float(success) for *_, success in rows]),
        source_trials=np.ones(count, dtype=np.float32),
        persistent_successes=np.asarray([float(p) for _, _, _, p, _ in rows]),
        persistent_trials=np.ones(count, dtype=np.float32),
        arm_ids=np.asarray([0, 1], dtype=np.int64),
        arm_teacher_step_quantiles=np.tile(
            np.asarray([[[0, 0, 0], [80, 100, 120]]], dtype=np.float32),
            (count, 1, 1),
        ),
    )
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text("{}\n")
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({"label_quality": {"passed": True}}) + "\n")
    protocol_hash = "a" * 64
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "status": "complete", "dataset_sha256": sha256(dataset),
        "exclusions_sha256": sha256(exclusions), "protocol_sha256": protocol_hash,
    }) + "\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "status": "frozen", "boundaries": [0, 8, 16],
        "transitions": [[0, 8], [8, 16]],
        "inputs": {
            "dataset_sha256": sha256(dataset),
            "dataset_report_sha256": sha256(report),
            "exclusions_sha256": sha256(exclusions),
            "readiness_sha256": sha256(readiness),
            "protocol_sha256": protocol_hash,
        },
        "cohort_policy": {
            "formal_oracle_roles": ["natural"],
            "label_support_roles": ["natural", "enrichment"],
        },
        "gates": {
            "minimum_complete_support_groups": 4,
            "minimum_hard_hazard_positive_groups": 2,
            "minimum_hard_hazard_positive_tasks": 2,
            "minimum_hard_hazard_positive_per_suite": 0,
            "minimum_positive_transitions_per_horizon": 0,
            "maximum_ambiguous_transition_fraction": 0.0,
            "minimum_natural_groups_per_policy": 4,
            "minimum_natural_tasks_per_policy": 4,
            "oracle_success_gap_vs_t0_min": -0.05,
            "oracle_teacher_savings_vs_t0_min": 0.30,
            "oracle_expected_paired_harm_max": 0.05,
            "minimum_oracle_arms_used": 2,
            "require_four_suites": True,
            "minimum_policy_pairs_passing_oracle": 1,
        },
    }) + "\n")
    output = tmp_path / "audit.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(root / "scripts/audit_r8a_recoverability_hazard.py"),
        "--dataset", str(dataset), "--dataset-report", str(report),
        "--exclusions", str(exclusions), "--readiness", str(readiness),
        "--config", str(config), "--output", str(output),
    ], cwd=root, check=True)
    result = json.loads(output.read_text())
    assert result["status"] == "PASS"
    assert result["hard_hazard_positive_groups"] == 2
    assert result["by_horizon"]["0_to_8"]["hard_positive_transitions"] == 2
    assert result["natural_cost_aware_oracle"]["pi0fast_libero"]["status"] == "PASS"
