#!/usr/bin/env python3
"""Assemble benchmark-release manifests and claim-safe evidence checklist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_EVIDENCE = [
    {
        "id": "clean_baseline_seed0",
        "path": "progress/2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md",
        "status_if_missing": "missing",
    },
    {
        "id": "w7_w8_recoverability",
        "path": "progress/2026-07-29_w8_direct_escalation_results.md",
        "status_if_missing": "missing",
    },
    {
        "id": "w9c_selector_kill",
        "path": "progress/2026-07-31_w9c_selector_gate_result.md",
        "status_if_missing": "missing",
    },
    {
        "id": "w10_suite_boundary",
        "path": "progress/2026-07-31_w10_object_spatial_benchmark.md",
        "status_if_missing": "missing",
    },
    {
        "id": "pre_a0_candidate_neg",
        "path": "progress/2026-08-03_rase_pre_a0_candidate_opportunity.md",
        "status_if_missing": "missing",
    },
    {
        "id": "pre_a1_short_replan_neg",
        "path": "progress/2026-08-03_rase_pre_a1_replan_mechanism.md",
        "status_if_missing": "missing",
    },
    {
        "id": "pre_a2_duration_signal",
        "path": "progress/2026-08-03_rase_pre_a2_recovery_duration.md",
        "status_if_missing": "missing",
    },
    {
        "id": "pre_a3_protocol",
        "path": "protocol/pre_a3_recovery_duration_v1.md",
        "status_if_missing": "missing",
    },
]


PENDING_ITEMS = [
    {
        "id": "clean_baseline_seed1",
        "action": "Run configs/eval_base.yaml with seed=1 and record mean/CI",
    },
    {
        "id": "clean_baseline_seed2",
        "action": "Run configs/eval_base.yaml with seed=2 and record mean/CI",
    },
    {
        "id": "second_policy_pair",
        "action": "Evaluate a second backbone pair under the same snapshot protocol",
    },
    {
        "id": "pre_a3_confirmatory_results",
        "action": "Collect 120-state pool and finish live closed-loop duration sweep",
    },
    {
        "id": "cost_pareto_tables",
        "action": "Export success/OFT-steps/harm/latency Pareto from PRE-A3 arms",
    },
    {
        "id": "paper_figures",
        "action": "Render recoverability matrix, duration curve, Pareto, calibration",
    },
]


def assemble(method_gate: dict[str, Any] | None) -> dict[str, Any]:
    evidence = []
    for item in REQUIRED_EVIDENCE:
        path = ROOT / item["path"]
        evidence.append(
            {
                "id": item["id"],
                "path": item["path"],
                "present": path.exists(),
                "status": "ready" if path.exists() else item["status_if_missing"],
            }
        )
    track = (
        method_gate.get("paper_track")
        if method_gate is not None
        else "benchmark_diagnosis_pending_pre_a3"
    )
    return {
        "schema_version": "rase-benchmark-release-manifest/v1",
        "paper_track": track,
        "method_gate": method_gate,
        "evidence": evidence,
        "pending": PENDING_ITEMS,
        "metrics_required": [
            "task_success",
            "clean_regret",
            "false_handback_harm",
            "strong_policy_steps",
            "wall_clock_s",
            "gpu_seconds",
            "peak_vram_mib",
        ],
        "statistics_required": [
            "task_cluster_bootstrap_95",
            "exact_mcnemar",
            "wilson_interval",
            "preregistered_multiplicity_control",
        ],
        "figures_required": [
            "recoverability_matrix",
            "duration_response_and_harm",
            "success_cost_pareto",
            "cross_task_generalization_calibration",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-gate", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/rase_benchmark_release_manifest_v1.json",
    )
    args = parser.parse_args()
    gate = None
    if args.method_gate is not None and args.method_gate.exists():
        gate = json.loads(args.method_gate.read_text(encoding="utf-8"))
    manifest = assemble(gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    ready = sum(1 for row in manifest["evidence"] if row["present"])
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evidence_ready": ready,
                "evidence_total": len(manifest["evidence"]),
                "pending": len(manifest["pending"]),
                "paper_track": manifest["paper_track"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
