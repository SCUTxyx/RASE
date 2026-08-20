#!/usr/bin/env python3
"""Audit the privileged cost ceiling for Student-vs-persistent-OFT takeover."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--qc-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-savings", type=float, default=0.20)
    parser.add_argument("--method-margin-savings", type=float, default=0.30)
    args = parser.parse_args()

    rows = read_jsonl(args.matrix)
    qc = json.loads(args.qc_audit.read_text())
    excluded = {str(value) for value in qc.get("qc_excluded_state_keys", [])}
    rows = [row for row in rows if str(row["state_key"]) not in excluded]
    if len(rows) != int(qc["n_complete_states"]):
        raise ValueError("QC matrix cardinality does not match the frozen audit")

    persistent_success = sum(bool(row["operator_success"]["OFT_PERSISTENT"]) for row in rows)
    student_success = sum(bool(row["operator_success"]["CONTINUE"]) for row in rows)
    privileged_success = sum(
        bool(row["operator_success"]["CONTINUE"])
        if bool(row["base_success"])
        else bool(row["operator_success"]["OFT_PERSISTENT"])
        for row in rows
    )
    persistent_steps = sum(int(row["operator_executed_oft_steps"]["OFT_PERSISTENT"]) for row in rows)
    privileged_steps = sum(
        0 if bool(row["base_success"])
        else int(row["operator_executed_oft_steps"]["OFT_PERSISTENT"])
        for row in rows
    )
    savings = 1.0 - privileged_steps / max(1, persistent_steps)
    required_oracle_capture = args.target_savings / savings if savings > 0 else float("inf")
    suite_support = {
        suite: {
            "states": sum(str(row["suite"]) == suite for row in rows),
            "student_success_states": sum(
                str(row["suite"]) == suite and bool(row["base_success"]) for row in rows
            ),
        }
        for suite in sorted({str(row["suite"]) for row in rows})
    }
    result = {
        "schema_version": "rase-r6-source-risk-opportunity/v1",
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix),
        "qc_audit": str(args.qc_audit.resolve()),
        "qc_audit_sha256": sha256(args.qc_audit),
        "n_states": len(rows),
        "n_tasks": len({str(row["task_id"]) for row in rows}),
        "student_success_states": student_success,
        "persistent_success_states": persistent_success,
        "privileged_trigger_success_states": privileged_success,
        "privileged_trigger_success_gap_vs_persistent": (
            privileged_success - persistent_success
        ) / max(1, len(rows)),
        "persistent_teacher_steps": persistent_steps,
        "privileged_trigger_teacher_steps": privileged_steps,
        "privileged_trigger_teacher_savings": savings,
        "target_teacher_savings": args.target_savings,
        "fraction_of_privileged_savings_required": required_oracle_capture,
        "method_margin_teacher_savings": args.method_margin_savings,
        "legacy_20pct_opportunity_ready": (
            privileged_success == persistent_success and savings >= args.target_savings
        ),
        "top_conference_margin_ready": (
            privileged_success == persistent_success and savings >= args.method_margin_savings
        ),
        "suite_support": suite_support,
        "decision": (
            "model_free_multi_policy_pair_screen"
            if savings < args.method_margin_savings else "risk_model_may_train"
        ),
        "scope": "privileged train-development upper bound; not deployable performance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
