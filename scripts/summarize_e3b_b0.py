#!/usr/bin/env python3
"""Audit the full B0 on-policy collection and enforce its engineering gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SUITES = ("spatial", "object", "goal", "long")
ARMS = ("source_h8", "one_shot_h8", "persistent_h8")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def classify(source: bool, candidate: bool) -> str:
    if source and candidate:
        return "both"
    if source:
        return "source_only"
    if candidate:
        return "candidate_only"
    return "neither"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--partition-dir", type=Path, required=True)
    parser.add_argument("--teacher-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-correction-steps", type=int, default=200)
    args = parser.parse_args()

    partition_dir = args.partition_dir.resolve()
    partitions = {role: load(partition_dir / f"{role}.json") for role in ("b0_smoke", "b1_collect", "b2_qualification")}
    roots = {role: set(value["state_keys"]) for role, value in partitions.items()}
    rows = []
    suite_summaries = {}
    for suite in SUITES:
        summary = load(args.input_dir.resolve() / suite / "summary.json")
        suite_summaries[suite] = summary
        rows.extend(summary["per_state_arm"])
    by_root: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_root.setdefault(str(row["state_key"]), {})[str(row["arm"])] = row
    one_shot_counts = Counter()
    persistent_counts = Counter()
    for arms in by_root.values():
        source = bool(arms["source_h8"]["success"])
        one_shot_counts[classify(source, bool(arms["one_shot_h8"]["success"]))] += 1
        persistent_counts[classify(source, bool(arms["persistent_h8"]["success"]))] += 1
    correction_steps = sum(int(row["correction_steps"]) for row in rows)
    teacher_samples = sum(int(row["teacher_samples"]) for row in rows)
    shapes = {tuple(row["teacher_native_chunk_shape"]) for row in rows if row["teacher_native_chunk_shape"]}
    teacher_gate = load(args.teacher_gate.resolve())
    checks = {
        "teacher_gate_pass": teacher_gate.get("decision") == "PASS",
        "all_12_b0_roots_complete": set(by_root) == roots["b0_smoke"] and len(by_root) == 12,
        "three_arms_per_root": all(set(arms) == set(ARMS) for arms in by_root.values()),
        "b0_b1_root_disjoint": roots["b0_smoke"].isdisjoint(roots["b1_collect"]),
        "b0_b2_root_disjoint": roots["b0_smoke"].isdisjoint(roots["b2_qualification"]),
        "b1_b2_root_disjoint": roots["b1_collect"].isdisjoint(roots["b2_qualification"]),
        "correction_steps_ge_threshold": correction_steps >= args.min_correction_steps,
        "teacher_native_shape_exactly_8x7": shapes == {(8, 7)},
        "teacher_samples_nonempty": teacher_samples > 0,
        "all_common_horizon_8": all(int(row["common_horizon"]) == 8 for row in rows),
        "smolvla_tail_discard_declared": all(
            row["smolvla_tail_policy"] == "discard_after_index_7_and_requery" for row in rows
        ),
    }
    artifact = {
        "schema_version": "rase-e3b-b0-gate/v1",
        "status": "complete",
        "n_roots": len(by_root),
        "n_arm_rollouts": len(rows),
        "correction_steps": correction_steps,
        "teacher_samples": teacher_samples,
        "teacher_native_chunk_shapes": [list(value) for value in sorted(shapes)],
        "one_shot_vs_source": dict(one_shot_counts),
        "persistent_vs_source": dict(persistent_counts),
        "suite_success": {suite: summary["success"] for suite, summary in suite_summaries.items()},
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "interpretation": (
            "B0 PASS qualifies the collection protocol and teacher labels only; "
            "candidate eligibility remains reserved for task-disjoint B2."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": artifact["decision"], "correction_steps": correction_steps, "teacher_samples": teacher_samples}, sort_keys=True))
    return 0 if artifact["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
