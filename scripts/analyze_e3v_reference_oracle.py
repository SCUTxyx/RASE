#!/usr/bin/env python3
"""Audit E3-V reference coverage, trace integrity, diversity, and Gate V."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_oft_rows(
    rows: list[dict[str, Any]],
    oft_summary: Mapping[str, Any],
    trajectory_dir: Path,
) -> None:
    """Normalize existing exact-root OFT trajectory artifacts into E3-V rows."""
    from rase.collect.candidates import load_artifact

    baseline = {str(row["state_key"]): row for row in rows}
    model_info = dict(oft_summary.get("model_info") or {})
    reference_id = f"oft:{model_info.get('suite') or oft_summary.get('suite') or 'unknown'}"
    for record in oft_summary.get("records") or []:
        key = str(record["state_key"])
        if key not in baseline:
            raise ValueError(f"OFT state {key} is absent from the E3-V base collection")
        result = record.get("direct_oft_result")
        if not isinstance(result, Mapping):
            raise ValueError(f"OFT state {key} lacks a live direct_oft_result")
        trace_path = trajectory_dir / f"{key}.npz"
        artifact = load_artifact(trace_path)
        actions = np.asarray(artifact.actions, dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 7:
            raise ValueError(f"invalid OFT trajectory {trace_path}: {actions.shape}")
        initial = actions[0, : min(8, actions.shape[1])]
        source = baseline[key]
        rows.append(
            {
                "schema_version": "rase-e3v-reference-trace/v1",
                "state_key": key,
                "task_id": source["task_id"],
                "suite": source["suite"],
                "source_success": False,
                "single_reference_success": source["single_reference_success"],
                "rollout_index": 0,
                "rollout_seed": 0,
                "policy_id": reference_id,
                "reference_id": reference_id,
                "result": dict(result),
                "trace_path": str(trace_path.resolve()),
                "trace_sha256": file_sha256(trace_path),
                "executed_action_steps": int(actions.shape[1]),
                "initial_chunk_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                "inference_events": len(record.get("chunk_query_records") or []),
            }
        )


def analyze(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_roots: int = 40,
    min_tasks: int = 6,
    min_successful_trajectories: int = 20,
    min_successful_tasks: int = 4,
    min_oracle_coverage: float = 0.20,
    min_single_failure_rescue_tasks: int = 2,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["state_key"])].append(row)
    if not grouped:
        raise ValueError("no E3-V rollouts")

    per_root: list[dict[str, Any]] = []
    trace_errors: list[str] = []
    for state_key, attempts in sorted(grouped.items()):
        identities = [
            (str(row.get("reference_id") or row.get("policy_id") or "unknown"), int(row["rollout_seed"]))
            for row in attempts
        ]
        if len(identities) != len(set(identities)):
            trace_errors.append(f"{state_key}:duplicate_seed")
        for row in attempts:
            if int(row.get("executed_action_steps", -1)) != int(row["result"]["continuation_steps"]):
                trace_errors.append(f"{state_key}:trace_length")
            if not row.get("trace_sha256") or not row.get("initial_chunk_sha256"):
                trace_errors.append(f"{state_key}:missing_trace_hash")
        successes = sum(bool(row["result"]["success"]) for row in attempts)
        first = attempts[0]
        per_root.append(
            {
                "state_key": state_key,
                "task_id": str(first["task_id"]),
                "suite": str(first["suite"]),
                "attempts": len(attempts),
                "successful_trajectories": successes,
                "reference_oracle_success": successes > 0,
                "single_reference_success": bool(first["single_reference_success"]),
                "rescues_single_reference_failure": (
                    not bool(first["single_reference_success"]) and successes > 0
                ),
                "unique_initial_chunks": len({str(row["initial_chunk_sha256"]) for row in attempts}),
                "reference_ids": sorted(
                    {str(row.get("reference_id") or row.get("policy_id") or "unknown") for row in attempts}
                ),
            }
        )

    n_roots = len(per_root)
    task_ids = {row["task_id"] for row in per_root}
    successful = [row for row in per_root if row["reference_oracle_success"]]
    baseline_failures = [row for row in per_root if not row["single_reference_success"]]
    rescued_baseline = [row for row in baseline_failures if row["rescues_single_reference_failure"]]
    successful_trajectories = sum(int(row["successful_trajectories"]) for row in per_root)
    successful_tasks = {row["task_id"] for row in successful}
    rescued_tasks = {row["task_id"] for row in rescued_baseline}
    diverse_roots = sum(int(row["unique_initial_chunks"] > 1) for row in per_root)

    cohort_sufficient = n_roots >= min_roots and len(task_ids) >= min_tasks
    checks = {
        "trace_integrity": not trace_errors,
        "successful_trajectories": successful_trajectories >= min_successful_trajectories,
        "successful_task_coverage": len(successful_tasks) >= min_successful_tasks,
        "reference_oracle_coverage": len(successful) / n_roots >= min_oracle_coverage,
        "single_reference_failure_rescue_tasks": len(rescued_tasks) >= min_single_failure_rescue_tasks,
    }
    if not cohort_sufficient:
        decision = "EXPAND_REQUIRED"
    else:
        decision = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": "rase-e3v-reference-viability-audit/v1",
        "decision": decision,
        "scientific_scope": "development_only_reference_viability",
        "cohort": {
            "n_roots": n_roots,
            "n_tasks": len(task_ids),
            "n_rollouts": len(rows),
            "cohort_sufficient_for_formal_gate": cohort_sufficient,
        },
        "metrics": {
            "successful_trajectories": successful_trajectories,
            "successful_roots": len(successful),
            "reference_oracle_coverage": len(successful) / n_roots,
            "successful_tasks": len(successful_tasks),
            "single_reference_failure_roots": len(baseline_failures),
            "rescued_single_reference_failure_roots": len(rescued_baseline),
            "rescued_single_reference_failure_tasks": len(rescued_tasks),
            "roots_with_multiple_unique_initial_chunks": diverse_roots,
            "initial_chunk_diversity_rate": diverse_roots / n_roots,
        },
        "thresholds": {
            "min_roots": min_roots,
            "min_tasks": min_tasks,
            "min_successful_trajectories": min_successful_trajectories,
            "min_successful_tasks": min_successful_tasks,
            "min_oracle_coverage": min_oracle_coverage,
            "min_single_reference_failure_rescue_tasks": min_single_failure_rescue_tasks,
        },
        "checks": checks,
        "trace_errors": trace_errors,
        "per_root": per_root,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-roots", type=int, default=40)
    parser.add_argument("--min-tasks", type=int, default=6)
    parser.add_argument("--additional-oft-summary", type=Path)
    parser.add_argument("--additional-oft-trajectory-dir", type=Path)
    args = parser.parse_args()
    summary = read_json(args.summary.resolve())
    if summary.get("status") != "complete":
        raise ValueError("E3-V collection is incomplete")
    rows = [dict(row) for row in summary.get("per_rollout") or []]
    if bool(args.additional_oft_summary) != bool(args.additional_oft_trajectory_dir):
        raise ValueError("both OFT summary and trajectory directory are required")
    if args.additional_oft_summary:
        append_oft_rows(
            rows,
            read_json(args.additional_oft_summary.resolve()),
            args.additional_oft_trajectory_dir.resolve(),
        )
    result = analyze(
        rows,
        min_roots=args.min_roots,
        min_tasks=args.min_tasks,
    )
    result["collection_summary"] = str(args.summary.resolve())
    result["protocol_sha256"] = summary.get("protocol_sha256")
    write_json(args.output.resolve(), result)
    print(json.dumps({"decision": result["decision"], **result["metrics"]}, sort_keys=True))
    return 0 if result["decision"] in {"PASS", "EXPAND_REQUIRED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
