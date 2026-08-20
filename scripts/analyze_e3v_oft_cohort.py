#!/usr/bin/env python3
"""Audit an exhaustive E3-V exact-root OFT recovery-trajectory cohort.

This gate answers whether the collected successful trajectories are sufficient
to train a small failure-specialized residual candidate.  It deliberately does
not claim that the residual already beats the strongest fixed policy; that is a
later, ungated same-root eligibility experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_cohort(
    protocol: Mapping[str, Any],
    summaries: Sequence[tuple[str, Mapping[str, Any]]],
    trajectory_dir: Path,
    *,
    action_loader: Callable[[Path], np.ndarray],
    min_successful_trajectories: int = 20,
    min_successful_tasks: int = 4,
    min_coverage: float = 0.20,
) -> dict[str, Any]:
    records = [dict(row) for row in protocol.get("records") or []]
    if not records:
        raise ValueError("protocol has no records")
    expected = {str(row["state_key"]): row for row in records}
    if len(expected) != len(records):
        raise ValueError("protocol contains duplicate state keys")

    collected: dict[str, dict[str, Any]] = {}
    duplicate_keys: list[str] = []
    summary_errors: list[str] = []
    for summary_path, summary in summaries:
        if summary.get("status") != "complete":
            summary_errors.append(f"{summary_path}:incomplete")
        suite = str(summary.get("suite") or (summary.get("model_info") or {}).get("suite") or "")
        for row in summary.get("records") or []:
            state_key = str(row["state_key"])
            if state_key in collected:
                duplicate_keys.append(state_key)
                continue
            collected[state_key] = {
                "summary_path": summary_path,
                "summary_suite": suite,
                "record": dict(row),
            }

    missing_keys = sorted(set(expected) - set(collected))
    unexpected_keys = sorted(set(collected) - set(expected))
    trace_errors: list[str] = []
    outcome_mismatches: list[str] = []
    per_root: list[dict[str, Any]] = []
    successful_steps: list[int] = []
    for state_key in sorted(set(expected) & set(collected)):
        frozen = expected[state_key]
        live = collected[state_key]["record"].get("direct_oft_result")
        if not isinstance(live, Mapping):
            summary_errors.append(f"{state_key}:missing_live_result")
            continue
        success = bool(live.get("success"))
        frozen_success = bool(frozen.get("single_reference_success"))
        if success != frozen_success:
            outcome_mismatches.append(state_key)

        trace_path = trajectory_dir / f"{state_key}.npz"
        shape: list[int] | None = None
        trace_hash: str | None = None
        initial_hash: str | None = None
        if not trace_path.is_file():
            trace_errors.append(f"{state_key}:missing_trace")
        else:
            try:
                actions = np.asarray(action_loader(trace_path), dtype=np.float32)
                shape = list(actions.shape)
                if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 7:
                    trace_errors.append(f"{state_key}:invalid_shape:{shape}")
                elif not np.isfinite(actions).all():
                    trace_errors.append(f"{state_key}:nonfinite_actions")
                elif int(actions.shape[1]) != int(live.get("continuation_steps", -1)):
                    trace_errors.append(f"{state_key}:length_mismatch")
                trace_hash = file_sha256(trace_path)
                initial_hash = hashlib.sha256(actions[0, : min(8, actions.shape[1])].tobytes()).hexdigest()
            except Exception as exc:  # keep all cohort failures in one audit
                trace_errors.append(f"{state_key}:load_error:{type(exc).__name__}:{exc}")

        if success:
            successful_steps.append(int(live.get("continuation_steps", 0)))
        per_root.append(
            {
                "state_key": state_key,
                "task_id": str(frozen["task_id"]),
                "suite": str(frozen["suite"]),
                "success": success,
                "frozen_single_reference_success": frozen_success,
                "outcome_reproduced": success == frozen_success,
                "continuation_steps": int(live.get("continuation_steps", 0)),
                "stop_reason": str(live.get("stop_reason", "")),
                "trace_path": str(trace_path.resolve()),
                "trace_shape": shape,
                "trace_sha256": trace_hash,
                "initial_chunk_sha256": initial_hash,
                "summary_path": collected[state_key]["summary_path"],
            }
        )

    successful = [row for row in per_root if row["success"]]
    successful_tasks = {row["task_id"] for row in successful}
    suite_totals = Counter(row["suite"] for row in per_root)
    suite_successes = Counter(row["suite"] for row in successful)
    protocol_complete = (
        not missing_keys
        and not unexpected_keys
        and not duplicate_keys
        and len(per_root) == len(expected)
    )
    trace_integrity = not trace_errors and len(per_root) == len(expected)
    outcome_reproducibility = not outcome_mismatches and len(per_root) == len(expected)
    coverage = len(successful) / len(expected)
    checks = {
        "protocol_complete": protocol_complete,
        "summary_complete": not summary_errors,
        "trace_integrity": trace_integrity,
        "outcome_reproducibility": outcome_reproducibility,
        "successful_trajectories": len(successful) >= min_successful_trajectories,
        "successful_task_coverage": len(successful_tasks) >= min_successful_tasks,
        "reference_coverage": coverage >= min_coverage,
    }
    return {
        "schema_version": "rase-e3v-oft-cohort-audit/v1",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "scientific_scope": "development_only_residual_supervision_viability",
        "claim_boundary": (
            "PASS authorizes residual dataset construction and a small development model only; "
            "it does not establish oracle gain over the strongest fixed policy."
        ),
        "protocol_sha256": protocol.get("protocol_sha256"),
        "cohort": {
            "expected_roots": len(expected),
            "collected_roots": len(per_root),
            "tasks": len({str(row["task_id"]) for row in records}),
            "suites": dict(sorted(suite_totals.items())),
            "exhaustive_available_source_failure_pool": True,
        },
        "metrics": {
            "successful_trajectories": len(successful),
            "successful_roots": len(successful),
            "successful_tasks": len(successful_tasks),
            "reference_coverage": coverage,
            "suite_successes": dict(sorted(suite_successes.items())),
            "outcome_mismatches": len(outcome_mismatches),
            "successful_continuation_steps": {
                "min": min(successful_steps) if successful_steps else None,
                "median": float(np.median(successful_steps)) if successful_steps else None,
                "max": max(successful_steps) if successful_steps else None,
            },
        },
        "thresholds": {
            "min_successful_trajectories": min_successful_trajectories,
            "min_successful_tasks": min_successful_tasks,
            "min_reference_coverage": min_coverage,
            "require_complete_exhaustive_protocol": True,
            "require_trace_integrity": True,
            "require_exact_outcome_reproducibility": True,
        },
        "checks": checks,
        "errors": {
            "summary": summary_errors,
            "trace": trace_errors,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "duplicate_keys": sorted(set(duplicate_keys)),
            "outcome_mismatches": outcome_mismatches,
        },
        "next_gate": {
            "name": "E3-U ungated same-root eligibility",
            "required_claim": (
                "continue-only and residual-only both occur, H_within >= 5%, and oracle gain "
                "over the best fixed candidate >= 5 percentage points across >= 2 tasks"
            ),
        },
        "per_root": per_root,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--oft-summary", type=Path, action="append", required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-successful-trajectories", type=int, default=20)
    parser.add_argument("--min-successful-tasks", type=int, default=4)
    parser.add_argument("--min-coverage", type=float, default=0.20)
    args = parser.parse_args()

    from rase.collect.candidates import load_artifact

    result = audit_cohort(
        read_json(args.protocol.resolve()),
        [(str(path.resolve()), read_json(path.resolve())) for path in args.oft_summary],
        args.trajectory_dir.resolve(),
        action_loader=lambda path: load_artifact(path).actions,
        min_successful_trajectories=args.min_successful_trajectories,
        min_successful_tasks=args.min_successful_tasks,
        min_coverage=args.min_coverage,
    )
    write_json(args.output.resolve(), result)
    print(json.dumps({"decision": result["decision"], **result["metrics"]}, sort_keys=True))
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
