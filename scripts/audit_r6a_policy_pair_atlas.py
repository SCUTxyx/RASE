#!/usr/bin/env python3
"""Apply frozen R6-A model-free opportunity gates to paired source/OFT outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_pair(
    source_rows: list[dict[str, Any]], oft_rows: dict[str, dict[str, Any]],
    *, min_savings: float, min_source_safe_tasks: int,
) -> dict[str, Any]:
    if {row["state_key"] for row in source_rows} != set(oft_rows):
        raise ValueError("source and persistent state keys differ")
    persistent_successes = 0
    privileged_successes = 0
    persistent_steps = 0
    privileged_steps = 0
    source_safe = []
    source_safe_tasks = set()
    saved_steps_by_suite: dict[str, int] = defaultdict(int)
    suite_source_safe: dict[str, int] = defaultdict(int)
    for source in source_rows:
        oft = oft_rows[str(source["state_key"])]
        source_success = bool(source["source_success"])
        oft_success = bool(oft["success"])
        oft_steps = int(oft["env_steps"])
        persistent_successes += oft_success
        persistent_steps += oft_steps
        if source_success:
            privileged_successes += 1
            source_safe.append(str(source["state_key"]))
            source_safe_tasks.add(f'{source["suite"]}:{source["task_id"]}')
            saved_steps_by_suite[str(source["suite"])] += oft_steps
            suite_source_safe[str(source["suite"])] += 1
        else:
            privileged_successes += oft_success
            privileged_steps += oft_steps
    savings = 1.0 - privileged_steps / max(1, persistent_steps)
    max_suite_share = max(saved_steps_by_suite.values(), default=0) / max(
        1, sum(saved_steps_by_suite.values())
    )
    suites = sorted({str(row["suite"]) for row in source_rows})
    gates = {
        "success_no_worse_than_persistent": privileged_successes >= persistent_successes,
        "teacher_savings_at_least_margin": savings >= min_savings,
        "source_safe_task_support": len(source_safe_tasks) >= min_source_safe_tasks,
        "all_four_suites_have_source_safe_support": all(suite_source_safe[suite] > 0 for suite in suites) and len(suites) == 4,
        "no_suite_over_half_savings": max_suite_share <= 0.5,
    }
    return {
        "n_states": len(source_rows),
        "persistent_successes": persistent_successes,
        "privileged_trigger_successes": privileged_successes,
        "success_gap_vs_persistent": (privileged_successes - persistent_successes) / len(source_rows),
        "persistent_teacher_steps": persistent_steps,
        "privileged_trigger_teacher_steps": privileged_steps,
        "privileged_teacher_savings": savings,
        "source_safe_states": len(source_safe),
        "source_safe_tasks": len(source_safe_tasks),
        "source_safe_task_ids": sorted(source_safe_tasks),
        "source_safe_state_keys": sorted(source_safe),
        "suite_source_safe_counts": dict(sorted(suite_source_safe.items())),
        "saved_steps_by_suite": dict(sorted(saved_steps_by_suite.items())),
        "maximum_suite_savings_share": max_suite_share,
        "gates": gates,
        "all_seed_gates_passed": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oft-analysis", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.manifest.resolve())
    oft_analysis = read_json(args.oft_analysis.resolve())
    if manifest.get("status") != "frozen_before_source_outcomes":
        raise ValueError("R6-A manifest is not frozen")
    if manifest["oft_analysis_sha256"] != sha256(args.oft_analysis.resolve()):
        raise ValueError("persistent OFT analysis changed after R6-A freeze")
    oft_rows = {
        str(row["state_key"]): dict(row["oft_only_result"])
        for row in oft_analysis["per_task"]
    }
    if len(oft_rows) != 48:
        raise ValueError("R6-A requires 48 persistent OFT rows")

    summaries: dict[str, dict[int, tuple[Path, dict[str, Any]]]] = defaultdict(dict)
    for path in args.source_summary:
        report = read_json(path.resolve())
        if report.get("status") != "complete" or int(report.get("n_states", -1)) != 48:
            raise ValueError(f"incomplete source report: {path}")
        if report["initial_keys_sha256"] != manifest["initial_keys_sha256"]:
            raise ValueError(f"initial-key hash mismatch: {path}")
        policy = str(report["policy_id"])
        seed = int(report["seed_index"])
        if seed in summaries[policy]:
            raise ValueError(f"duplicate policy/seed report: {policy}/{seed}")
        summaries[policy][seed] = (path.resolve(), report)

    expected_policies = [str(row["policy_id"]) for row in manifest["source_policies"]]
    pair_reports = {}
    for policy in expected_policies:
        if set(summaries[policy]) != {0, 1}:
            raise ValueError(f"policy {policy} lacks both frozen seeds")
        seeds = {}
        for seed in (0, 1):
            path, source = summaries[policy][seed]
            seeds[str(seed)] = {
                "source_summary": str(path),
                "source_summary_sha256": sha256(path),
                **evaluate_pair(
                    source["per_state"], oft_rows,
                    min_savings=float(manifest["gates"]["minimum_privileged_teacher_savings"]),
                    min_source_safe_tasks=int(manifest["gates"]["minimum_source_safe_tasks"]),
                ),
            }
        pair_reports[policy] = {
            "seeds": seeds,
            "both_seeds_pass": all(seeds[str(seed)]["all_seed_gates_passed"] for seed in (0, 1)),
        }
    passing = sorted(policy for policy, report in pair_reports.items() if report["both_seeds_pass"])
    result = {
        "schema_version": "rase-r6a-policy-pair-atlas/v1",
        "scientific_scope": "development-only initial-state model-free opportunity screen",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest.resolve()),
        "oft_analysis": str(args.oft_analysis.resolve()),
        "oft_analysis_sha256": sha256(args.oft_analysis.resolve()),
        "pairs": pair_reports,
        "passing_policy_pairs": passing,
        "n_passing_policy_pairs": len(passing),
        "minimum_passing_policy_pairs": int(manifest["gates"]["minimum_passing_policy_pairs"]),
        "atlas_gate_status": (
            "ready" if len(passing) >= int(manifest["gates"]["minimum_passing_policy_pairs"])
            else "not_ready"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["atlas_gate_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
