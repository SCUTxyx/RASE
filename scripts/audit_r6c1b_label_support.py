#!/usr/bin/env python3
"""Replica-aware label-support gate for R6-C.1B.

The old audit counted only rep0.  That is a useful sensitivity analysis, but it
throws away the rep1/rep2 adjudication that R6-C.1B paid to collect.  This
version aggregates each (policy, seed, state) triple once, uses the empirical
replica probability at every boundary, and applies majority adjudication only
for the discrete support counts.  Replicas never increase the number of
groups, tasks, or OOF examples.

Enrichment is training-only.  Natural and enrichment support are always
reported separately.  A policy passes only when it has enough failure and
early-rescue support, failure support spans real tasks, every suite contains
both source failures and matched source successes, and early rescues are not
concentrated in one suite.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path


Triple = tuple[str, int, str]


def role_of(path: Path) -> str:
    return "enrichment" if "train_enrichment" in path.parts else "natural"


def _replica_index(data: dict) -> int:
    return int(data.get("rollout_index", 0))


def load_groups(input_roots: list[Path], exclusions_path: Path) -> list[dict]:
    """Load one replica-aggregated record per (policy, seed, state)."""
    exclusions_data = json.loads(exclusions_path.read_text())
    if exclusions_data.get("status") != "frozen":
        raise ValueError("reproducibility exclusions are not frozen")
    excluded = {
        (str(policy), int(seed), str(state))
        for policy, seed, state in exclusions_data.get("excluded", [])
    }

    by_triple: dict[Triple, list[dict]] = defaultdict(list)
    for root in input_roots:
        paths = sorted(glob.glob(
            str(root / "suite_*" / "*" / "**" / "*__seed*.json"),
            recursive=True,
        ))
        for value in paths:
            path = Path(value)
            if path.name == "report.json":
                continue
            data = json.loads(path.read_text())
            if not data.get("rows"):
                continue
            first = data["rows"][0]
            triple = (
                str(first["policy_id"]),
                int(first["seed_index"]),
                str(first["state_key"]),
            )
            if triple in excluded:
                continue
            by_triple[triple].append({
                "path": path,
                "data": data,
                "replicate_index": _replica_index(data),
                "cohort_role": role_of(path),
            })

    groups: list[dict] = []
    for triple, replicas in sorted(by_triple.items()):
        replicas.sort(key=lambda item: (item["replicate_index"], str(item["path"])))
        canonical = replicas[0]
        first = canonical["data"]["rows"][0]
        roles = {item["cohort_role"] for item in replicas}
        if len(roles) != 1:
            raise ValueError(f"mixed cohort roles for {triple}: {sorted(roles)}")
        source_values = [bool(item["data"]["source_success"]) for item in replicas]
        if len(set(source_values)) != 1:
            # A frozen repro manifest must already exclude these groups.  Do
            # not silently turn a source flip into a hard majority label.
            raise ValueError(f"unexcluded source-success flip for {triple}")
        boundary_trials: dict[int, list[dict]] = defaultdict(list)
        for item in replicas:
            for row in item["data"]["rows"]:
                elapsed = int(row["elapsed_source_steps"])
                boundary_trials[elapsed].append({
                    "success": bool(row["persistent_success_if_enter_now"]),
                    "teacher_steps": float(
                        row["persistent_teacher_steps_if_enter_now"] or 0.0
                    ),
                })
        boundaries = {}
        for elapsed, trials in sorted(boundary_trials.items()):
            successes = sum(t["success"] for t in trials)
            probability = successes / len(trials)
            costs = sorted(float(t["teacher_steps"]) for t in trials)
            boundaries[elapsed] = {
                "successes": successes,
                "trials": len(trials),
                "success_probability": probability,
                "majority_success": probability > 0.5,
                "teacher_steps": costs,
                "teacher_steps_median": costs[len(costs) // 2]
                if len(costs) % 2 else 0.5 * (costs[len(costs)//2 - 1] + costs[len(costs)//2]),
            }
        source_success = source_values[0]
        early_rescuable = (not source_success) and any(
            boundaries.get(elapsed, {}).get("majority_success", False)
            for elapsed in (0, 8, 16)
        )
        groups.append({
            "key": list(triple),
            "policy_id": triple[0],
            "seed_index": triple[1],
            "state_key": triple[2],
            "task_id": str(first["task_id"]),
            "suite": str(first["suite"]),
            "instruction": str(first.get("instruction", "")),
            "cohort_role": canonical["cohort_role"],
            "source_success": source_success,
            "early_rescuable": early_rescuable,
            "n_replicas": len(replicas),
            "canonical_path": str(canonical["path"]),
            "boundaries": boundaries,
        })
    return groups


def _stats(values: list[dict]) -> dict:
    failures = [v for v in values if not v["source_success"]]
    rescues = [v for v in values if v["early_rescuable"]]
    return {
        "groups": len(values),
        "tasks": len({v["task_id"] for v in values}),
        "source_failures": len(failures),
        "source_failure_tasks": len({v["task_id"] for v in failures}),
        "early_rescuable": len(rescues),
        "early_rescue_tasks": len({v["task_id"] for v in rescues}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-source-failures", type=int, default=30)
    parser.add_argument("--min-early-rescuable", type=int, default=20)
    parser.add_argument("--min-suites", type=int, default=4)
    parser.add_argument("--min-failure-tasks", type=int, default=12)
    parser.add_argument("--min-early-rescue-suites", type=int, default=3)
    parser.add_argument("--max-early-rescue-suite-fraction", type=float, default=0.60)
    args = parser.parse_args()

    groups = load_groups(args.input_root, args.exclusions)
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for group in groups:
        by_policy[group["policy_id"]].append(group)

    policy_results = {}
    for policy, values in sorted(by_policy.items()):
        overall = _stats(values)
        suite_support = {}
        for suite in sorted({v["suite"] for v in values}):
            subset = [v for v in values if v["suite"] == suite]
            suite_support[suite] = _stats(subset)
            suite_support[suite]["matched_source_successes"] = sum(
                v["source_success"] for v in subset
            )
        early_by_suite = {
            suite: stats["early_rescuable"] for suite, stats in suite_support.items()
        }
        early_total = max(1, overall["early_rescuable"])
        max_early_share = max(early_by_suite.values(), default=0) / early_total
        rescue_suites = sum(value > 0 for value in early_by_suite.values())
        suite_balance = (
            len(suite_support) >= args.min_suites
            and all(
                stats["source_failures"] > 0
                and stats["matched_source_successes"] > 0
                for stats in suite_support.values()
            )
        )
        gates = {
            "source_failures_ge_min": overall["source_failures"] >= args.min_source_failures,
            "early_rescuable_ge_min": overall["early_rescuable"] >= args.min_early_rescuable,
            "failure_tasks_ge_min": overall["source_failure_tasks"] >= args.min_failure_tasks,
            "every_suite_has_failure_and_matched_success": suite_balance,
            "early_rescue_spans_min_suites": rescue_suites >= args.min_early_rescue_suites,
            "early_rescue_not_single_suite_concentrated": (
                max_early_share <= args.max_early_rescue_suite_fraction
            ),
        }
        cohort_support = {
            role: _stats([v for v in values if v["cohort_role"] == role])
            for role in sorted({v["cohort_role"] for v in values})
        }
        policy_results[policy] = {
            **overall,
            "suite_support": suite_support,
            "cohort_support": cohort_support,
            "early_rescue_suites": rescue_suites,
            "max_early_rescue_suite_fraction": max_early_share,
            "gates": gates,
            "passed": all(gates.values()),
        }

    result = {
        "schema_version": "rase-r6c1b-label-support/v2",
        "status": "complete",
        "scientific_scope": (
            "replica-aggregated support gate; one vote per policy/seed/state triple; "
            "majority adjudication for counts; enrichment is training-only"
        ),
        "input_roots": [str(path.resolve()) for path in args.input_root],
        "exclusions": str(args.exclusions.resolve()),
        "n_groups": len(groups),
        "policy_results": policy_results,
        "policies_passing": sorted(
            policy for policy, value in policy_results.items() if value["passed"]
        ),
        "policies_failing": sorted(
            policy for policy, value in policy_results.items() if not value["passed"]
        ),
    }
    # The project may advance policy-by-policy.  Pi0.5 being underpowered must
    # not block a qualified Pi0Fast experiment, nor be hidden by a pooled gate.
    result["pi0fast_primary_gate_passed"] = bool(
        policy_results.get("pi0fast_libero", {}).get("passed", False)
    )
    result["dual_vla_gate_passed"] = all(
        policy_results.get(policy, {}).get("passed", False)
        for policy in ("pi0fast_libero", "pi05_libero")
    )
    result["decision"] = (
        "PROCEED_PI0FAST_READINESS"
        if result["pi0fast_primary_gate_passed"]
        else "STOP_LEARNED_SELECTOR_PRIMARY_LINE"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pi0fast_primary_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
