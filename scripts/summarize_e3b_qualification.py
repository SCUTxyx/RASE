#!/usr/bin/env python3
"""Summarize task-disjoint E3-B candidate qualification.

The collector writes one ``summary.json`` per suite with three exact-root
arms.  This report deliberately computes the qualification statistics from
the per-state rows (rather than episode-level aggregates), then bootstraps
over logical tasks so a single easy task cannot dominate the decision.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUITES = ("spatial", "object", "goal", "long")
ARMS = ("source_h8", "one_shot_h8", "persistent_h8")


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def classify(source: bool, candidate: bool) -> str:
    if source and candidate:
        return "both"
    if source:
        return "source_only"
    if candidate:
        return "candidate_only"
    return "neither"


def bootstrap(values: dict[str, list[dict[str, bool]]], seed: int = 20260821, n: int = 4000) -> dict[str, Any]:
    """Cluster bootstrap by logical task; return point estimates and 95% CI."""
    import random

    tasks = sorted(values)
    if not tasks:
        return {"n_tasks": 0, "n_resamples": 0}

    def metric(sample: list[str]) -> tuple[float, float, float, float]:
        rows = [row for task in sample for row in values[task]]
        n_rows = max(1, len(rows))
        source = sum(row["source"] for row in rows) / n_rows
        candidate = sum(row["candidate"] for row in rows) / n_rows
        union = sum(row["source"] or row["candidate"] for row in rows) / n_rows
        return source, candidate, union, union - max(source, candidate)

    point = metric(tasks)
    rng = random.Random(seed)
    samples = [metric([rng.choice(tasks) for _ in tasks]) for _ in range(n)]
    names = ("source_rate", "candidate_rate", "union_rate", "oracle_gain")
    result: dict[str, Any] = {"n_tasks": len(tasks), "n_resamples": n}
    for index, name in enumerate(names):
        series = sorted(item[index] for item in samples)
        result[name] = {
            "estimate": point[index],
            "ci95": [series[int(0.025 * (n - 1))], series[int(0.975 * (n - 1))]],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--partition-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seen-collection-dir", action="append", type=Path, default=[])
    parser.add_argument("--min-h-within", type=float, default=0.05)
    parser.add_argument("--min-oracle-gain", type=float, default=0.05)
    parser.add_argument("--min-candidate-tasks", type=int, default=2)
    args = parser.parse_args()

    partition_dir = args.partition_dir.resolve()
    partitions = {
        role: load(partition_dir / f"{role}.json")
        for role in ("b0_smoke", "b1_collect", "b2_qualification")
    }
    b2 = partitions["b2_qualification"]
    state_to_task = {str(row["state_key"]): str(row["logical_task_id"]) for row in b2["records"]}
    state_to_suite = {str(row["state_key"]): str(row["suite"]).lower() for row in b2["records"]}
    rows: list[dict[str, Any]] = []
    suite_summaries: dict[str, Any] = {}
    input_dir = args.input_dir.resolve()
    for suite in SUITES:
        summary_path = input_dir / suite / "summary.json"
        if not summary_path.exists():
            raise SystemExit(f"missing summary: {summary_path}")
        summary = load(summary_path)
        suite_summaries[suite] = summary
        rows.extend(summary.get("per_state_arm", []))

    by_state: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = str(row["state_key"])
        if key not in state_to_task:
            raise SystemExit(f"qualification row not in B2 partition: {key}")
        by_state[key][str(row["arm"])] = row
    root_keys = set(state_to_task)
    complete = set(by_state) == root_keys and len(by_state) == len(root_keys)
    three_arms = complete and all(set(arms) == set(ARMS) for arms in by_state.values())

    arm_reports: dict[str, Any] = {}
    for candidate_arm in ("one_shot_h8", "persistent_h8"):
        counts = Counter()
        task_rows: dict[str, list[dict[str, bool]]] = defaultdict(list)
        for state_key, arms in by_state.items():
            if not all(arm in arms for arm in ARMS):
                continue
            source = bool(arms["source_h8"]["success"])
            candidate = bool(arms[candidate_arm]["success"])
            counts[classify(source, candidate)] += 1
            task_rows[state_to_task[state_key]].append({"source": source, "candidate": candidate})
        n = sum(counts.values())
        source_hits = counts["both"] + counts["source_only"]
        candidate_hits = counts["both"] + counts["candidate_only"]
        union_hits = n - counts["neither"]
        h_within = (counts["source_only"] + counts["candidate_only"]) / n if n else 0.0
        source_rate = source_hits / n if n else 0.0
        candidate_rate = candidate_hits / n if n else 0.0
        union_rate = union_hits / n if n else 0.0
        oracle_gain = union_rate - max(source_rate, candidate_rate)
        candidate_tasks = sorted(
            task for task, task_values in task_rows.items() if any(v["candidate"] and not v["source"] for v in task_values)
        )
        arm_reports[candidate_arm] = {
            "counts": dict(counts),
            "n_states": n,
            "source_rate": source_rate,
            "candidate_rate": candidate_rate,
            "union_rate": union_rate,
            "h_within": h_within,
            "oracle_gain": oracle_gain,
            "candidate_only_tasks": candidate_tasks,
            "n_candidate_only_tasks": len(candidate_tasks),
            "task_cluster_bootstrap": bootstrap(task_rows),
        }

    seen_state_keys: set[str] = set()
    for collection_dir in args.seen_collection_dir:
        for path in collection_dir.resolve().glob("**/summary.json"):
            summary = load(path)
            for row in summary.get("per_state_arm", []):
                seen_state_keys.add(str(row["state_key"]))

    one = arm_reports["one_shot_h8"]
    checks = {
        "all_24_b2_roots_complete": complete,
        "three_arms_per_root": three_arms,
        "b0_b1_b2_root_disjoint": (
            set(partitions["b0_smoke"]["state_keys"]).isdisjoint(b2["state_keys"])
            and set(partitions["b1_collect"]["state_keys"]).isdisjoint(b2["state_keys"])
        ),
        "qualification_state_not_seen_in_dagger": not (root_keys & seen_state_keys),
        "source_and_candidate_only_both_exist": one["counts"].get("source_only", 0) > 0
        and one["counts"].get("candidate_only", 0) > 0,
        "h_within_ge_threshold": one["h_within"] >= args.min_h_within,
        "oracle_gain_ge_threshold": one["oracle_gain"] >= args.min_oracle_gain,
        "candidate_only_tasks_ge_threshold": one["n_candidate_only_tasks"] >= args.min_candidate_tasks,
    }
    artifact = {
        "schema_version": "rase-e3b-qualification/v1",
        "status": "complete" if complete else "incomplete",
        "n_roots": len(by_state),
        "n_tasks": len(set(state_to_task.values())),
        "suite_success": {suite: summary["success"] for suite, summary in suite_summaries.items()},
        "arms": arm_reports,
        "seen_state_keys": len(seen_state_keys),
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "interpretation": (
            "One-shot is the primary RASE decision-point candidate; persistent is an on-policy compounding-error ablation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": artifact["decision"], "one_shot": one}, sort_keys=True))
    return 0 if artifact["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
