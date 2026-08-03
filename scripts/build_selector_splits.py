#!/usr/bin/env python3
"""Build frozen episode- or task-disjoint splits for selector JSONL rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--grouping", choices=("episode", "task"), default="episode")
    parser.add_argument("--leave-suite-out", action="store_true")
    parser.add_argument("--requirements", type=Path)
    parser.add_argument(
        "--fail-not-ready",
        action="store_true",
        help="exit non-zero after writing an artifact whose support audit is NOT_READY",
    )
    args = parser.parse_args()

    from rase.collect.dataset_export import (
        audit_split_support,
        build_grouped_benchmark_splits,
        build_leave_one_suite_out_splits,
    )

    rows = []
    for path in args.dataset:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    group_fields = ("task_id", "episode_id") if args.grouping == "episode" else ("task_id",)
    if args.leave_suite_out:
        result = build_leave_one_suite_out_splits(rows, group_fields=group_fields)
    else:
        result = build_grouped_benchmark_splits(
            rows,
            seed=args.seed,
            group_fields=group_fields,
            stratify_fields=("suite", "perturb_dim", "level", "cohort"),
        )
        result["schema_version"] = "rase-selector-benchmark-splits/v1"
    result["grouping"] = args.grouping
    result["sources"] = [str(path.resolve()) for path in args.dataset]
    if args.requirements:
        requirements = json.loads(args.requirements.read_text(encoding="utf-8"))
        if args.leave_suite_out:
            requirements = {"required_splits": ["train", "test"], **requirements}
            audits = {
                suite: audit_split_support(rows, fold, requirements=requirements)
                for suite, fold in result["folds"].items()
            }
            result["requirements_audit"] = {
                "status": "READY"
                if all(audit["ready"] for audit in audits.values())
                else "NOT_READY",
                "ready": all(audit["ready"] for audit in audits.values()),
                "folds": audits,
            }
        else:
            result["requirements_audit"] = audit_split_support(
                rows, result, requirements=requirements
            )
        result["status"] = result["requirements_audit"]["status"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(result.get("requirements_audit", result.get("audit", {})), indent=2), flush=True
    )
    print(f"WROTE {args.output}", flush=True)
    if args.fail_not_ready and result.get("status") == "NOT_READY":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
