#!/usr/bin/env python3
"""Analyze the R6-B1.2 candidate-arm data before training the R6-C baseline.

Produces the per-boundary statistics required by the R6-C plan:

- source failure prevalence per policy / suite / boundary;
- per-boundary rescuability: P(persistent success | enter now) and its teacher
  cost (teacher/action steps) for ``ENTER_PERSISTENT_OFT``;
- temporal non-monotonicity: does P(source success within 8/16/32) and the final
  source success change non-monotonically across elapsed source steps;
- candidate-arm opportunity: given the source outcome at each boundary, how many
  trajectories would have been rescued by a persistent takeover that fired at
  that boundary and how many teacher steps it would have cost.

Every statistic is computed from the same frozen B1.2 metadata rows used to
build the training dataset.  ``--grouping`` keeps all rows of a trajectory group
in one fold when the same keys are reused for the 5-seed OOF.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_exclusions(path: Path) -> set[tuple[str, int, str]]:
    """Load a frozen exclusion manifest into {(policy_id, seed_index, state_key)}."""
    if path is None:
        return set()
    data = json.loads(path.read_text())
    excluded = set()
    for entry in data["excluded"]:
        policy, seed, state = entry
        excluded.add((str(policy), int(seed), str(state)))
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="B1.2 collection output root (contains suite_*)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, default=None,
                        help="frozen exclusion manifest of known nondeterministic groups")
    args = parser.parse_args()

    metadata_paths = sorted(glob.glob(str(args.input_root / "suite_*" / "*" / "seed_*" / "*__seed*.json")))
    metadata_paths = [path for path in metadata_paths if Path(path).name != "report.json"]
    if not metadata_paths:
        raise SystemExit(f"no trajectory metadata under {args.input_root}")

    excluded = load_exclusions(args.exclusions)

    groups: dict[str, list[dict]] = defaultdict(list)
    for path_string in metadata_paths:
        data = json.loads(Path(path_string).read_text())
        if not data["rows"]:
            continue
        policy_id = str(data["rows"][0]["policy_id"])
        seed_index = int(data["rows"][0]["seed_index"])
        state_key = str(data["rows"][0]["state_key"])
        if (policy_id, seed_index, state_key) in excluded:
            continue
        group = str(data["rows"][0]["group_id"])
        for boundary in data["rows"]:
            boundary["group_id"] = group
            groups[group].append(boundary)

    rows: list[dict] = []
    for group in sorted(groups):
        members = sorted(groups[group], key=lambda row: int(row["elapsed_source_steps"]))
        for position, boundary in enumerate(members):
            boundary["_position"] = position
            boundary["_n_boundaries"] = len(members)
            rows.append(boundary)

    by_policy: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_policy[row["policy_id"]].append(row)

    def rate(values: list[bool]) -> float:
        return float(np.mean(values)) if values else float("nan")

    policies = sorted(by_policy)
    policy_stats: dict[str, dict] = {}
    for policy in policies:
        member_rows = by_policy[policy]
        policy_stats[policy] = {
            "n_rows": len(member_rows),
            "n_groups": len({row["group_id"] for row in member_rows}),
            "n_states": len({row["state_key"] for row in member_rows}),
            "source_final_success_rate": rate([bool(row["source_final_success"]) for row in member_rows]),
            "persistent_success_rate": rate([bool(row["persistent_success_if_enter_now"]) for row in member_rows]),
            "persistent_teacher_steps_mean": float(np.mean(
                [float(row["persistent_teacher_steps_if_enter_now"] or 0.0) for row in member_rows])),
            "source_failure_rows": int(sum(not bool(row["source_final_success"]) for row in member_rows)),
            "source_failure_rescued_rate": rate([
                bool(row["persistent_success_if_enter_now"])
                for row in member_rows if not bool(row["source_final_success"])
            ]),
            "source_failure_rescue_teacher_steps_mean": float(np.mean([
                float(row["persistent_teacher_steps_if_enter_now"] or 0.0)
                for row in member_rows if not bool(row["source_final_success"])
            ])),
        }

    # Per-boundary rescuability (pooled across groups; boundaries are t={0,16,32,64,96,128}).
    by_elapsed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_elapsed[int(row["elapsed_source_steps"])].append(row)
    boundary_stats = []
    for elapsed in sorted(by_elapsed):
        members = by_elapsed[elapsed]
        boundary_stats.append({
            "elapsed_source_steps": elapsed,
            "n_rows": len(members),
            "source_success_rate": rate([bool(row["source_final_success"]) for row in members]),
            "source_within_8_rate": rate([bool(row["source_success_within_8"]) for row in members]),
            "source_within_16_rate": rate([bool(row["source_success_within_16"]) for row in members]),
            "source_within_32_rate": rate([bool(row["source_success_within_32"]) for row in members]),
            "persistent_success_rate": rate([bool(row["persistent_success_if_enter_now"]) for row in members]),
            "persistent_teacher_steps_mean": float(np.mean(
                [float(row["persistent_teacher_steps_if_enter_now"] or 0.0) for row in members])),
        })

    # Temporal non-monotonicity: for each group, does the boundary-ordered
    # within-8/16/32 horizon series change direction more than once?
    non_monotonic = 0
    monotonic_groups = 0
    for group, members in groups.items():
        ordered = sorted(members, key=lambda row: int(row["elapsed_source_steps"]))
        if len(ordered) < 3:
            continue
        monotonic_groups += 1
        horizon = [int(row["source_success_within_16"]) for row in ordered]
        direction = []
        for left, right in zip(horizon, horizon[1:]):
            if right != left:
                direction.append(1 if right > left else -1)
        changes = sum(1 for left, right in zip(direction, direction[1:]) if left != right)
        if changes > 0:
            non_monotonic += 1

    # Candidate-arm opportunity: per group, if the controller fires at boundary
    # index i (the earliest risky boundary with dwell), how does success change
    # and how many teacher steps are spent.
    rescue_opportunity = []
    for group, members in groups.items():
        ordered = sorted(members, key=lambda row: int(row["elapsed_source_steps"]))
        for position, boundary in enumerate(ordered):
            if not bool(boundary["persistent_success_if_enter_now"]):
                continue
            source_success = bool(ordered[0]["source_final_success"])
            rescue_opportunity.append({
                "group_id": group,
                "state_key": str(boundary["state_key"]),
                "policy_id": str(boundary["policy_id"]),
                "boundary_index": position,
                "elapsed_source_steps": int(boundary["elapsed_source_steps"]),
                "source_success": source_success,
                "persistent_success": bool(boundary["persistent_success_if_enter_now"]),
                "teacher_steps": float(boundary["persistent_teacher_steps_if_enter_now"] or 0.0),
                "creates_new_success": bool(boundary["persistent_success_if_enter_now"]) and not source_success,
                "is_rescue_after_failure_visible": not bool(boundary["source_success_within_32"]),
            })

    report = {
        "schema_version": "rase-r6c-candidate-arm-analysis/v1",
        "scientific_scope": "per-boundary candidate-arm statistics from the frozen B1.2 collection",
        "input_root": str(args.input_root.resolve()),
        "n_rows": len(rows),
        "n_groups": len(groups),
        "n_states": len({row["state_key"] for row in rows}),
        "n_tasks": len({row["task_id"] for row in rows}),
        "policies": policies,
        "exclusions": str(args.exclusions.resolve()) if args.exclusions is not None else None,
        "n_excluded_groups": len(excluded),
        "policy_stats": policy_stats,
        "boundary_stats": boundary_stats,
        "temporal": {
            "n_groups_ge_3_boundaries": monotonic_groups,
            "n_groups_with_nonmonotonic_within16": non_monotonic,
            "nonmonotonic_fraction": float(non_monotonic / max(1, monotonic_groups)),
        },
        "rescue_opportunity": {
            "n_rescue_opportunities": len(rescue_opportunity),
            "n_groups_with_rescue": len({row["group_id"] for row in rescue_opportunity}),
            "n_created_success": int(sum(row["creates_new_success"] for row in rescue_opportunity)),
            "n_rescue_after_failure_visible": int(
                sum(row["is_rescue_after_failure_visible"] for row in rescue_opportunity)),
            "mean_teacher_steps": float(np.mean([row["teacher_steps"] for row in rescue_opportunity]))
            if rescue_opportunity else float("nan"),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "n_rows": report["n_rows"], "n_groups": report["n_groups"],
        "policy_stats": policy_stats,
        "boundary_stats": boundary_stats,
        "temporal": report["temporal"],
        "rescue_opportunity": report["rescue_opportunity"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
