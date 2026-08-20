#!/usr/bin/env python3
"""Build a capability-aware per-policy operator opportunity/support atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_rows(path: Path, policy_id: str) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("policy_id")) == policy_id:
            rows.append(row)
    return rows


def group_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["root_id"]), str(row["decision_point_id"]),
        int(row["exact_repeat_replica"]),
    )


def analyze_set(
    rows: Sequence[dict[str, Any]], operators: Sequence[str], *, tie_margin: float,
) -> dict[str, Any]:
    operators = tuple(operators)
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("available") is True and row.get("utility") is not None:
            grouped[group_key(row)][str(row["operator_id"])] = row
    eligible = {
        key: values for key, values in grouped.items()
        if all(operator in values for operator in operators)
    }
    wins = {operator: 0 for operator in operators}
    co_best = {operator: 0 for operator in operators}
    practical_ties = 0
    by_suite: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pairwise_vs_continue: dict[str, dict[str, int]] = {}
    for operator in operators:
        if operator != "continue.source":
            pairwise_vs_continue[operator] = {"operator_win": 0, "continue_win": 0, "tie": 0}
    utilities = {operator: [] for operator in operators}
    oracle_values: list[float] = []
    group_rows: list[dict[str, Any]] = []
    for key, values in sorted(eligible.items()):
        scores = {operator: float(values[operator]["utility"]) for operator in operators}
        best = max(scores.values())
        winners = [operator for operator, value in scores.items() if best - value <= tie_margin]
        if len(winners) == len(operators):
            practical_ties += 1
        elif len(winners) == 1:
            wins[winners[0]] += 1
            by_suite[str(values[winners[0]]["suite"])][winners[0]] += 1
        else:
            for operator in winners:
                co_best[operator] += 1
        for operator, value in scores.items():
            utilities[operator].append(value)
        oracle_values.append(best)
        continue_value = scores.get("continue.source")
        if continue_value is not None:
            for operator, counts in pairwise_vs_continue.items():
                difference = scores[operator] - continue_value
                if difference > tie_margin:
                    counts["operator_win"] += 1
                elif difference < -tie_margin:
                    counts["continue_win"] += 1
                else:
                    counts["tie"] += 1
        group_rows.append({
            "root_id": key[0], "decision_point_id": key[1], "replica": key[2],
            "suite": str(next(iter(values.values()))["suite"]),
            "task_id": str(next(iter(values.values()))["task_id"]),
            "utilities": scores, "practical_winners": winners,
        })
    fixed_means = {
        operator: float(np.mean(values)) if values else float("nan")
        for operator, values in utilities.items()
    }
    best_fixed = max(fixed_means, key=fixed_means.get) if fixed_means else None
    gap = (
        float(np.mean(oracle_values)) - fixed_means[best_fixed]
        if oracle_values and best_fixed is not None else float("nan")
    )
    support_operators = [operator for operator, count in wins.items() if count >= 5]
    result = {
        "operators": list(operators), "eligible_groups": len(eligible),
        "tie_margin": tie_margin, "unique_wins": wins, "co_best": co_best,
        "all_operator_practical_ties": practical_ties,
        "pairwise_vs_continue": pairwise_vs_continue,
        "mean_utility": fixed_means, "best_fixed_operator": best_fixed,
        "oracle_minus_best_fixed": gap,
        "support_operators_at_least_5_unique_wins": support_operators,
        "support_gate": "PASS" if len(support_operators) >= 2 else "FAIL",
        "unique_wins_by_suite": {suite: dict(value) for suite, value in by_suite.items()},
        "groups": group_rows,
    }
    for value in fixed_means.values():
        if not math.isfinite(value):
            raise ValueError("atlas contains non-finite fixed utility")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie-margin", type=float, default=0.01)
    args = parser.parse_args()
    rows = load_rows(args.branches, args.policy_id)
    declared = [
        operator for operator in (
            "continue.source", "requery.source", "resample.source", "fallback.persistent",
        )
        if any(row.get("operator_id") == operator and row.get("available") is True for row in rows)
    ]
    source = [operator for operator in declared if operator in {
        "continue.source", "requery.source", "resample.source",
    }]
    result = {
        "schema_version": "rase-vnext-policy-operator-atlas/v1",
        "policy_id": args.policy_id, "branches": str(args.branches.resolve()),
        "branches_sha256": sha256(args.branches),
        "tie_margin": args.tie_margin,
        "source_only": analyze_set(rows, source, tie_margin=args.tie_margin),
        "non_abort_with_fallback": analyze_set(rows, declared, tie_margin=args.tie_margin),
    }
    atomic_json(args.output, result)
    summary = {
        name: {
            key: value for key, value in result[name].items()
            if key in {"operators", "eligible_groups", "unique_wins", "all_operator_practical_ties", "pairwise_vs_continue", "best_fixed_operator", "oracle_minus_best_fixed", "support_gate"}
        }
        for name in ("source_only", "non_abort_with_fallback")
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
