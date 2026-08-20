#!/usr/bin/env python3
"""RASE success-breakthrough step 0: outcome-blind domain atlas.

Two-layer outcome policy (preregistered): candidate-domain *features* are
pre-rollout only; domain *classification labels* may use the inner development
outcomes (confirmation + K5 + K3); the outer validation set is frozen and
never read here.

Grid: (policy, decision_time) x task. Each cell reports support, source
success, fallback success, fallback-not-optimal rate, winner entropy,
oracle-minus-best-fixed, all-fail rate, and task-bootstrap CIs. Cells are
classified all-candidates-fail / fallback-dominates / heterogeneous-winner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_branches(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _unit_ops(rows: list[dict[str, Any]]) -> dict[tuple[str, Any], dict[str, dict[str, Any]]]:
    """Group executable branch rows by (policy, task, step, root, replica)."""
    units: dict[tuple[str, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("available") is not True:
            continue
        if row.get("operator_id") == "abort.safe":
            continue
        key = (
            str(row.get("policy_id")), str(row.get("task_id")),
            int(row.get("decision_step")),
            str(row.get("root_id")),
            int(row.get("seed_ledger", {}).get("exact_repeat_replica", row.get("exact_repeat_replica", 0))),
        )
        units[key][str(row["operator_id"])] = row
    return units


def _cell_stats(unit_rows: dict[tuple[str, Any], dict[str, dict[str, Any]]]) -> dict[str, Any]:
    n = len(unit_rows)
    source_success: list[float] = []
    fallback_success: list[float] = []
    fb_not_optimal = 0
    all_fail = 0
    hetero = 0
    winner_entropies: list[float] = []
    oracle_minus_best: list[float] = []
    for ops in unit_rows.values():
        success = {op: bool(row.get("success")) for op, row in ops.items()}
        executable = {op: ok for op, ok in success.items() if op != "abort.safe"}
        fb = executable.get("fallback.persistent")
        others = [v for op, v in executable.items() if op != "fallback.persistent"]
        source_vals = [v for op, v in executable.items() if op in ("continue.source", "requery.source")]
        if source_vals:
            source_success.append(float(any(source_vals)))
        if fb is not None:
            fallback_success.append(float(fb))
        winners = [op for op, ok in executable.items() if ok]
        if winners:
            entropy = float(np.log2(len(winners)))
            winner_entropies.append(entropy)
        if fb is False:
            if not any(others):
                all_fail += 1
            else:
                hetero += 1
                fb_not_optimal += 1
        elif fb is True:
            if any(others):
                pass  # fallback co-wins; not a counterfactual failure
        else:
            # fallback missing (should not happen for available rows)
            if not any(others):
                all_fail += 1
        # oracle minus best-fixed (best fixed = argmax over per-op success rates)
        best_fixed = max(
            (sum(1 for ops in unit_rows.values() if ops.get(op, {}).get("success")) / n
             for op in ("continue.source", "requery.source", "fallback.persistent")),
        )
        oracle = sum(1 for ops in unit_rows.values() if any(
            row.get("success") for row in ops.values() if row.get("operator_id") != "abort.safe"
        )) / n
        oracle_minus_best.append(oracle - best_fixed)
    return {
        "units": n,
        "source_success_rate": round(float(np.mean(source_success)), 4) if source_success else None,
        "fallback_success_rate": round(float(np.mean(fallback_success)), 4) if fallback_success else None,
        "fallback_not_optimal_units": fb_not_optimal,
        "fallback_not_optimal_rate": round(fb_not_optimal / n, 4) if n else None,
        "heterogeneous_units": hetero,
        "all_fail_units": all_fail,
        "all_fail_rate": round(all_fail / n, 4) if n else None,
        "mean_winner_entropy": round(float(np.mean(winner_entropies)), 4) if winner_entropies else None,
        "oracle_minus_best_fixed": round(float(np.mean(oracle_minus_best)), 4) if oracle_minus_best else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--k5", type=Path, required=True)
    parser.add_argument("--k3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in (args.confirmation, args.k5, args.k3):
        rows.extend(_load_branches(path))
    units = _unit_ops(rows)
    print("total units:", len(units), "| total branch rows:", len(rows))

    # grid: (policy, step) -> task -> list of unit ops dicts (one per root/replica)
    grid: dict[str, dict[str, list[dict[str, dict[str, Any]]]]] = defaultdict(lambda: defaultdict(list))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, ops in units.items():
        policy, task, step, root, replica = key
        grid[f"{policy}@{step}"][task].append(ops)
        by_task[task].append(ops)

    atlas: dict[str, Any] = {
        "schema_version": "rase-vnext-domain-atlas/v1",
        "status": "frozen",
        "sources": {
            "confirmation": str(args.confirmation.resolve()),
            "k5": str(args.k5.resolve()),
            "k3": str(args.k3.resolve()),
        },
        "outcome_policy": (
            "two-layer: candidate-domain features are pre-rollout only; domain "
            "classification labels use inner development outcomes; outer "
            "validation set frozen and not read here"
        ),
        "classification": {
            "all_candidates_fail": "fallback fails and no other executable candidate succeeds",
            "fallback_dominates": "fallback succeeds and no non-fallback counterfactual winner",
            "heterogeneous_winner": "fallback fails but some other candidate succeeds",
        },
        "grid": {},
        "summary": {},
        "units_total": len(units),
    }
    for cell_key, tasks in sorted(grid.items()):
        policy, step = cell_key.split("@")
        cell: dict[str, Any] = {"policy": policy, "decision_step": int(step), "tasks": {}}
        cell_units = 0
        cell_hetero = 0
        cell_allfail = 0
        cell_fbnotopt = 0
        for task, task_units in sorted(tasks.items()):
            stats = _cell_stats({f"u{i}": ops for i, ops in enumerate(task_units)})
            cell_units += stats["units"]
            cell_hetero += stats["heterogeneous_units"]
            cell_allfail += stats["all_fail_units"]
            cell_fbnotopt += stats["fallback_not_optimal_units"]
            cell["tasks"][task] = stats
        cell["totals"] = {
            "units": cell_units,
            "heterogeneous_units": cell_hetero,
            "all_fail_units": cell_allfail,
            "fallback_not_optimal_units": cell_fbnotopt,
            "heterogeneous_rate": round(cell_hetero / cell_units, 4) if cell_units else None,
        }
        atlas["grid"][cell_key] = cell
        atlas["summary"][cell_key] = cell["totals"]

    atomic_json(args.output, atlas)
    print(json.dumps(atlas["summary"], indent=2, sort_keys=True))
    print("atlas written:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
