"""Model-free confirmation opportunity audit at independent-root granularity."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


RAW_FIELDS = ("success", "harm", "query_cost", "fallback_cost", "latency_cost")


def utility(row: dict, weights: dict) -> float:
    return (
        float(weights["success_reward"]) * float(row["success"])
        - float(weights["harm_weight"]) * float(row["harm"])
        - float(weights["query_weight"]) * float(row["query_cost"])
        - float(weights["fallback_weight"]) * float(row["fallback_cost"])
        - float(weights["latency_weight"]) * float(row["latency_cost"])
    )


def aggregate_confirmation(rows: list[dict], *, repeats: int, weights: dict) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        missing = set(RAW_FIELDS) - row.keys()
        if missing:
            raise ValueError(f"branch row missing raw fields: {sorted(missing)}")
        if row.get("available", True) is False:
            continue
        key = (str(row["root_id"]), str(row["policy_id"]), str(row["decision_point_id"]), str(row["operator_id"]))
        grouped[key].append(row)
    point_values = []
    for key, trials in grouped.items():
        replicas = [int(row["exact_repeat_replica"]) for row in trials]
        if sorted(replicas) != list(range(repeats)):
            raise ValueError(f"branch {key} requires exact replicas 0..{repeats - 1}, got {sorted(replicas)}")
        exemplar = trials[0]
        point_values.append({
            "root_id": key[0], "policy_id": key[1], "decision_point_id": key[2],
            "operator_id": key[3], "task_id": str(exemplar["task_id"]),
            "suite": str(exemplar["suite"]),
            "utility": float(np.mean([utility(row, weights) for row in trials])),
            **{field: float(np.mean([float(row[field]) for row in trials])) for field in RAW_FIELDS},
        })
    if not point_values:
        raise ValueError("no available confirmation branches")
    by_root_operator: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in point_values:
        by_root_operator[(row["root_id"], row["policy_id"], row["operator_id"])].append(row)
    result = []
    for (root_id, policy_id, operator_id), values in sorted(by_root_operator.items()):
        exemplar = values[0]
        result.append({
            "root_id": root_id, "operator_id": operator_id,
            "policy_id": policy_id, "task_id": exemplar["task_id"],
            "suite": exemplar["suite"], "decision_points": len(values),
            "utility": float(np.mean([row["utility"] for row in values])),
            **{field: float(np.mean([row[field] for row in values])) for field in RAW_FIELDS},
        })
    return result


def _gap(rows: list[dict]) -> float:
    by_root_policy: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_operator: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_root_policy[(row["root_id"], row["policy_id"])].append(row)
        by_operator[row["operator_id"]].append(row["utility"])
    oracle = np.mean([
        max(item["utility"] for item in values) for values in by_root_policy.values()
    ])
    best_fixed = max(np.mean(values) for values in by_operator.values())
    return float(oracle - best_fixed)


def nested_task_root_bootstrap(rows: list[dict], *, samples: int, seed: int) -> tuple[float, float]:
    by_task_root: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_task_root[row["task_id"]][row["root_id"]].append(row)
    tasks = sorted(by_task_root)
    if len(tasks) < 2:
        raise ValueError("task/root bootstrap requires at least two tasks")
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(samples):
        sampled: list[dict] = []
        for task_draw, task_index in enumerate(rng.integers(0, len(tasks), len(tasks))):
            task = tasks[int(task_index)]
            roots = sorted(by_task_root[task])
            for root_draw, root_index in enumerate(rng.integers(0, len(roots), len(roots))):
                root = roots[int(root_index)]
                synthetic = f"boot-t{task_draw}-r{root_draw}"
                sampled.extend({**row, "root_id": synthetic} for row in by_task_root[task][root])
        gaps.append(_gap(sampled))
    return float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))


def audit_opportunity(
    rows: list[dict], *, repeats: int, weights: dict, gate: dict,
    bootstrap_samples: int = 2000, bootstrap_seed: int = 202708,
) -> dict:
    aggregated = aggregate_confirmation(rows, repeats=repeats, weights=weights)
    roots = sorted({row["root_id"] for row in aggregated})
    tasks = sorted({row["task_id"] for row in aggregated})
    suites = sorted({row["suite"] for row in aggregated})
    policies = sorted({row["policy_id"] for row in aggregated})
    gap = _gap(aggregated)
    lower, upper = nested_task_root_bootstrap(
        aggregated, samples=bootstrap_samples, seed=bootstrap_seed,
    )
    # G-O2's denominator is the physical independent root, not the number of
    # decision points or the number of source policies. Average over policies
    # before assigning co-winners to a root.
    by_root_operator: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in aggregated:
        by_root_operator[(row["root_id"], row["operator_id"])].append(row["utility"])
    by_root: dict[str, list[dict]] = defaultdict(list)
    for (root_id, operator_id), values in by_root_operator.items():
        by_root[root_id].append({
            "operator_id": operator_id,
            "utility": float(np.mean(values)),
        })
    winner_roots: dict[str, set[str]] = defaultdict(set)
    for root, values in by_root.items():
        best = max(row["utility"] for row in values)
        for row in values:
            if np.isclose(row["utility"], best, atol=1e-12, rtol=0):
                winner_roots[row["operator_id"]].add(root)
    winner_summary = {
        operator: {"roots": len(wins), "fraction": len(wins) / len(roots)}
        for operator, wins in sorted(winner_roots.items())
    }
    qualifying = [
        operator for operator, values in winner_summary.items()
        if values["fraction"] >= float(gate["minimum_root_winner_fraction"])
    ]
    policy_results = {}
    for policy in policies:
        subset = [row for row in aggregated if row["policy_id"] == policy]
        policy_results[policy] = {"oracle_minus_best_fixed": _gap(subset)}
    required_policies = set(gate["required_policies"])
    meaningful = float(gate["minimum_oracle_minus_best_fixed"])
    checks = {
        "G_O1_mean_effect": gap >= meaningful,
        "G_O1_bootstrap_lower": lower > 0,
        "G_O2_operator_diversity": len(qualifying) >= int(gate["minimum_winner_operators"]),
        "G_O2_task_coverage": len(tasks) >= int(gate["minimum_tasks"]),
        "G_O2_suite_coverage": len(suites) >= int(gate["minimum_suites"]),
        "G_O3_policy_coverage": required_policies <= set(policies),
        "G_O3_policy_opportunity": all(
            policy_results.get(policy, {}).get("oracle_minus_best_fixed", -np.inf) >= meaningful
            for policy in required_policies
        ),
    }
    raw_summary = {
        operator: {
            field: float(np.mean([row[field] for row in aggregated if row["operator_id"] == operator]))
            for field in RAW_FIELDS
        }
        for operator in sorted({row["operator_id"] for row in aggregated})
    }
    return {
        "schema_version": "rase-vnext-opportunity-audit/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "independence_unit": "root_id",
        "bootstrap_unit": "task_with_roots_nested",
        "roots": len(roots), "tasks": len(tasks), "suites": suites,
        "policies": policies, "oracle_minus_best_fixed": gap,
        "bootstrap_95_ci": [lower, upper], "winner_summary": winner_summary,
        "qualifying_winner_operators": qualifying, "policy_results": policy_results,
        "raw_metrics_by_operator": raw_summary, "checks": checks,
        "remains_locked_on_fail": ["information_gate", "mvp", "selector", "closed_loop"],
    }
