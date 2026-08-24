#!/usr/bin/env python3
"""Audit Gate-B schedule-screen records and estimate schedule-oracle headroom.

For root i and fixed re-planning period E, Y_i(E) is terminal success.  The
reported quantity is:

  AC_schedule = mean_i max_E Y_i(E) - max_E mean_i Y_i(E).

It is an *eligibility diagnostic*, not a deployable selector result: the
per-root max is never run at test time.  A positive value only licenses the
subsequent, separately controlled trajectory-learning experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_schedules(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--schedules must be a non-empty list of distinct integers")
    return values


def read_roots(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    roots_dir = path / "roots" if path.is_dir() and (path / "roots").is_dir() else path
    if not roots_dir.is_dir():
        raise ValueError(f"expected Gate-B directory or roots directory, got {path}")
    complete: list[dict[str, Any]] = []
    errors: dict[str, list[str]] = {}
    for item in sorted(roots_dir.glob("*.json")):
        record = json.loads(item.read_text(encoding="utf-8"))
        root = record.get("root") if isinstance(record.get("root"), dict) else {}
        root_id = str(root.get("root_id", item.stem))
        messages: list[str] = []
        if record.get("schema_version") != "rase-v6-gateb-schedule-root/v1":
            messages.append("wrong_schema")
        if record.get("status") != "complete":
            messages.append(f"root_status={record.get('status')!r}")
        schedules = record.get("schedules")
        if not isinstance(schedules, list):
            messages.append("missing_schedules")
        if messages:
            errors[root_id] = messages
            continue
        complete.append(record)
    return complete, errors


def audit_root(record: Mapping[str, Any], expected: list[int]) -> tuple[dict[str, Any] | None, list[str]]:
    root = record.get("root")
    if not isinstance(root, Mapping):
        return None, ["missing_root"]
    root_id = str(root.get("root_id", ""))
    rows = record.get("schedules")
    if not isinstance(rows, list):
        return None, ["missing_schedules"]
    errors: list[str] = []
    by_e: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("nonmapping_schedule")
            continue
        try:
            e = int(row.get("schedule"))
        except (TypeError, ValueError):
            errors.append("invalid_schedule")
            continue
        if e in by_e:
            errors.append(f"duplicate_schedule={e}")
        by_e[e] = row
    if sorted(by_e) != sorted(expected):
        errors.append(f"schedules={sorted(by_e)} expected={sorted(expected)}")
    expected_initial: str | None = None
    expected_target_set: str | None = None
    for e in expected:
        row = by_e.get(e)
        if row is None:
            continue
        if row.get("status") != "complete":
            errors.append(f"schedule_{e}_not_complete")
            continue
        if not isinstance(row.get("success"), bool):
            errors.append(f"schedule_{e}_success_not_bool")
        initial = row.get("initial_qpos_sha256")
        if not isinstance(initial, str):
            errors.append(f"schedule_{e}_missing_initial_qpos")
        elif expected_initial is None:
            expected_initial = initial
        elif expected_initial != initial:
            errors.append("initial_qpos_not_matched")
        perturb = row.get("perturbation")
        if not isinstance(perturb, Mapping):
            errors.append(f"schedule_{e}_missing_perturbation")
        else:
            if perturb.get("applied_after_env_step") != root.get("perturb_at_step"):
                errors.append(f"schedule_{e}_wrong_perturb_step")
            if perturb.get("pre_qpos_sha256") == perturb.get("post_qpos_sha256"):
                errors.append(f"schedule_{e}_no_qpos_change")
        if int(row.get("n_inference_events", 0)) < 1:
            errors.append(f"schedule_{e}_no_inference")
        targets = row.get("perturbation_targets")
        target_set = json.dumps(targets, sort_keys=True)
        if expected_target_set is None:
            expected_target_set = target_set
        elif target_set != expected_target_set:
            errors.append("perturbation_targets_not_matched")
    if errors:
        return None, errors
    outcomes = {e: int(bool(by_e[e]["success"])) for e in expected}
    winners = [e for e in expected if outcomes[e] == max(outcomes.values())]
    return {
        "root_id": root_id,
        "task_id": str(root.get("task_id")),
        "task_number": int(root.get("task_number")),
        "replicate": int(root.get("replicate")),
        "init_state_id": int(root.get("init_state_id")),
        "perturb_at_step": int(root.get("perturb_at_step")),
        "outcomes": outcomes,
        "winner_schedules": winners,
        "unique_winner": winners[0] if len(winners) == 1 else None,
    }, []


def schedule_means(roots: list[Mapping[str, Any]], schedules: list[int]) -> dict[int, float]:
    return {
        e: float(np.mean([int(root["outcomes"][e]) for root in roots]))
        for e in schedules
    }


def task_internals(roots: list[Mapping[str, Any]], schedules: list[int]) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in roots:
        by_task[str(row["task_id"])].append(row)
    detail: dict[str, Any] = {}
    heterogeneous: list[str] = []
    for task, rows in sorted(by_task.items()):
        unique = [int(row["unique_winner"]) for row in rows if row["unique_winner"] is not None]
        # A task supplies strong schedule heterogeneity only if different
        # schedules each uniquely win at some matched root; arbitrary tie-breaks
        # never count as evidence.
        is_heterogeneous = len(set(unique)) >= 2
        if is_heterogeneous:
            heterogeneous.append(task)
        detail[task] = {
            "n_roots": len(rows),
            "schedule_success": schedule_means(rows, schedules),
            "unique_winner_counts": dict(sorted(Counter(unique).items())),
            "heterogeneous_unique_winners": is_heterogeneous,
        }
    return {"per_task": detail, "heterogeneous_tasks": heterogeneous}


def summarize(roots: list[Mapping[str, Any]], schedules: list[int]) -> dict[str, Any]:
    if not roots:
        return {"n_roots": 0, "ac_schedule": None}
    means = schedule_means(roots, schedules)
    oracle = float(np.mean([max(int(row["outcomes"][e]) for e in schedules) for row in roots]))
    best_e = max(schedules, key=lambda e: (means[e], -e))
    internals = task_internals(roots, schedules)
    return {
        "n_roots": len(roots),
        "n_tasks": len({str(row["task_id"]) for row in roots}),
        "roots_per_task": dict(sorted(Counter(str(row["task_id"]) for row in roots).items())),
        "schedule_success": means,
        "best_fixed_schedule": int(best_e),
        "best_fixed_success": float(means[best_e]),
        "schedule_oracle_success_diagnostic": oracle,
        "ac_schedule": float(oracle - means[best_e]),
        "n_oracle_better_than_best_fixed_roots": int(sum(
            max(int(row["outcomes"][e]) for e in schedules) > int(row["outcomes"][best_e])
            for row in roots
        )),
        "per_task": internals["per_task"],
        "heterogeneous_tasks": internals["heterogeneous_tasks"],
        "n_heterogeneous_tasks": len(internals["heterogeneous_tasks"]),
    }


def bootstrap(roots: list[Mapping[str, Any]], schedules: list[int], *, n: int, seed: int) -> dict[str, Any]:
    tasks = sorted({str(row["task_id"]) for row in roots})
    if len(tasks) < 2 or n < 1:
        return {"cluster": "task_id", "n_bootstrap": 0, "ac_schedule_ci95": None}
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in roots:
        by_task[str(row["task_id"])].append(row)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n):
        sample: list[Mapping[str, Any]] = []
        for _draw in tasks:
            sample.extend(by_task[rng.choice(tasks)])
        values.append(float(summarize(sample, schedules)["ac_schedule"]))
    return {
        "cluster": "task_id", "n_bootstrap": n,
        "ac_schedule_ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "ac_schedule_bootstrap_mean": float(np.mean(values)),
    }


def decision(summary: Mapping[str, Any], boot: Mapping[str, Any]) -> dict[str, Any]:
    if not summary.get("n_roots"):
        return {"decision": "FAIL", "reason": "no_auditable_roots", "next_action": "DO_NOT_TRAIN_RL"}
    ci = boot.get("ac_schedule_ci95")
    lower = float(ci[0]) if isinstance(ci, list) and len(ci) == 2 else math.nan
    passed = (
        float(summary["ac_schedule"]) >= 0.05
        and math.isfinite(lower) and lower > 0.0
        and int(summary["n_heterogeneous_tasks"]) >= 2
    )
    if passed:
        return {
            "decision": "PASS",
            "reason": "schedule_headroom_positive_with_task_internal_heterogeneity",
            "next_action": "RUN_INDEPENDENT_10_TASK_X_8_ROOT_CONFIRMATION_THEN_TRAJECTORY_RL",
        }
    return {
        "decision": "FAIL",
        "reason": "schedule_headroom_gate_not_met",
        "next_action": "DO_NOT_TRAIN_TRAJECTORY_RL; REPORT_LOCAL_CF_ONLY_OR_CHANGE_DOMAIN",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedules", default="2,4,6,8,10")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    schedules = parse_schedules(args.schedules)
    raw, root_errors = read_roots(args.input)
    auditable: list[dict[str, Any]] = []
    for record in raw:
        root_id = str((record.get("root") or {}).get("root_id", "unknown"))
        value, errors = audit_root(record, schedules)
        if errors:
            root_errors[root_id] = errors
        elif value is not None:
            auditable.append(value)
    summary = summarize(auditable, schedules)
    boot = bootstrap(auditable, schedules, n=args.bootstrap, seed=args.seed)
    result = {
        "schema_version": "rase-v6-gateb-schedule-analysis/v1",
        "input": str(args.input), "expected_schedules": schedules,
        "n_input_complete_roots": len(raw), "n_auditable_roots": len(auditable),
        "n_audit_failures": len(root_errors), "audit_failures": root_errors,
        "summary": summary, "bootstrap": boot, "gate": decision(summary, boot),
        "roots": auditable,
    }
    atomic_json(args.output, result)
    print(json.dumps({"summary": summary, "bootstrap": boot, "gate": result["gate"]}, sort_keys=True), flush=True)
    return 0 if auditable else 2


if __name__ == "__main__":
    raise SystemExit(main())
