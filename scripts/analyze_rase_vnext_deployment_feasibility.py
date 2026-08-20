#!/usr/bin/env python3
"""A2: deployment feasibility — oracle upper bound and per-unit statistics.

Reads the frozen cost ledger and frozen selector dataset, and computes:

  1. per-unit statistics: fallback-not-optimal rate, continue/requery
     substitutability, recoverable success, fallback-call reduction rate,
     reported at task / root / unit levels with task bootstrap CIs;
  2. oracle Pareto upper bound under S1 latency budgets and S2 fallback
     quotas with real costs: if the oracle cannot beat always-fallback in a
     preregistered budget interval, that scenario is closed (stop rule).
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=str, default="10,30,60,120")
    parser.add_argument("--quotas", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20270818)
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text())
    scenarios = json.loads(args.scenarios.read_text())
    if scenarios.get("status") != "frozen":
        raise SystemExit("scenario protocol is not frozen")
    budgets = [float(v) for v in args.budgets.split(",")]
    quotas = [float(v) for v in args.quotas.split(",")]

    units: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in ledger["ledger"]:
        units[(str(entry["root_id"]), int(entry["replica"]))][str(entry["operator"])] = entry

    # ---- 1. per-unit statistics -------------------------------------------
    fallback_not_optimal: list[str] = []
    substitutable: list[str] = []
    recoverable: list[float] = []
    fallback_calls: list[str] = []
    fallback_success: list[float] = []
    per_task: dict[str, list[float]] = defaultdict(list)
    per_root: dict[str, list[float]] = defaultdict(list)
    for unit, ops in units.items():
        success = {op: float(entry["success"]) for op, entry in ops.items()}
        if "fallback.persistent" not in success:
            continue
        fb = success["fallback.persistent"]
        others_better = any(
            value > fb for op, value in success.items() if op != "fallback.persistent"
        )
        task = str(next(iter(ops.values()))["task_id"])
        root = str(unit[0])
        per_task[task].append(float(others_better))
        per_root[root].append(float(others_better))
        fallback_success.append(float(fb))
        if others_better:
            fallback_not_optimal.append(f"{unit[0]}/{unit[1]}")
        # continue/requery substitutable: any non-fallback operator succeeds
        if any(value for op, value in success.items() if op != "fallback.persistent"):
            substitutable.append(f"{unit[0]}/{unit[1]}")
        # recoverable success: units where continue fails but fallback succeeds
        if not success.get("continue.source", 0.0) and fb:
            recoverable.append(1.0)
        else:
            recoverable.append(0.0)
        fallback_calls.append(unit[0])

    def _bootstrap_ci(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"mean": None, "ci95": None, "n": 0}
        array = np.asarray(values, dtype=np.float64)
        rng = np.random.default_rng(args.seed)
        indices = rng.integers(0, len(array), size=(args.replicates, len(array)))
        samples = array[indices].mean(axis=1)
        return {
            "mean": round(float(array.mean()), 4),
            "ci95": [round(float(np.quantile(samples, 0.025)), 4),
                     round(float(np.quantile(samples, 0.975)), 4)],
            "n": len(array),
        }

    task_level = {
        task: float(np.mean(values)) for task, values in per_task.items()
    }
    root_level = {
        root: float(np.mean(values)) for root, values in per_root.items()
    }
    stats = {
        "units_total": len(units),
        "fallback_not_optimal_units": len(fallback_not_optimal),
        "fallback_not_optimal_rate_unit": round(
            len(fallback_not_optimal) / len(units), 4,
        ),
        "fallback_not_optimal_tasks": len({u.split("/")[0] for u in fallback_not_optimal}),
        "fallback_not_optimal_roots": len({u.split("/")[0] for u in fallback_not_optimal}),
        "continue_or_requery_substitutable_units": len(substitutable),
        "recoverable_success_rate": _bootstrap_ci(recoverable),
        "fallback_success_rate": _bootstrap_ci(fallback_success),
        "task_level_fallback_not_optimal": {
            "mean": round(float(np.mean(list(task_level.values()))), 4) if task_level else None,
            "ci95": _bootstrap_ci(list(task_level.values()))["ci95"],
            "tasks": len(task_level),
            "positive_tasks": int(sum(1 for v in task_level.values() if v > 0)),
        },
        "root_level_fallback_not_optimal": {
            "mean": round(float(np.mean(list(root_level.values()))), 4) if root_level else None,
            "roots": len(root_level),
            "positive_roots": int(sum(1 for v in root_level.values() if v > 0)),
        },
    }

    # ---- 2. oracle Pareto upper bound under real costs ----------------------
    # Cost components per operator (per unit): normalized incremental cost and
    # incremental wall time (vs continue, may be None).
    def _costs(op: str, unit: dict[str, dict[str, Any]]) -> dict[str, Any]:
        entry = unit.get(op)
        if entry is None:
            return {"success": 0.0, "cost": 0.0, "wall": 0.0, "present": False}
        return {
            "success": float(entry.get("success") or 0.0),
            "cost": float(entry.get("incremental_cost") or 0.0),
            "wall": float(entry.get("incremental_wall_s_vs_continue") or 0.0),
            "present": True,
        }

    report: dict[str, Any] = {
        "schema_version": "rase-vnext-deployment-feasibility/v1",
        "ledger_sha256": args.ledger.read_text() and __import__(
            "hashlib"
        ).sha256(args.ledger.read_bytes()).hexdigest(),
        "scenarios_sha256": __import__(
            "hashlib"
        ).sha256(args.scenarios.read_bytes()).hexdigest(),
        "unit_stats": stats,
        "oracle": {},
    }

    # S0: current cheap fallback (normalized costs only)
    def _utility(unit: dict[str, dict[str, Any]], op: str, lam: float) -> float:
        info = _costs(op, unit)
        return info["success"] - lam * info["cost"]

    for lam in (0.05, 0.1, 0.2, 0.5, 1.0):
        oracle_values = [
            max(_utility(unit, op, lam) for op in unit)
            for unit in units.values()
        ]
        fb_values = [
            _utility(unit, "fallback.persistent", lam) for unit in units.values()
        ]
        report["oracle"].setdefault("S0_current_cheap_fallback", {})[str(lam)] = {
            "oracle_mean": round(float(np.mean(oracle_values)), 5),
            "fallback_mean": round(float(np.mean(fb_values)), 5),
            "oracle_beats_fallback": bool(
                float(np.mean(oracle_values)) > float(np.mean(fb_values))
            ),
        }

    # S1: latency-budgeted — oracle with hard wall budget T; beyond T the
    # candidate is treated as continue (downgrade). Use incremental wall time.
    def _s1_utility(unit: dict[str, dict[str, Any]], op: str, budget: float) -> float:
        info = _costs(op, unit)
        if not info["present"]:
            return 0.0
        if info["wall"] > budget:
            cont = _costs("continue.source", unit)
            return cont["success"]
        return info["success"]

    for budget in budgets:
        oracle_values = [
            max(_s1_utility(unit, op, budget) for op in unit)
            for unit in units.values()
        ]
        fb_values = [
            _s1_utility(unit, "fallback.persistent", budget) for unit in units.values()
        ]
        report["oracle"].setdefault("S1_latency_budgeted", {})[str(budget)] = {
            "oracle_mean_success": round(float(np.mean(oracle_values)), 5),
            "fallback_mean_success": round(float(np.mean(fb_values)), 5),
            "oracle_beats_fallback": bool(
                float(np.mean(oracle_values)) > float(np.mean(fb_values))
            ),
        }

    # S2: fallback-constrained — quota q: q fraction of units get fallback
    # (deterministic hash selection, frozen seed), rest continue; oracle picks
    # the best available operator under the quota.
    def _hash_choice(unit: tuple[str, int], quota: float) -> bool:
        token = f"{unit[0]}/{unit[1]}/{quota}".encode()
        digest = __import__("hashlib").sha256(token).digest()
        return int.from_bytes(digest[:4], "big") / (2 ** 32) < quota

    for quota in quotas:
        oracle_values: list[float] = []
        fb_values: list[float] = []
        for unit_key, unit in units.items():
            if "fallback.persistent" in unit and _hash_choice(unit_key, quota):
                available = unit
            else:
                available = {op: entry for op, entry in unit.items() if op != "fallback.persistent"}
                if not available:
                    available = unit
            oracle_values.append(max(
                float(entry.get("success") or 0.0) for entry in available.values()
            ))
            cont = _costs("continue.source", unit)
            fb = _costs("fallback.persistent", unit)
            fb_values.append(fb["success"] if "fallback.persistent" in available else cont["success"])
        report["oracle"].setdefault("S2_fallback_constrained", {})[str(quota)] = {
            "oracle_mean_success": round(float(np.mean(oracle_values)), 5),
            "quota_fallback_mean_success": round(float(np.mean(fb_values)), 5),
            "oracle_beats_quota_fallback": bool(
                float(np.mean(oracle_values)) > float(np.mean(fb_values))
            ),
        }

    # Stop-rule summary
    report["stop_rule_summary"] = {
        "S0": "frozen comparison (expected FAIL); oracle vs fallback: %s" % (
            report["oracle"]["S0_current_cheap_fallback"]["0.1"]["oracle_beats_fallback"]
        ),
        "S1": {
            budget: report["oracle"]["S1_latency_budgeted"][str(budget)]["oracle_beats_fallback"]
            for budget in budgets
        },
        "S2": {
            quota: report["oracle"]["S2_fallback_constrained"][str(quota)]["oracle_beats_quota_fallback"]
            for quota in quotas
        },
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
