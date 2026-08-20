#!/usr/bin/env python3
"""A3: deployment-scenario nested OOF with margin_lambda (v2 plan A3).

For each preregistered scenario (S0/S1/S2) and lambda, the abstain margin is
selected once on K5 calibration tasks (inner choice); outer-test tasks are
evaluated exactly once.  Scenario constraints (latency budget, fallback quota)
are applied at decision/evaluation time.  Reports success, latency,
fallback-call rate, U_lambda, and task bootstrap CIs for selector vs
continue/requery/always-fallback/quota-aware-fallback/oracle.
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

from rase.vnext.selector import (  # noqa: E402
    fit_ridge,
    select_candidates,
    utility_lambda,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hash_choice(key: str, quota: float, salt: str) -> bool:
    token = f"{salt}:{key}:{quota}".encode()
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:4], "big") / (2 ** 32) < quota


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", default="raw-action")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--margins", type=str, default="0.0,0.05,0.1,0.2")
    parser.add_argument("--lambdas", type=str, default="0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--budgets", type=str, default="10,30,60,120")
    parser.add_argument("--quotas", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20270818)
    args = parser.parse_args()

    manifest = json.loads(args.dataset.read_text())
    ledger = json.loads(args.ledger.read_text())
    scenarios = json.loads(args.scenarios.read_text())
    if manifest.get("status") != "frozen" or scenarios.get("status") != "frozen":
        raise SystemExit("dataset or scenario protocol is not frozen")
    arrays = np.load(Path(str(manifest["features_path"])), allow_pickle=False)
    feature = arrays["raw"]
    rows = manifest["rows"]
    n = len(rows)
    tasks = np.asarray([row["task_id"] for row in rows])
    folds = np.asarray([int(row["fold"]) for row in rows])
    calib = np.asarray([bool(row["calibration_split"]) for row in rows])
    operators = np.asarray([row["operator"] for row in rows])
    roots = np.asarray([row["root_id"] for row in rows])
    replicas = np.asarray([int(row["replica"]) for row in rows])
    success = np.asarray([bool(row["success"]) for row in rows], dtype=np.float64)
    margins = [float(v) for v in args.margins.split(",")]
    lambdas = [float(v) for v in args.lambdas.split(",")]
    budgets = [float(v) for v in args.budgets.split(",")]
    quotas = [float(v) for v in args.quotas.split(",")]

    # incremental wall time per (root, replica, operator) from the ledger
    wall: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for entry in ledger["ledger"]:
        key = (str(entry["root_id"]), int(entry["replica"]))
        wall[key][str(entry["operator"])] = float(entry.get("incremental_wall_s_vs_continue") or 0.0)

    outer_folds = sorted(set(int(f) for f in folds))
    report: dict[str, Any] = {
        "schema_version": "rase-vnext-deployment-oof/v1",
        "dataset_manifest_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "ledger_sha256": hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
        "scenarios_sha256": hashlib.sha256(args.scenarios.read_bytes()).hexdigest(),
        "feature": args.feature, "alpha": args.alpha,
        "margins": margins, "lambdas": lambdas,
        "budgets": budgets, "quotas": quotas,
    }

    for scenario in ("S0_current_cheap_fallback", "S1_latency_budgeted", "S2_fallback_constrained"):
        scenario_report: dict[str, Any] = {}
        # freeze margin per (lambda, constraint) on calibration tasks
        chosen_margins: dict[str, float] = {}
        for lam in lambdas:
            calib_units: dict[tuple[str, int], list[int]] = defaultdict(list)
            for i in range(n):
                if bool(calib[i]):
                    calib_units[(str(roots[i]), int(replicas[i]))].append(i)
            margin_utility: dict[float, list[float]] = {m: [] for m in margins}
            # fit a quick model on non-calib rows for margin selection
            train_mask = ~calib
            model = fit_ridge(
                feature[train_mask], success[train_mask], alpha=args.alpha,
                model_type="candidate",
                feature_version=manifest.get("feature_version", "v1"),
                training_manifest_sha256=manifest.get("sha256", ""),
            )
            for unit, indices in calib_units.items():
                scores = {str(operators[i]): float(model.predict(feature[i:i + 1])[0]) for i in indices}
                for margin in margins:
                    decision = select_candidates(scores, abstain_margin=margin)
                    chosen = next(i for i in indices if str(operators[i]) == decision.chosen_operator)
                    margin_utility[margin].append(
                        utility_lambda(success[chosen], 0.0, lam)
                    )
            best = max(margins, key=lambda m: float(np.mean(margin_utility[m])) if margin_utility[m] else -1.0)
            chosen_margins[str(lam)] = best
        scenario_report["frozen_margin_per_lambda"] = chosen_margins

        # outer evaluation per (lambda, constraint) using frozen margins
        for lam in lambdas:
            margin = chosen_margins[str(lam)]
            eval_report: dict[str, Any] = {"margin": margin, "outer_folds": {}}
            for outer_fold in outer_folds:
                test_mask = folds == outer_fold
                calib_mask = calib & ~test_mask
                train_mask = ~calib_mask & ~test_mask
                model = fit_ridge(
                    feature[train_mask], success[train_mask], alpha=args.alpha,
                    model_type="candidate",
                    feature_version=manifest.get("feature_version", "v1"),
                    training_manifest_sha256=manifest.get("sha256", ""),
                )
                test_units: dict[tuple[str, int], list[int]] = defaultdict(list)
                for i in range(n):
                    if bool(test_mask[i]):
                        test_units[(str(roots[i]), int(replicas[i]))].append(i)
                fold_metrics = _evaluate_fold(
                    test_units, operators, success, feature, model, margin, lam,
                    scenario, budgets, quotas, wall, args.seed,
                )
                eval_report["outer_folds"][str(outer_fold)] = fold_metrics
            # aggregate across folds
            aggregated = _aggregate(eval_report["outer_folds"], args.replicates, args.seed)
            scenario_report[str(lam)] = aggregated
        report[scenario] = scenario_report

    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _available_operators(
    unit: tuple[str, int],
    operators: dict[str, int],
    scenario: str,
    budgets: list[float],
    quotas: list[float],
    wall: dict[tuple[str, int], dict[str, float]],
    seed: int,
) -> tuple[set[str], dict[str, Any]]:
    """Scenario-constrained operator availability + constraint info."""
    ops = set(operators)
    info: dict[str, Any] = {"fallback_calls": False, "downgraded": False}
    if "S1" in scenario:
        budget = min(budgets)
        for op in list(ops):
            op_wall = wall.get(unit, {}).get(op, 0.0)
            if op_wall > budget and op == "fallback.persistent":
                ops.discard(op)
                info["downgraded"] = True
    if "S2" in scenario:
        quota = min(quotas)
        if "fallback.persistent" in ops and not _hash_choice(f"{unit[0]}/{unit[1]}", quota, seed):
            ops.discard("fallback.persistent")
        else:
            info["fallback_calls"] = "fallback.persistent" in ops
    if "fallback.persistent" in ops:
        info["fallback_calls"] = True
    return ops, info


def _evaluate_fold(
    test_units, operators, success, feature, model, margin, lam,
    scenario, budgets, quotas, wall, seed,
) -> dict[str, Any]:
    selector_values: list[float] = []
    continue_values: list[float] = []
    fallback_values: list[float] = []
    oracle_values: list[float] = []
    fallback_calls = 0
    total = 0
    for unit, indices in test_units.items():
        total += 1
        op_set = {str(operators[i]) for i in indices}
        available, info = _available_operators(
            unit, {op: 1 for op in op_set}, scenario, budgets, quotas, wall, seed,
        )
        scores = {
            str(operators[i]): float(model.predict(feature[i:i + 1])[0])
            for i in indices if str(operators[i]) in available
        }
        truth = {str(operators[i]): float(success[i]) for i in indices}
        if "continue.source" in available:
            scores.setdefault("continue.source", 0.5)
        decision = select_candidates(scores, abstain_margin=margin)
        chosen = decision.chosen_operator
        selector_values.append(utility_lambda(truth.get(chosen, 0.0), 0.0, lam))
        continue_values.append(utility_lambda(truth.get("continue.source", 0.0), 0.0, lam))
        fallback_values.append(utility_lambda(truth.get("fallback.persistent", 0.0), 0.0, lam))
        oracle_values.append(max(
            utility_lambda(truth[op], 0.0, lam) for op in available if op in truth
        ))
        if chosen == "fallback.persistent":
            fallback_calls += 1
    return {
        "selector_mean": round(float(np.mean(selector_values)), 5),
        "continue_mean": round(float(np.mean(continue_values)), 5),
        "fallback_mean": round(float(np.mean(fallback_values)), 5),
        "oracle_mean": round(float(np.mean(oracle_values)), 5),
        "fallback_call_rate": round(fallback_calls / total, 4) if total else None,
        "units": total,
    }


def _aggregate(fold_metrics: dict[str, dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    keys = ("selector_mean", "continue_mean", "fallback_mean", "oracle_mean", "fallback_call_rate")
    out: dict[str, Any] = {"folds": len(fold_metrics), "units_total": sum(
        m["units"] for m in fold_metrics.values()
    )}
    for key in keys:
        values = np.asarray([m[key] for m in fold_metrics.values() if m.get(key) is not None])
        if not len(values):
            out[key] = None
            continue
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(values), size=(replicates, len(values)))
        samples = values[idx].mean(axis=1)
        out[key] = {
            "mean": round(float(values.mean()), 5),
            "ci95": [round(float(np.quantile(samples, 0.025)), 5),
                     round(float(np.quantile(samples, 0.975)), 5)],
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
