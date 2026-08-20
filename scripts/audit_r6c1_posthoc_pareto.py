#!/usr/bin/env python3
"""Post-hoc R6-C.1 controller Pareto diagnosis on frozen OOF predictions.

This script is deliberately diagnostic.  It averages the five already-OOF
prediction sets, compares ensemble mean versus LCB, decomposes source-risk and
fallback-recoverability gates, corrects absolute paired harm, and performs an
outer-fold cross-fit threshold sensitivity.  Thresholds selected here have
seen development labels and MUST NOT be reported as a new formal R6 result.
Their only purpose is to choose a pre-registered R7 mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    p = int(labels.sum())
    n = len(labels) - p
    if p == 0 or n == 0:
        return float("nan")
    return float((ranks(scores)[labels].sum() - p * (p + 1) / 2) / (p * n))


def threshold_grid(values: np.ndarray, *, condition: str) -> list[float]:
    finite = values[np.isfinite(values)]
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, 21)).tolist()
    if condition == "less":
        return sorted(set([-math.inf, *quantiles, math.inf]))
    return sorted(set([-math.inf, *quantiles, math.inf]))


def metric(records: list[dict]) -> dict:
    n = max(1, len(records))
    baseline_success = sum(r["baseline_success"] for r in records)
    success = sum(r["success"] for r in records)
    false_continue = sum(r["false_continue"] for r in records)
    harm = sum(r["baseline_success"] and not r["success"] for r in records)
    fallback_harm = sum(r["source_success"] and r["entered"] and not r["success"]
                        for r in records)
    late_entry_harm = sum(r["baseline_success"] and not r["success"]
                          and (r["enter_elapsed"] or 0) > 0 for r in records)
    baseline_steps = sum(r["baseline_teacher"] for r in records)
    teacher_steps = sum(r["teacher"] for r in records)
    return {
        "episodes": len(records), "successes": success,
        "baseline_successes": baseline_success,
        "success_gap": (success - baseline_success) / n,
        "false_continue": false_continue,
        "false_continue_rate": false_continue / max(1, baseline_success),
        "absolute_paired_harm": harm / n,
        "fallback_harm": fallback_harm / n,
        "late_entry_harm": late_entry_harm / n,
        "teacher_steps": teacher_steps, "baseline_teacher_steps": baseline_steps,
        "savings": 1.0 - teacher_steps / max(1.0, baseline_steps),
    }


def gate(metrics: dict, suite_metrics: dict[str, dict], *, require_savings: bool) -> bool:
    ok = (metrics["success_gap"] >= -0.05
          and metrics["false_continue_rate"] <= 0.05
          and metrics["absolute_paired_harm"] <= 0.05
          and all(value["success_gap"] >= -0.05
                  and value["absolute_paired_harm"] <= 0.05
                  for value in suite_metrics.values()))
    return ok and (not require_savings or metrics["savings"] >= 0.20)


def evaluate(groups: list[dict], source_key: str, arm_key: str | None,
             risk_threshold: float, arm_threshold: float,
             *, t0_only: bool = False) -> tuple[dict, dict, list[dict]]:
    records = []
    for group in groups:
        rows = group["rows"]
        source_success = bool(rows[0]["source_success"] > 0.5)
        baseline_success = bool(rows[0]["persistent_success"] > 0.5)
        enter = None
        for row in rows:
            if t0_only and row["elapsed"] != 0:
                continue
            risky = float(row[source_key]) < risk_threshold
            arm_ok = True if arm_key is None else float(row[arm_key]) > arm_threshold
            if risky and arm_ok:
                enter = row
                break
            if t0_only:
                break
        if enter is None:
            success = source_success
            teacher = 0.0
        else:
            success = bool(enter["persistent_success"] > 0.5)
            teacher = float(enter["teacher_steps"])
        records.append({
            "group_id": group["group_id"], "task_id": group["task_id"],
            "suite": group["suite"], "fold": group["fold"],
            "source_success": source_success, "baseline_success": baseline_success,
            "success": success, "entered": enter is not None,
            "enter_elapsed": None if enter is None else enter["elapsed"],
            "teacher": teacher, "baseline_teacher": rows[0]["teacher_steps"],
            "false_continue": (enter is None and not source_success and baseline_success),
        })
    overall = metric(records)
    by_suite = {
        suite: metric([row for row in records if row["suite"] == suite])
        for suite in sorted({row["suite"] for row in records})
    }
    return overall, by_suite, records


def best_rule(groups: list[dict], source_key: str, arm_key: str | None,
              *, t0_only: bool, require_savings: bool) -> dict:
    rows = [row for group in groups for row in group["rows"]
            if not t0_only or row["elapsed"] == 0]
    risk_grid = threshold_grid(np.asarray([row[source_key] for row in rows]), condition="less")
    arm_grid = ([-math.inf] if arm_key is None else
                threshold_grid(np.asarray([row[arm_key] for row in rows]), condition="greater"))
    best = None
    for risk_threshold in risk_grid:
        for arm_threshold in arm_grid:
            overall, suites, records = evaluate(
                groups, source_key, arm_key, risk_threshold, arm_threshold,
                t0_only=t0_only,
            )
            if not gate(overall, suites, require_savings=require_savings):
                continue
            rank = (overall["savings"], overall["success_gap"],
                    -overall["absolute_paired_harm"])
            if best is None or rank > best[0]:
                best = (rank, risk_threshold, arm_threshold, overall, suites, records)
    if best is None:
        return {"found": False}
    return {
        "found": True, "risk_threshold": best[1], "arm_threshold": best[2],
        "metrics": best[3], "metrics_by_suite": best[4], "records": best[5],
    }


def aggregate_record_metrics(records: list[dict]) -> dict:
    overall = metric(records)
    suites = {suite: metric([row for row in records if row["suite"] == suite])
              for suite in sorted({row["suite"] for row in records})}
    return {
        "metrics": overall, "metrics_by_suite": suites,
        "formal_gate_passed": gate(overall, suites, require_savings=True),
    }


def corrected_original_records(report: dict) -> list[dict]:
    """Normalize legacy controller records and recompute harm from outcomes.

    The original trainer marked ``paired_harm`` only for no-entry misses.  A
    controller can also enter at t8/t16 and lose a trajectory that the t0
    persistent baseline would have rescued.  Downstream metrics must therefore
    derive harm from paired outcomes, never trust the legacy boolean field.
    """
    records = []
    for row in report["trajectory_records"]:
        records.append({
            "group_id": str(row["group_id"]),
            "task_id": str(row["task_id"]),
            "suite": str(row["suite"]),
            "source_success": bool(row["source_success"]),
            "baseline_success": bool(row["baseline_success"]),
            "success": bool(row["controller_success"]),
            "entered": bool(row["entered_persistent"]),
            "enter_elapsed": row.get("enter_elapsed_source_steps"),
            "teacher": float(row["controller_teacher_steps"]),
            "baseline_teacher": float(row["baseline_teacher_steps"]),
            "false_continue": bool(row["false_continue"]),
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = np.load(args.dataset)
    data = {key: raw[key] for key in raw.files}
    reports = [json.loads(path.read_text()) for path in args.report]
    dataset_hash = sha256(args.dataset)
    if any(report.get("dataset_sha256") != dataset_hash for report in reports):
        raise ValueError("report/dataset hash mismatch")

    by_index: dict[int, list[dict]] = defaultdict(list)
    for report in reports:
        for row in report["predictions"]:
            by_index[int(row["index"])].append(row)
    groups_by_id: dict[str, list[dict]] = defaultdict(list)
    for index, predictions in sorted(by_index.items()):
        if len(predictions) != len(reports):
            raise ValueError(f"index {index} missing seed prediction")
        first = predictions[0]
        row = {
            "index": index, "elapsed": int(data["elapsed_source_steps"][index]),
            "source_success": float(data["arm_success"][index, 0]),
            "persistent_success": float(data["arm_success"][index, 1]),
            "teacher_steps": float(data["arm_teacher_steps"][index, 1]),
            "source_mean": float(np.mean([value["source_mean"] for value in predictions])),
            "source_lcb": float(np.mean([value["source_lcb"] for value in predictions])),
            "advantage_mean": float(np.mean([value["advantage_mean"] for value in predictions])),
            "advantage_lcb": float(np.mean([value["advantage_lcb"] for value in predictions])),
            "persistent_mean": float(np.mean([
                value["persistent_success_mean"] for value in predictions
            ])),
        }
        groups_by_id[str(first["group_id"])].append(row)
    groups = []
    for group_id, rows in sorted(groups_by_id.items()):
        rows.sort(key=lambda row: row["elapsed"])
        prediction = next(value for value in reports[0]["predictions"]
                          if value["group_id"] == group_id)
        folds = {int(value["fold"]) for report in reports for value in report["predictions"]
                 if value["group_id"] == group_id}
        if len(folds) != 1:
            raise ValueError(f"group crosses folds: {group_id}: {folds}")
        groups.append({
            "group_id": group_id, "task_id": str(prediction["task_id"]),
            "suite": str(prediction["suite"]), "fold": folds.pop(), "rows": rows,
        })

    families = {
        "lcb_joint_early": ("source_lcb", "advantage_lcb", False),
        "mean_joint_early": ("source_mean", "advantage_mean", False),
        "lcb_risk_only_early": ("source_lcb", None, False),
        "mean_risk_only_early": ("source_mean", None, False),
        "mean_persistent_guard_early": ("source_mean", "persistent_mean", False),
        "mean_joint_t0": ("source_mean", "advantage_mean", True),
        "mean_risk_only_t0": ("source_mean", None, True),
        "mean_persistent_guard_t0": ("source_mean", "persistent_mean", True),
    }
    diagnostics = {}
    for name, (source_key, arm_key, t0_only) in families.items():
        safe = best_rule(groups, source_key, arm_key, t0_only=t0_only,
                         require_savings=False)
        formal = best_rule(groups, source_key, arm_key, t0_only=t0_only,
                           require_savings=True)
        # Cross-fit: select the maximum-savings safe threshold on four OOF
        # folds, then apply it untouched to the fifth fold.
        crossfit_records = []
        fold_rules = []
        for fold in sorted({group["fold"] for group in groups}):
            train = [group for group in groups if group["fold"] != fold]
            heldout = [group for group in groups if group["fold"] == fold]
            selected = best_rule(train, source_key, arm_key, t0_only=t0_only,
                                 require_savings=False)
            if not selected["found"]:
                raise RuntimeError("always-enter rule should be safety-feasible")
            overall, suites, records = evaluate(
                heldout, source_key, arm_key, selected["risk_threshold"],
                selected["arm_threshold"], t0_only=t0_only,
            )
            crossfit_records.extend(records)
            fold_rules.append({
                "fold": fold, "risk_threshold": selected["risk_threshold"],
                "arm_threshold": selected["arm_threshold"],
                "train_metrics": selected["metrics"], "heldout_metrics": overall,
            })
        diagnostics[name] = {
            "full_oof_best_safety_feasible": {k: v for k, v in safe.items() if k != "records"},
            "full_oof_formal_solution_exists": formal["found"],
            "full_oof_best_formal": ({k: v for k, v in formal.items() if k != "records"}
                                     if formal["found"] else None),
            "crossfit": {**aggregate_record_metrics(crossfit_records),
                         "fold_rules": fold_rules},
        }


    corrected_original = []
    for report_path, report in zip(args.report, reports, strict=True):
        records = corrected_original_records(report)
        audited = aggregate_record_metrics(records)
        legacy_harm = float(report.get("metrics", {}).get("absolute_paired_harm", float("nan")))
        corrected_harm = float(audited["metrics"]["absolute_paired_harm"])
        corrected_original.append({
            "report": str(report_path),
            "seed": int(report["seed"]),
            **audited,
            "legacy_absolute_paired_harm": legacy_harm,
            "legacy_harm_undercount": corrected_harm - legacy_harm,
        })

    # Privileged source-risk oracle: continue every true source success and
    # enter t0 for every true source failure.  It quantifies the value of
    # perfect source-risk without requiring a recoverability oracle.
    oracle_records = []
    for group in groups:
        row = group["rows"][0]
        source_success = bool(row["source_success"] > 0.5)
        baseline_success = bool(row["persistent_success"] > 0.5)
        entered = not source_success
        success = source_success if not entered else baseline_success
        oracle_records.append({
            "group_id": group["group_id"], "task_id": group["task_id"],
            "suite": group["suite"], "fold": group["fold"],
            "source_success": source_success, "baseline_success": baseline_success,
            "success": success, "entered": entered, "enter_elapsed": 0 if entered else None,
            "teacher": row["teacher_steps"] if entered else 0.0,
            "baseline_teacher": row["teacher_steps"], "false_continue": False,
        })
    t0_rows = [group["rows"][0] for group in groups]
    source_failure = np.asarray([row["source_success"] <= 0.5 for row in t0_rows])
    rescue = np.asarray([row["source_success"] <= 0.5
                         and row["persistent_success"] > 0.5 for row in t0_rows])
    ranking = {}
    for score in ("source_mean", "source_lcb", "advantage_mean", "advantage_lcb",
                  "persistent_mean"):
        values = np.asarray([row[score] for row in t0_rows])
        ranking[score] = {
            "source_failure_auc": auc(source_failure, -values)
            if score.startswith("source") else auc(source_failure, values),
            "rescue_benefit_auc": auc(rescue, -values)
            if score.startswith("source") else auc(rescue, values),
        }
    result = {
        "schema_version": "rase-r6c1-posthoc-pareto/v1",
        "status": "complete", "formal_r6_decision_unchanged": True,
        "scientific_scope": "post-hoc mechanism selection for a new preregistered R7 only",
        "dataset": str(args.dataset), "dataset_sha256": dataset_hash,
        "reports": [str(path) for path in args.report],
        "groups": len(groups), "tasks": len({group["task_id"] for group in groups}),
        "ranking_at_t0": ranking,
        "corrected_original_formal_controller": {
            "per_seed": corrected_original,
            "seeds_passing": sum(row["formal_gate_passed"] for row in corrected_original),
            "mean_metrics": {
                key: float(np.mean([row["metrics"][key] for row in corrected_original]))
                for key in ("success_gap", "false_continue_rate",
                            "absolute_paired_harm", "fallback_harm",
                            "late_entry_harm", "savings")
            },
            "note": (
                "Corrected audit counts every t0-baseline-success/controller-failure "
                "pair as harm, including late-entry rescue decay. Formal R6 remains FAIL."
            ),
        },
        "privileged_source_risk_oracle": aggregate_record_metrics(oracle_records),
        "controller_families": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "corrected_original_formal_controller": result["corrected_original_formal_controller"],
        "privileged_source_risk_oracle": result["privileged_source_risk_oracle"]["metrics"],
        "families": {name: {
            "full_oof_formal_solution_exists": value["full_oof_formal_solution_exists"],
            "crossfit_metrics": value["crossfit"]["metrics"],
            "crossfit_formal_gate_passed": value["crossfit"]["formal_gate_passed"],
        } for name, value in diagnostics.items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
