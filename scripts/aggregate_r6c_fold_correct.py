#!/usr/bin/env python3
"""R6-C.0: fold-correct aggregation of the R6-C candidate-arm OOF reports.

The original `train_r6c_candidate_arm_student.py` summary used an
*avg-threshold* aggregation: it took the mean of the per-fold thresholds and
re-applied it to every pooled prediction (lines 393-407).  Different outer
folds legitimately have different train-derived thresholds; averaging them and
re-evaluating all OOF data is methodologically wrong.

This script computes the official fold-correct aggregation:

- Per fold, the train-derived threshold is kept exactly as selected on the
  calibration split (already stored in `fold_reports[*].threshold`).
- Per fold, the held-out decision is replayed on that fold's validation rows
  using that fold's own threshold.
- Episode-level counts are pooled across folds and across the 5 training seeds.
- The avg-threshold aggregation is reported separately as a sensitivity
  analysis (labelled as the methodologically-weak summary).

Additional metrics reported (point estimates + task-cluster bootstrap
intervals, NOT hard gates):

- success / success gap vs ENTER_PERSISTENT_OFT@t0
- teacher-step savings
- original-protocol false-continue (denominator = baseline successes)
- conditional missed-rescue rate
    = not-entered and source failed and t0 OFT would have succeeded
      / source failed and t0 OFT would have succeeded
- absolute paired harm
    = not-entered and source failed and t0 OFT would have succeeded
      / all groups
- rescue count and intervention burden (entered / episodes)

The R6-C FAIL 0/5 verdict is NOT revisited; this report only freezes the
numbers with the correct methodology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DWELL = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_decisions(predictions: list[dict], arm_teacher: np.ndarray | None = None,
                    arm_success: np.ndarray | None = None) -> list[dict]:
    """Replay the two-boundary dwell controller per group using fold thresholds.

    Each prediction row carries `group_id`, `elapsed_source_steps`,
    `source_lcb`, `threshold` (the threshold of *that row's fold*), and
    `fold`.  A group is a trajectory; its rows are its boundaries.

    ``arm_teacher`` / ``arm_success`` are the dataset arrays (indexed by the
    row's ``index`` field) used to recover the *true* persistent teacher steps.
    """
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_group[str(row["group_id"])].append(row)

    decisions: list[dict] = []
    for group, members in by_group.items():
        members.sort(key=lambda r: int(r["elapsed_source_steps"]))
        threshold = float(members[0]["threshold"])  # same for all rows of a fold
        lcbs = np.asarray([float(r["source_lcb"]) for r in members], dtype=float)
        source_success = bool(members[0]["source_success"])
        # persistent success at t=0 (first boundary of the group is elapsed 0)
        persistent0 = bool(members[0]["persistent_success"])
        # two-boundary dwell: enter at the position where 2 consecutive
        # boundaries have lcb < threshold.
        streak = 0
        enter_pos = None
        for position, row in enumerate(members):
            if lcbs[position] < threshold:
                streak += 1
                if streak >= DWELL:
                    enter_pos = position
                    break
            else:
                streak = 0
        entered = enter_pos is not None
        if entered:
            controller_success = bool(members[enter_pos]["persistent_success"])
            controller_teacher = _persistent_teacher(members[enter_pos], arm_teacher, arm_success)
        else:
            controller_success = source_success
            controller_teacher = 0.0
        decisions.append({
            "group_id": group,
            "policy_id": str(members[0]["policy_id"]),
            "task_id": str(members[0]["task_id"]),
            "state_key": str(members[0]["state_key"]),
            "fold": int(members[0]["fold"]),
            "threshold": threshold,
            "n_boundaries": len(members),
            "entered": entered,
            "enter_position": enter_pos,
            "source_success": source_success,
            "persistent_success_at_0": persistent0,
            "controller_success": controller_success,
            "baseline_success": persistent0,
            "baseline_teacher_steps": _persistent_teacher(members[0], arm_teacher, arm_success),
            "false_continue": (not entered) and (not source_success) and persistent0,
            "conditional_missed_rescue": (not entered) and (not source_success) and persistent0,
            "absolute_paired_harm": (not entered) and (not source_success) and persistent0,
            "rescue": entered and bool(members[enter_pos]["persistent_success"]),
            "controller_teacher_steps": controller_teacher,
        })
    return decisions


def _persistent_teacher(row: dict, arm_teacher: np.ndarray | None,
                        arm_success: np.ndarray | None) -> float:
    """Executed persistent-OFT calls, including unsuccessful rollouts."""
    if arm_teacher is not None and "index" in row:
        index = int(row["index"])
        return float(arm_teacher[index, 1])
    return float(row.get("arm_cost_q50", 0.0))


def aggregate_counts(validation_metrics_list: list[dict]) -> dict[str, float]:
    """Pool per-fold validation_metrics episode counts (fold-correct)."""
    keys = ["episodes", "entered", "successes", "baseline_successes",
            "false_continue", "teacher_steps", "baseline_teacher_steps"]
    total = {key: sum(float(vm.get(key, 0.0)) for vm in validation_metrics_list) for key in keys}
    episodes = max(1.0, total["episodes"])
    base_succ = max(1.0, total["baseline_successes"])
    return {
        "episodes": total["episodes"],
        "entered": total["entered"],
        "successes": total["successes"],
        "baseline_successes": total["baseline_successes"],
        "success_gap": (total["successes"] - total["baseline_successes"]) / episodes,
        "false_continue": total["false_continue"],
        "false_continue_rate": total["false_continue"] / base_succ,
        "teacher_steps": total["teacher_steps"],
        "baseline_teacher_steps": total["baseline_teacher_steps"],
        "savings": 1.0 - total["teacher_steps"] / max(1.0, total["baseline_teacher_steps"]),
    }


def pool_decision_metrics(decisions: list[dict]) -> dict[str, float]:
    n = max(1, len(decisions))
    base_succ = max(1, sum(d["baseline_success"] for d in decisions))
    base_steps = sum(d["baseline_teacher_steps"] for d in decisions)
    ctrl_steps = sum(d["controller_teacher_steps"] for d in decisions)
    return {
        "n_groups": len(decisions),
        "entered": float(sum(d["entered"] for d in decisions)),
        "intervention_burden": float(sum(d["entered"] for d in decisions)) / n,
        "rescue": float(sum(d["rescue"] for d in decisions)),
        "rescue_rate": float(sum(d["rescue"] for d in decisions)) / n,
        "controller_success": float(sum(d["controller_success"] for d in decisions)),
        "baseline_success": float(sum(d["baseline_success"] for d in decisions)),
        "success_gap": (float(sum(d["controller_success"] for d in decisions))
                        - float(sum(d["baseline_success"] for d in decisions))) / n,
        "false_continue": float(sum(d["false_continue"] for d in decisions)),
        "false_continue_rate": float(sum(d["false_continue"] for d in decisions)) / base_succ,
        "conditional_missed_rescue_rate": (
            float(sum(d["conditional_missed_rescue"] for d in decisions))
            / max(1, float(sum((not d["source_success"]) and d["baseline_success"] for d in decisions)))),
        "absolute_paired_harm_rate": float(sum(d["absolute_paired_harm"] for d in decisions)) / n,
        "missed_rescue_denominator": float(
            sum((not d["source_success"]) and d["baseline_success"] for d in decisions)),
        "teacher_savings": 1.0 - ctrl_steps / max(1.0, base_steps),
    }


def task_cluster_bootstrap(decisions: list[dict], rng_seed: int,
                           n_iter: int = 2000) -> dict[str, list[float]]:
    """Bootstrap by resampling whole tasks (clusters) with replacement."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for d in decisions:
        by_task[d["task_id"]].append(d)
    tasks = list(by_task)
    rng = random.Random(rng_seed)
    series: dict[str, list[float]] = {
        "success_gap": [], "false_continue_rate": [],
        "conditional_missed_rescue_rate": [], "absolute_paired_harm_rate": [],
        "teacher_savings": [],
    }
    for _ in range(n_iter):
        sampled: list[dict] = []
        for _ in range(len(tasks)):
            sampled.extend(by_task[rng.choice(tasks)])
        m = pool_decision_metrics(sampled)
        series["success_gap"].append(m["success_gap"])
        series["false_continue_rate"].append(m["false_continue_rate"])
        series["conditional_missed_rescue_rate"].append(m["conditional_missed_rescue_rate"])
        series["absolute_paired_harm_rate"].append(m["absolute_paired_harm_rate"])
        series["teacher_savings"].append(m["teacher_savings"])
    out = {}
    for key, values in series.items():
        values = np.asarray(values, dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
    return out


def avg_threshold_sensitivity(predictions: list[dict], arm_teacher: np.ndarray | None = None,
                              arm_success: np.ndarray | None = None) -> dict[str, float]:
    """Replicate the original (methodologically-weak) avg-threshold summary.

    This is exactly what the old trainer did: take the mean of per-fold
    thresholds, re-run the controller on every pooled prediction.  Reported
    only as a sensitivity analysis.
    """
    rows = sorted(predictions, key=lambda r: (str(r["group_id"]), int(r["elapsed_source_steps"])))
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)
    avg_thr = float(np.mean([float(r["threshold"]) for r in predictions]))
    n = max(1, len(by_group))
    base_succ = 0
    entered = 0; succ = 0; base = 0; fc = 0
    base_steps = 0.0; ctrl_steps = 0.0
    for group, members in by_group.items():
        members.sort(key=lambda r: int(r["elapsed_source_steps"]))
        lcbs = np.asarray([float(r["source_lcb"]) for r in members])
        src = bool(members[0]["source_success"])
        p0 = bool(members[0]["persistent_success"])
        base += int(p0)
        base_steps += _persistent_teacher(members[0], arm_teacher, arm_success)
        streak = 0; enter_pos = None
        for position in range(len(members)):
            if lcbs[position] < avg_thr:
                streak += 1
                if streak >= DWELL:
                    enter_pos = position; break
            else:
                streak = 0
        if enter_pos is not None:
            entered += 1
            succ += int(bool(members[enter_pos]["persistent_success"]))
            ctrl_steps += _persistent_teacher(members[enter_pos], arm_teacher, arm_success)
        else:
            succ += int(src)
            if (not src) and p0:
                fc += 1
    return {
        "avg_threshold": float(avg_thr),
        "episodes": float(len(by_group)),
        "entered": float(entered),
        "success_gap": (succ - base) / n,
        "false_continue_rate": fc / max(1, base_succ or base),
        "savings": 1.0 - ctrl_steps / max(1.0, base_steps),
        "sensitivity_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-root", type=Path, required=True,
                        help="runs/pre_c0_r6/r6c_candidate_arm_oof_v1")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--collector-report", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=None,
                        help="report modes; default all in oof-root/seed_*/*.json")
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    args = parser.parse_args()

    # Hash inventory
    hashes = {
        "collector_sha256": json.loads(args.collector_report.read_text()).get("collector_sha256"),
        "protocol_sha256": sha256(args.protocol),
        "initial_keys_sha256": json.loads(
            (ROOT / "runs/rase_ui_phase1a_replacement48_initial_keys_v2.json").read_text()
        ).get("state_keys_sha256"),
        "dataset_sha256": sha256(args.dataset),
        "dataset_report_sha256": sha256(args.dataset_report),
        "exclusions_sha256": sha256(args.exclusions),
    }

    # Real persistent-teacher-step labels for decision replay
    raw = np.load(args.dataset)
    arm_teacher = raw["arm_teacher_steps"]
    arm_success = raw["arm_success"]

    # Collect per-mode reports across seeds
    import glob as _glob
    candidates = sorted(_glob.glob(str(args.oof_root / "seed_*" / "*.json")))
    mode_reports: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for path_string in candidates:
        path = Path(path_string)
        result = json.loads(path.read_text())
        mode = str(result["mode"])
        if args.modes and mode not in args.modes:
            continue
        mode_reports[mode].append((int(result["seed"]), result))

    out: dict = {
        "schema_version": "rase-r6c-fold-correct-final/v1",
        "status": "complete",
        "scientific_scope": ("fold-correct official aggregation of the R6-C no-WM OOF; "
                             "avg-threshold reported only as sensitivity analysis; "
                             "R6-C gate verdict is NOT revisited (kept FAIL 0/5)"),
        "verdict_frozen": {"r6c_stage_gate": "FAIL", "passing_seeds_per_vla": {"pi05_libero": 0, "pi0fast_libero": 0}},
        "hash_inventory": hashes,
        "modes": {},
    }
    for mode in sorted(mode_reports):
        per_seed = sorted(mode_reports[mode], key=lambda item: item[0])
        seed_results = []
        for seed, result in per_seed:
            fold_vms = [fr["validation_metrics"] for fr in result["fold_reports"]]
            fold_correct = aggregate_counts(fold_vms)
            decisions = group_decisions(result["predictions"], arm_teacher, arm_success)
            decision_metrics = pool_decision_metrics(decisions)
            bootstrap = task_cluster_bootstrap(decisions, args.bootstrap_seed + seed)
            sensitivity = avg_threshold_sensitivity(result["predictions"], arm_teacher, arm_success)
            seed_results.append({
                "seed": seed,
                "target_policy": result.get("target_policy"),
                "source_policy": result.get("source_policy"),
                "fold_correct_metrics": fold_correct,
                "decision_metrics": decision_metrics,
                "task_cluster_bootstrap_95": bootstrap,
                "avg_threshold_sensitivity": sensitivity,
            })
        # Aggregate across seeds: pool fold-correct counts
        pooled_counts = {
            k: sum(float(sr["fold_correct_metrics"][k]) for sr in seed_results)
            for k in ["episodes", "entered", "successes", "baseline_successes",
                      "false_continue", "teacher_steps", "baseline_teacher_steps"]
        }
        eps = max(1.0, pooled_counts["episodes"])
        bsucc = max(1.0, pooled_counts["baseline_successes"])
        pooled = {
            "episodes": pooled_counts["episodes"],
            "entered": pooled_counts["entered"],
            "successes": pooled_counts["successes"],
            "baseline_successes": pooled_counts["baseline_successes"],
            "success_gap": (pooled_counts["successes"] - pooled_counts["baseline_successes"]) / eps,
            "false_continue": pooled_counts["false_continue"],
            "false_continue_rate": pooled_counts["false_continue"] / bsucc,
            "savings": 1.0 - pooled_counts["teacher_steps"] / max(1.0, pooled_counts["baseline_teacher_steps"]),
            "intervention_burden": pooled_counts["entered"] / eps,
        }
        out["modes"][mode] = {
            "n_seeds": len(seed_results),
            "pooled_fold_correct_metrics": pooled,
            "seed_results": seed_results,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": out["status"],
        "hash_inventory": out["hash_inventory"],
        "modes": {m: out["modes"][m]["pooled_fold_correct_metrics"] for m in out["modes"]},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
