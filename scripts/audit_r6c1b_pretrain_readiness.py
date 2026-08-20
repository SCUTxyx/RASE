#!/usr/bin/env python3
"""R6-C.1B pre-training readiness audit.

This is the mandatory stop between counterfactual collection and R6-C.1C
training.  It answers three different questions without conflating them:

1. Are the replica-adjudicated labels reproducible enough to use?
2. Does the *natural development cohort* contain a model-free early-window
   success/cost Pareto opportunity?
3. Can a deliberately small, deployable, task-held-out linear probe rank the
   takeover advantage, or is the privileged opportunity observationally
   inaccessible?

Enrichment rows may train the probe, but calibration and OOF evaluation use
natural rows only.  All splits are by true task id.  This audit does not train
or approve the full selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_r6c1b_label_support import load_groups  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary_entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and s[order[end]] == s[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + 1 + end)
        index = end
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def instruction_hash(text: str, dim: int = 32) -> np.ndarray:
    values = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "little")
        values[number % dim] += 1.0 if ((number >> 8) & 1) else -1.0
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 0 else values


def recent_history(trace: np.ndarray, elapsed: int, window: int = 4) -> np.ndarray:
    action_dim = int(trace.shape[1])
    result = np.zeros((window, action_dim), dtype=np.float32)
    end = min(max(0, int(elapsed)), len(trace))
    start = max(0, end - window)
    values = trace[start:end]
    if len(values):
        result[-len(values):] = values
    return result.reshape(-1)


def feature_rows(groups: list[dict]) -> list[dict]:
    """Extract lightweight deployable features from each canonical replica."""
    rows: list[dict] = []
    for group in groups:
        metadata = json.loads(Path(group["canonical_path"]).read_text())
        npz_path = Path(metadata["npz"])
        if not npz_path.exists():
            candidate = Path(group["canonical_path"]).with_suffix(".npz")
            npz_path = candidate
        data = np.load(npz_path)
        image = data["image"].astype(np.float32) / 255.0
        proprio = data["proprio"].astype(np.float32)
        action_summary = data["source_action_summary"].astype(np.float32)
        trace = data["source_action_trace"].astype(np.float32)
        metadata_by_elapsed = {
            int(row["elapsed_source_steps"]): position
            for position, row in enumerate(metadata["rows"])
        }
        for elapsed in (0, 8, 16):
            boundary = group["boundaries"].get(elapsed)
            position = metadata_by_elapsed.get(elapsed)
            if boundary is None or position is None:
                continue
            frame = image[position]
            # 2 views x 3 channels x 4 x 4 spatial means, plus channel moments.
            pooled = frame.reshape(2, 3, 4, 24, 4, 24).mean(axis=(3, 5)).reshape(-1)
            moments = np.concatenate([
                frame.mean(axis=(2, 3)).reshape(-1),
                frame.std(axis=(2, 3)).reshape(-1),
            ])
            feature = np.concatenate([
                pooled,
                moments,
                proprio[position],
                action_summary[position],
                recent_history(trace, elapsed),
                np.asarray([elapsed / 490.0], dtype=np.float32),
                instruction_hash(group["instruction"]),
            ]).astype(np.float32)
            persistent_probability = float(boundary["success_probability"])
            rows.append({
                "feature": feature,
                "policy_id": group["policy_id"],
                "task_id": group["task_id"],
                "suite": group["suite"],
                "cohort_role": group["cohort_role"],
                "key": tuple(group["key"]),
                "elapsed": elapsed,
                "source_success": float(group["source_success"]),
                "source_risk": float(not group["source_success"]),
                "persistent_probability": persistent_probability,
                "persistent_majority": float(persistent_probability > 0.5),
                "advantage_probability": float(
                    (not group["source_success"]) * persistent_probability
                ),
                "advantage_majority": float(
                    (not group["source_success"]) and persistent_probability > 0.5
                ),
                "teacher_steps": float(boundary["teacher_steps_median"]),
            })
    return rows


def _strategy_summary(groups: list[dict]) -> dict:
    if not groups:
        return {"n_groups": 0, "passed": False}
    base_success = 0
    base_teacher = 0.0
    source_successes = 0
    oracle_success = 0
    oracle_teacher = 0.0
    arm_usage: Counter[str] = Counter()
    rescue = 0
    harm = 0
    fixed = {elapsed: {"eligible": 0, "successes": 0, "teacher_steps": 0.0}
             for elapsed in (0, 8, 16)}
    for group in groups:
        source = bool(group["source_success"])
        source_successes += int(source)
        b0 = group["boundaries"].get(0)
        if b0 is None:
            continue
        base_success += int(b0["majority_success"])
        base_teacher += float(b0["teacher_steps_median"])
        options: list[tuple[str, bool, float]] = [("source", source, 0.0)]
        any_early_success = False
        for elapsed in (0, 8, 16):
            boundary = group["boundaries"].get(elapsed)
            if boundary is None:
                continue
            success = bool(boundary["majority_success"])
            cost = float(boundary["teacher_steps_median"])
            fixed[elapsed]["eligible"] += 1
            fixed[elapsed]["successes"] += int(success)
            fixed[elapsed]["teacher_steps"] += cost
            options.append((f"t{elapsed}", success, cost))
            any_early_success = any_early_success or success
        rescue += int((not source) and any_early_success)
        harm += int(source and any(not success for name, success, _ in options if name != "source"))
        feasible = [(name, cost) for name, success, cost in options if success]
        if feasible:
            name, cost = min(feasible, key=lambda value: (value[1], value[0]))
            oracle_success += 1
            oracle_teacher += cost
            arm_usage[name] += 1
        else:
            arm_usage["none"] += 1
    n = len(groups)
    base_rate = base_success / n
    oracle_rate = oracle_success / n
    savings = 1.0 - oracle_teacher / max(1.0, base_teacher)
    used_arms = {name: count for name, count in arm_usage.items()
                 if name != "none" and count >= 3}
    gates = {
        "oracle_success_gap_ge_minus_5pp": oracle_rate - base_rate >= -0.05,
        "oracle_teacher_savings_ge_30pct": savings >= 0.30,
        "at_least_two_oracle_arms_used": len(used_arms) >= 2,
        "rescue_and_harm_support_exist": rescue > 0 and harm > 0,
        "four_suites": len({group["suite"] for group in groups}) >= 4,
        "at_least_12_tasks": len({group["task_id"] for group in groups}) >= 12,
    }
    return {
        "n_groups": n,
        "n_tasks": len({group["task_id"] for group in groups}),
        "suites": sorted({group["suite"] for group in groups}),
        "source_success_rate": source_successes / n,
        "persistent_t0_success_rate": base_rate,
        "persistent_t0_total_teacher_steps": base_teacher,
        "cost_aware_oracle_success_rate": oracle_rate,
        "oracle_success_gap_vs_t0": oracle_rate - base_rate,
        "cost_aware_oracle_total_teacher_steps": oracle_teacher,
        "oracle_teacher_savings_vs_t0": savings,
        "oracle_arm_usage": dict(sorted(arm_usage.items())),
        "early_rescue_groups": rescue,
        "fallback_harm_groups": harm,
        "fixed_boundaries": {
            f"t{elapsed}": {
                **values,
                "success_rate": values["successes"] / max(1, values["eligible"]),
            }
            for elapsed, values in fixed.items()
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def label_quality(groups: list[dict], repro: dict) -> dict:
    by_elapsed: dict[int, list[dict]] = defaultdict(list)
    all_costs: list[float] = []
    nonmonotonic = 0
    complete_sequences = 0
    for group in groups:
        sequence = []
        for elapsed in (0, 8, 16):
            boundary = group["boundaries"].get(elapsed)
            if boundary is None:
                continue
            if group["n_replicas"] >= 2:
                by_elapsed[elapsed].append(boundary)
            all_costs.extend(boundary["teacher_steps"])
            sequence.append((elapsed, int(boundary["majority_success"])))
        if len(sequence) == 3:
            complete_sequences += 1
            values = [value for _, value in sequence]
            nonmonotonic += int(any(values[i + 1] > values[i]
                                    for i in range(len(values) - 1)))

    boundary_report = {}
    for elapsed, values in sorted(by_elapsed.items()):
        variable = sum(0 < v["successes"] < v["trials"] for v in values)
        entropies = [binary_entropy(v["success_probability"]) for v in values]
        posterior_variances = []
        concentrations = []
        for value in values:
            alpha = 1.0 + value["successes"]
            beta = 1.0 + value["trials"] - value["successes"]
            concentration = alpha + beta
            concentrations.append(concentration)
            posterior_variances.append(alpha * beta /
                                       (concentration ** 2 * (concentration + 1.0)))
        boundary_report[f"t{elapsed}"] = {
            "replicated_groups": len(values),
            "variable_groups": variable,
            "variability_rate": variable / max(1, len(values)),
            "mean_label_entropy_bits": float(np.mean(entropies)) if entropies else 0.0,
            "mean_beta_posterior_concentration": float(np.mean(concentrations)) if concentrations else 0.0,
            "mean_beta_posterior_variance": float(np.mean(posterior_variances)) if posterior_variances else 0.0,
        }
    total_replicated = sum(v["replicated_groups"] for v in boundary_report.values())
    total_variable = sum(v["variable_groups"] for v in boundary_report.values())
    overall_variability = total_variable / max(1, total_replicated)
    t0 = boundary_report.get("t0", {}).get("variability_rate", 1.0)
    later = [boundary_report.get(name, {}).get("variability_rate", 0.0)
             for name in ("t8", "t16")]
    summary = repro.get("repro_summary", {})
    collected = max(1, int(summary.get("n_collected_triples", 0)))
    gates = {
        "seed_mismatch_zero": int(summary.get("n_seed_mismatch", 0)) == 0,
        "source_success_flip_rate_le_5pct": (
            int(summary.get("n_success_flip", 0)) / collected <= 0.05
        ),
        "overall_boundary_variability_le_10pct": overall_variability <= 0.10,
        "t0_variability_le_10pct": t0 <= 0.10,
        "t0_not_materially_less_stable_than_later": t0 <= max(later, default=0.0) + 0.02,
    }
    costs = np.asarray(all_costs, dtype=float)
    return {
        "repro_summary": summary,
        "boundary_quality": boundary_report,
        "overall_boundary_variability_rate": overall_variability,
        "complete_t0_t8_t16_sequences": complete_sequences,
        "nonmonotonic_sequences": nonmonotonic,
        "nonmonotonic_fraction": nonmonotonic / max(1, complete_sequences),
        "teacher_cost_quantiles": {
            "q10": float(np.quantile(costs, 0.10)) if len(costs) else 0.0,
            "q50": float(np.quantile(costs, 0.50)) if len(costs) else 0.0,
            "q90": float(np.quantile(costs, 0.90)) if len(costs) else 0.0,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def fit_logistic(x: np.ndarray, y: np.ndarray, seed: int) -> dict:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-5] = 1.0
    z = (x - mean) / std
    torch.manual_seed(seed)
    xt = torch.as_tensor(z, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    weight = torch.zeros(z.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.035)
    positive = max(1e-3, float(y.sum()))
    negative = max(1e-3, float(len(y) - y.sum()))
    pos_weight = torch.tensor(min(10.0, negative / positive), dtype=torch.float32)
    for _ in range(350):
        optimizer.zero_grad()
        logits = xt @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, yt, pos_weight=pos_weight)
        loss = loss + 2e-4 * weight.square().mean()
        loss.backward()
        optimizer.step()
    return {
        "mean": mean,
        "std": std,
        "weight": weight.detach().numpy(),
        "bias": float(bias.detach()),
    }


def predict_logistic(model: dict, x: np.ndarray) -> np.ndarray:
    logits = ((x - model["mean"]) / model["std"]) @ model["weight"] + model["bias"]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> dict:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-5] = 1.0
    z = (x - mean) / std
    design = np.concatenate([z, np.ones((len(z), 1), dtype=z.dtype)], axis=1)
    gram = design.T @ design
    regularizer = np.eye(gram.shape[0], dtype=gram.dtype) * ridge
    regularizer[-1, -1] = 0.0
    coefficient = np.linalg.solve(gram + regularizer, design.T @ y)
    return {"mean": mean, "std": std, "coefficient": coefficient}


def predict_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["std"]
    design = np.concatenate([z, np.ones((len(z), 1), dtype=z.dtype)], axis=1)
    return design @ model["coefficient"]


def controller_metrics(rows: list[dict], scores: dict[tuple, float], threshold: float) -> dict:
    t0 = [row for row in rows if row["elapsed"] == 0]
    successes = baseline_successes = missed = 0
    teacher = baseline_teacher = 0.0
    entered = 0
    records = []
    for row in t0:
        enter = scores[row["key"]] >= threshold
        source = bool(row["source_success"])
        persistent = bool(row["persistent_majority"])
        baseline_successes += int(persistent)
        baseline_teacher += row["teacher_steps"]
        if enter:
            entered += 1
            success = persistent
            teacher += row["teacher_steps"]
        else:
            success = source
        rescue = (not source) and persistent
        missed_here = (not enter) and rescue
        missed += int(missed_here)
        successes += int(success)
        records.append({"task_id": row["task_id"], "missed": missed_here,
                        "success": success, "baseline_success": persistent,
                        "teacher": row["teacher_steps"] if enter else 0.0,
                        "baseline_teacher": row["teacher_steps"]})
    n = max(1, len(t0))
    return {
        "groups": len(t0),
        "entered": entered,
        "success_gap": (successes - baseline_successes) / n,
        "false_continue_rate": missed / max(1, baseline_successes),
        "absolute_paired_harm": missed / n,
        "teacher_savings": 1.0 - teacher / max(1.0, baseline_teacher),
        "records": records,
    }


def select_threshold(rows: list[dict], scores: dict[tuple, float]) -> float:
    values = sorted({scores[row["key"]] for row in rows if row["elapsed"] == 0})
    candidates = [-1.0, *values, 2.0]
    best = None
    for threshold in candidates:
        metrics = controller_metrics(rows, scores, threshold)
        if (metrics["success_gap"] < -0.05
                or metrics["false_continue_rate"] > 0.05
                or metrics["absolute_paired_harm"] > 0.05):
            continue
        rank = (metrics["teacher_savings"], metrics["success_gap"])
        if best is None or rank > best[0]:
            best = (rank, threshold)
    # Safe fallback: enter OFT at t0 for every group.
    return float(best[1] if best is not None else -1.0)


def bootstrap_auc(rows: list[dict], predictions: list[float], iterations: int,
                  seed: int) -> dict:
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_task[row["task_id"]].append(index)
    tasks = sorted(by_task)
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        selected = [rng.choice(tasks) for _ in tasks]
        indices = [index for task in selected for index in by_task[task]]
        value = auc_score(
            np.asarray([rows[index]["advantage_majority"] for index in indices]),
            np.asarray([predictions[index] for index in indices]),
        )
        if math.isfinite(value):
            values.append(value)
    array = np.asarray(values, dtype=float)
    return {
        "iterations_valid": len(values),
        "lower_95": float(np.quantile(array, 0.025)) if len(array) else float("nan"),
        "median": float(np.quantile(array, 0.5)) if len(array) else float("nan"),
        "upper_95": float(np.quantile(array, 0.975)) if len(array) else float("nan"),
    }


def probe_policy(rows: list[dict], policy: str, folds: int,
                 bootstrap_iterations: int, seed: int) -> dict:
    policy_rows = [row for row in rows if row["policy_id"] == policy]
    natural = [row for row in policy_rows if row["cohort_role"] == "natural"]
    tasks = sorted({row["task_id"] for row in natural})
    random.Random(seed).shuffle(tasks)
    task_folds = [set(tasks[index::folds]) for index in range(folds)]
    oof_rows: list[dict] = []
    oof_source: list[float] = []
    oof_persistent: list[float] = []
    oof_advantage: list[float] = []
    oof_cost: list[float] = []
    decisions = []
    fold_reports = []
    for fold, validation_tasks in enumerate(task_folds):
        remaining = sorted(set(tasks) - validation_tasks)
        random.Random(seed + fold + 101).shuffle(remaining)
        n_cal = max(4, len(remaining) // 5)
        calibration_tasks = set(remaining[:n_cal])
        fit_tasks = set(remaining[n_cal:])
        fit = [row for row in policy_rows if row["task_id"] in fit_tasks]
        calibration = [row for row in natural if row["task_id"] in calibration_tasks]
        validation = [row for row in natural if row["task_id"] in validation_tasks]
        if min(len(fit), len(calibration), len(validation)) == 0:
            raise ValueError(f"empty probe partition for {policy} fold {fold}")
        x_fit = np.stack([row["feature"] for row in fit])
        x_cal = np.stack([row["feature"] for row in calibration])
        x_val = np.stack([row["feature"] for row in validation])
        source_model = fit_logistic(
            x_fit, np.asarray([row["source_risk"] for row in fit]), seed + fold * 31
        )
        persistent_model = fit_logistic(
            x_fit, np.asarray([row["persistent_probability"] for row in fit]),
            seed + fold * 31 + 1,
        )
        advantage_model = fit_logistic(
            x_fit, np.asarray([row["advantage_probability"] for row in fit]),
            seed + fold * 31 + 2,
        )
        cost_model = fit_ridge(
            x_fit, np.log1p(np.asarray([row["teacher_steps"] for row in fit]))
        )
        cal_adv = predict_logistic(advantage_model, x_cal)
        val_source = predict_logistic(source_model, x_val)
        val_persistent = predict_logistic(persistent_model, x_val)
        val_adv = predict_logistic(advantage_model, x_val)
        val_cost = np.expm1(predict_ridge(cost_model, x_val))
        cal_scores = {
            row["key"]: float(score)
            for row, score in zip(calibration, cal_adv)
            if row["elapsed"] == 0
        }
        threshold = select_threshold(calibration, cal_scores)
        val_scores = {
            row["key"]: float(score)
            for row, score in zip(validation, val_adv)
            if row["elapsed"] == 0
        }
        fold_controller = controller_metrics(validation, val_scores, threshold)
        decisions.extend(fold_controller.pop("records"))
        fold_reports.append({
            "fold": fold,
            "fit_rows": len(fit),
            "calibration_rows": len(calibration),
            "validation_rows": len(validation),
            "advantage_threshold": threshold,
            "controller": fold_controller,
        })
        oof_rows.extend(validation)
        oof_source.extend(val_source.tolist())
        oof_persistent.extend(val_persistent.tolist())
        oof_advantage.extend(val_adv.tolist())
        oof_cost.extend(val_cost.tolist())

    advantage_label = np.asarray([row["advantage_majority"] for row in oof_rows])
    source_label = np.asarray([row["source_risk"] for row in oof_rows])
    persistent_label = np.asarray([row["persistent_majority"] for row in oof_rows])
    advantage_prediction = np.asarray(oof_advantage)
    prevalence = float(advantage_label.mean())
    suite_metrics = {}
    for suite in sorted({row["suite"] for row in oof_rows}):
        indices = [i for i, row in enumerate(oof_rows) if row["suite"] == suite]
        suite_metrics[suite] = {
            "rows": len(indices),
            "positives": int(advantage_label[indices].sum()),
            "advantage_auc": auc_score(advantage_label[indices], advantage_prediction[indices]),
            "advantage_ap": average_precision(advantage_label[indices], advantage_prediction[indices]),
        }
    decision_n = max(1, len(decisions))
    decision_metrics = {
        "groups": len(decisions),
        "success_gap": (
            sum(d["success"] for d in decisions)
            - sum(d["baseline_success"] for d in decisions)
        ) / decision_n,
        "false_continue_rate": (
            sum(d["missed"] for d in decisions)
            / max(1, sum(d["baseline_success"] for d in decisions))
        ),
        "absolute_paired_harm": sum(d["missed"] for d in decisions) / decision_n,
        "teacher_savings": 1.0 - (
            sum(d["teacher"] for d in decisions)
            / max(1.0, sum(d["baseline_teacher"] for d in decisions))
        ),
    }
    advantage_auc = auc_score(advantage_label, advantage_prediction)
    advantage_ap = average_precision(advantage_label, advantage_prediction)
    bootstrap = bootstrap_auc(
        oof_rows, oof_advantage, bootstrap_iterations, seed + 999
    )
    finite_suite_aucs = [value["advantage_auc"] for value in suite_metrics.values()
                         if math.isfinite(value["advantage_auc"])]
    # Readiness asks whether a deployable signal and a Pareto direction exist.
    # The strict 5% false-continue/harm constraints belong to the decisive
    # task-bootstrap ensemble OOF gate, not to this deliberately weak linear
    # probe.  Keep them visible as yellow diagnostics without using them to
    # suppress a scientifically justified final experiment.
    gates = {
        "advantage_auc_ge_0_65": math.isfinite(advantage_auc) and advantage_auc >= 0.65,
        "advantage_ap_above_prevalence": math.isfinite(advantage_ap) and advantage_ap > prevalence,
        "at_least_3_suites_above_random": sum(value > 0.5 for value in finite_suite_aucs) >= 3,
        "task_bootstrap_auc_lower_above_random": (
            math.isfinite(bootstrap["lower_95"]) and bootstrap["lower_95"] > 0.5
        ),
        "probe_controller_success_gap_ge_minus_5pp": decision_metrics["success_gap"] >= -0.05,
        "probe_controller_has_positive_savings": decision_metrics["teacher_savings"] > 0.0,
    }
    return {
        "policy_id": policy,
        "natural_oof_rows": len(oof_rows),
        "natural_oof_tasks": len(tasks),
        "feature_dim": int(oof_rows[0]["feature"].shape[0]),
        "advantage_prevalence": prevalence,
        "source_risk_auc": auc_score(source_label, np.asarray(oof_source)),
        "persistent_success_auc": auc_score(persistent_label, np.asarray(oof_persistent)),
        "advantage_auc": advantage_auc,
        "advantage_ap": advantage_ap,
        "advantage_auc_task_bootstrap": bootstrap,
        "teacher_cost_log_mae": float(np.mean(np.abs(
            np.log1p(np.maximum(0.0, np.asarray(oof_cost)))
            - np.log1p(np.asarray([row["teacher_steps"] for row in oof_rows]))
        ))),
        "suite_metrics": suite_metrics,
        "controller": decision_metrics,
        "formal_safety_diagnostics": {
            "false_continue_le_5pct": decision_metrics["false_continue_rate"] <= 0.05,
            "harm_le_5pct": decision_metrics["absolute_paired_harm"] <= 0.05,
            "note": "diagnostic only at readiness; enforced as a hard gate in formal 5-seed OOF",
        },
        "fold_reports": fold_reports,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--label-support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--bootstrap-iterations", type=int, default=300)
    args = parser.parse_args()

    support = json.loads(args.label_support.read_text())
    if support.get("schema_version") != "rase-r6c1b-label-support/v2":
        raise ValueError("readiness requires replica-aware label-support/v2")
    repro = json.loads(args.exclusions.read_text())
    if repro.get("status") != "frozen":
        raise ValueError("reproducibility audit is not frozen")
    groups = load_groups(args.input_root, args.exclusions)
    quality = label_quality(groups, repro)
    rows = feature_rows(groups)
    policies = sorted({group["policy_id"] for group in groups})
    policy_results = {}
    for policy in policies:
        natural = [group for group in groups
                   if group["policy_id"] == policy and group["cohort_role"] == "natural"]
        opportunity = _strategy_summary(natural)
        probe = probe_policy(
            rows, policy, args.folds, args.bootstrap_iterations, args.seed
        )
        support_passed = bool(
            support.get("policy_results", {}).get(policy, {}).get("passed", False)
        )
        gates = {
            "replica_quality_passed": quality["passed"],
            "label_support_passed": support_passed,
            "natural_model_free_opportunity_passed": opportunity["passed"],
            "task_heldout_probe_passed": probe["passed"],
        }
        policy_results[policy] = {
            "support_passed": support_passed,
            "natural_opportunity": opportunity,
            "probe": probe,
            "gates": gates,
            "passed": all(gates.values()),
        }

    pi0fast_pass = bool(policy_results.get("pi0fast_libero", {}).get("passed", False))
    pi05_pass = bool(policy_results.get("pi05_libero", {}).get("passed", False))
    result = {
        "schema_version": "rase-r6c1b-pretrain-readiness/v1",
        "status": "complete",
        "scientific_scope": (
            "development-only readiness; natural task-held-out OOF; enrichment train-only; "
            "no selector training authorization"
        ),
        "input_roots": [str(path.resolve()) for path in args.input_root],
        "input_hashes": {
            "exclusions": sha256(args.exclusions),
            "label_support": sha256(args.label_support),
        },
        "n_replica_aggregated_groups": len(groups),
        "n_probe_rows": len(rows),
        "label_quality": quality,
        "policy_results": policy_results,
        "pi0fast_formal_training_ready": pi0fast_pass,
        "pi05_formal_training_ready": pi05_pass,
        "decision": (
            "APPROVE_PI0FAST_ONLY"
            if pi0fast_pass and not pi05_pass
            else "APPROVE_BOTH" if pi0fast_pass and pi05_pass
            else "STOP_BEFORE_FORMAL_SELECTOR_TRAINING"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if pi0fast_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
