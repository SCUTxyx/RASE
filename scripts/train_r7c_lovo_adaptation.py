#!/usr/bin/env python3
"""Leave-one-VLA-out zero-shot and 0/8/16/32 trajectory adaptation curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.risk.multi_vla_descriptor import behavior_descriptor  # noqa: E402
from rase.risk.r7_source_protocol import (  # noqa: E402
    FOLD_SEED,
    N_FOLDS,
    calibration_tasks,
    task_folds,
)
from scripts.train_r7a_source_risk_probe import fit_platt, metrics, sha256, task_bootstrap  # noqa: E402
from scripts.train_r7c_multivla_source_risk import predict, train_member  # noqa: E402

ADAPTATION_ROWS = (0, 8, 16, 32)
SELECTION_SALT = "rase-r7c-lovo-adaptation/v1/20260812"


def select_unlabeled(indices: np.ndarray, state_key: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.asarray([], dtype=np.int64)
    ranked = sorted(
        indices.tolist(),
        key=lambda index: hashlib.sha256(
            f"{SELECTION_SALT}:{state_key[index]}".encode()
        ).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError(f"need {count} calibration rows, have {len(ranked)}")
    return np.asarray(ranked[:count], dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--heldout-policy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold-seed", type=int, default=FOLD_SEED)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    report = json.loads(args.dataset_report.read_text())
    if report.get("status") != "frozen" or report.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("multi-VLA dataset/report is not frozen")
    with np.load(args.dataset, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    policies = sorted(set(data["policy_id"].tolist()))
    if args.heldout_policy not in policies or len(policies) < 2:
        raise ValueError("heldout policy must be one of at least two policies")
    source_policies = [policy for policy in policies if policy != args.heldout_policy]
    heldout_all = np.flatnonzero(data["policy_id"] == args.heldout_policy)
    labels = data["source_failure"].astype(np.float64)
    n_rows = len(data["source_failure"])
    probability_unlabeled = {
        count: np.full(n_rows, np.nan, dtype=np.float64) for count in ADAPTATION_ROWS
    }
    probability_labeled = {
        count: np.full(n_rows, np.nan, dtype=np.float64) for count in ADAPTATION_ROWS if count
    }
    labeled_available = {count: [] for count in ADAPTATION_ROWS if count}
    fold_reports = []
    all_tasks = set(data["task_id"].tolist())
    folds = task_folds(data["task_id"], data["suite"], count=args.folds, seed=args.fold_seed)

    for fold, validation_tasks in enumerate(folds):
        train_tasks = all_tasks - validation_tasks
        cal_tasks = calibration_tasks(
            train_tasks, data["task_id"], data["suite"], fold=fold, seed=args.fold_seed,
        )
        fit_tasks = train_tasks - cal_tasks
        is_source = np.isin(data["policy_id"], source_policies)
        is_heldout = data["policy_id"] == args.heldout_policy
        source_fit = np.flatnonzero(is_source & np.isin(data["task_id"], list(fit_tasks)))
        source_cal = np.flatnonzero(is_source & np.isin(data["task_id"], list(cal_tasks)))
        heldout_cal = np.flatnonzero(is_heldout & np.isin(data["task_id"], list(cal_tasks)))
        heldout_val = np.flatnonzero(is_heldout & np.isin(data["task_id"], list(validation_tasks)))
        if len(heldout_cal) != 32:
            raise ValueError(f"fold {fold} must expose exactly 32 heldout calibration rows")
        for partition, indices in (("source_fit", source_fit), ("source_cal", source_cal)):
            if len(np.unique(labels[indices])) != 2:
                raise ValueError(f"fold {fold} {partition} lacks both labels")

        members = []
        source_cal_logits = []
        for member in range(args.members):
            member_seed = args.seed + fold * 1009 + member * 7919
            model, stats, source_desc = train_member(
                data, source_fit, mode="shared_desc", policy_count=len(policies),
                seed=member_seed, epochs=args.epochs, device=args.device,
            )
            members.append((model, stats, source_desc))
            source_cal_logits.append(predict(
                model, stats, source_desc, data, source_cal,
                mode="shared_desc", device=args.device,
            ))
        pooled_temperature, pooled_bias = fit_platt(
            np.mean(source_cal_logits, axis=0), labels[source_cal],
        )
        fold_adaptation = {}
        for count in ADAPTATION_ROWS:
            selected = select_unlabeled(heldout_cal, data["state_key"], count)
            if count:
                heldout_descriptor = behavior_descriptor(
                    data["image"][selected], data["proprio"][selected],
                    data["action_summary"][selected],
                )
            else:
                # Zero-shot has no heldout behavior.  Use the mean descriptor of
                # source policies, never heldout outcomes or evaluation states.
                heldout_descriptor = np.mean(np.stack([
                    value for _, _, mapping in members for value in mapping.values()
                ]), axis=0).astype(np.float32)
            cal_logits, val_logits = [], []
            for model, stats, source_desc in members:
                mapping = dict(source_desc)
                mapping[args.heldout_policy] = heldout_descriptor
                cal_logits.append(predict(
                    model, stats, mapping, data, heldout_cal,
                    mode="shared_desc", device=args.device,
                ))
                val_logits.append(predict(
                    model, stats, mapping, data, heldout_val,
                    mode="shared_desc", device=args.device,
                ))
            mean_cal = np.mean(cal_logits, axis=0)
            mean_val = np.mean(val_logits, axis=0)
            probability_unlabeled[count][heldout_val] = 1.0 / (
                1.0 + np.exp(-(mean_val / pooled_temperature + pooled_bias))
            )
            entry = {
                "unlabeled_rows": int(count),
                "selected_state_keys": data["state_key"][selected].tolist(),
                "labeled_calibration_available": False,
            }
            if count:
                selected_mask = np.isin(heldout_cal, selected)
                selected_labels = labels[heldout_cal][selected_mask]
                if len(np.unique(selected_labels)) == 2:
                    temperature, bias = fit_platt(
                        mean_cal[selected_mask], selected_labels,
                    )
                    probability_labeled[count][heldout_val] = 1.0 / (
                        1.0 + np.exp(-(mean_val / temperature + bias))
                    )
                    labeled_available[count].append(True)
                    entry.update({
                        "labeled_calibration_available": True,
                        "temperature": temperature, "bias": bias,
                    })
                else:
                    labeled_available[count].append(False)
            fold_adaptation[str(count)] = entry
        fold_reports.append({
            "fold": fold, "fit_tasks": sorted(fit_tasks),
            "calibration_tasks": sorted(cal_tasks),
            "validation_tasks": sorted(validation_tasks),
            "adaptation": fold_adaptation,
        })

    curves = {"unlabeled": {}, "labeled_calibration": {}}
    for count, probability in probability_unlabeled.items():
        if not np.isfinite(probability[heldout_all]).all():
            raise AssertionError(f"incomplete heldout OOF curve for count={count}")
        row_metrics = metrics(labels[heldout_all], probability[heldout_all])
        curves["unlabeled"][str(count)] = {
            "metrics": row_metrics,
            "task_bootstrap": task_bootstrap(
                labels[heldout_all], probability[heldout_all], data["task_id"][heldout_all],
                seed=args.seed + count, samples=args.bootstrap_samples,
            ),
        }
    for count, probability in probability_labeled.items():
        available = all(labeled_available[count]) and len(labeled_available[count]) == args.folds
        if available and np.isfinite(probability[heldout_all]).all():
            curves["labeled_calibration"][str(count)] = {
                "available_all_folds": True,
                "metrics": metrics(labels[heldout_all], probability[heldout_all]),
                "task_bootstrap": task_bootstrap(
                    labels[heldout_all], probability[heldout_all], data["task_id"][heldout_all],
                    seed=args.seed + 1000 + count, samples=args.bootstrap_samples,
                ),
            }
        else:
            curves["labeled_calibration"][str(count)] = {
                "available_all_folds": False,
                "available_folds": int(sum(labeled_available[count])),
                "reason": "hash-selected labeled subset lacks both classes in at least one fold",
            }
    result = {
        "schema_version": "rase-r7c-lovo-adaptation/v1",
        "status": "COMPLETE",
        "seed": args.seed, "heldout_policy": args.heldout_policy,
        "source_policies": source_policies,
        "dataset_sha256": sha256(args.dataset),
        "selection_salt": SELECTION_SALT,
        "adaptation_rows": list(ADAPTATION_ROWS),
        "curves": curves, "fold_reports": fold_reports,
        "zero_shot_is_primary_gate": False,
        "adaptation_failure_definition": "32-unlabeled AUROC <0.65 or AP gain <0.05",
        "forbidden_and_absent": [
            "heldout validation labels in descriptor/calibration",
            "OFT labels/actions/cost", "policy outcome rate as descriptor",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    arrays = {}
    for count, probability in probability_unlabeled.items():
        arrays[f"unlabeled_{count}"] = probability[heldout_all].astype(np.float32)
    for count, probability in probability_labeled.items():
        arrays[f"labeled_{count}"] = probability[heldout_all].astype(np.float32)
    np.savez_compressed(
        args.output.with_suffix(".predictions.npz"),
        state_key=data["state_key"][heldout_all], task_id=data["task_id"][heldout_all],
        source_failure=labels[heldout_all].astype(np.float32), **arrays,
    )
    print(json.dumps({
        "status": "COMPLETE", "heldout_policy": args.heldout_policy,
        "curves": curves,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
