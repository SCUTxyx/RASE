#!/usr/bin/env python3
"""Audit R7-A2 source-outcome support before any source-risk model is trained."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.risk.r7_source_protocol import (  # noqa: E402
    FOLD_SEED,
    N_FOLDS,
    calibration_tasks,
    task_folds,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary_entropy(successes: int, total: int) -> float:
    if successes in (0, total):
        return 0.0
    p = successes / total
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path)
    parser.add_argument("--policy-id", default="pi0fast_libero")
    parser.add_argument("--min-failures", type=int, default=48)
    parser.add_argument("--min-successes", type=int, default=32)
    parser.add_argument("--min-failure-tasks", type=int, default=16)
    parser.add_argument("--min-mixed-tasks", type=int, default=8)
    parser.add_argument("--min-suite-per-class", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(args.initial_keys.read_text())
    expected_all = {str(row["state_key"]): row for row in manifest["records"]}
    excluded: set[str] = set()
    exclusion = None
    if args.exclusion_manifest is not None:
        exclusion = json.loads(args.exclusion_manifest.read_text())
        if exclusion.get("status") != "frozen":
            raise ValueError("R7 exclusion manifest is not frozen")
        excluded = {str(key) for key in exclusion.get("excluded_state_keys", [])}
        if not excluded or not excluded <= set(expected_all):
            raise ValueError("R7 exclusion keys are empty or outside the frozen cohort")
        if exclusion.get("initial_keys_sha256") != sha256(args.initial_keys):
            raise ValueError("R7 exclusion / initial-key hash mismatch")
    expected = {key: row for key, row in expected_all.items() if key not in excluded}
    observed: dict[str, dict] = {}
    errors: list[dict] = []
    for path in sorted(args.input_root.glob("suite_*/seed_0/*__seed0.json")):
        payload = json.loads(path.read_text())
        boundaries = payload.get("rows") or []
        if len(boundaries) != 1:
            errors.append({"metadata": str(path), "reason": "not_exactly_one_t0_boundary"})
            continue
        boundary = boundaries[0]
        key = str(boundary.get("state_key", ""))
        if key in excluded:
            continue
        if key in observed:
            errors.append({"state_key": key, "reason": "duplicate"})
            continue
        if key not in expected:
            errors.append({"state_key": key, "reason": "not_in_frozen_manifest"})
            continue
        frozen = expected[key]
        stop_reason = str(payload.get("stop_reason", ""))
        inference_error = payload.get("policy_inference_error")
        if stop_reason == "policy_inference_error":
            inference_error_valid = (
                isinstance(inference_error, dict)
                and inference_error.get("type") == "invalid_action_token_sequence"
                and bool(inference_error.get("initial_10_step_proposal_complete"))
                and int(inference_error.get("elapsed_source_steps", -1))
                == int(payload.get("source_steps", -2))
                and not bool(payload.get("source_success"))
            )
        else:
            inference_error_valid = inference_error is None
        npz = Path(str(payload.get("npz", "")))
        if not npz.is_file() or sha256(npz) != str(payload.get("npz_sha256", "")):
            errors.append({"state_key": key, "reason": "npz_missing_or_hash_mismatch"})
            continue
        metadata_checks = {
            "elapsed_source_steps": int(boundary.get("elapsed_source_steps", -1)) == 0,
            "policy_id": boundary.get("policy_id") == args.policy_id,
            "seed_index": int(boundary.get("seed_index", -1)) == 0,
            "task_id": boundary.get("task_id") == frozen.get("task_id"),
            "suite": boundary.get("suite") == frozen.get("suite"),
            "source_label": bool(boundary.get("source_final_success"))
            == bool(payload.get("source_success")),
            "source_steps": int(boundary.get("source_total_steps", -1))
            == int(payload.get("source_steps", -2)),
            "no_persistent_label": boundary.get("persistent_success_if_enter_now") is None,
            "no_persistent_cost": boundary.get("persistent_teacher_steps_if_enter_now") is None,
            "counterfactual_skipped": boundary.get("counterfactual_timing") == "skipped",
            "instruction": bool(str(boundary.get("instruction", "")).strip()),
            "policy_inference_error_contract": inference_error_valid,
        }
        failed_metadata = sorted(name for name, ok in metadata_checks.items() if not ok)
        if failed_metadata:
            errors.append({
                "state_key": key, "reason": "metadata_contract_failure",
                "failed_checks": failed_metadata,
            })
            continue
        try:
            with np.load(npz, allow_pickle=False) as raw:
                required = {
                    "image", "proprio", "source_action", "source_action_summary",
                    "source_action_trace", "oft_action", "oft_action_summary",
                }
                missing_arrays = sorted(required - set(raw.files))
                array_checks = {
                    "image": not missing_arrays and raw["image"].shape == (1, 2, 3, 96, 96),
                    "proprio": not missing_arrays and raw["proprio"].shape == (1, 8),
                    "source_action": not missing_arrays and raw["source_action"].shape == (1, 7),
                    "source_action_summary": not missing_arrays
                    and raw["source_action_summary"].shape == (1, 20),
                    "source_action_trace": not missing_arrays
                    and raw["source_action_trace"].ndim == 2
                    and raw["source_action_trace"].shape[1] == 7
                    and raw["source_action_trace"].shape[0] == int(payload["source_steps"]),
                    "initial_10_step_proposal_complete": not missing_arrays
                    and raw["source_action_trace"].ndim == 2
                    and raw["source_action_trace"].shape[0] >= 10,
                    "no_oft_action": not missing_arrays and raw["oft_action"].size == 0,
                    "no_oft_summary": not missing_arrays and raw["oft_action_summary"].size == 0,
                }
        except Exception as exc:
            errors.append({"state_key": key, "reason": "npz_read_failure", "detail": repr(exc)})
            continue
        failed_arrays = sorted(name for name, ok in array_checks.items() if not ok)
        if missing_arrays or failed_arrays:
            errors.append({
                "state_key": key, "reason": "feature_contract_failure",
                "missing_arrays": missing_arrays, "failed_checks": failed_arrays,
            })
            continue
        observed[key] = {
            **frozen,
            "source_success": bool(payload["source_success"]),
            "source_steps": int(payload["source_steps"]),
            "stop_reason": stop_reason,
            "metadata": str(path.resolve()),
            "npz": str(npz.resolve()),
        }

    missing = sorted(set(expected) - set(observed))
    rows = list(observed.values())
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    mixed_tasks = sum(
        len({row["source_success"] for row in members}) > 1
        for members in by_task.values()
    )
    failures = sum(not row["source_success"] for row in rows)
    successes = len(rows) - failures
    failure_tasks = len({row["task_id"] for row in rows if not row["source_success"]})
    suite_stats = {}
    for suite in ("Spatial", "Object", "Goal", "Long"):
        subset = [row for row in rows if row["suite"] == suite]
        suite_stats[suite] = {
            "states": len(subset),
            "successes": sum(row["source_success"] for row in subset),
            "failures": sum(not row["source_success"] for row in subset),
            "failure_tasks": len({row["task_id"] for row in subset if not row["source_success"]}),
        }
    task_entropy = {
        task: binary_entropy(sum(row["source_success"] for row in members), len(members))
        for task, members in by_task.items()
    }
    fold_support = []
    folds_have_fit_cal_both_classes = False
    expected_count = len(expected)
    if len(rows) == expected_count and len(by_task) == 48:
        task_array = np.asarray([str(row["task_id"]) for row in rows])
        suite_array = np.asarray([str(row["suite"]) for row in rows])
        label_array = np.asarray([not row["source_success"] for row in rows], dtype=np.int64)
        all_tasks = set(task_array.tolist())
        for fold, validation_tasks in enumerate(task_folds(task_array, suite_array)):
            train_tasks = all_tasks - validation_tasks
            cal_tasks = calibration_tasks(train_tasks, task_array, suite_array, fold=fold)
            fit_tasks = train_tasks - cal_tasks
            fit_idx = np.flatnonzero(np.isin(task_array, list(fit_tasks)))
            cal_idx = np.flatnonzero(np.isin(task_array, list(cal_tasks)))
            val_idx = np.flatnonzero(np.isin(task_array, list(validation_tasks)))
            fold_support.append({
                "fold": fold, "fit_rows": int(len(fit_idx)),
                "calibration_rows": int(len(cal_idx)), "validation_rows": int(len(val_idx)),
                "fit_classes": sorted(np.unique(label_array[fit_idx]).tolist()),
                "calibration_classes": sorted(np.unique(label_array[cal_idx]).tolist()),
                "validation_classes": sorted(np.unique(label_array[val_idx]).tolist()),
            })
        folds_have_fit_cal_both_classes = all(
            row["fit_classes"] == [0, 1] and row["calibration_classes"] == [0, 1]
            for row in fold_support
        )
    complete = not errors and not missing and len(rows) == expected_count
    support_gate = (
        complete and failures >= args.min_failures and successes >= args.min_successes
        and failure_tasks >= args.min_failure_tasks
        and mixed_tasks >= args.min_mixed_tasks
        and all(value["failures"] >= args.min_suite_per_class
                and value["successes"] >= args.min_suite_per_class
                for value in suite_stats.values())
        and folds_have_fit_cal_both_classes
    )
    policy_inference_failures = sum(
        row.get("stop_reason") == "policy_inference_error" for row in rows
    )
    result = {
        "schema_version": "rase-r7-source-label-support/v4",
        "status": "PASS" if support_gate else "FAIL",
        "policy_id": args.policy_id,
        "scientific_scope": "development label-support gate; not a model-performance result",
        "input_root": str(args.input_root.resolve()),
        "initial_keys": str(args.initial_keys.resolve()),
        "initial_keys_sha256": sha256(args.initial_keys),
        "expected_states": expected_count,
        "excluded_state_keys": sorted(excluded),
        "exclusion_manifest": (str(args.exclusion_manifest.resolve())
                               if args.exclusion_manifest is not None else None),
        "exclusion_manifest_sha256": (sha256(args.exclusion_manifest)
                                      if args.exclusion_manifest is not None else None),
        "states": len(rows), "tasks": len(by_task),
        "successes": successes, "failures": failures,
        "policy_inference_failures": policy_inference_failures,
        "source_success_rate": successes / max(1, len(rows)),
        "failure_tasks": failure_tasks, "mixed_outcome_tasks": mixed_tasks,
        "mean_within_task_label_entropy_bits": (
            sum(task_entropy.values()) / max(1, len(task_entropy))
        ),
        "fold_seed": FOLD_SEED, "n_folds": N_FOLDS,
        "fold_label_support": fold_support,
        "suite_stats": suite_stats,
        "missing_state_keys": missing, "errors": errors,
        "gate": {
            "complete_expected_cohort": complete,
            "failures_at_least_minimum": failures >= args.min_failures,
            "successes_at_least_minimum": successes >= args.min_successes,
            "failure_tasks_at_least_minimum": failure_tasks >= args.min_failure_tasks,
            "mixed_tasks_at_least_minimum": mixed_tasks >= args.min_mixed_tasks,
            "each_suite_at_least_minimum_per_class": all(
                value["failures"] >= args.min_suite_per_class
                and value["successes"] >= args.min_suite_per_class
                for value in suite_stats.values()
            ),
            "all_folds_fit_and_calibration_have_both_classes": folds_have_fit_cal_both_classes,
        },
        "thresholds": {
            "min_failures": args.min_failures,
            "min_successes": args.min_successes,
            "min_failure_tasks": args.min_failure_tasks,
            "min_mixed_tasks": args.min_mixed_tasks,
            "min_suite_per_class": args.min_suite_per_class,
        },
        "unlocks_on_pass": [
            "task-clustered source-risk representation probe",
            "hash-selected exact-repeat stability audit",
        ],
        "remains_locked_even_on_pass": [
            "OFT counterfactual collection", "selector training",
            "world-model feature ablation", "validation", "test",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
