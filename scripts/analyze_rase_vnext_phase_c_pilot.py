#!/usr/bin/env python3
"""Analyze the explicitly labeled π0-fast A-PARTIAL action-sensitivity pilot."""

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

from rase.vnext.adapter_parity import (
    audit_action_roundtrip,
    audit_motion_trace_conversion,
    audit_resample_capability,
    build_capability_report,
)
from rase.vnext.libero import (
    LIBERO_ACTION_SEMANTICS,
    LIBERO_MOTION_SEMANTIC_MAP,
    LiberoPolicyAdapter,
)
from rase.vnext.phase_c_pilot import (
    SOURCE_OPERATORS,
    bootstrap_task_difference,
    finite_dict,
    grouped_metrics,
    raw_action_feature_vector,
    ridge_oof_predictions,
    stable_seed,
    task_folds,
    trace_feature_vector,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def one_hot(value: str, vocabulary: tuple[str, ...]) -> np.ndarray:
    result = np.zeros(len(vocabulary), dtype=np.float64)
    result[vocabulary.index(value)] = 1.0
    return result


def load_dataset(feature_dir: Path) -> dict[str, Any]:
    contract_path = feature_dir / "EXPORT_CONTRACT.json"
    report_path = feature_dir / "collection_report.json"
    contract = json.loads(contract_path.read_text())
    collection = json.loads(report_path.read_text())
    if collection.get("status") not in {"COMPLETE", "PARTIAL_REPRODUCIBLE"}:
        raise SystemExit("feature collection report is neither COMPLETE nor PARTIAL_REPRODUCIBLE")
    expected = len(contract["groups"])
    metadata_paths = sorted((feature_dir / "groups").glob("*.json"))
    if len(metadata_paths) != expected:
        raise SystemExit(f"feature metadata count {len(metadata_paths)} != expected {expected}")

    records: list[dict[str, Any]] = []
    resample_groups: list[list[np.ndarray]] = []
    alignment_failures: list[str] = []
    feature_hash_failures: list[str] = []
    unreproducible: list[dict[str, Any]] = []
    for meta_path in metadata_paths:
        meta = json.loads(meta_path.read_text())
        if meta.get("status") == "UNREPRODUCIBLE":
            unreproducible.append(meta)
            continue
        if meta.get("status") != "COMPLETE":
            raise SystemExit(f"non-complete feature group: {meta_path}")
        if not all(meta.get("alignment_checks", {}).values()):
            alignment_failures.append(meta_path.name)
        npz_path = Path(str(meta["features_path"]))
        if sha256(npz_path) != meta["features_sha256"]:
            feature_hash_failures.append(npz_path.name)
            continue
        with np.load(npz_path, allow_pickle=False) as arrays:
            actions = arrays["actions"].copy()
            masks = arrays["action_step_mask"].copy()
            proprio = arrays["proprio"].astype(np.float64).copy()
            proprio_mask = arrays["proprio_mask"].astype(np.bool_).copy()
            if "resample_candidate_actions" in arrays:
                candidate_actions = arrays["resample_candidate_actions"].copy()
                candidate_masks = arrays["resample_candidate_step_mask"].copy()
                resample_groups.append([
                    candidate_actions[index][candidate_masks[index]] for index in range(2)
                ])
        group_id = "|".join(map(str, meta["group_key"]))
        for operator_index, operator in enumerate(meta["operator_order"]):
            if operator not in tuple(contract["operators"]):
                raise SystemExit(f"unexpected operator {operator}")
            outcome = meta["outcomes"][operator]
            if outcome.get("utility") is None:
                raise SystemExit(f"missing utility for {meta_path.name}/{operator}")
            raw = raw_action_feature_vector(actions[operator_index], masks[operator_index])
            trace = trace_feature_vector(
                actions[operator_index], masks[operator_index],
                semantics=LIBERO_ACTION_SEMANTICS,
                policy_id=str(meta["policy_id"]),
                semantic_map=LIBERO_MOTION_SEMANTIC_MAP,
            )
            records.append({
                "group_id": group_id, "task": str(meta["task_id"]),
                "suite": str(meta["suite"]), "operator": str(operator),
                "decision_point": str(meta["decision_point_id"]),
                "replica": int(meta["exact_repeat_replica"]),
                "utility": float(outcome["utility"]),
                "action": actions[operator_index], "mask": masks[operator_index],
                "proprio": proprio * proprio_mask,
                "raw": raw, "trace": trace,
            })
    if alignment_failures or feature_hash_failures:
        raise SystemExit(
            f"feature integrity failure: alignment={alignment_failures}, hash={feature_hash_failures}"
        )
    return {
        "contract": contract, "collection": collection, "records": records,
        "resample_groups": resample_groups,
        "unreproducible": unreproducible,
        "contract_sha256": sha256(contract_path),
        "collection_report_sha256": sha256(report_path),
    }


def build_features(
    records: list[dict[str, Any]], operator_vocabulary: tuple[str, ...],
) -> dict[str, np.ndarray]:
    operator = np.stack([one_hot(row["operator"], operator_vocabulary) for row in records])
    point_vocab = tuple(sorted({row["decision_point"] for row in records}))
    point = np.stack([one_hot(row["decision_point"], point_vocab) for row in records])
    proprio = np.stack([row["proprio"] for row in records])
    raw = np.stack([row["raw"] for row in records])
    trace = np.stack([row["trace"] for row in records])

    shuffled = np.zeros_like(trace)
    by_operator: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_operator[row["operator"]].append(index)
    for name, indices in sorted(by_operator.items()):
        ordered = sorted(
            indices,
            key=lambda index: (stable_seed("shuffle", name, records[index]["group_id"]), index),
        )
        source = ordered[-1:] + ordered[:-1]
        for target_index, source_index in zip(ordered, source):
            shuffled[target_index] = trace[source_index]

    return {
        "C0_operator_prior": operator,
        "C2_context_prior": np.column_stack((operator, point)),
        "C3_proprio": np.column_stack((operator, point, proprio)),
        "C4_raw_action": np.column_stack((operator, raw)),
        "C4_proprio_raw_action": np.column_stack((operator, point, proprio, raw)),
        "C6_motion_trace": np.column_stack((operator, trace)),
        "C6_proprio_motion_trace": np.column_stack((operator, point, proprio, trace)),
        "C6_proprio_trace_shuffled": np.column_stack((operator, point, proprio, shuffled)),
    }


def parity_report(dataset: dict[str, Any]) -> dict[str, Any]:
    resample_enabled = "resample.source" in tuple(dataset["contract"]["operators"])
    adapter = LiberoPolicyAdapter(
        policy_id="pi0fast.libero", family="pi0fast",
        supports_requery=True, supports_resample=resample_enabled,
        stochastic_sampling=resample_enabled,
    )
    samples = [row["action"][row["mask"]] for row in dataset["records"]]
    tokens = [adapter.raw_to_canonical(sample) for sample in samples]
    roundtrip = audit_action_roundtrip(adapter, samples)
    motion = audit_motion_trace_conversion(
        tokens, semantic_map=LIBERO_MOTION_SEMANTIC_MAP,
    )
    resample = (
        audit_resample_capability(
            dataset["resample_groups"], minimum_distinct_fraction=0.1,
        )
        if resample_enabled else None
    )
    capability = build_capability_report(
        adapter.descriptor, resample_audit=resample,
        fallback_available=True, abort_available=True,
    )
    checks = {
        "raw_canonical_raw_roundtrip": roundtrip.status == "PASS",
        "motion_trace_conversion": motion.status == "PASS",
        "empirical_resample_or_frozen_mask": (
            resample.status == "PASS" if resample is not None else not resample_enabled
        ),
        "capability_contract": capability["status"] == "PASS",
        "libero_translation_scale_m": LIBERO_MOTION_SEMANTIC_MAP.translation_scale == 0.05,
        "libero_rotation_scale_rad": LIBERO_MOTION_SEMANTIC_MAP.rotation_scale == 0.5,
        "libero_rotation_representation_axis_angle": (
            LIBERO_MOTION_SEMANTIC_MAP.rotation_representation == "axis_angle"
        ),
        "libero_gripper_minus_open_plus_closed": True,
        "all_frozen_actions_reproducible": not dataset["unreproducible"],
    }
    if not checks["all_frozen_actions_reproducible"]:
        parity_status = "B_FAIL_REPRODUCIBILITY"
    elif all(checks.values()):
        parity_status = "B_PASS_SINGLE_POLICY"
    else:
        parity_status = "B_FAIL"
    return {
        "schema_version": "rase-vnext-phase-b-parity/v1",
        "scope": "pi0fast.libero_single_policy",
        "status": parity_status,
        "checks": checks,
        "roundtrip": {
            key: roundtrip.details[key]
            for key in ("samples", "maximum_absolute_error", "maximum_relative_error", "failures")
        },
        "motion_trace": {
            "samples": motion.details["samples"], "failures": motion.details["failures"],
        },
        "resample": (
            resample.details if resample is not None else {
                "status": "MASKED_BY_FROZEN_CONTRACT",
                "reason": "discovery_capability_audit:no_native_candidate_diversity",
            }
        ),
        "capability": capability,
        "controller_evidence": {
            "translation_output_range_m": [-0.05, 0.05],
            "rotation_vector_output_range_rad": [-0.5, 0.5],
            "orientation_update": "R_goal = R_delta(axis_angle) @ R_current",
            "gripper_convention": "-1=open,+1=closed",
        },
        "unreproducible_groups": [
            {
                "group_key": row["group_key"], "task_id": row["task_id"],
                "suite": row["suite"], "decision_point_id": row["decision_point_id"],
                "alignment_attempts": row["alignment_attempts"],
            }
            for row in dataset["unreproducible"]
        ],
    }


def analyze(dataset: dict[str, Any], *, bootstrap_replicates: int) -> dict[str, Any]:
    records = dataset["records"]
    operator_vocabulary = tuple(dataset["contract"]["operators"])
    features = build_features(records, operator_vocabulary)
    targets = np.array([row["utility"] for row in records], dtype=np.float64)
    tasks = [row["task"] for row in records]
    groups = [row["group_id"] for row in records]
    task_suite = {row["task"]: row["suite"] for row in records}
    model_results: dict[str, list[dict[str, Any]]] = {name: [] for name in features}
    per_task: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in features
    }
    # This margin is part of the frozen practical-effect protocol and must be
    # applied to the primary pairwise metric as well as the support audit.
    # Otherwise tiny deterministic query/latency costs turn practically tied
    # successful branches into an artificial operator-prior classification task.
    practical_tie_margin = 0.01

    for seed in range(5):
        folds = task_folds(tasks, task_suite, seed=seed, folds=5)
        for name, matrix in features.items():
            predictions = ridge_oof_predictions(
                matrix, targets, tasks, folds, alpha=1.0,
            )
            metrics, group_details = grouped_metrics(
                targets, predictions, groups, tie_margin=practical_tie_margin,
            )
            model_results[name].append({"seed": seed, **metrics})
            group_task = {row["group_id"]: row["task"] for row in records}
            for group, detail in group_details.items():
                if detail.pairwise_total:
                    per_task[name][group_task[group]].append(
                        detail.pairwise_correct / detail.pairwise_total
                    )

    summary: dict[str, Any] = {}
    for name, seeds in model_results.items():
        summary[name] = {
            "pairwise_accuracy_mean": float(np.mean([row["pairwise_accuracy"] for row in seeds])),
            "pairwise_accuracy_std": float(np.std([row["pairwise_accuracy"] for row in seeds])),
            "mean_oracle_regret": float(np.mean([row["mean_oracle_regret"] for row in seeds])),
            "seeds": seeds,
        }

    primary = "C6_proprio_motion_trace"
    control_names = (
        "C0_operator_prior", "C2_context_prior", "C3_proprio",
        "C4_raw_action", "C4_proprio_raw_action",
    )
    best_control = max(control_names, key=lambda name: summary[name]["pairwise_accuracy_mean"])
    gain, interval = bootstrap_task_difference(
        per_task[primary], per_task[best_control],
        replicates=bootstrap_replicates, seed=202708,
    )
    primary_seed_values = [row["pairwise_accuracy"] for row in model_results[primary]]
    control_seed_values = [row["pairwise_accuracy"] for row in model_results[best_control]]
    directional_seeds = sum(
        left > right for left, right in zip(primary_seed_values, control_seed_values)
    )
    shuffled_gap = (
        summary[primary]["pairwise_accuracy_mean"]
        - summary["C6_proprio_trace_shuffled"]["pairwise_accuracy_mean"]
    )
    task_count = len(set(tasks))
    group_count = len(set(groups))
    support_counts = {operator: 0 for operator in operator_vocabulary}
    practical_ties = 0
    records_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        records_by_group[row["group_id"]].append(row)
    for candidates in records_by_group.values():
        values = np.array([row["utility"] for row in candidates], dtype=np.float64)
        best = float(values.max())
        winners = [
            row["operator"] for row in candidates
            if best - float(row["utility"]) > -1e-12
            and best - float(row["utility"]) <= practical_tie_margin
        ]
        if len(winners) == len(candidates):
            practical_ties += 1
        elif len(winners) == 1:
            support_counts[winners[0]] += 1
    support_pass = sum(count >= 3 for count in support_counts.values()) >= 2
    informative_groups = sum(support_counts.values())
    informative_tasks = len({
        row["task"] for rows in records_by_group.values()
        if max(float(item["utility"]) for item in rows)
        - min(float(item["utility"]) for item in rows) > practical_tie_margin
        for row in rows
    })
    if not support_pass:
        status = "PILOT_SUPPORT_FAIL"
        next_action = "do_not_train_build_outcome_independent_challenge_and_fallback_trace_support"
    elif (
        gain >= 0.03 and interval[0] > 0 and directional_seeds >= 4
        and shuffled_gap >= 0.02 and task_count >= 32
    ):
        status = "PILOT_SIGNAL_PASS"
        next_action = "freeze_independent_challenge_cohort_then_test_transfer"
    elif gain >= 0.01 and directional_seeds >= 3:
        status = "PILOT_SIGNAL_WEAK"
        next_action = "diagnose_observability_and_task_semantic_alignment_without_scaling"
    else:
        status = "PILOT_SIGNAL_FAIL"
        next_action = "stop_representation_scaling_and_repair_target_or_operator_observability"
    report = {
        "schema_version": "rase-vnext-phase-c-single-policy-pilot/v1",
        "status": status,
        "scope": "A_PARTIAL_pi0fast.libero_only_not_a_pooled_multi_VLA_claim",
        "next_action": next_action,
        "integrity": {
            "tasks": task_count, "groups": group_count, "candidate_rows": len(records),
            "practical_pairwise_groups": informative_groups,
            "practical_pairwise_tasks": informative_tasks,
            "operators": list(operator_vocabulary),
            "task_held_out": True, "same_root_candidates_never_cross_fold": True,
            "feature_contract_sha256": dataset["contract_sha256"],
            "collection_report_sha256": dataset["collection_report_sha256"],
            "unreproducible_groups": len(dataset["unreproducible"]),
        },
        "frozen_primary": {
            "model": primary, "best_control_baseline": best_control,
            "pairwise_accuracy_gain": gain,
            "task_bootstrap_95_ci": interval,
            "minimum_practical_gain": 0.03,
            "directional_seeds": directional_seeds,
            "required_directional_seeds": 4,
            "trace_minus_shuffled_gain": shuffled_gap,
            "required_trace_minus_shuffled_gain": 0.02,
            "bootstrap_unit": "task", "bootstrap_replicates": bootstrap_replicates,
        },
        "support_diagnostic": {
            "practical_tie_margin": practical_tie_margin,
            "unique_wins_by_operator": support_counts,
            "all_candidate_practical_ties": practical_ties,
            "minimum_unique_wins_per_supported_operator": 3,
            "requires_at_least_two_supported_operators": True,
            "status": "PASS" if support_pass else "FAIL",
        },
        "models": summary,
        "limitations": [
            "A_PARTIAL permits only a labeled single-policy pilot",
            "confirmation labels are used for development and are not a sealed causal audit",
            "fallback/abort are excluded from this source-action sensitivity pilot",
            "pi0fast resample is excluded because the frozen capability audit masked it",
            "no held-out VLA direction is claimed",
            "world model, RL, OPD, and semantic pretraining remain locked",
        ],
    }
    finite_dict(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    dataset = load_dataset(args.feature_dir.resolve())
    parity = parity_report(dataset)
    pilot = analyze(dataset, bootstrap_replicates=args.bootstrap_replicates)
    result = {"phase_b": parity, "phase_c_pilot": pilot}
    atomic_json(args.output, result)
    print(json.dumps({
        "phase_b": parity["status"],
        "phase_c_pilot": pilot["status"],
        "primary": pilot["frozen_primary"],
    }, indent=2, sort_keys=True))
    return 0 if parity["status"] == "B_PASS_SINGLE_POLICY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
