#!/usr/bin/env python3
"""R8-A0 model-free audit of losing fallback recoverability while waiting.

This script consumes only the frozen replica-aggregated R6 dynamic-boundary
dataset.  It trains no model and writes no new rollout.  Natural groups define
the formal cost-aware oracle; enrichment groups may contribute label-support
counts but never formal opportunity metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")


def binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(probability * math.log2(probability)
             + (1.0 - probability) * math.log2(1.0 - probability))


def as_python(value):
    return value.item() if hasattr(value, "item") else value


def group_rows(data: dict[str, np.ndarray]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(data["group_id"]):
        groups[str(group)].append(index)
    return groups


def summarize_transition(records: list[dict]) -> dict:
    if not records:
        return {"transitions": 0, "hard_positive_transitions": 0}
    drops = np.asarray([row["probability_drop"] for row in records], dtype=float)
    return {
        "transitions": len(records),
        "hard_positive_transitions": sum(row["hard_positive"] for row in records),
        "hard_gain_transitions": sum(row["hard_gain"] for row in records),
        "ambiguous_transitions": sum(row["ambiguous"] for row in records),
        "ambiguous_transition_fraction": float(np.mean([row["ambiguous"] for row in records])),
        "mean_probability_drop": float(drops.mean()),
        "median_probability_drop": float(np.median(drops)),
        "fraction_probability_drop_positive": float(np.mean(drops > 0.0)),
        "fraction_probability_drop_at_least_0p5": float(np.mean(drops >= 0.5)),
    }


def oracle_summary(groups: list[dict], gates: dict) -> dict:
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for group in groups:
        if group["cohort_role"] == "natural":
            by_policy[group["policy_id"]].append(group)
    result = {}
    for policy, rows in sorted(by_policy.items()):
        baseline_probability = 0.0
        oracle_probability = 0.0
        baseline_steps = 0.0
        oracle_steps = 0.0
        paired_harm = 0.0
        arm_usage = Counter()
        for row in rows:
            options = [
                ("source", row["source_probability"], 0.0),
                ("t0", row["persistent_probability"][0], row["teacher_cost_q50"][0]),
                ("t8", row["persistent_probability"][8], row["teacher_cost_q50"][8]),
                ("t16", row["persistent_probability"][16], row["teacher_cost_q50"][16]),
            ]
            best_probability = max(item[1] for item in options)
            best = min((item for item in options
                        if abs(item[1] - best_probability) <= 1e-12),
                       key=lambda item: (item[2], item[0]))
            p0 = row["persistent_probability"][0]
            baseline_probability += p0
            oracle_probability += best[1]
            baseline_steps += row["teacher_cost_q50"][0]
            oracle_steps += best[2]
            paired_harm += max(0.0, p0 - best[1])
            arm_usage[best[0]] += 1
        count = len(rows)
        suites = sorted({row["suite"] for row in rows})
        tasks = sorted({row["task_id"] for row in rows})
        success_gap = ((oracle_probability - baseline_probability) / count
                       if count else float("nan"))
        savings = (1.0 - oracle_steps / baseline_steps
                   if baseline_steps > 0 else float("nan"))
        harm = paired_harm / count if count else float("nan")
        checks = {
            "natural_groups_at_least_minimum": count >= gates["minimum_natural_groups_per_policy"],
            "natural_tasks_at_least_minimum": len(tasks) >= gates["minimum_natural_tasks_per_policy"],
            "four_suites": (len(suites) == 4 if gates["require_four_suites"] else True),
            "oracle_success_gap_ge_minimum": success_gap >= gates["oracle_success_gap_vs_t0_min"],
            "oracle_teacher_savings_ge_minimum": savings >= gates["oracle_teacher_savings_vs_t0_min"],
            "oracle_expected_paired_harm_le_maximum": harm <= gates["oracle_expected_paired_harm_max"],
            "oracle_uses_minimum_arms": len([name for name, n in arm_usage.items() if n])
            >= gates["minimum_oracle_arms_used"],
        }
        result[policy] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "natural_groups": count, "natural_tasks": len(tasks), "suites": suites,
            "t0_expected_success_rate": baseline_probability / count if count else float("nan"),
            "oracle_expected_success_rate": oracle_probability / count if count else float("nan"),
            "oracle_success_gap_vs_t0": success_gap,
            "t0_total_median_teacher_steps": baseline_steps,
            "oracle_total_median_teacher_steps": oracle_steps,
            "oracle_teacher_savings_vs_t0": savings,
            "oracle_expected_paired_harm": harm,
            "oracle_arm_usage": dict(sorted(arm_usage.items())),
            "gate": checks,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if config.get("status") != "frozen":
        raise ValueError("R8-A0 protocol is not frozen")
    expected = config["inputs"]
    require_hash(args.dataset, expected["dataset_sha256"], "dataset")
    require_hash(args.dataset_report, expected["dataset_report_sha256"], "dataset report")
    require_hash(args.exclusions, expected["exclusions_sha256"], "exclusions")
    require_hash(args.readiness, expected["readiness_sha256"], "readiness")
    report = json.loads(args.dataset_report.read_text())
    readiness = json.loads(args.readiness.read_text())
    if report.get("status") != "complete" or report.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("dataset report is incomplete or unbound")
    if report.get("exclusions_sha256") != sha256(args.exclusions):
        raise ValueError("dataset report / exclusion hash mismatch")
    if report.get("protocol_sha256") != expected["protocol_sha256"]:
        raise ValueError("dataset report / frozen collector protocol mismatch")
    if not bool((readiness.get("label_quality") or {}).get("passed")):
        raise ValueError("upstream replica label-quality gate did not pass")

    with np.load(args.dataset, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    required = {
        "group_id", "base_group_id", "cohort_role", "policy_id", "task_id", "suite",
        "state_key", "elapsed_source_steps", "source_successes", "source_trials",
        "persistent_successes", "persistent_trials", "arm_teacher_step_quantiles",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"R8-A0 dataset missing fields: {missing}")
    if len(data["arm_ids"]) != 2 or data["arm_teacher_step_quantiles"].shape[1:] != (2, 3):
        raise ValueError("unexpected candidate-arm cost schema")

    boundaries = [int(value) for value in config["boundaries"]]
    transitions = [tuple(map(int, value)) for value in config["transitions"]]
    support_roles = set(config["cohort_policy"]["label_support_roles"])
    complete_groups: list[dict] = []
    incomplete: list[dict] = []
    duplicate_boundaries: list[dict] = []
    for group_id, indices in sorted(group_rows(data).items()):
        by_elapsed: dict[int, int] = {}
        for index in indices:
            elapsed = int(data["elapsed_source_steps"][index])
            if elapsed in by_elapsed:
                duplicate_boundaries.append({"group_id": group_id, "elapsed": elapsed})
            by_elapsed[elapsed] = index
        if any(elapsed not in by_elapsed for elapsed in boundaries):
            incomplete.append({"group_id": group_id,
                               "available_boundaries": sorted(by_elapsed)})
            continue
        selected = [by_elapsed[elapsed] for elapsed in boundaries]
        constant_fields = ("base_group_id", "cohort_role", "policy_id", "task_id",
                           "suite", "state_key", "source_successes", "source_trials")
        if any(len({as_python(data[name][index]) for index in selected}) != 1
               for name in constant_fields):
            raise ValueError(f"nonconstant group metadata/label: {group_id}")
        role = str(data["cohort_role"][selected[0]])
        if role not in support_roles:
            continue
        persistent_probability, persistent_safe, persistent_unsafe = {}, {}, {}
        ambiguous, entropy, q50 = {}, {}, {}
        for elapsed, index in zip(boundaries, selected):
            successes = float(data["persistent_successes"][index])
            trials = float(data["persistent_trials"][index])
            if trials < 1 or successes < 0 or successes > trials:
                raise ValueError(f"invalid persistent trial counts: {group_id}@{elapsed}")
            probability = successes / trials
            persistent_probability[elapsed] = probability
            persistent_safe[elapsed] = successes == trials
            persistent_unsafe[elapsed] = successes == 0
            ambiguous[elapsed] = 0 < successes < trials
            entropy[elapsed] = binary_entropy(probability)
            q50[elapsed] = float(data["arm_teacher_step_quantiles"][index, 1, 1])
        source_successes = float(data["source_successes"][selected[0]])
        source_trials = float(data["source_trials"][selected[0]])
        transition_rows = []
        for start, end in transitions:
            transition_rows.append({
                "start": start, "end": end,
                "hard_positive": persistent_safe[start] and persistent_unsafe[end],
                "hard_gain": persistent_unsafe[start] and persistent_safe[end],
                "ambiguous": ambiguous[start] or ambiguous[end],
                "probability_drop": persistent_probability[start] - persistent_probability[end],
            })
        sequence = [persistent_probability[elapsed] for elapsed in boundaries]
        complete_groups.append({
            "group_id": group_id,
            "base_group_id": str(data["base_group_id"][selected[0]]),
            "cohort_role": role,
            "policy_id": str(data["policy_id"][selected[0]]),
            "task_id": str(data["task_id"][selected[0]]),
            "suite": str(data["suite"][selected[0]]),
            "state_key": str(data["state_key"][selected[0]]),
            "source_probability": source_successes / source_trials,
            "persistent_probability": persistent_probability,
            "persistent_certain_safe": persistent_safe,
            "persistent_certain_unsafe": persistent_unsafe,
            "teacher_cost_q50": q50,
            "mean_boundary_entropy_bits": float(np.mean(list(entropy.values()))),
            "transitions": transition_rows,
            "hard_hazard_positive": any(row["hard_positive"] for row in transition_rows),
            "nonmonotonic_gain_after_loss": (sequence[0] > sequence[1]
                                             and sequence[2] > sequence[1]),
        })

    if duplicate_boundaries:
        raise ValueError(f"duplicate boundary rows: {duplicate_boundaries[:3]}")
    transition_records = [
        {**transition, "group_id": group["group_id"], "task_id": group["task_id"],
         "suite": group["suite"], "policy_id": group["policy_id"],
         "cohort_role": group["cohort_role"]}
        for group in complete_groups for transition in group["transitions"]
    ]
    positives = [group for group in complete_groups if group["hard_hazard_positive"]]
    positive_tasks = {group["task_id"] for group in positives}
    positive_by_suite = Counter(group["suite"] for group in positives)
    by_horizon = {
        f"{start}_to_{end}": summarize_transition([
            row for row in transition_records if row["start"] == start and row["end"] == end
        ]) for start, end in transitions
    }
    per_policy_support = {}
    for policy in sorted({group["policy_id"] for group in complete_groups}):
        rows = [group for group in complete_groups if group["policy_id"] == policy]
        positive_rows = [group for group in rows if group["hard_hazard_positive"]]
        per_policy_support[policy] = {
            "complete_groups": len(rows),
            "natural_groups": sum(group["cohort_role"] == "natural" for group in rows),
            "enrichment_groups": sum(group["cohort_role"] == "enrichment" for group in rows),
            "hard_hazard_positive_groups": len(positive_rows),
            "hard_hazard_positive_tasks": len({group["task_id"] for group in positive_rows}),
            "hard_hazard_positive_by_suite": dict(sorted(Counter(
                group["suite"] for group in positive_rows).items())),
            "mean_boundary_entropy_bits": float(np.mean([
                group["mean_boundary_entropy_bits"] for group in rows
            ])) if rows else float("nan"),
            "nonmonotonic_gain_after_loss_fraction": float(np.mean([
                group["nonmonotonic_gain_after_loss"] for group in rows
            ])) if rows else float("nan"),
        }

    gates = config["gates"]
    ambiguous_fraction = float(np.mean([row["ambiguous"] for row in transition_records]))
    support_gate = {
        "complete_groups_at_least_minimum": len(complete_groups)
        >= gates["minimum_complete_support_groups"],
        "hard_hazard_positive_groups_at_least_minimum": len(positives)
        >= gates["minimum_hard_hazard_positive_groups"],
        "hard_hazard_positive_tasks_at_least_minimum": len(positive_tasks)
        >= gates["minimum_hard_hazard_positive_tasks"],
        "each_suite_has_minimum_hard_hazard_positives": all(
            positive_by_suite[suite] >= gates["minimum_hard_hazard_positive_per_suite"]
            for suite in ("Spatial", "Object", "Goal", "Long")
        ),
        "each_horizon_has_minimum_positive_transitions": all(
            by_horizon[f"{start}_to_{end}"]["hard_positive_transitions"]
            >= gates["minimum_positive_transitions_per_horizon"]
            for start, end in transitions
        ),
        "ambiguous_transition_fraction_le_maximum": ambiguous_fraction
        <= gates["maximum_ambiguous_transition_fraction"],
    }
    oracles = oracle_summary(complete_groups, gates)
    oracle_passes = sum(row["status"] == "PASS" for row in oracles.values())
    oracle_gate = {
        "minimum_policy_pairs_pass_natural_oracle": oracle_passes
        >= gates["minimum_policy_pairs_passing_oracle"]
    }
    passed = all(support_gate.values()) and all(oracle_gate.values())
    result = {
        "schema_version": "rase-r8a-recoverability-hazard-audit/v1",
        "status": "PASS" if passed else "FAIL",
        "decision": "UNLOCK_R8A1_PROBABILISTIC_PILOT" if passed else "STOP_BEFORE_COLLECTION",
        "scientific_scope": "model-free opportunity and label-support only; no feature separability claim",
        "config": str(args.config.resolve()), "config_sha256": sha256(args.config),
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
        "dataset_report_sha256": sha256(args.dataset_report),
        "exclusions_sha256": sha256(args.exclusions),
        "readiness_sha256": sha256(args.readiness),
        "input_rows": int(len(data["group_id"])),
        "input_groups": len(group_rows(data)),
        "complete_support_groups": len(complete_groups),
        "incomplete_groups": len(incomplete),
        "incomplete_boundary_distribution": {
            ",".join(map(str, key)): value for key, value in sorted(Counter(
                tuple(row["available_boundaries"]) for row in incomplete
            ).items(), key=lambda item: str(item[0]))
        },
        "hard_hazard_positive_groups": len(positives),
        "hard_hazard_positive_tasks": len(positive_tasks),
        "hard_hazard_positive_by_suite": dict(sorted(positive_by_suite.items())),
        "ambiguous_transition_fraction": ambiguous_fraction,
        "mean_boundary_entropy_bits": float(np.mean([
            group["mean_boundary_entropy_bits"] for group in complete_groups
        ])) if complete_groups else float("nan"),
        "nonmonotonic_gain_after_loss_groups": sum(
            group["nonmonotonic_gain_after_loss"] for group in complete_groups),
        "nonmonotonic_gain_after_loss_fraction": float(np.mean([
            group["nonmonotonic_gain_after_loss"] for group in complete_groups
        ])) if complete_groups else float("nan"),
        "by_horizon": by_horizon,
        "per_policy_support": per_policy_support,
        "natural_cost_aware_oracle": oracles,
        "support_gate": support_gate,
        "oracle_gate": oracle_gate,
        "thresholds": gates,
        "unlocks_on_pass": ["R8-A1 repeated probabilistic boundary pilot only"],
        "remains_locked": [
            "risk model training", "selector training", "world-model features",
            "multi-VLA main claim", "validation", "test",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
