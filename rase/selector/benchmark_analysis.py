"""Read-only descriptive analysis for frozen selector benchmark rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np

from rase.selector.lightweight import ACTIONS

BENCHMARK_ANALYSIS_SCHEMA_VERSION = "rase-selector-benchmark-analysis/v1"
DEFAULT_COMPOSITION_FIELDS = ("cohort", "suite", "episode_outcome", "direct_outcome")
DIRECT_OUTCOME_LABELS = (
    "both_success",
    "smol_only",
    "oft_only",
    "both_fail",
    "missing_smol",
    "missing_oft",
    "missing_both",
)


def _split_index(splits: Mapping[str, Any] | None) -> dict[str, str]:
    if splits is None:
        return {}
    payload = splits.get("splits", splits)
    result: dict[str, str] = {}
    for split, keys in payload.items():
        for raw in keys:
            key = str(raw)
            if key in result:
                raise ValueError(f"state {key} occurs in multiple splits")
            result[key] = str(split)
    return result


def _utility(arm: Mapping[str, Any], success_reward: float) -> float:
    return success_reward * float(bool(arm["success"])) - float(arm["cost"])


def _oracle(row: Mapping[str, Any], success_reward: float) -> tuple[str, float] | None:
    arms = dict(row.get("arms") or {})
    if any(not dict(arms.get(action) or {}).get("observed", False) for action in ACTIONS):
        return None
    utilities = {action: _utility(dict(arms[action]), success_reward) for action in ACTIONS}
    action = max(ACTIONS, key=lambda name: (utilities[name], -ACTIONS.index(name)))
    return action, utilities[action]


def paired_bootstrap_gap(
    differences: Sequence[float], *, seed: int = 20260731, bootstrap_samples: int = 5000
) -> dict[str, Any]:
    """Return a deterministic percentile CI for paired differences."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    values = np.asarray(differences, dtype=np.float64)
    if not len(values):
        return {
            "n_pairs": 0,
            "mean_difference": None,
            "bootstrap_ci_95": {"lower": None, "upper": None},
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "n_pairs": len(values),
        "mean_difference": float(values.mean()),
        "bootstrap_ci_95": {"lower": float(lower), "upper": float(upper)},
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def _direct_outcome_label(row: Mapping[str, Any]) -> str:
    arms = dict(row.get("arms") or {})
    smol = dict(arms.get(ACTIONS[0]) or {})
    oft = dict(arms.get(ACTIONS[1]) or {})
    smol_observed = bool(smol.get("observed", False))
    oft_observed = bool(oft.get("observed", False))
    if not smol_observed and not oft_observed:
        return "missing_both"
    if not smol_observed:
        return "missing_smol"
    if not oft_observed:
        return "missing_oft"
    smol_success = bool(smol.get("success", False))
    oft_success = bool(oft.get("success", False))
    if smol_success and oft_success:
        return "both_success"
    if smol_success:
        return "smol_only"
    if oft_success:
        return "oft_only"
    return "both_fail"


def _composition(
    rows: list[dict[str, Any]], split_by_state: Mapping[str, str], fields: tuple[str, ...]
) -> dict[str, Any]:
    domains = {}
    for field in fields:
        if field == "direct_outcome":
            domains[field] = list(DIRECT_OUTCOME_LABELS)
        else:
            domains[field] = sorted(
                {str(row.get(field) if row.get(field) is not None else "unlabeled") for row in rows}
            )
    split_names = sorted(set(split_by_state.values())) if split_by_state else ["all"]
    counts = Counter()
    for row in rows:
        split = split_by_state.get(str(row["state_key"]), "all")
        cell = tuple(
            _direct_outcome_label(row)
            if field == "direct_outcome"
            else str(row.get(field) if row.get(field) is not None else "unlabeled")
            for field in fields
        )
        counts[(split, *cell)] += 1
    cells = []
    for split, values in product(split_names, product(*(domains[field] for field in fields))):
        cells.append(
            {"split": split, **dict(zip(fields, values)), "n_states": counts[(split, *values)]}
        )
    return {"fields": ["split", *fields], "domains": domains, "cells": cells}


def _direct_outcomes(rows: list[dict[str, Any]], success_reward: float) -> dict[str, Any]:
    result = {}
    for action in ACTIONS:
        observed = [dict((row.get("arms") or {}).get(action) or {}) for row in rows]
        observed = [arm for arm in observed if arm.get("observed", False)]
        successes = sum(bool(arm.get("success", False)) for arm in observed)
        result[action] = {
            "n_observed": len(observed),
            "n_missing": len(rows) - len(observed),
            "n_success": successes,
            "n_failure": len(observed) - successes,
            "success_rate": successes / len(observed) if observed else None,
            "mean_cost": float(np.mean([float(arm["cost"]) for arm in observed]))
            if observed
            else None,
            "mean_utility": (
                float(np.mean([_utility(arm, success_reward) for arm in observed]))
                if observed
                else None
            ),
        }
    return result


def _task_split_warnings(
    rows: list[dict[str, Any]],
    split_by_state: Mapping[str, str],
    reward: float,
) -> list[str]:
    if not split_by_state:
        return []
    task_splits: dict[str, set[str]] = defaultdict(set)
    split_rows: dict[str, list[dict[str, Any]]] = {
        split: [] for split in sorted(set(split_by_state.values()))
    }
    for row in rows:
        split = split_by_state.get(str(row["state_key"]))
        if split is not None:
            task_splits[str(row.get("task_id") or "unlabeled")].add(split)
            split_rows[split].append(row)
    warnings = [
        f"task {task} occurs in multiple splits: {sorted(values)}"
        for task, values in sorted(task_splits.items())
        if len(values) > 1
    ]
    for split, selected in split_rows.items():
        tasks = {str(row.get("task_id") or "unlabeled") for row in selected}
        if not tasks:
            warnings.append(f"split {split} contains no tasks")
        cohort_counts = Counter(str(row.get("cohort") or "unlabeled") for row in selected)
        clean = cohort_counts["clean_control"]
        failure = cohort_counts["failure_challenge"]
        if clean == 0 or failure == 0:
            warnings.append(
                f"split {split} cohort composition: clean_control={clean}, "
                f"failure_challenge={failure}"
            )
        oracle_counts = Counter(
            oracle[0] for row in selected if (oracle := _oracle(row, reward)) is not None
        )
        if oracle_counts[ACTIONS[1]] == 0:
            warnings.append(f"split {split} has 0 escalation oracle support")
        learned = [
            row.get("learned_action") for row in selected if row.get("learned_action") is not None
        ]
        if not learned:
            warnings.append(
                f"split {split} learned action unavailable: no learned_action annotations"
            )
        elif len(learned) != len(selected):
            warnings.append(
                f"split {split} learned action partially available: "
                f"{len(learned)}/{len(selected)} states"
            )
        elif sum(action == ACTIONS[1] for action in learned) == 0:
            warnings.append(f"split {split} has 0 learned escalation actions")
    return warnings


def _evaluate_shortcut(
    rows: list[dict[str, Any]],
    *,
    mapping: Mapping[str, str],
    fallback_action: str,
    reward: float,
) -> dict[str, Any]:
    action_correct = 0
    utilities: list[float] = []
    oracle_utilities: list[float] = []
    evaluable_actions = 0
    unseen = 0
    action_counts = Counter()
    for row in rows:
        suite = str(row.get("suite") or "unlabeled")
        if suite in mapping:
            action = mapping[suite]
        else:
            action = fallback_action
            unseen += 1
        action_counts[action] += 1
        oracle = _oracle(row, reward)
        arm = dict((row.get("arms") or {}).get(action) or {})
        if oracle is None or not arm.get("observed", False):
            continue
        evaluable_actions += 1
        action_correct += int(action == oracle[0])
        utilities.append(_utility(arm, reward))
        oracle_utilities.append(oracle[1])
    return {
        "n_states": len(rows),
        "n_evaluable": evaluable_actions,
        "n_unseen_suite_fallback": unseen,
        "action_counts": {action: action_counts[action] for action in ACTIONS},
        "oracle_action_accuracy": action_correct / evaluable_actions if evaluable_actions else None,
        "mean_utility": float(np.mean(utilities)) if utilities else None,
        "mean_oracle_utility": float(np.mean(oracle_utilities)) if oracle_utilities else None,
        "mean_oracle_utility_gap": (
            float(np.mean(np.asarray(oracle_utilities) - np.asarray(utilities)))
            if utilities
            else None
        ),
    }


def _suite_shortcut(
    train: list[dict[str, Any]],
    evaluation_splits: Mapping[str, list[dict[str, Any]]],
    reward: float,
    preregistered_fallback_action: str,
) -> dict[str, Any]:
    labeled = [(row, _oracle(row, reward)) for row in train]
    labeled = [(row, oracle) for row, oracle in labeled if oracle is not None]
    global_counts = Counter(oracle[0] for _, oracle in labeled)
    global_action = (
        max(ACTIONS, key=lambda action: (global_counts[action], -ACTIONS.index(action)))
        if labeled
        else None
    )
    fallback_action = global_action or preregistered_fallback_action
    suite_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row, oracle in labeled:
        suite_counts[str(row.get("suite") or "unlabeled")][oracle[0]] += 1
    mapping = {
        suite: max(ACTIONS, key=lambda action: (counts[action], -ACTIONS.index(action)))
        for suite, counts in sorted(suite_counts.items())
    }
    return {
        "fit_scope": "train_only",
        "n_train_evaluable": len(labeled),
        "suite_majority_action": mapping,
        "fallback": {
            "semantics": "train_global_majority_then_preregistered",
            "train_global_majority_action": global_action,
            "preregistered_action": preregistered_fallback_action,
            "applied_action": fallback_action,
        },
        "per_split": {
            split: _evaluate_shortcut(
                selected,
                mapping=mapping,
                fallback_action=fallback_action,
                reward=reward,
            )
            for split, selected in evaluation_splits.items()
        },
    }


def _loso_descriptive(
    rows: list[dict[str, Any]], reward: float, preregistered_fallback_action: str
) -> list[dict[str, Any]]:
    folds = []
    for suite in sorted({str(row.get("suite") or "unlabeled") for row in rows}):
        test = [row for row in rows if str(row.get("suite") or "unlabeled") == suite]
        train = [row for row in rows if str(row.get("suite") or "unlabeled") != suite]
        shortcut = _suite_shortcut(
            train,
            {"train": train, "test": test},
            reward,
            preregistered_fallback_action,
        )
        folds.append(
            {
                "held_out_suite": suite,
                "n_train": len(train),
                "n_test": len(test),
                "train_oracle_action_counts": {
                    action: sum(
                        (oracle := _oracle(row, reward)) is not None and oracle[0] == action
                        for row in train
                    )
                    for action in ACTIONS
                },
                "test_oracle_action_counts": {
                    action: sum(
                        (oracle := _oracle(row, reward)) is not None and oracle[0] == action
                        for row in test
                    )
                    for action in ACTIONS
                },
                "test_direct_outcome": _direct_outcomes(test, reward),
                "train_only_suite_shortcut": shortcut,
            }
        )
    return folds


def analyze_selector_benchmark(
    rows: list[Mapping[str, Any]],
    *,
    splits: Mapping[str, Any] | None = None,
    success_reward: float = 1.0,
    bootstrap_seed: int = 20260731,
    bootstrap_samples: int = 5000,
    composition_fields: tuple[str, ...] = DEFAULT_COMPOSITION_FIELDS,
    shortcut_fallback_action: str = ACTIONS[0],
) -> dict[str, Any]:
    """Compute CPU-only benchmark diagnostics; never fit a model or run rollouts."""
    if shortcut_fallback_action not in ACTIONS:
        raise ValueError(f"unsupported shortcut fallback action: {shortcut_fallback_action}")
    required_composition = ("cohort", "suite", "episode_outcome", "direct_outcome")
    if composition_fields != required_composition:
        raise ValueError(f"composition_fields must be {required_composition}")
    materialized = [dict(row) for row in rows]
    keys = [str(row.get("state_key") or "") for row in materialized]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("benchmark rows require unique non-empty state_key values")
    split_by_state = _split_index(splits)
    if split_by_state:
        missing = sorted(set(keys) - set(split_by_state))
        extra = sorted(set(split_by_state) - set(keys))
        if missing or extra:
            raise ValueError(
                f"dataset/split state mismatch: missing={missing[:5]}, extra={extra[:5]}"
            )

    oracle_rows = [(row, _oracle(row, success_reward)) for row in materialized]
    oracle_counts = Counter(result[0] for _, result in oracle_rows if result is not None)
    gaps = {}
    for index, action in enumerate(ACTIONS):
        differences = []
        for row, oracle in oracle_rows:
            arm = dict((row.get("arms") or {}).get(action) or {})
            if oracle is not None and arm.get("observed", False):
                differences.append(oracle[1] - _utility(arm, success_reward))
        gaps[action] = paired_bootstrap_gap(
            differences, seed=bootstrap_seed + index, bootstrap_samples=bootstrap_samples
        )
    return {
        "schema_version": BENCHMARK_ANALYSIS_SCHEMA_VERSION,
        "n_states": len(materialized),
        "composition": _composition(materialized, split_by_state, composition_fields),
        "direct_outcome": _direct_outcomes(materialized, success_reward),
        "oracle_action_support": {
            "n_evaluable": sum(oracle_counts.values()),
            "n_not_evaluable": len(materialized) - sum(oracle_counts.values()),
            "action_counts": {action: oracle_counts[action] for action in ACTIONS},
        },
        "oracle_minus_fixed_policy_utility_gaps": gaps,
        "task_split_composition_warnings": _task_split_warnings(
            materialized, split_by_state, success_reward
        ),
        "train_only_suite_shortcut": _suite_shortcut(
            [
                row
                for row in materialized
                if not split_by_state or split_by_state.get(str(row["state_key"])) == "train"
            ],
            {
                split: [
                    row
                    for row in materialized
                    if split_by_state.get(str(row["state_key"])) == split
                ]
                for split in sorted(set(split_by_state.values()))
            }
            if split_by_state
            else {"all": materialized},
            success_reward,
            shortcut_fallback_action,
        ),
        "leave_suite_out_descriptive_folds": _loso_descriptive(
            materialized, success_reward, shortcut_fallback_action
        ),
    }
