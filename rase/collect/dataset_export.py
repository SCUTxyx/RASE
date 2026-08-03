"""Build selector/QC rows from dual-oracle summaries and frozen artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rase.collect.candidates import CandidateArtifact
from rase.collect.schema import StateMetadata

EXPORT_SCHEMA_VERSION = "rase-recovery-dataset/v1"
BENCHMARK_SPLIT_SCHEMA_VERSION = "rase-recovery-benchmark-splits/v1"
LEAVE_SUITE_OUT_SCHEMA_VERSION = "rase-selector-leave-suite-out-splits/v1"
SPLIT_SUPPORT_AUDIT_SCHEMA_VERSION = "rase-selector-split-support-audit/v1"


def _candidate_gt_index(summary: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in summary.get("per_candidate_gt") or []:
        key = (str(row["state_key"]), int(row["candidate_id"]))
        if key in index:
            raise ValueError(f"duplicate per-candidate GT row: {key}")
        index[key] = dict(row)
    return index


def build_recovery_rows(
    summary: Mapping[str, Any],
    *,
    metadata_for: Callable[[str], StateMetadata],
    artifact_for: Callable[[str], tuple[Path, CandidateArtifact]],
    traces_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return one portable training/QC row per state-candidate pair."""
    gt = _candidate_gt_index(summary)
    rows: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    for state in summary.get("per_state") or []:
        state_key = str(state["state_key"])
        if state_key in seen_states:
            raise ValueError(f"duplicate state row: {state_key}")
        seen_states.add(state_key)
        meta = metadata_for(state_key)
        artifact_path, artifact = artifact_for(state_key)
        for candidate_id in range(int(artifact.actions.shape[0])):
            candidate_gt = gt.get((state_key, candidate_id), {})
            trace_root = (
                traces_dir / state_key / f"c{candidate_id}" if traces_dir is not None else None
            )
            trace_manifests = (
                sorted(trace_root.glob("r*/trace.json"))
                if trace_root is not None and trace_root.is_dir()
                else []
            )
            trace_videos = (
                sorted(trace_root.glob("r*/rollout.mp4"))
                if trace_root is not None and trace_root.is_dir()
                else []
            )
            rows.append(
                {
                    "schema_version": EXPORT_SCHEMA_VERSION,
                    "state_key": state_key,
                    "candidate_id": candidate_id,
                    "candidate_seed": int(artifact.metadata.seeds[candidate_id]),
                    "candidate_temperature": float(artifact.metadata.temperature),
                    "candidate_policy_hash": artifact.metadata.policy_hash,
                    "candidate_artifact": str(artifact_path),
                    "candidate_shape": list(artifact.metadata.shape),
                    "task_id": meta.task_id,
                    "instruction": meta.instruction,
                    "suite": state.get("suite") or meta.suite,
                    "episode_id": meta.episode_id,
                    "perturb_dim": state.get("perturb_dim") or meta.perturb_dim,
                    "perturb_sub": meta.perturb_sub,
                    "level": int(state.get("level") or meta.level),
                    "t0": int(state.get("t0", state.get("step", meta.step))),
                    "episode_outcome": meta.episode_outcome,
                    "state_seed": int(meta.seed),
                    "set_label_smolvla": state.get("set_label_smolvla"),
                    "dual_track_label": state.get(
                        "dual_track_label",
                        state.get("cross_label", state.get("split")),
                    ),
                    "recoverable_smolvla": bool(
                        candidate_gt.get(
                            "recoverable_smolvla",
                            state.get("recoverable_smolvla", False),
                        )
                    ),
                    "recoverable_oft": bool(
                        candidate_gt.get(
                            "recoverable_oft",
                            state.get("recoverable_oft", False),
                        )
                    ),
                    "successes_smolvla": int(candidate_gt.get("successes_smolvla", 0)),
                    "trials_smolvla": int(candidate_gt.get("trials_smolvla", 0)),
                    "successes_oft": int(candidate_gt.get("successes_oft", 0)),
                    "trials_oft": int(candidate_gt.get("trials_oft", 0)),
                    "trace_manifests": [str(path) for path in trace_manifests],
                    "videos": [str(path) for path in trace_videos],
                }
            )
    return rows


def split_state_keys(rows: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Group unique state keys by dual-track label for release splits."""
    groups: dict[str, set[str]] = {}
    for row in rows:
        label = str(row.get("dual_track_label") or "unlabeled")
        groups.setdefault(label, set()).add(str(row["state_key"]))
    return {label: sorted(keys) for label, keys in sorted(groups.items())}


def _normalized_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    if not fractions:
        raise ValueError("at least one split fraction is required")
    normalized = {str(name): float(value) for name, value in fractions.items()}
    if any(not name for name in normalized):
        raise ValueError("split names must be non-empty")
    if any(value <= 0 for value in normalized.values()):
        raise ValueError("split fractions must be positive")
    total = sum(normalized.values())
    return {name: value / total for name, value in normalized.items()}


def _state_rows(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collapse candidate rows while rejecting inconsistent state metadata."""
    fields = (
        "task_id",
        "episode_id",
        "suite",
        "perturb_dim",
        "perturb_sub",
        "level",
        "dual_track_label",
        "episode_outcome",
        "cohort",
    )
    states: dict[str, dict[str, Any]] = {}
    for raw in rows:
        key = str(raw["state_key"])
        row = {field: raw.get(field) for field in fields}
        missing = [field for field in ("task_id", "episode_id") if not row.get(field)]
        if missing:
            raise ValueError(f"state {key!r} missing grouping field(s): {', '.join(missing)}")
        previous = states.get(key)
        if previous is not None and previous != row:
            raise ValueError(f"inconsistent metadata across candidate rows for state {key!r}")
        states[key] = row
    return states


def _stable_tiebreak(seed: int, *parts: str) -> int:
    payload = json.dumps([seed, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_grouped_benchmark_splits(
    rows: list[Mapping[str, Any]],
    *,
    fractions: Mapping[str, float] | None = None,
    seed: int = 20260727,
    group_fields: tuple[str, ...] = ("task_id", "episode_id"),
    stratify_fields: tuple[str, ...] = (
        "suite",
        "perturb_dim",
        "level",
        "dual_track_label",
    ),
) -> dict[str, Any]:
    """Create deterministic group-disjoint splits with greedy stratification.

    Candidate rows are first collapsed to unique states. All snapshots from an
    episode are assigned together, preventing temporal leakage between train,
    validation, and test. The greedy objective balances both split size and the
    configured research strata; exact balance is not promised for indivisible
    episode groups, so the returned audit exposes every resulting count.
    """
    split_fractions = _normalized_fractions(fractions or {"train": 0.70, "val": 0.15, "test": 0.15})
    if not group_fields:
        raise ValueError("group_fields must not be empty")
    if not stratify_fields:
        raise ValueError("stratify_fields must not be empty")
    states = _state_rows(rows)
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for state_key, row in states.items():
        group = tuple(str(row.get(field, "")) for field in group_fields)
        if any(not value for value in group):
            raise ValueError(f"state {state_key!r} has an empty group field")
        groups[group].append(state_key)

    strata_by_state = {
        key: tuple(str(row.get(field, "")) for field in stratify_fields)
        for key, row in states.items()
    }
    overall_strata = Counter(strata_by_state.values())
    target_sizes = {name: len(states) * fraction for name, fraction in split_fractions.items()}
    target_strata = {
        name: {cell: count * split_fractions[name] for cell, count in overall_strata.items()}
        for name in split_fractions
    }
    assigned: dict[str, list[str]] = {name: [] for name in split_fractions}
    assigned_strata: dict[str, Counter[tuple[str, ...]]] = {
        name: Counter() for name in split_fractions
    }

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            _stable_tiebreak(seed, *item[0]),
            item[0],
        ),
    )
    for group, state_keys in ordered_groups:
        group_counts = Counter(strata_by_state[key] for key in state_keys)
        scores: list[tuple[float, int, str]] = []
        for candidate_split in split_fractions:
            size_score = 0.0
            stratum_score = 0.0
            stratum_terms = 0
            for split in split_fractions:
                addition = len(state_keys) if split == candidate_split else 0
                size_after = len(assigned[split]) + addition
                size_scale = max(target_sizes[split], 1.0)
                size_score += ((size_after - target_sizes[split]) / size_scale) ** 2
                for cell, target in sorted(target_strata[split].items()):
                    cell_addition = group_counts[cell] if split == candidate_split else 0
                    after = assigned_strata[split][cell] + cell_addition
                    stratum_score += ((after - target) / max(target, 1.0)) ** 2
                    stratum_terms += 1
            # A mean stratum error prevents a large Cartesian grid from
            # overwhelming the primary split-size objective.
            score = size_score + stratum_score / max(stratum_terms, 1)
            scores.append(
                (
                    score,
                    _stable_tiebreak(seed, *group, candidate_split),
                    candidate_split,
                )
            )
        chosen = min(scores)[2]
        assigned[chosen].extend(state_keys)
        assigned_strata[chosen].update(group_counts)

    state_to_split: dict[str, str] = {}
    group_to_split: dict[tuple[str, ...], str] = {}
    for split, state_keys in assigned.items():
        for key in state_keys:
            if key in state_to_split:
                raise AssertionError(f"state assigned twice: {key}")
            state_to_split[key] = split
            group = tuple(str(states[key][field]) for field in group_fields)
            previous = group_to_split.setdefault(group, split)
            if previous != split:
                raise AssertionError(f"group leaked across splits: {group}")
    if set(state_to_split) != set(states):
        raise AssertionError("not all states were assigned")

    def cell_rows(split: str) -> list[dict[str, Any]]:
        return [
            {
                **dict(zip(stratify_fields, cell)),
                "n_states": count,
            }
            for cell, count in sorted(assigned_strata[split].items())
        ]

    return {
        "schema_version": BENCHMARK_SPLIT_SCHEMA_VERSION,
        "seed": int(seed),
        "fractions": split_fractions,
        "group_fields": list(group_fields),
        "stratify_fields": list(stratify_fields),
        "splits": {name: sorted(keys) for name, keys in assigned.items()},
        "audit": {
            "n_rows": len(rows),
            "n_states": len(states),
            "n_groups": len(groups),
            "group_leakage": False,
            "per_split": {
                name: {
                    "n_states": len(assigned[name]),
                    "n_groups": sum(value == name for value in group_to_split.values()),
                    "strata": cell_rows(name),
                }
                for name in split_fractions
            },
        },
    }


def build_leave_one_suite_out_splits(
    rows: list[Mapping[str, Any]],
    *,
    group_fields: tuple[str, ...] = ("task_id", "episode_id"),
) -> dict[str, Any]:
    """Build deterministic folds with exactly one complete suite held out."""
    if not group_fields:
        raise ValueError("group_fields must not be empty")
    states = _state_rows(rows)
    group_suites: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for key, row in states.items():
        suite = str(row.get("suite") or "")
        if not suite:
            raise ValueError(f"state {key!r} missing suite")
        group = tuple(str(row.get(field) or "") for field in group_fields)
        if any(not value for value in group):
            raise ValueError(f"state {key!r} has an empty group field")
        group_suites[group].add(suite)
    mixed = sorted(group for group, suites in group_suites.items() if len(suites) > 1)
    if mixed:
        raise ValueError(f"group spans multiple suites: {mixed[:5]}")

    suites = sorted({str(row["suite"]) for row in states.values()})
    folds: dict[str, dict[str, Any]] = {}
    for suite in suites:
        test = sorted(key for key, row in states.items() if str(row["suite"]) == suite)
        train = sorted(set(states) - set(test))
        folds[suite] = {
            "held_out_suite": suite,
            "splits": {"train": train, "test": test},
            "audit": {"n_train": len(train), "n_test": len(test), "group_leakage": False},
        }
    return {
        "schema_version": LEAVE_SUITE_OUT_SCHEMA_VERSION,
        "group_fields": list(group_fields),
        "n_states": len(states),
        "suites": suites,
        "folds": folds,
    }


def audit_split_support(
    rows: list[Mapping[str, Any]],
    splits: Mapping[str, Any],
    *,
    requirements: Mapping[str, Any] | None = None,
    success_reward: float = 1.0,
) -> dict[str, Any]:
    """Audit per-split support without changing frozen assignments.

    New per-split requirements accept either a scalar/list applied to every
    split or a mapping keyed by split. Count requirements may additionally map
    a split to per-label/per-action minima. Legacy train-only requirement keys
    remain supported and are applied in addition to the new generic gates.
    Episode groups are always defined as ``(task_id, episode_id)``.
    """
    from rase.selector.lightweight import ACTIONS

    defaults: dict[str, Any] = {
        "required_splits": ["train", "val", "test"],
        "min_states_per_split": 1,
        "min_train_states": 1,
        "min_train_suites": 1,
        "min_train_optimal_actions": 2,
        "min_train_arm_observed": 1,
        "min_train_arm_successes": 0,
        "min_train_arm_failures": 0,
        "required_cohorts": [],
        "min_states_per_cohort": 0,
        "required_suites": [],
        "min_states_per_suite": 0,
        "min_episode_groups_per_split": 0,
        "min_optimal_actions_per_split": 0,
        "required_optimal_actions": [],
        "min_states_per_optimal_action": 1,
        "min_arm_observed_per_split": 0,
        "min_arm_successes_per_split": 0,
        "min_arm_failures_per_split": 0,
    }
    supplied = dict(requirements or {})
    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        raise ValueError(f"unknown split support requirement(s): {unknown}")
    req = {**defaults, **supplied}

    def split_value(value: Any, split: str, default: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return value.get(split, default)

    def required_labels(value: Any, split: str) -> list[str]:
        selected = split_value(value, split, [])
        if selected is None:
            return []
        if isinstance(selected, str):
            return [selected]
        if isinstance(selected, Mapping):
            return [str(label) for label in selected]
        return [str(label) for label in selected]

    def label_minimum(value: Any, split: str, label: str, default: int = 0) -> int:
        selected = split_value(value, split, default)
        if isinstance(selected, Mapping):
            return int(selected.get(label, default))
        return int(selected)

    def action_minimum(value: Any, split: str, action: str) -> int:
        if not isinstance(value, Mapping):
            return int(value)
        selected = value.get(split, value.get(action, 0))
        if isinstance(selected, Mapping):
            return int(selected.get(action, 0))
        return int(selected)

    states = _state_rows(rows)
    payload = splits.get("splits", splits)
    if not isinstance(payload, Mapping):
        raise ValueError("splits must contain a mapping")
    split_keys = {str(name): [str(key) for key in keys] for name, keys in payload.items()}
    reasons: list[str] = []
    seen: dict[str, str] = {}
    for name, keys in split_keys.items():
        for key in keys:
            if key in seen:
                reasons.append(f"state {key} occurs in both {seen[key]} and {name}")
            seen[key] = name
    missing = sorted(set(states) - set(seen))
    extra = sorted(set(seen) - set(states))
    if missing:
        reasons.append(f"dataset states missing from splits: {missing[:5]}")
    if extra:
        reasons.append(f"split states missing from dataset: {extra[:5]}")
    required_splits = [str(name) for name in req["required_splits"]]
    for name in required_splits:
        if name not in split_keys:
            reasons.append(f"required split {name!r} is missing")

    row_by_state = {str(row["state_key"]): row for row in rows}
    cohort_domain = sorted({str(row.get("cohort") or "unlabeled") for row in states.values()})
    suite_domain = sorted({str(row.get("suite") or "unlabeled") for row in states.values()})
    episode_group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    per_split: dict[str, Any] = {}
    for name, keys in sorted(split_keys.items()):
        valid = [key for key in keys if key in states]
        minimum = int(split_value(req["min_states_per_split"], name, 0))
        if len(valid) < minimum:
            reasons.append(f"split {name} has {len(valid)} states; requires at least {minimum}")

        cohort_counts = Counter(str(states[key].get("cohort") or "unlabeled") for key in valid)
        suite_counts = Counter(str(states[key].get("suite") or "unlabeled") for key in valid)
        episode_group_counts: Counter[tuple[str, str]] = Counter()
        arm_counts = {action: {"observed": 0, "success": 0, "failure": 0} for action in ACTIONS}
        optimal = Counter()
        for key in valid:
            state = states[key]
            group = (str(state.get("task_id") or ""), str(state.get("episode_id") or ""))
            episode_group_counts[group] += 1
            episode_group_splits[group].add(name)
            arms = dict(row_by_state[key].get("arms") or {})
            utilities: dict[str, float] = {}
            for action in ACTIONS:
                arm = dict(arms.get(action) or {})
                if not arm.get("observed", False):
                    continue
                arm_counts[action]["observed"] += 1
                hit = bool(arm.get("success", False))
                arm_counts[action]["success" if hit else "failure"] += 1
                try:
                    utilities[action] = success_reward * float(hit) - float(arm["cost"])
                except (KeyError, TypeError, ValueError):
                    reasons.append(f"state {key} has invalid {action} outcome")
            if len(utilities) == len(ACTIONS):
                best = max(ACTIONS, key=lambda action: (utilities[action], -ACTIONS.index(action)))
                optimal[best] += 1

        for cohort in required_labels(req["required_cohorts"], name):
            required = label_minimum(req["min_states_per_cohort"], name, cohort, 1)
            if cohort_counts[cohort] < required:
                reasons.append(
                    f"split {name} cohort {cohort!r} has {cohort_counts[cohort]} states; "
                    f"requires at least {required}"
                )
        for suite in required_labels(req["required_suites"], name):
            required = label_minimum(req["min_states_per_suite"], name, suite, 1)
            if suite_counts[suite] < required:
                reasons.append(
                    f"split {name} suite {suite!r} has {suite_counts[suite]} states; "
                    f"requires at least {required}"
                )
        min_groups = int(split_value(req["min_episode_groups_per_split"], name, 0))
        if len(episode_group_counts) < min_groups:
            reasons.append(
                f"split {name} has {len(episode_group_counts)} episode groups; "
                f"requires at least {min_groups} (group=task_id+episode_id)"
            )
        supported_optimal = sum(count > 0 for count in optimal.values())
        min_optimal = int(split_value(req["min_optimal_actions_per_split"], name, 0))
        if supported_optimal < min_optimal:
            reasons.append(
                f"split {name} has {supported_optimal} supported optimal actions; "
                f"requires at least {min_optimal}"
            )
        for action in required_labels(req["required_optimal_actions"], name):
            if action not in ACTIONS:
                raise ValueError(f"unsupported required optimal action: {action}")
            required = label_minimum(req["min_states_per_optimal_action"], name, action, 1)
            if optimal[action] < required:
                reasons.append(
                    f"split {name} optimal action {action} has {optimal[action]} states; "
                    f"requires at least {required}"
                )
        for action, counts in arm_counts.items():
            for metric, requirement_name in (
                ("observed", "min_arm_observed_per_split"),
                ("success", "min_arm_successes_per_split"),
                ("failure", "min_arm_failures_per_split"),
            ):
                required = action_minimum(req[requirement_name], name, action)
                if counts[metric] < required:
                    reasons.append(
                        f"split {name} {action} {metric} support is {counts[metric]}; "
                        f"requires at least {required}"
                    )
        per_split[name] = {
            "n_states": len(valid),
            "cohort_counts": {cohort: cohort_counts[cohort] for cohort in cohort_domain},
            "cohorts": sorted(cohort_counts),
            "suite_counts": {suite: suite_counts[suite] for suite in suite_domain},
            "suites": sorted(suite_counts),
            "n_suites": len(suite_counts),
            "episode_group_fields": ["task_id", "episode_id"],
            "episode_group_counts": [
                {"task_id": group[0], "episode_id": group[1], "n_states": count}
                for group, count in sorted(episode_group_counts.items())
            ],
            "n_episode_groups": len(episode_group_counts),
            "arm_counts": arm_counts,
            "optimal_action_counts": {action: optimal[action] for action in ACTIONS},
            "n_supported_optimal_actions": supported_optimal,
        }

    leaking_groups = sorted(
        group for group, assigned in episode_group_splits.items() if len(assigned) > 1
    )
    if leaking_groups:
        reasons.append(
            f"episode groups leak across splits: {leaking_groups[:5]} (group=task_id+episode_id)"
        )

    # Legacy train-only gates remain additive and preserve old requirements.
    train = per_split.get("train", {})
    if int(train.get("n_states", 0)) < int(req["min_train_states"]):
        reasons.append(
            f"train split has {train.get('n_states', 0)} states; "
            f"requires at least {req['min_train_states']}"
        )
    if int(train.get("n_suites", 0)) < int(req["min_train_suites"]):
        reasons.append(
            f"train split has {train.get('n_suites', 0)} suites; "
            f"requires at least {req['min_train_suites']}"
        )
    if int(train.get("n_supported_optimal_actions", 0)) < int(req["min_train_optimal_actions"]):
        reasons.append(
            f"train has {train.get('n_supported_optimal_actions', 0)} supported optimal actions; "
            f"requires at least {req['min_train_optimal_actions']}"
        )
    train_arms = train.get("arm_counts", {})
    for action in ACTIONS:
        counts = train_arms.get(action, {"observed": 0, "success": 0, "failure": 0})
        for metric, requirement_name in (
            ("observed", "min_train_arm_observed"),
            ("success", "min_train_arm_successes"),
            ("failure", "min_train_arm_failures"),
        ):
            required = int(req[requirement_name])
            if counts[metric] < required:
                reasons.append(
                    f"train {action} {metric} support is {counts[metric]}; "
                    f"requires at least {required}"
                )

    reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": SPLIT_SUPPORT_AUDIT_SCHEMA_VERSION,
        "status": "READY" if not reasons else "NOT_READY",
        "ready": not reasons,
        "reasons": reasons,
        "requirements": req,
        "episode_group_fields": ["task_id", "episode_id"],
        "episode_group_leakage": bool(leaking_groups),
        "per_split": per_split,
        # Preserve the old convenience field while making per_split canonical.
        "train": per_split.get("train", {}),
    }
