"""Dependency-free baseline for cost-sensitive policy escalation.

The module deliberately separates *diagnostic portfolio matrices* from
deployable action outcomes.  An any-of-K portfolio hit is useful for measuring
policy-relative recoverability, but is not the outcome of a selector action.
Rows derived from such matrices are therefore marked as proxies and fail the
default training-readiness gate.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CONTINUE_SMOL = "continue_smol"
ESCALATE_OFT = "escalate_oft"
ABSTAIN = "abstain"
ACTIONS = (CONTINUE_SMOL, ESCALATE_OFT, ABSTAIN)
SELECTOR_DATASET_SCHEMA_VERSION = "rase-escalation-dataset/v1"
SELECTOR_MODEL_SCHEMA_VERSION = "rase-lightweight-selector/v1"
DEFAULT_SMOL_COST = 0.02
DEFAULT_OFT_COST = 0.10
DEFAULT_ABSTAIN_COST = 0.0
FORBIDDEN_DEPLOYMENT_FEATURES = frozenset(
    {
        "cohort",
        "episode_outcome",
        "level",
        "outcome",
        "perturb_dim",
        "perturb_level",
        "perturb_sub",
        "success",
    }
)


def _utility(outcome: dict[str, Any], success_reward: float) -> float:
    return success_reward * float(bool(outcome["success"])) - float(outcome["cost"])


def build_policy_matrix_proxy_rows(
    matrix: dict[str, Any],
    *,
    metadata_by_state: dict[str, dict[str, Any]],
    smol_cost: float = DEFAULT_SMOL_COST,
    oft_cost: float = DEFAULT_OFT_COST,
    abstain_cost: float = DEFAULT_ABSTAIN_COST,
) -> list[dict[str, Any]]:
    """Convert a paired matrix to diagnostic rows, never silent train labels.

    Both continuation arms are tagged ``proxy=True`` because the matrix uses an
    oracle any-of-K portfolio.  A later direct-arm rollout may replace these
    outcomes and clear the proxy flag.
    """
    if matrix.get("schema_version") != "rase-one-shot-policy-matrix/v1":
        raise ValueError("unsupported policy matrix schema")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in matrix.get("per_state") or []:
        key = str(raw["state_key"])
        if key in seen:
            raise ValueError(f"duplicate state row: {key}")
        seen.add(key)
        meta = dict(metadata_by_state.get(key) or {})
        if not meta:
            raise ValueError(f"metadata missing for state {key}")
        rows.append(
            {
                "schema_version": SELECTOR_DATASET_SCHEMA_VERSION,
                "state_key": key,
                "task_id": meta.get("task_id"),
                "episode_id": meta.get("episode_id"),
                "suite": meta.get("suite", raw.get("suite")),
                "perturb_dim": meta.get("perturb_dim", raw.get("dim")),
                "perturb_sub": meta.get("perturb_sub"),
                "level": int(meta.get("level", raw.get("level", 0))),
                "t0": int(meta.get("step", meta.get("t0", 0))),
                "episode_outcome": meta.get("episode_outcome"),
                "cohort": matrix.get("cohort"),
                "features": {"t0": float(meta.get("step", meta.get("t0", 0)))},
                "arms": {
                    CONTINUE_SMOL: {
                        "success": bool(raw["smol_portfolio_hit"]),
                        "cost": float(smol_cost),
                        "observed": True,
                        "proxy": True,
                        "outcome_semantics": "oracle_any_of_k_portfolio",
                    },
                    ESCALATE_OFT: {
                        "success": bool(raw["oft_portfolio_hit"]),
                        "cost": float(oft_cost),
                        "observed": True,
                        "proxy": True,
                        "outcome_semantics": "oracle_any_of_k_portfolio",
                    },
                    ABSTAIN: {
                        "success": False,
                        "cost": float(abstain_cost),
                        "observed": True,
                        "proxy": False,
                        "outcome_semantics": "defined_termination",
                    },
                },
            }
        )
    return rows


def build_direct_escalation_rows(
    smol_summary: dict[str, Any],
    direct_oft_summaries: list[dict[str, Any]],
    *,
    metadata_by_state: dict[str, dict[str, Any]],
    candidate_index: int = 0,
    smol_cost: float = DEFAULT_SMOL_COST,
    oft_cost: float = DEFAULT_OFT_COST,
    abstain_cost: float = DEFAULT_ABSTAIN_COST,
    cohort: str = "failure_challenge",
    features_by_state: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Build deployable fixed-Smol versus direct-OFT action outcomes."""
    smol_rows = {
        str(row["state_key"]): row for row in smol_summary.get("per_state") or []
    }
    if len(smol_rows) != len(smol_summary.get("per_state") or []):
        raise ValueError("duplicate state in Smol summary")
    oft_rows: dict[str, dict[str, Any]] = {}
    for summary in direct_oft_summaries:
        if summary.get("schema_version") != "rase-oft-direct-escalation/v1":
            raise ValueError("unexpected direct OFT summary schema")
        if summary.get("status") != "complete":
            raise ValueError("direct OFT summary is incomplete")
        for raw in summary.get("per_state") or []:
            key = str(raw["state_key"])
            if key in oft_rows:
                raise ValueError(f"duplicate direct OFT state: {key}")
            oft_rows[key] = dict(raw)
    expected = set(metadata_by_state)
    if set(smol_rows) != expected or set(oft_rows) != expected:
        raise ValueError("Smol/direct-OFT/metadata state unions do not match")

    rows: list[dict[str, Any]] = []
    for key in sorted(expected):
        meta = dict(metadata_by_state[key])
        candidates = list(smol_rows[key].get("candidates") or [])
        if not 0 <= candidate_index < len(candidates):
            raise ValueError(f"state {key} lacks Smol candidate {candidate_index}")
        chosen = dict(candidates[candidate_index])
        if int(chosen.get("trials", 0)) != 1:
            raise ValueError(f"state {key} Smol candidate is not one-shot")
        rows.append(
            {
                "schema_version": SELECTOR_DATASET_SCHEMA_VERSION,
                "state_key": key,
                "task_id": meta.get("task_id"),
                "episode_id": meta.get("episode_id"),
                "suite": meta.get("suite"),
                "perturb_dim": meta.get("perturb_dim"),
                "perturb_sub": meta.get("perturb_sub"),
                "level": int(meta.get("level", 0)),
                "t0": int(meta.get("step", meta.get("t0", 0))),
                "episode_outcome": meta.get("episode_outcome"),
                "cohort": cohort,
                "features": dict(
                    (features_by_state or {}).get(
                        key, {"t0": float(meta.get("step", meta.get("t0", 0)))}
                    )
                ),
                "arms": {
                    CONTINUE_SMOL: {
                        "success": bool(int(chosen.get("successes", 0))),
                        "cost": float(smol_cost),
                        "observed": True,
                        "proxy": False,
                        "outcome_semantics": (
                            f"frozen_candidate_{candidate_index}_then_smol_continuation"
                        ),
                    },
                    ESCALATE_OFT: {
                        "success": bool(oft_rows[key]["direct_oft_success"]),
                        "cost": float(oft_cost),
                        "observed": True,
                        "proxy": False,
                        "outcome_semantics": "direct_oft_from_snapshot",
                    },
                    ABSTAIN: {
                        "success": False,
                        "cost": float(abstain_cost),
                        "observed": True,
                        "proxy": False,
                        "outcome_semantics": "defined_termination",
                    },
                },
            }
        )
    return rows


def build_direct_policy_rows(
    direct_smol_summary: dict[str, Any],
    direct_oft_summaries: list[dict[str, Any]],
    *,
    metadata_by_state: dict[str, dict[str, Any]],
    features_by_state: dict[str, dict[str, float]],
    smol_cost: float = DEFAULT_SMOL_COST,
    oft_cost: float = DEFAULT_OFT_COST,
    abstain_cost: float = DEFAULT_ABSTAIN_COST,
    cohort: str,
) -> list[dict[str, Any]]:
    """Build deployable direct-Smol/direct-OFT/abstain counterfactual rows."""
    if direct_smol_summary.get("schema_version") != "rase-smol-direct-continuation/v1":
        raise ValueError("unexpected direct Smol summary schema")
    if direct_smol_summary.get("status") != "complete":
        raise ValueError("direct Smol summary is incomplete")
    smol_rows = {
        str(row["state_key"]): row
        for row in direct_smol_summary.get("per_state") or []
    }
    if len(smol_rows) != len(direct_smol_summary.get("per_state") or []):
        raise ValueError("duplicate direct Smol state")
    oft_rows: dict[str, dict[str, Any]] = {}
    for summary in direct_oft_summaries:
        if summary.get("schema_version") != "rase-oft-direct-escalation/v1":
            raise ValueError("unexpected direct OFT summary schema")
        if summary.get("status") != "complete":
            raise ValueError("direct OFT summary is incomplete")
        for raw in summary.get("per_state") or []:
            key = str(raw["state_key"])
            if key in oft_rows:
                raise ValueError(f"duplicate direct OFT state: {key}")
            oft_rows[key] = dict(raw)
    expected = set(metadata_by_state)
    if set(smol_rows) != expected or set(oft_rows) != expected:
        raise ValueError("direct-Smol/direct-OFT/metadata state unions do not match")
    if set(features_by_state) != expected:
        raise ValueError("selector feature state union does not match outcomes")
    costs = (float(smol_cost), float(oft_cost), float(abstain_cost))
    if any(value < 0 for value in costs):
        raise ValueError("selector action costs must be non-negative")

    rows: list[dict[str, Any]] = []
    for key in sorted(expected):
        meta = dict(metadata_by_state[key])
        rows.append(
            {
                "schema_version": SELECTOR_DATASET_SCHEMA_VERSION,
                "state_key": key,
                "task_id": meta.get("task_id"),
                "episode_id": meta.get("episode_id"),
                "suite": meta.get("suite"),
                "perturb_dim": meta.get("perturb_dim"),
                "perturb_sub": meta.get("perturb_sub"),
                "level": int(meta.get("level", 0)),
                "t0": int(meta.get("step", meta.get("t0", 0))),
                "episode_outcome": meta.get("episode_outcome"),
                "cohort": cohort,
                "features": dict(features_by_state[key]),
                "arms": {
                    CONTINUE_SMOL: {
                        "success": bool(smol_rows[key]["direct_smol_success"]),
                        "cost": costs[0],
                        "observed": True,
                        "proxy": False,
                        "outcome_semantics": "direct_smol_from_snapshot",
                    },
                    ESCALATE_OFT: {
                        "success": bool(oft_rows[key]["direct_oft_success"]),
                        "cost": costs[1],
                        "observed": True,
                        "proxy": False,
                        "outcome_semantics": "direct_oft_from_snapshot",
                    },
                    ABSTAIN: {
                        "success": False,
                        "cost": costs[2],
                        "observed": True,
                        "proxy": False,
                        "outcome_semantics": "defined_termination",
                    },
                },
            }
        )
    return rows


@dataclass(frozen=True)
class DatasetAudit:
    ready: bool
    reasons: list[str]
    n_rows: int
    n_train: int
    n_groups: int
    group_leakage: bool
    cohort_counts: dict[str, int]
    episode_outcome_counts: dict[str, int]
    arm_counts: dict[str, dict[str, int]]
    optimal_action_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_index(splits: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    payload = splits.get("splits", splits)
    state_to_split: dict[str, str] = {}
    duplicate: list[str] = []
    for split, keys in payload.items():
        for raw_key in keys:
            key = str(raw_key)
            if key in state_to_split:
                duplicate.append(key)
            state_to_split[key] = str(split)
    return state_to_split, sorted(set(duplicate))


def audit_selector_dataset(
    rows: list[dict[str, Any]],
    splits: dict[str, Any],
    *,
    success_reward: float = 1.0,
    min_train_states: int = 30,
) -> DatasetAudit:
    """Audit leakage, arm observability, proxy labels, and label support."""
    reasons: list[str] = []
    state_to_split, split_duplicates = _split_index(splits)
    if split_duplicates:
        reasons.append(f"state keys occur in multiple splits: {split_duplicates[:5]}")
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("state_key", ""))
        if not key:
            reasons.append("row missing state_key")
            continue
        if key in keyed:
            reasons.append(f"duplicate dataset state: {key}")
        keyed[key] = row
    missing_split = sorted(set(keyed) - set(state_to_split))
    extra_split = sorted(set(state_to_split) - set(keyed))
    if missing_split:
        reasons.append(f"dataset states missing from splits: {missing_split[:5]}")
    if extra_split:
        reasons.append(f"split states missing from dataset: {extra_split[:5]}")

    group_splits: dict[tuple[str, str], set[str]] = {}
    cohort_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    arm_counts = {
        action: {"observed": 0, "success": 0, "failure": 0, "proxy": 0}
        for action in ACTIONS
    }
    optimal_counts = {action: 0 for action in ACTIONS}
    train_rows = [
        row
        for key, row in keyed.items()
        if state_to_split.get(key) == "train"
    ]
    for key, row in keyed.items():
        task_id = str(row.get("task_id") or "")
        episode_id = str(row.get("episode_id") or "")
        if not task_id or not episode_id:
            reasons.append(f"state {key} missing task_id/episode_id")
        else:
            group_splits.setdefault((task_id, episode_id), set()).add(
                state_to_split.get(key, "unassigned")
            )
        cohort = str(row.get("cohort") or "unlabeled")
        cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1
        outcome = str(row.get("episode_outcome") or "unlabeled")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        features = row.get("features")
        if not isinstance(features, dict) or not features:
            reasons.append(f"state {key} has no deployable numeric features")
        elif any(
            not isinstance(value, (int, float)) or not np.isfinite(float(value))
            for value in features.values()
        ):
            reasons.append(f"state {key} has non-finite/non-numeric features")
        else:
            forbidden = sorted(set(features) & FORBIDDEN_DEPLOYMENT_FEATURES)
            if forbidden:
                reasons.append(
                    f"state {key} uses forbidden deployment features: {forbidden}"
                )

    leaking_groups = [group for group, values in group_splits.items() if len(values) > 1]
    if leaking_groups:
        reasons.append(f"episode groups leak across splits: {leaking_groups[:5]}")
    if len(train_rows) < min_train_states:
        reasons.append(
            f"train split has {len(train_rows)} states; requires at least {min_train_states}"
        )

    for row in train_rows:
        utilities: dict[str, float] = {}
        arms = dict(row.get("arms") or {})
        for action in ACTIONS:
            arm = dict(arms.get(action) or {})
            if not arm.get("observed", False):
                continue
            arm_counts[action]["observed"] += 1
            hit = bool(arm.get("success", False))
            arm_counts[action]["success" if hit else "failure"] += 1
            arm_counts[action]["proxy"] += int(bool(arm.get("proxy", False)))
            try:
                utilities[action] = _utility(arm, success_reward)
            except (KeyError, TypeError, ValueError):
                reasons.append(f"state {row.get('state_key')} has invalid {action} outcome")
        if len(utilities) == len(ACTIONS):
            best = max(ACTIONS, key=lambda action: (utilities[action], -ACTIONS.index(action)))
            optimal_counts[best] += 1

    for action in ACTIONS:
        counts = arm_counts[action]
        if counts["observed"] != len(train_rows):
            reasons.append(f"{action} is not observed for every train state")
        if counts["proxy"]:
            reasons.append(f"{action} has {counts['proxy']} proxy outcomes in train")
    for action in (CONTINUE_SMOL, ESCALATE_OFT):
        counts = arm_counts[action]
        if counts["success"] == 0 or counts["failure"] == 0:
            reasons.append(f"{action} lacks both success and failure support in train")
    if sum(value > 0 for value in optimal_counts.values()) < 2:
        reasons.append("train optimal-action labels collapse to fewer than two actions")

    def is_clean_success(row: dict[str, Any]) -> bool:
        return (
            str(row.get("cohort")) == "clean_control"
            and str(row.get("episode_outcome")) == "success"
        )

    def is_failure_challenge(row: dict[str, Any]) -> bool:
        return (
            str(row.get("cohort")) == "failure_challenge"
            and str(row.get("episode_outcome")) == "failure"
        )

    for row in keyed.values():
        cohort = str(row.get("cohort"))
        outcome = str(row.get("episode_outcome"))
        if cohort == "clean_control" and outcome != "success":
            reasons.append(
                f"state {row.get('state_key')} is a clean_control without success outcome"
            )
        if cohort == "failure_challenge" and outcome != "failure":
            reasons.append(
                f"state {row.get('state_key')} is a failure_challenge without failure outcome"
            )

    n_clean_success = sum(is_clean_success(row) for row in keyed.values())
    n_failure_challenge = sum(is_failure_challenge(row) for row in keyed.values())
    n_train_clean_success = sum(is_clean_success(row) for row in train_rows)
    n_train_failure_challenge = sum(is_failure_challenge(row) for row in train_rows)
    if n_clean_success == 0:
        reasons.append("dataset has no clean-success control states")
    if n_failure_challenge == 0:
        reasons.append("dataset has no failure-challenge states")
    if n_train_clean_success == 0:
        reasons.append("train split has no clean-success control states")
    if n_train_failure_challenge == 0:
        reasons.append("train split has no failure-challenge states")

    # Preserve order while removing repeated row-level messages.
    reasons = list(dict.fromkeys(reasons))
    return DatasetAudit(
        ready=not reasons,
        reasons=reasons,
        n_rows=len(keyed),
        n_train=len(train_rows),
        n_groups=len(group_splits),
        group_leakage=bool(leaking_groups),
        cohort_counts=dict(sorted(cohort_counts.items())),
        episode_outcome_counts=dict(sorted(outcome_counts.items())),
        arm_counts=arm_counts,
        optimal_action_counts=optimal_counts,
    )


@dataclass
class LightweightSelector:
    numeric_features: list[str]
    categorical_fields: list[str]
    categories: dict[str, list[str]]
    means: dict[str, float]
    scales: dict[str, float]
    coefficients: dict[str, list[float]]
    ridge: float
    success_reward: float

    @property
    def n_features(self) -> int:
        return len(self.numeric_features) + sum(len(v) for v in self.categories.values())

    @property
    def n_parameters(self) -> int:
        return (self.n_features + 1) * len(ACTIONS)

    def _vector(self, row: dict[str, Any]) -> np.ndarray:
        raw_features = dict(row.get("features") or {})
        values = [
            (float(raw_features.get(name, self.means[name])) - self.means[name])
            / self.scales[name]
            for name in self.numeric_features
        ]
        for field in self.categorical_fields:
            current = str(row.get(field) or "")
            values.extend(float(current == category) for category in self.categories[field])
        return np.asarray([1.0, *values], dtype=np.float64)

    def scores(self, row: dict[str, Any]) -> dict[str, float]:
        vector = self._vector(row)
        return {
            action: float(vector @ np.asarray(self.coefficients[action]))
            for action in ACTIONS
        }

    def predict(self, row: dict[str, Any]) -> str:
        scores = self.scores(row)
        return max(ACTIONS, key=lambda action: (scores[action], -ACTIONS.index(action)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SELECTOR_MODEL_SCHEMA_VERSION,
            "actions": list(ACTIONS),
            "numeric_features": self.numeric_features,
            "categorical_fields": self.categorical_fields,
            "categories": self.categories,
            "means": self.means,
            "scales": self.scales,
            "coefficients": self.coefficients,
            "ridge": self.ridge,
            "success_reward": self.success_reward,
            "n_features": self.n_features,
            "n_parameters": self.n_parameters,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LightweightSelector":
        if payload.get("schema_version") != SELECTOR_MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported selector model schema")
        return cls(
            numeric_features=list(payload["numeric_features"]),
            categorical_fields=list(payload["categorical_fields"]),
            categories={key: list(value) for key, value in payload["categories"].items()},
            means={key: float(value) for key, value in payload["means"].items()},
            scales={key: float(value) for key, value in payload["scales"].items()},
            coefficients={key: list(value) for key, value in payload["coefficients"].items()},
            ridge=float(payload["ridge"]),
            success_reward=float(payload["success_reward"]),
        )


def fit_lightweight_selector(
    rows: list[dict[str, Any]],
    *,
    ridge: float = 1.0,
    success_reward: float = 1.0,
    categorical_fields: tuple[str, ...] = ("suite",),
) -> LightweightSelector:
    """Fit one ridge utility head per action using full-information outcomes."""
    if not rows:
        raise ValueError("cannot fit an empty selector dataset")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    numeric = sorted(
        {
            str(name)
            for row in rows
            for name, value in dict(row.get("features") or {}).items()
            if isinstance(value, (int, float)) and np.isfinite(float(value))
        }
    )
    if not numeric:
        raise ValueError("no numeric deployable features")
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in numeric:
        values = np.asarray(
            [
                float(dict(row.get("features") or {})[name])
                for row in rows
                if name in dict(row.get("features") or {})
            ]
        )
        means[name] = float(values.mean())
        scale = float(values.std())
        scales[name] = scale if scale > 1e-12 else 1.0
    categories = {
        field: sorted({str(row.get(field) or "") for row in rows})
        for field in categorical_fields
    }
    model = LightweightSelector(
        numeric_features=numeric,
        categorical_fields=list(categorical_fields),
        categories=categories,
        means=means,
        scales=scales,
        coefficients={},
        ridge=float(ridge),
        success_reward=float(success_reward),
    )
    design = np.stack([model._vector(row) for row in rows])
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    for action in ACTIONS:
        selected: list[int] = []
        targets: list[float] = []
        for index, row in enumerate(rows):
            outcome = dict((row.get("arms") or {}).get(action) or {})
            if outcome.get("observed", False):
                selected.append(index)
                targets.append(_utility(outcome, success_reward))
        if not selected:
            raise ValueError(f"no observed outcomes for action {action}")
        x = design[np.asarray(selected)]
        y = np.asarray(targets, dtype=np.float64)
        coefficients = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
        model.coefficients[action] = coefficients.tolist()
    return model


def _policy_metrics(
    rows: list[dict[str, Any]], actions: list[str], *, success_reward: float
) -> dict[str, Any]:
    successes: list[float] = []
    costs: list[float] = []
    utilities: list[float] = []
    clean_regrets: list[float] = []
    evaluable = 0
    for row, action in zip(rows, actions):
        arm = dict((row.get("arms") or {}).get(action) or {})
        if not arm.get("observed", False):
            continue
        evaluable += 1
        success = float(bool(arm["success"]))
        cost = float(arm["cost"])
        successes.append(success)
        costs.append(cost)
        utilities.append(success_reward * success - cost)
        if row.get("episode_outcome") == "success":
            base = dict((row.get("arms") or {}).get(CONTINUE_SMOL) or {})
            if base.get("observed", False) and bool(base.get("success", False)):
                clean_regrets.append(
                    max(0.0, float(bool(base["success"])) - success)
                )
    total = len(rows)
    return {
        "n": total,
        "n_evaluable": evaluable,
        "coverage": evaluable / total if total else None,
        "success_rate": float(np.mean(successes)) if successes else None,
        "mean_cost": float(np.mean(costs)) if costs else None,
        "mean_utility": float(np.mean(utilities)) if utilities else None,
        "strong_policy_usage": actions.count(ESCALATE_OFT) / total if total else None,
        "abstain_rate": actions.count(ABSTAIN) / total if total else None,
        "clean_regret": float(np.mean(clean_regrets)) if clean_regrets else None,
        "n_clean_regret_evaluable": len(clean_regrets),
        "action_counts": {action: actions.count(action) for action in ACTIONS},
    }


def _paired_utility_difference(
    rows: list[dict[str, Any]],
    left_actions: list[str],
    right_actions: list[str],
    *,
    success_reward: float,
    seed: int = 20260730,
    bootstrap_samples: int = 5000,
) -> dict[str, Any]:
    """Paired utility delta with a deterministic percentile bootstrap CI."""
    differences: list[float] = []
    for row, left, right in zip(rows, left_actions, right_actions):
        left_arm = dict((row.get("arms") or {}).get(left) or {})
        right_arm = dict((row.get("arms") or {}).get(right) or {})
        if not left_arm.get("observed", False) or not right_arm.get("observed", False):
            continue
        differences.append(
            _utility(left_arm, success_reward) - _utility(right_arm, success_reward)
        )
    if not differences:
        return {
            "n_pairs": 0,
            "mean_difference": None,
            "bootstrap_ci_95": {"lower": None, "upper": None},
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        }
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sample = rng.integers(0, len(values), size=len(values))
        bootstrap_means[index] = float(values[sample].mean())
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "n_pairs": len(differences),
        "mean_difference": float(values.mean()),
        "bootstrap_ci_95": {"lower": float(lower), "upper": float(upper)},
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def evaluate_selector(
    model: LightweightSelector,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate learned, fixed, oracle, and matched-escalation baselines."""
    learned = [model.predict(row) for row in rows]
    fixed = {action: [action] * len(rows) for action in ACTIONS}
    oracle = []
    for row in rows:
        utilities = {
            action: _utility(dict(row["arms"][action]), model.success_reward)
            for action in ACTIONS
        }
        oracle.append(max(ACTIONS, key=lambda action: (utilities[action], -ACTIONS.index(action))))
    escalation_budget = learned.count(ESCALATE_OFT)
    ranked = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(
            str(rows[index]["state_key"]).encode()
        ).digest(),
    )
    chosen = set(ranked[:escalation_budget])
    matched_random = [
        ESCALATE_OFT if index in chosen else CONTINUE_SMOL
        for index in range(len(rows))
    ]
    abstain_budget = learned.count(ABSTAIN)
    action_matched_random = [CONTINUE_SMOL] * len(rows)
    for index in ranked[:escalation_budget]:
        action_matched_random[index] = ESCALATE_OFT
    for index in ranked[escalation_budget : escalation_budget + abstain_budget]:
        action_matched_random[index] = ABSTAIN
    paired = {
        "learned_minus_always_continue": _paired_utility_difference(
            rows,
            learned,
            fixed[CONTINUE_SMOL],
            success_reward=model.success_reward,
        ),
        "learned_minus_always_escalate": _paired_utility_difference(
            rows,
            learned,
            fixed[ESCALATE_OFT],
            success_reward=model.success_reward,
        ),
        "learned_minus_matched_random_trigger": _paired_utility_difference(
            rows,
            learned,
            matched_random,
            success_reward=model.success_reward,
        ),
        "learned_minus_matched_random_actions": _paired_utility_difference(
            rows,
            learned,
            action_matched_random,
            success_reward=model.success_reward,
        ),
    }
    return {
        "learned": _policy_metrics(rows, learned, success_reward=model.success_reward),
        "always_continue": _policy_metrics(
            rows, fixed[CONTINUE_SMOL], success_reward=model.success_reward
        ),
        "always_escalate": _policy_metrics(
            rows, fixed[ESCALATE_OFT], success_reward=model.success_reward
        ),
        "always_abstain": _policy_metrics(
            rows, fixed[ABSTAIN], success_reward=model.success_reward
        ),
        "matched_random_trigger": _policy_metrics(
            rows, matched_random, success_reward=model.success_reward
        ),
        "matched_random_actions": _policy_metrics(
            rows, action_matched_random, success_reward=model.success_reward
        ),
        "oracle_upper_bound": _policy_metrics(
            rows, oracle, success_reward=model.success_reward
        ),
        "paired_utility_differences": paired,
    }
