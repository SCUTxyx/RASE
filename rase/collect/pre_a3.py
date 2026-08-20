"""PRE-A3 recovery-duration protocol: cohort freeze, analysis, and method gates."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rase.collect.policy_matrix import exact_mcnemar_p

SUITES = ("Spatial", "Object", "Goal", "Long")
CELLS = (("clean", 0), ("camera", 1), ("robot", 1))
DEFAULT_DURATIONS = (0, 8, 16, 32, 64, 96, 128)
ARTIFACT_VERSION = "rase-pre-a3-state-keys/v1"
DESIGN_VERSION = "rase-pre-a3-design/v1"
AUDIT_VERSION = "rase-pre-a3-recovery-duration-audit/v1"
GATE_VERSION = "rase-pre-a3-method-gate/v1"


def cell_name(dimension: str, level: int) -> str:
    return f"{dimension}:L{level}"


def task_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    value_fn,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    """Cluster bootstrap over unique task_id with paired unit values."""
    if not rows:
        return [0.0, 0.0]
    by_task: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    task_ids = sorted(by_task)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(replicates)):
        sampled = rng.choice(task_ids, size=len(task_ids), replace=True)
        units = [unit for task in sampled for unit in by_task[str(task)]]
        draws.append(float(value_fn(units)))
    return [float(value) for value in np.quantile(np.asarray(draws), [0.025, 0.975])]


def assign_task_disjoint_splits(
    tasks_by_suite: Mapping[str, Sequence[str]],
    *,
    train_n: int = 6,
    val_n: int = 2,
    test_n: int = 2,
    seed: int = 2_026_080_401,
) -> dict[str, str]:
    """Assign each unique task to train/val/test with fixed per-suite counts."""
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for suite in SUITES:
        tasks = sorted({str(task) for task in tasks_by_suite.get(suite, ())})
        if len(tasks) != train_n + val_n + test_n:
            raise ValueError(
                f"{suite}: expected {train_n + val_n + test_n} tasks, got {len(tasks)}"
            )
        order = list(tasks)
        rng.shuffle(order)
        for task in order[:train_n]:
            assignment[task] = "train"
        for task in order[train_n : train_n + val_n]:
            assignment[task] = "val"
        for task in order[train_n + val_n :]:
            assignment[task] = "test"
    return assignment


def build_design(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    excluded_task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate and package an outcome-independent 120-state design."""
    rows = [dict(row) for row in records]
    if len(rows) != 120:
        raise ValueError(f"PRE-A3 design requires 120 records, got {len(rows)}")
    task_ids = [str(row["task_id"]) for row in rows]
    if len(set(task_ids)) != 40:
        raise ValueError(f"expected 40 unique tasks, got {len(set(task_ids))}")
    excluded = set(excluded_task_ids or ())
    overlap = sorted(set(task_ids) & excluded)
    if overlap:
        raise ValueError(f"design overlaps excluded development tasks: {overlap[:8]}")

    by_suite_task: dict[str, set[str]] = defaultdict(set)
    cell_counts: Counter[str] = Counter()
    for row in rows:
        suite = str(row["suite"])
        task = str(row["task_id"])
        dim = str(row["dimension"])
        level = int(row["level"])
        by_suite_task[suite].add(task)
        cell_counts[f"{suite}|{cell_name(dim, level)}"] += 1
        row["cell"] = cell_name(dim, level)

    for suite in SUITES:
        if len(by_suite_task[suite]) != 10:
            raise ValueError(f"{suite} must contain exactly 10 tasks")
    for suite in SUITES:
        for dim, level in CELLS:
            key = f"{suite}|{cell_name(dim, level)}"
            if cell_counts[key] != 10:
                raise ValueError(f"{key} must contain 10 states, got {cell_counts[key]}")

    # Each task must appear under all three conditions.
    conditions_by_task: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        conditions_by_task[str(row["task_id"])].add(str(row["cell"]))
    expected_cells = {cell_name(dim, level) for dim, level in CELLS}
    bad = {
        task: sorted(cells)
        for task, cells in conditions_by_task.items()
        if set(cells) != expected_cells
    }
    if bad:
        raise ValueError(f"tasks missing full condition coverage: {list(bad)[:5]}")

    assignment = assign_task_disjoint_splits(
        {suite: sorted(tasks) for suite, tasks in by_suite_task.items()},
        seed=seed,
    )
    for row in rows:
        row["split"] = assignment[str(row["task_id"])]

    split_counts = Counter(row["split"] for row in rows)
    payload = {
        "artifact_version": DESIGN_VERSION,
        "n_requests": len(rows),
        "n_unique_tasks": 40,
        "n_unique_episodes": len({str(row.get("episode_id", row["task_id"])) for row in rows}),
        "selection_uses_outcomes": False,
        "split_seed": seed,
        "split_policy": "per_suite_tasks_6_train_2_val_2_test",
        "split_counts": {
            "train": split_counts["train"],
            "val": split_counts["val"],
            "test": split_counts["test"],
        },
        "cell_counts": dict(sorted(cell_counts.items())),
        "excluded_task_ids": sorted(excluded),
        "durations": list(DEFAULT_DURATIONS),
        "execution_mode": "live_closed_loop_oft_prefix",
        "records": rows,
    }
    digest = hashlib.sha256(
        repr(sorted((r["task_id"], r["dimension"], r["level"], r["split"]) for r in rows)).encode()
    ).hexdigest()
    payload["design_sha256"] = digest
    return payload


def freeze_keys_from_pool_records(
    records: Sequence[Mapping[str, Any]],
    *,
    design: Mapping[str, Any],
    pool: str,
) -> dict[str, Any]:
    """Join collected pool states onto a frozen design without using outcomes."""
    by_episode = {str(row["episode_id"]): dict(row) for row in design["records"]}
    if len(by_episode) != 120:
        raise ValueError("design episode_ids must be unique and length 120")
    joined = []
    for record in records:
        if int(record.get("step", -1)) != 0:
            continue
        episode_id = str(record["episode_id"])
        design_row = by_episode.get(episode_id)
        if design_row is None:
            raise ValueError(f"unplanned episode in PRE-A3 pool: {episode_id}")
        dim = str(record.get("perturbation_dimension") or record.get("dimension"))
        level = int(record.get("perturbation_level", record.get("level")))
        if dim != str(design_row["dimension"]) or level != int(design_row["level"]):
            raise ValueError(
                f"episode identity mismatch for {episode_id}: "
                f"pool=({dim},L{level}) design="
                f"({design_row['dimension']},L{design_row['level']})"
            )
        concrete = str(design_row.get("concrete_task_id") or "")
        pool_task = str(record["task_id"])
        if concrete and pool_task != concrete:
            raise ValueError(
                f"concrete task mismatch for {episode_id}: "
                f"pool={pool_task} design={concrete}"
            )
        joined.append(
            {
                **design_row,
                "state_key": str(record["state_key"]),
                "episode_id": episode_id,
                "step": int(record.get("step", 0)),
                "snapshot_policy_step": int(record.get("snapshot_policy_step", 0)),
                "snapshot_simulator_timestep": int(
                    record.get("snapshot_simulator_timestep", record.get("step", 0))
                ),
                "suite": str(record.get("suite") or design_row["suite"]),
                "perturbation_dimension": dim,
                "perturbation_level": level,
                "cell": cell_name(dim, level),
                "pool_task_id": pool_task,
            }
        )
    if len(joined) != 120:
        raise ValueError(f"frozen cohort requires 120 joined states, got {len(joined)}")
    if len({row["state_key"] for row in joined}) != 120:
        raise ValueError("duplicate state_key in frozen cohort")
    ordered = sorted(joined, key=lambda r: int(r["request_index"]))
    state_keys = [row["state_key"] for row in ordered]
    checksum = hashlib.sha256("\n".join(state_keys).encode()).hexdigest()
    return {
        "artifact_version": ARTIFACT_VERSION,
        "selection_uses_outcomes": False,
        "exclude_from_flagship_hidden_test": False,
        "n_states": 120,
        "n_tasks": 40,
        "n_episodes": len({row["episode_id"] for row in ordered}),
        "pool": pool,
        "design_sha256": design["design_sha256"],
        "state_keys": state_keys,
        "state_keys_sha256": checksum,
        "split_counts": dict(design["split_counts"]),
        "records": ordered,
        "durations": list(DEFAULT_DURATIONS),
        "execution_mode": "live_closed_loop_oft_prefix",
        "gates_preregistered": True,
    }


def analyze_recovery_duration(
    duration: Mapping[str, Any],
    *,
    keys: Mapping[str, Any] | None = None,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 2_026_080_403,
    split: str | None = None,
) -> dict[str, Any]:
    """Analyze live or replay duration sweeps with PRE-A3 confirmatory gates."""
    lengths = [int(value) for value in duration["prefix_lengths"]]
    if not lengths or lengths[0] != 0:
        raise ValueError("duration sweep must start at zero")
    key_rows = {}
    if keys is not None:
        key_rows = {str(row["state_key"]): dict(row) for row in keys.get("records") or []}

    rows = []
    for row in duration["per_state"]:
        state_key = str(row["state_key"])
        meta = key_rows.get(state_key, {})
        split_name = str(meta.get("split") or row.get("split") or "train")
        if split is not None and split_name != split:
            continue
        outcomes = [bool(arm["success"]) for arm in row["arms"]]
        if len(outcomes) != len(lengths):
            raise ValueError(f"incomplete arms for {state_key}")
        direct = row.get("direct_oft_success")
        if direct is None:
            # optional trailing persistent arm encoded separately
            direct = bool(row.get("persistent_oft_success", False))
        base = outcomes[0]
        successful = [
            length
            for length, success in zip(lengths[1:], outcomes[1:], strict=True)
            if success
        ]
        minimum = min(successful) if successful else None
        rescue = (not base) and minimum is not None
        harmed = [
            length
            for length, success in zip(lengths[1:], outcomes[1:], strict=True)
            if base and not success
        ]
        nonmonotonic = any(
            earlier and not later
            for earlier, later in zip(outcomes[1:-1], outcomes[2:], strict=True)
        )
        rows.append(
            {
                "state_key": state_key,
                "task_id": str(row.get("task_id") or meta.get("task_id")),
                "suite": str(row.get("suite") or meta.get("suite")),
                "cell": str(
                    row.get("cell")
                    or meta.get("cell")
                    or cell_name(
                        str(row.get("perturbation_dimension") or meta.get("perturbation_dimension")),
                        int(row.get("perturbation_level", meta.get("perturbation_level", 0))),
                    )
                ),
                "split": split_name,
                "outcomes": dict(zip((str(v) for v in lengths), outcomes, strict=True)),
                "base_success": base,
                "minimum_successful_duration": minimum,
                "fixed_duration_rescue": rescue,
                "harmed_durations": harmed,
                "nonmonotonic_finite_duration": nonmonotonic,
                "direct_oft_success": bool(direct),
                "direct_only_rescue": (not base) and minimum is None and bool(direct),
                "oft_steps_best_fixed": None,
            }
        )

    n = len(rows)
    if n == 0:
        raise ValueError("no states selected for analysis")
    successes = {
        str(length): sum(row["outcomes"][str(length)] for row in rows) for length in lengths
    }
    base_n = successes["0"]
    fixed_oracle_n = sum(
        any(row["outcomes"][str(length)] for length in lengths[1:]) for row in rows
    )
    fixed_rescues = sum(row["fixed_duration_rescue"] for row in rows)
    rescue_tasks = sorted({row["task_id"] for row in rows if row["fixed_duration_rescue"]})
    rescue_suites = sorted({row["suite"] for row in rows if row["fixed_duration_rescue"]})
    rescue_cells = sorted({row["cell"] for row in rows if row["fixed_duration_rescue"]})
    direct_n = sum(row["direct_oft_success"] for row in rows)
    direct_only = sum(row["direct_only_rescue"] for row in rows)
    harms = {
        str(length): sum(
            row["base_success"] and not row["outcomes"][str(length)] for row in rows
        )
        for length in lengths[1:]
    }
    best_fixed_n = max(successes[str(length)] for length in lengths[1:])
    best_fixed = [
        length for length in lengths[1:] if successes[str(length)] == best_fixed_n
    ]
    best_h = best_fixed[0]
    best_fixed_harm = harms[str(best_h)] / max(1, sum(row["base_success"] for row in rows))
    oracle_gap_pp = 100.0 * (fixed_oracle_n - base_n) / n
    adaptive_vs_best_pp = 100.0 * (fixed_oracle_n - best_fixed_n) / n
    # duration heterogeneity: rescues whose minimum duration is not unique
    mins = [row["minimum_successful_duration"] for row in rows if row["fixed_duration_rescue"]]
    heterogeneous = len(set(mins)) >= 2

    def _gap(units: Sequence[Mapping[str, Any]]) -> float:
        if not units:
            return 0.0
        oracle = sum(
            any(unit["outcomes"][str(length)] for length in lengths[1:]) for unit in units
        )
        base = sum(unit["base_success"] for unit in units)
        return (oracle - base) / len(units)

    ci = task_bootstrap_ci(
        rows, _gap, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    base_only = sum(row["base_success"] and not row["outcomes"][str(best_h)] for row in rows)
    best_only = sum((not row["base_success"]) and row["outcomes"][str(best_h)] for row in rows)
    mcnemar = exact_mcnemar_p(best_only, base_only)

    pass_conditions = {
        "oracle_gap_ge_8pp": oracle_gap_pp >= 8.0 and ci[0] > 0.0,
        "rescues_ge_4_task_disjoint": len(rescue_tasks) >= 4,
        "rescues_cover_ge_2_suites": len(rescue_suites) >= 2,
        "rescues_cover_ge_2_cells": len(rescue_cells) >= 2,
        "duration_heterogeneity": heterogeneous,
        "best_fixed_harm_le_5pct": best_fixed_harm <= 0.05,
        "adaptive_headroom_ge_5pp": adaptive_vs_best_pp >= 5.0,
    }
    gate_pass = all(pass_conditions.values())
    if gate_pass:
        status = "pre_a3_gate_pass"
        termination_gate = "open"
        next_step = (
            "Build PRE-B safe-handback dataset and fit calibrated termination/"
            "competence baselines against fixed-duration and always-OFT."
        )
    elif fixed_rescues >= 2 and oracle_gap_pp >= 8.0:
        status = "duration_structure_signal_unconfirmed"
        termination_gate = "replication_required"
        next_step = "Signal present but confirmatory gate failed; do not train."
    elif direct_only >= 2 and direct_n > fixed_oracle_n:
        status = "episode_persistent_fallback"
        termination_gate = "closed"
        next_step = (
            "Finite handback insufficient; prefer episode-long fallback or "
            "distilled recovery policy."
        )
    else:
        status = "not_ready"
        termination_gate = "closed"
        next_step = "No stable duration structure; stop learned-switching method line."

    return {
        "schema_version": AUDIT_VERSION,
        "status": status,
        "split": split or "all",
        "n_states": n,
        "durations": lengths,
        "successes_by_duration": successes,
        "base_successes": base_n,
        "fixed_duration_oracle_successes": fixed_oracle_n,
        "fixed_duration_rescues": fixed_rescues,
        "fixed_duration_rescue_tasks": rescue_tasks,
        "fixed_duration_rescue_suites": rescue_suites,
        "fixed_duration_rescue_cells": rescue_cells,
        "base_harmed_by_duration": harms,
        "best_fixed_duration_successes": best_fixed_n,
        "best_fixed_durations": best_fixed,
        "best_fixed_harm_rate_on_base_successes": best_fixed_harm,
        "oracle_minus_base_pp": oracle_gap_pp,
        "adaptive_minus_best_fixed_pp": adaptive_vs_best_pp,
        "oracle_minus_base_task_bootstrap_95": {
            "lower": 100.0 * ci[0],
            "upper": 100.0 * ci[1],
            "unit": "task_cluster_rate_difference",
        },
        "best_fixed_vs_base_mcnemar_exact_p": mcnemar,
        "direct_oft_successes": direct_n,
        "direct_only_rescues": direct_only,
        "nonmonotonic_finite_duration_states": [
            row["state_key"] for row in rows if row["nonmonotonic_finite_duration"]
        ],
        "pass_conditions": pass_conditions,
        "gate_pass": gate_pass,
        "termination_model_gate": termination_gate,
        "critic_gate": "closed",
        "world_model_gate": "closed",
        "next_step": next_step,
        "per_state": rows,
        "limitations": [
            "Confirmatory claims require the frozen hidden test split.",
            "Live closed-loop OFT prefixes are required for main evidence.",
            "Deterministic replay is diagnostic only.",
        ],
    }


def decide_method_gate(
    hidden_audit: Mapping[str, Any],
    *,
    val_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the post-PRE-A3 method branch from preregistered hidden-test gates."""
    hidden_pass = bool(hidden_audit.get("gate_pass"))
    val_pass = True if val_audit is None else bool(val_audit.get("gate_pass"))
    if hidden_pass and val_pass:
        decision = "enter_safe_handback_method"
        paper_track = "method_plus_benchmark_candidate"
        allowed = [
            "pre_b_safe_handback_dataset",
            "calibrated_termination_baselines",
            "second_backbone_generalization",
        ]
        forbidden = [
            "ridge_mlp_rl_three_arm_selector",
            "generative_world_model_training",
            "same_profile_temperature_candidate_scaling",
        ]
    else:
        decision = "benchmark_diagnosis_only"
        paper_track = "benchmark_diagnosis"
        allowed = [
            "policy_relative_recoverability_tables",
            "mechanism_falsification_chain",
            "release_benchmark_artifacts",
        ]
        forbidden = [
            "termination_model_training",
            "candidate_critic_training",
            "generative_world_model_training",
            "ridge_mlp_rl_three_arm_selector",
        ]
    return {
        "schema_version": GATE_VERSION,
        "decision": decision,
        "paper_track": paper_track,
        "hidden_status": hidden_audit.get("status"),
        "hidden_gate_pass": hidden_pass,
        "val_gate_pass": val_pass,
        "termination_model_gate": "open" if decision.startswith("enter_") else "closed",
        "world_model_gate": "closed",
        "allowed_next": allowed,
        "forbidden_next": forbidden,
        "rationale": hidden_audit.get("next_step"),
    }
