#!/usr/bin/env python3
"""Audit and analyse V6 Stage-0 exact-root refresh opportunity records.

Primary value estimates use the *mean* of the four independently seeded
R-new branches, i.e. an estimate of expected K=1 refresh performance.  The
script never substitutes ``max(R-new)``; that would turn the eligibility gate
into best-of-K test-time search, which V6 explicitly excludes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA = "rase-v6-stage0-analysis/v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_file() and path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
        return
    if path.is_dir():
        roots = path / "roots" if (path / "roots").is_dir() else path
        for candidate in sorted(roots.glob("*.json")):
            value = read_json(candidate)
            if isinstance(value, dict):
                for row in value.get("branches") or []:
                    if isinstance(row, dict):
                        yield row
        return
    if path.is_file() and path.suffix == ".json":
        value = read_json(path)
        if isinstance(value, list):
            yield from (row for row in value if isinstance(row, dict))
            return
        if isinstance(value, dict):
            yield from (row for row in value.get("branches") or [] if isinstance(row, dict))
            return
    raise ValueError(f"cannot read Stage-0 records from {path}")


def _success(row: Mapping[str, Any]) -> float:
    value = row.get("success")
    if not isinstance(value, bool):
        raise ValueError("success must be boolean")
    return float(value)


def audit_root(
    rows: list[dict[str, Any]], *, expected_k: int | None,
    allowed_k: set[int] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not rows:
        return None, ["empty_root"]
    root_id = str(rows[0].get("root_id", ""))
    required_constant = (
        "root_id", "state_key", "task_id", "suite", "perturb_dim", "perturb_level",
        "cursor", "native_chunk_horizon", "root_snapshot_sha256", "source_generation_seed",
        "source_temperature", "downstream_controller", "downstream_seed", "downstream_temperature",
    )
    for field in required_constant:
        values = {json.dumps(row.get(field), sort_keys=True) for row in rows}
        if len(values) != 1:
            errors.append(f"inconsistent:{field}")
    schemas = {str(row.get("schema_version")) for row in rows}
    if len(schemas) != 1:
        errors.append("inconsistent:schema_version")
    schema = next(iter(schemas), "")
    if schema not in {"rase-v6-stage0-branch/v1", "rase-v6-stage0-branch/v2"}:
        errors.append("wrong_schema")
    if schema == "rase-v6-stage0-branch/v2":
        required_v2 = (
            "old_chunk_source_qpos_sha256", "pre_perturb_boundary_qpos_sha256",
            "perturbation_timing", "position_target_mode",
        )
        for field in required_v2:
            values = {json.dumps(row.get(field), sort_keys=True) for row in rows}
            if len(values) != 1 or any(row.get(field) is None for row in rows):
                errors.append(f"missing_or_inconsistent:{field}")
        if rows[0].get("perturbation_timing") != "after_old_prefix_before_branch_decision":
            errors.append("invalid_perturbation_timing")
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "complete":
            errors.append("incomplete_branch")
        arm = str(row.get("arm"))
        by_arm[arm].append(row)
    c_rows = by_arm.get("C", [])
    same_rows = by_arm.get("R_same", [])
    new_rows = sorted(by_arm.get("R_new", []), key=lambda row: int(row.get("arm_index", -1)))
    if len(c_rows) != 1:
        errors.append(f"expected_one_C_got_{len(c_rows)}")
    if len(same_rows) != 1:
        errors.append(f"expected_one_R_same_got_{len(same_rows)}")
    if expected_k is not None and len(new_rows) != expected_k:
        errors.append(f"expected_{expected_k}_R_new_got_{len(new_rows)}")
    if allowed_k is not None and len(new_rows) not in allowed_k:
        allowed_text = "_or_".join(str(value) for value in sorted(allowed_k))
        errors.append(f"expected_R_new_k_{allowed_text}_got_{len(new_rows)}")
    if not new_rows:
        errors.append("missing_R_new")
    if errors:
        return None, errors
    source_seed = int(c_rows[0]["source_generation_seed"])
    if int(same_rows[0].get("candidate_generation_seed")) != source_seed:
        errors.append("r_same_seed_not_matched")
    new_seeds = [row.get("candidate_generation_seed") for row in new_rows]
    if any(seed is None for seed in new_seeds):
        errors.append("missing_r_new_seed")
    elif source_seed in {int(seed) for seed in new_seeds}:
        errors.append("r_new_reused_source_seed")
    elif len({int(seed) for seed in new_seeds}) != len(new_seeds):
        errors.append("r_new_seed_not_unique")
    candidate_steps = {int(row.get("candidate_chunk_steps", -1)) for row in rows}
    if len(candidate_steps) != 1 or next(iter(candidate_steps)) <= 0:
        errors.append("candidate_horizon_mismatch")
    if errors:
        return None, errors
    q_c = _success(c_rows[0])
    q_same = _success(same_rows[0])
    new_outcomes = [_success(row) for row in new_rows]
    q_new = float(np.mean(new_outcomes))
    return {
        "root_id": root_id,
        "state_key": c_rows[0]["state_key"],
        "task_id": c_rows[0]["task_id"],
        "suite": c_rows[0]["suite"],
        "perturb_dim": c_rows[0]["perturb_dim"],
        "perturb_level": c_rows[0]["perturb_level"],
        "cursor": c_rows[0]["cursor"],
        "actual_cursor_fraction": c_rows[0].get("actual_cursor_fraction"),
        "q_continue": q_c,
        "q_refresh_same": q_same,
        "q_refresh_new_mean": q_new,
        "q_refresh_new_best_diagnostic": float(max(new_outcomes)),
        "adv_refresh_new": q_new - q_c,
        "adv_refresh_same": q_same - q_c,
        "r_new_outcomes": new_outcomes,
        "r_new_seeds": [int(seed) for seed in new_seeds],
        "r_new_k": int(len(new_outcomes)),
        "root_snapshot_sha256": c_rows[0]["root_snapshot_sha256"],
        "downstream_seed": c_rows[0]["downstream_seed"],
    }, []


def summarize(roots: list[dict[str, Any]]) -> dict[str, Any]:
    if not roots:
        return {
            "n_roots": 0, "ac_refresh": None, "mean_continue": None,
            "mean_refresh_new": None, "mean_refresh_same": None,
        }
    c = np.asarray([row["q_continue"] for row in roots], dtype=float)
    r = np.asarray([row["q_refresh_new_mean"] for row in roots], dtype=float)
    same = np.asarray([row["q_refresh_same"] for row in roots], dtype=float)
    advantage = r - c
    tol = 1e-12
    task_counts = Counter(str(row["task_id"]) for row in roots)
    cursor_counts = Counter(str(row["cursor"]) for row in roots)
    return {
        "n_roots": int(len(roots)),
        "n_tasks": int(len(task_counts)),
        "roots_per_task": dict(sorted(task_counts.items())),
        "roots_per_cursor": dict(sorted(cursor_counts.items())),
        "mean_continue": float(c.mean()),
        "mean_refresh_same": float(same.mean()),
        "mean_refresh_new": float(r.mean()),
        "mean_best_of_k_diagnostic": float(np.mean([row["q_refresh_new_best_diagnostic"] for row in roots])),
        "mean_adv_refresh_new": float(advantage.mean()),
        "mean_adv_refresh_same": float((same - c).mean()),
        "refresh_better_roots": int(np.sum(advantage > tol)),
        "continue_better_roots": int(np.sum(advantage < -tol)),
        "ties": int(np.sum(np.abs(advantage) <= tol)),
        "p_advantage_positive": float(np.mean(advantage > tol)),
        "p_advantage_negative": float(np.mean(advantage < -tol)),
        "ac_refresh": float(np.maximum(c, r).mean() - max(c.mean(), r.mean())),
        "ac_refresh_same": float(np.maximum(c, same).mean() - max(c.mean(), same.mean())),
    }


def cluster_bootstrap(roots: list[dict[str, Any]], *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    tasks = sorted({str(row["task_id"]) for row in roots})
    if not roots or len(tasks) < 2 or n_bootstrap < 1:
        return {"n_bootstrap": 0, "cluster": "task_id", "ac_refresh_ci95": None}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in roots:
        by_task[str(row["task_id"])].append(row)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sample: list[dict[str, Any]] = []
        for _draw in tasks:
            sample.extend(by_task[rng.choice(tasks)])
        value = summarize(sample)["ac_refresh"]
        if value is not None:
            values.append(float(value))
    return {
        "n_bootstrap": n_bootstrap,
        "cluster": "task_id",
        "ac_refresh_ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "ac_refresh_bootstrap_mean": float(np.mean(values)),
    }


def gate(summary: Mapping[str, Any], bootstrap: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    if not summary["n_roots"]:
        return {"decision": "FAIL", "reason": "no_auditable_complete_roots", "next_action": "DO_NOT_TRAIN_HEAD"}
    refresh = int(summary["refresh_better_roots"])
    continue_ = int(summary["continue_better_roots"])
    if mode == "pilot":
        passed = refresh >= 3 and continue_ >= 3
        if passed:
            return {
                "decision": "PASS", "reason": "two_sided_statewise_opportunity_observed",
                "next_action": "RUN_FORMAL_STAGE0_AND_STAGE0B; DO_NOT_TRAIN_HEAD_YET",
            }
        if refresh >= 3 and continue_ == 0:
            return {
                "decision": "FAIL", "reason": "refresh_only; no evidence a selector beats always-refresh",
                "next_action": "RUN_ALWAYS_REFRESH_PROBE_OR_CHANGE_DOMAIN; DO_NOT_TRAIN_HEAD",
            }
        if continue_ >= 3 and refresh == 0:
            return {
                "decision": "FAIL", "reason": "continue_only; no refresh recovery opportunity",
                "next_action": "CHANGE_DOMAIN_OR_SOURCE; DO_NOT_TRAIN_HEAD",
            }
        return {
            "decision": "FAIL", "reason": "insufficient_two_sided_opportunity",
            "next_action": "CHANGE_DOMAIN_OR_EXPAND_PILOT_ONLY_IF_AUDITABLE; DO_NOT_TRAIN_HEAD",
        }
    if mode != "formal":
        raise ValueError(mode)
    ci = bootstrap.get("ac_refresh_ci95")
    ci_lower = float(ci[0]) if isinstance(ci, list) and len(ci) == 2 else math.nan
    passed = (
        float(summary["ac_refresh"]) >= 0.05
        and math.isfinite(ci_lower)
        and ci_lower > 0.0
        and float(summary["p_advantage_positive"]) >= 0.05
        and float(summary["p_advantage_negative"]) >= 0.05
    )
    if passed:
        return {
            "decision": "PASS", "reason": "formal_ACR_and_two_sided_coverage_pass",
            "next_action": "PROCEED_STAGE1_SAME_ROOT_DATA_COLLECTION",
        }
    ac = float(summary["ac_refresh"])
    if ac < 0.03:
        action = "CHANGE_DOMAIN_OR_SOURCE; DO_NOT_TRAIN_HEAD"
    else:
        action = "AUDIT_PROTOCOL_OR_RUN_ONE_CHEAP_VISUAL_RESIDUAL_ONLY; DO_NOT_TRAIN_HEAD"
    return {"decision": "FAIL", "reason": "formal_gate_not_met", "next_action": action}


def analyse(
    rows: Iterable[dict[str, Any]], *, expected_k: int | None,
    allowed_k: set[int] | None = None, mode: str, n_bootstrap: int, seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        root_id = row.get("root_id")
        if isinstance(root_id, str) and root_id:
            grouped[root_id].append(row)
    roots: list[dict[str, Any]] = []
    audit_failures: dict[str, list[str]] = {}
    for root_id, group in sorted(grouped.items()):
        root, errors = audit_root(group, expected_k=expected_k, allowed_k=allowed_k)
        if errors:
            audit_failures[root_id] = errors
        elif root is not None:
            roots.append(root)
    summary = summarize(roots)
    boot = cluster_bootstrap(roots, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "schema_version": SCHEMA,
        "analysis_mode": mode,
        "primary_estimand": "Q_R = mean(success of independently seeded R_new branches); no best-of-K selection",
        "n_input_roots": len(grouped),
        "n_auditable_roots": len(roots),
        "n_audit_failures": len(audit_failures),
        "audit_failures": audit_failures,
        "summary": summary,
        "bootstrap": boot,
        "gate": gate(summary, boot, mode=mode),
        "per_root": roots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--expected-r-new-k", type=int, default=4)
    parser.add_argument(
        "--allowed-r-new-k", default=None,
        help="Optional comma-separated allowed K values, e.g. 4,8 for an adaptive-K refinement.",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.expected_r_new_k < 1:
        parser.error("--expected-r-new-k must be positive")
    allowed_k: set[int] | None = None
    if args.allowed_r_new_k is not None:
        try:
            allowed_k = {int(value.strip()) for value in str(args.allowed_r_new_k).split(",") if value.strip()}
        except ValueError as exc:
            parser.error(f"invalid --allowed-r-new-k: {exc}")
        if not allowed_k or min(allowed_k) < 1:
            parser.error("--allowed-r-new-k must contain positive integers")
    artifact = analyse(
        iter_rows(args.input.resolve()),
        expected_k=None if allowed_k is not None else args.expected_r_new_k,
        allowed_k=allowed_k,
        mode=args.mode,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    atomic_json(args.output.resolve(), artifact)
    print(json.dumps(artifact["gate"], sort_keys=True), flush=True)
    return 0 if artifact["gate"]["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
