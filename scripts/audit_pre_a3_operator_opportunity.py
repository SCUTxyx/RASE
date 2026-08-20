#!/usr/bin/env python3
"""Audit whether PRE-A3 contains enough counterfactual signal to train a selector.

The script is deliberately model-free.  It groups exact restored states, checks
that every state has the same operator set, and compares the oracle selector to
the best fixed operator.  Training a world model before this audit passes would
only fit noise or a universally-best fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SUCCESS_KEYS = ("success", "task_success", "is_success", "completed")
STATE_KEYS = ("state_key", "snapshot_id", "snapshot_key", "state_id", "state", "key")
TASK_KEYS = ("task_id", "task", "task_name")
SUITE_KEYS = ("suite", "suite_name", "benchmark")
OPERATOR_KEYS = ("operator", "operator_id", "profile", "arm", "arm_name")
DURATION_KEYS = ("duration", "recovery_duration", "handover_duration", "prefix_steps")


def first(row: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def walk_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if any(key in value for key in SUCCESS_KEYS):
            yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "success", "passed"}
    return bool(value)


def normalize_operator(row: dict[str, Any], path: Path) -> str | None:
    raw = first(row, OPERATOR_KEYS)
    if raw is not None:
        raw_text = str(raw).strip().upper().replace("-", "_")
        if "PERSIST" in raw_text or raw_text in {"OFT", "B1"}:
            return "OFT_PERSISTENT"
    duration = first(row, DURATION_KEYS)
    if duration is None:
        match = re.search(r"(?:duration|prefix|h)[_-]?(persistent|\d+)", path.stem.lower())
        if match:
            duration = match.group(1)
    if duration is not None:
        text = str(duration).strip().lower()
        if text in {"persistent", "full", "oft", "inf", "infinite", "-1"}:
            return "OFT_PERSISTENT"
        try:
            steps = int(float(text))
            return "CONTINUE" if steps == 0 else f"OFT_H{steps}"
        except ValueError:
            pass
    if raw is None:
        return None
    text = str(raw).strip().upper().replace("-", "_")
    if text in {"BASE", "B0", "SOURCE", "CONTINUE", "H0", "OFT_H0"}:
        return "CONTINUE"
    if "PERSIST" in text or text in {"OFT", "B1"}:
        return "OFT_PERSISTENT"
    match = re.search(r"(?:^|_)H(?:_|)?(\d+)(?:$|_)", text)
    return f"OFT_H{int(match.group(1))}" if match else text


def infer_from_path(path: Path, root: Path, key: str) -> str | None:
    parts = path.relative_to(root).parts
    if key == "suite":
        for part in parts:
            if part.startswith("suite_"):
                return part.removeprefix("suite_")
    return None


def nominal_steps(operator: str) -> int | None:
    if operator == "CONTINUE":
        return 0
    match = re.fullmatch(r"OFT_H(\d+)", operator)
    return int(match.group(1)) if match else None


def executed_oft_steps(row: dict[str, Any], operator: str) -> int:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    measured = metrics.get("oft_action_select_calls")
    if measured is not None:
        try:
            return max(0, int(measured))
        except (TypeError, ValueError):
            pass
    # The PRE-A3 persistent arm predates action-call instrumentation.  Its
    # result has metrics={} and candidate_steps=0, so those zeroes must not
    # override the actual number of OFT-controlled environment steps.
    if operator == "OFT_PERSISTENT":
        return max(0, int(row.get("env_steps", 0)))
    for value in (row.get("prefix_length"), row.get("candidate_steps")):
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                pass
    fixed = nominal_steps(operator)
    if fixed is not None:
        return fixed
    # Persistent takeover runs until termination; env_steps is the closest
    # deployable compute/burden measure in the PRE-A3 scheduler result.
    return max(0, int(row.get("env_steps", 0)))


def load_state_metadata(path: Path | None) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Load the frozen PRE-A3 state-key artifact used to join logical tasks.

    The recovery scheduler only stores opaque state keys.  A state-name prefix
    is not a task ID; task-disjoint claims must use this outcome-independent
    artifact, which is itself tied to the frozen design by design_sha256.
    """
    if path is None:
        return {}, {}
    payload = json.loads(path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"state-key artifact has no records list: {path}")
    metadata: dict[str, dict[str, str]] = {}
    for row in records:
        if not isinstance(row, dict) or row.get("state_key") is None:
            continue
        state = str(row["state_key"])
        task = row.get("logical_task_id", row.get("task_id"))
        if task is None:
            raise ValueError(f"state-key record has no logical task ID: {state}")
        value = {
            "task_id": str(task),
            "suite": str(row.get("suite", "unknown")),
            "split": str(row.get("split", "unknown")),
            "concrete_task_id": str(row.get("concrete_task_id", row.get("pool_task_id", "unknown"))),
            "episode_id": str(row.get("episode_id", "unknown")),
        }
        if state in metadata and metadata[state] != value:
            raise ValueError(f"conflicting state metadata: {state}")
        metadata[state] = value
    provenance = {
        "path": str(path),
        "artifact_version": payload.get("artifact_version"),
        "design_sha256": payload.get("design_sha256"),
        "selection_uses_outcomes": payload.get("selection_uses_outcomes"),
        "n_states": payload.get("n_states"),
        "n_tasks": payload.get("n_tasks"),
    }
    return metadata, provenance


def load_records(root: Path, state_metadata: dict[str, dict[str, str]] | None = None) -> list[dict[str, Any]]:
    state_metadata = state_metadata or {}
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # PRE-A3 scheduler result envelope.  State identity and outcome live in
        # different nested objects and must be joined within the same file.
        scheduler_rows = []
        if isinstance(payload, dict) and isinstance(payload.get("key"), dict) \
                and isinstance(payload.get("result"), dict):
            result = payload["result"]
            if any(key in result for key in SUCCESS_KEYS):
                scheduler_rows.append({**payload.get("key", {}), **result})
        candidate_rows = scheduler_rows or list(walk_json(payload))
        for row in candidate_rows:
            operator = normalize_operator(row, path)
            state = first(row, STATE_KEYS)
            state_meta = state_metadata.get(str(state), {}) if state is not None else {}
            task = first(row, TASK_KEYS, state_meta.get("task_id"))
            suite = first(row, SUITE_KEYS, state_meta.get(
                "suite", infer_from_path(path, root, "suite")))
            if state is None:
                init_state = row.get("init_state_id")
                seed = row.get("seed")
                if task is not None and init_state is not None:
                    state = f"{suite}:{task}:{init_state}:{seed}"
            if state is None or task is None or operator is None:
                continue
            records.append({
                "state_key": str(state),
                "task_id": str(task),
                "suite": str(suite or "unknown"),
                "operator": operator,
                "success": bool_value(first(row, SUCCESS_KEYS)),
                "executed_oft_steps": executed_oft_steps(row, operator),
                "env_steps": max(0, int(row.get("env_steps", 0))),
                "stop_reason": str(row.get("stop_reason", "unknown")),
                "split": state_meta.get("split", "unknown"),
                "concrete_task_id": state_meta.get("concrete_task_id", "unknown"),
                "episode_id": state_meta.get("episode_id", "unknown"),
                "source": str(path),
            })
    if not records:
        raise SystemExit(f"No outcome records found under {root}")
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        key = (row["state_key"], row["operator"])
        if key in dedup and dedup[key]["success"] != row["success"]:
            raise ValueError(f"conflicting duplicate outcome: {key}")
        dedup[key] = row
    return list(dedup.values())


def wilson(successes: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def audit(records: list[dict[str, Any]], min_complete: int, min_gap: float,
          min_winners: int, min_tasks_per_winner: int, max_fixed_harm: float) -> dict[str, Any]:
    matrix: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    meta: dict[str, dict[str, str]] = {}
    for row in records:
        matrix[row["state_key"]][row["operator"]] = row
        meta[row["state_key"]] = {
            "task_id": row["task_id"], "suite": row["suite"],
            "split": row.get("split", "unknown"),
            "concrete_task_id": row.get("concrete_task_id", "unknown"),
            "episode_id": row.get("episode_id", "unknown"),
        }
    operators = sorted({row["operator"] for row in records}, key=lambda x: (x != "CONTINUE", x))
    if "CONTINUE" not in operators:
        return {"status": "not_ready", "reasons": ["strict CONTINUE operator is missing"],
                "n_records": len(records), "operators": operators}
    complete = {key: rows for key, rows in matrix.items() if set(rows) == set(operators)}
    incomplete = {key: sorted(set(operators) - set(rows)) for key, rows in matrix.items()
                  if set(rows) != set(operators)}
    n = len(complete)
    rates = {op: sum(rows[op]["success"] for rows in complete.values()) / max(1, n)
             for op in operators}
    base = rates["CONTINUE"]
    fixed_ops = [op for op in operators if op != "CONTINUE"]
    best_fixed = max(fixed_ops, key=lambda op: rates[op]) if fixed_ops else "CONTINUE"
    best_fixed_rate = rates.get(best_fixed, base)
    oracle_success = sum(any(row["success"] for row in rows.values()) for rows in complete.values())
    oracle_rate = oracle_success / max(1, n)
    winners: Counter[str] = Counter()
    winner_tasks: dict[str, set[str]] = defaultdict(set)
    best_fixed_harm = 0
    per_state = []
    for state, rows in complete.items():
        base_ok = rows["CONTINUE"]["success"]
        successful = [op for op in operators if rows[op]["success"]]
        strict_winners = [op for op in successful if not base_ok and op != "CONTINUE"]
        for op in strict_winners:
            winners[op] += 1
            winner_tasks[op].add(meta[state]["task_id"])
        if base_ok and not rows[best_fixed]["success"]:
            best_fixed_harm += 1
        per_state.append({**meta[state], "state_key": state, "base_success": base_ok,
                          "successful_operators": successful, "strict_winners": strict_winners,
                          "operator_success": {op: bool(rows[op]["success"]) for op in operators},
                          "operator_executed_oft_steps": {
                              op: int(rows[op].get("executed_oft_steps", 0)) for op in operators}})
    gap = oracle_rate - best_fixed_rate
    fixed_harm_rate = best_fixed_harm / max(1, n)
    diverse = [op for op, count in winners.items()
               if count > 0 and len(winner_tasks[op]) >= min_tasks_per_winner]
    reasons = []
    if n < min_complete:
        reasons.append(f"only {n} complete states (<{min_complete})")
    if gap < min_gap:
        reasons.append(f"oracle-minus-best-fixed {gap:.4f} (<{min_gap:.4f})")
    if len(diverse) < min_winners:
        reasons.append(f"only {len(diverse)} task-diverse winning operators (<{min_winners})")
    if fixed_harm_rate > max_fixed_harm:
        reasons.append(f"best-fixed harm {fixed_harm_rate:.4f} (>{max_fixed_harm:.4f})")
    persistent = "OFT_PERSISTENT" if "OFT_PERSISTENT" in operators else None
    finite_ops = sorted((op for op in operators if nominal_steps(op) is not None),
                        key=lambda op: nominal_steps(op))
    finite_recovery_ops = [op for op in finite_ops if op != "CONTINUE"]
    best_fixed_finite = max(finite_ops, key=lambda op: rates[op]) if finite_ops else "CONTINUE"
    finite_oracle_success = sum(any(rows[op]["success"] for op in finite_ops)
                                for rows in complete.values())
    min_success_counts: Counter[str] = Counter()
    oracle_cost = persistent_cost = 0
    nonmonotonic_states = 0
    deterministic_prefix_violations: list[dict[str, Any]] = []
    finite_safe_states = 0
    finite_safe_tasks: set[str] = set()
    cost_comparison_states = 0
    for state, rows in complete.items():
        successful_by_cost = sorted(
            (op for op in operators if rows[op]["success"]),
            key=lambda op: (int(rows[op].get("executed_oft_steps", 0)),
                            nominal_steps(op) is None, op))
        if persistent and rows[persistent]["success"]:
            # Paired burden comparison on exactly the persistent-success set.
            # This prevents failures of both policies from masquerading as free
            # oracle savings.
            cost_comparison_states += 1
            persistent_cost += int(rows[persistent].get("executed_oft_steps", 0))
            if successful_by_cost:
                chosen = successful_by_cost[0]
                min_success_counts[chosen] += 1
                oracle_cost += int(rows[chosen].get("executed_oft_steps", 0))
            if any(rows[op]["success"] for op in finite_recovery_ops):
                finite_safe_states += 1
                finite_safe_tasks.add(meta[state]["task_id"])
        finite_outcomes = [bool(rows[op]["success"]) for op in finite_ops]
        # A shorter successful handback followed by failure at a longer prefix
        # violates monotonic-duration assumptions and must be modeled explicitly.
        if any(finite_outcomes[i] and not finite_outcomes[j]
               for i in range(len(finite_outcomes)) for j in range(i + 1, len(finite_outcomes))):
            nonmonotonic_states += 1
        # For a deterministic OFT policy, finite arm h and persistent OFT must
        # have exactly the same OFT-controlled prefix through step h.  If
        # persistent succeeds at step t <= h, arm h cannot fail: handback has
        # not occurred yet.  Violations mean the counterfactual matrix is not
        # internally coherent enough to supervise a safe-handback model.
        if persistent and rows[persistent]["success"]:
            persistent_success_step = int(rows[persistent].get("env_steps", 0))
            violated_ops = [
                op for op in finite_recovery_ops
                if nominal_steps(op) is not None
                and persistent_success_step <= int(nominal_steps(op) or 0)
                and not rows[op]["success"]
            ]
            if violated_ops:
                deterministic_prefix_violations.append({
                    "state_key": state,
                    "task_id": meta[state]["task_id"],
                    "suite": meta[state]["suite"],
                    "persistent_success_step": persistent_success_step,
                    "violating_finite_operators": violated_ops,
                })
    cost_savings = (1.0 - oracle_cost / persistent_cost) if persistent_cost > 0 else 0.0
    safe_reasons = []
    if persistent is None:
        safe_reasons.append("persistent OFT upper bound is missing")
    if finite_safe_states < 20:
        safe_reasons.append(f"only {finite_safe_states} finite-safe states (<20)")
    if len(finite_safe_tasks) < 3:
        safe_reasons.append(f"only {len(finite_safe_tasks)} true tasks have finite-safe states (<3)")
    if cost_savings < 0.20:
        safe_reasons.append(f"oracle OFT-step savings {cost_savings:.4f} (<0.20)")
    duration_bins = [op for op, count in min_success_counts.items()
                     if op not in {"CONTINUE", "OFT_PERSISTENT"} and count >= 3]
    if len(duration_bins) < 2:
        safe_reasons.append(f"only {len(duration_bins)} populated finite stopping bins (<2)")
    if deterministic_prefix_violations:
        safe_reasons.append(
            f"{len(deterministic_prefix_violations)} states violate deterministic "
            "OFT-prefix consistency"
        )
    return {
        "schema_version": "rase-pre-a3-opportunity-audit/v1",
        "status": "ready" if not reasons else "not_ready",
        "reasons": reasons,
        "n_records": len(records), "n_states": len(matrix), "n_complete_states": n,
        "n_incomplete_states": len(incomplete), "incomplete_examples": dict(list(incomplete.items())[:10]),
        "operators": operators, "success_rates": rates, "base_rate": base,
        "best_fixed_operator": best_fixed, "best_fixed_rate": best_fixed_rate,
        "oracle_rate": oracle_rate, "oracle_ci95": wilson(oracle_success, n),
        "oracle_minus_best_fixed": gap, "best_fixed_harm_rate": fixed_harm_rate,
        "strict_rescues_by_operator": dict(winners),
        "winning_task_counts": {op: len(tasks) for op, tasks in winner_tasks.items()},
        "task_diverse_winning_operators": sorted(diverse), "per_state": per_state,
        "success_selector_status": "ready" if not reasons else "not_ready",
        "safe_handback_status": "ready" if not safe_reasons else "not_ready",
        "safe_handback_reasons": safe_reasons,
        "best_fixed_finite_operator": best_fixed_finite,
        "best_fixed_finite_rate": rates.get(best_fixed_finite, base),
        "finite_oracle_rate": finite_oracle_success / max(1, n),
        "finite_oracle_minus_best_fixed_finite": (
            finite_oracle_success / max(1, n) - rates.get(best_fixed_finite, base)),
        "finite_safe_states": finite_safe_states,
        "finite_safe_task_count": len(finite_safe_tasks),
        "finite_safe_tasks": sorted(finite_safe_tasks),
        "minimum_successful_operator_counts": dict(min_success_counts),
        "nonmonotonic_duration_states": nonmonotonic_states,
        "deterministic_prefix_consistency_status": (
            "ready" if not deterministic_prefix_violations else "not_ready"),
        "deterministic_prefix_violation_states": len(deterministic_prefix_violations),
        "deterministic_prefix_violations": deterministic_prefix_violations,
        "cost_comparison_states": cost_comparison_states,
        "persistent_total_executed_oft_steps": persistent_cost,
        "oracle_minimum_total_executed_oft_steps": oracle_cost,
        "oracle_oft_step_savings_fraction": cost_savings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--state-keys", type=Path, help=(
        "Frozen PRE-A3 state-key artifact for outcome-independent logical task joins"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--matrix-output", type=Path)
    ap.add_argument("--min-complete-states", type=int, default=60)
    ap.add_argument("--min-oracle-gap", type=float, default=0.05)
    ap.add_argument("--min-winning-operators", type=int, default=2)
    ap.add_argument("--min-tasks-per-winning-operator", type=int, default=2)
    ap.add_argument("--max-best-fixed-harm", type=float, default=0.05)
    ap.add_argument(
        "--exclude-prefix-inconsistent",
        action="store_true",
        help=("QC-filter states that violate deterministic OFT-prefix consistency, "
              "while recording every exclusion and the unfiltered status"),
    )
    args = ap.parse_args()
    state_metadata, state_keys_provenance = load_state_metadata(
        args.state_keys.resolve() if args.state_keys else None)
    records = load_records(args.input_root.resolve(), state_metadata)
    report = audit(records, args.min_complete_states, args.min_oracle_gap,
                   args.min_winning_operators, args.min_tasks_per_winning_operator,
                   args.max_best_fixed_harm)
    if args.exclude_prefix_inconsistent:
        violations = list(report.get("deterministic_prefix_violations") or [])
        excluded = {str(row["state_key"]) for row in violations}
        if excluded:
            raw_summary = {
                "n_complete_states": report.get("n_complete_states"),
                "safe_handback_status": report.get("safe_handback_status"),
                "safe_handback_reasons": report.get("safe_handback_reasons"),
                "deterministic_prefix_violation_states": len(excluded),
                "deterministic_prefix_violations": violations,
            }
            filtered = [row for row in records if row["state_key"] not in excluded]
            report = audit(
                filtered, args.min_complete_states, args.min_oracle_gap,
                args.min_winning_operators, args.min_tasks_per_winning_operator,
                args.max_best_fixed_harm,
            )
            report["qc_exclusion_policy"] = "deterministic_oft_prefix_inconsistency"
            report["qc_excluded_state_keys"] = sorted(excluded)
            report["qc_excluded_state_count"] = len(excluded)
            report["raw_before_qc_exclusion"] = raw_summary
        else:
            report["qc_exclusion_policy"] = "deterministic_oft_prefix_inconsistency"
            report["qc_excluded_state_keys"] = []
            report["qc_excluded_state_count"] = 0
    report["state_keys_provenance"] = state_keys_provenance
    report["true_task_join_status"] = "ready" if state_keys_provenance else "not_ready"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.matrix_output:
        args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
        with args.matrix_output.open("w") as handle:
            for row in report.get("per_state", []):
                handle.write(json.dumps(row) + "\n")
    print(json.dumps({key: report.get(key) for key in (
        "status", "reasons", "n_complete_states", "operators", "best_fixed_operator",
        "best_fixed_rate", "oracle_rate", "oracle_minus_best_fixed",
        "best_fixed_harm_rate", "task_diverse_winning_operators",
        "safe_handback_status", "safe_handback_reasons",
        "best_fixed_finite_operator", "best_fixed_finite_rate", "finite_oracle_rate",
        "finite_oracle_minus_best_fixed_finite", "finite_safe_states",
        "finite_safe_task_count",
        "minimum_successful_operator_counts", "nonmonotonic_duration_states",
        "deterministic_prefix_consistency_status",
        "deterministic_prefix_violation_states",
        "oracle_oft_step_savings_fraction")}, indent=2))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
