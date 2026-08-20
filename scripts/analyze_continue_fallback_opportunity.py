#!/usr/bin/env python3
"""Measure within-task opportunity for strict CONTINUE versus direct fallback."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _direct_arm(row: dict[str, Any]) -> dict[str, Any]:
    if "result" in row:
        arm = dict(row["result"])
    else:
        matches = [
            arm
            for arm in row.get("arms") or []
            if arm.get("arm_label") == "direct_oft"
        ]
        if len(matches) != 1:
            raise ValueError(f"state {row.get('state_key')} requires one direct_oft arm")
        arm = dict(matches[0])
    if (
        arm.get("prefix_source") != "direct"
        or int(arm.get("prefix_steps", -1)) != 0
        or int(arm.get("env_steps", -1)) < 0
        or arm.get("stop_reason") is None
    ):
        raise ValueError(f"state {row.get('state_key')} has invalid direct rollout provenance")
    return arm


def analyze(
    key_payload: dict[str, Any],
    continue_summary: dict[str, Any],
    fallback_summaries: list[dict[str, Any]],
    *,
    min_heterogeneity: float = 0.05,
    min_oracle_gain: float = 0.05,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260820,
) -> dict[str, Any]:
    records = key_payload.get("records") or []
    metadata = {str(row["state_key"]): dict(row) for row in records}
    frozen_keys = [str(value) for value in key_payload.get("state_keys") or []]
    if not frozen_keys or len(frozen_keys) != len(set(frozen_keys)):
        raise ValueError("frozen keys must be non-empty and unique")
    if set(metadata) != set(frozen_keys):
        raise ValueError("frozen key metadata coverage mismatch")

    if continue_summary.get("status") != "complete":
        raise ValueError("CONTINUE summary is incomplete")
    continue_rows = continue_summary.get("per_pair") or []
    continue_by_key: dict[str, dict[str, Any]] = {}
    for row in continue_rows:
        key = str(row.get("state_key"))
        if key in continue_by_key:
            raise ValueError(f"duplicate CONTINUE result for {key}")
        continue_by_key[key] = dict(row)
    if set(continue_by_key) != set(frozen_keys):
        raise ValueError("CONTINUE state coverage differs from frozen keys")

    fallback_by_key: dict[str, dict[str, Any]] = {}
    for summary in fallback_summaries:
        if summary.get("schema_version") not in {
            "rase-oft-direct-escalation/v1",
            "rase-oft-decision-suffix/v1",
        }:
            raise ValueError("unexpected OFT summary schema")
        if summary.get("status") != "complete":
            raise ValueError("OFT summary is incomplete")
        for row in summary.get("per_state") or []:
            key = str(row.get("state_key"))
            if key in fallback_by_key:
                raise ValueError(f"duplicate fallback result for {key}")
            fallback_by_key[key] = dict(row)
    if set(fallback_by_key) != set(frozen_keys):
        raise ValueError("fallback state coverage differs from frozen keys")

    per_state = []
    for key in frozen_keys:
        meta = metadata[key]
        continue_row = continue_by_key[key]
        fallback_row = fallback_by_key[key]
        direct = _direct_arm(fallback_row)
        continue_success = bool(continue_row["continue_smol_active_chunk"])
        fallback_success = bool(direct["success"])
        if continue_success and not fallback_success:
            winner = "continue"
        elif fallback_success and not continue_success:
            winner = "fallback"
        else:
            winner = "tie"
        per_state.append(
            {
                "state_key": key,
                "task_id": str(meta["task_id"]),
                "episode_id": str(meta["episode_id"]),
                "suite": str(meta["suite"]),
                "step": int(meta["step"]),
                "perturbation_dimension": str(meta["perturbation_dimension"]),
                "perturbation_level": int(meta["perturbation_level"]),
                "continue_success": continue_success,
                "fallback_success": fallback_success,
                "winner": winner,
                "continue_env_steps": int(
                    continue_row["continue_smol_active_chunk_env_steps"]
                ),
                "fallback_env_steps": int(direct["env_steps"]),
                "fallback_stop_reason": str(direct["stop_reason"]),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_state:
        grouped[row["task_id"]].append(row)
    task_rows = []
    for task_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["step"])
        winners = {row["winner"] for row in rows if row["winner"] != "tie"}
        continue_successes = sum(row["continue_success"] for row in rows)
        fallback_successes = sum(row["fallback_success"] for row in rows)
        oracle_successes = sum(
            row["continue_success"] or row["fallback_success"] for row in rows
        )
        best_fixed_successes = max(continue_successes, fallback_successes)
        task_rows.append(
            {
                "task_id": task_id,
                "suite": rows[0]["suite"],
                "perturbation_dimension": rows[0]["perturbation_dimension"],
                "perturbation_level": rows[0]["perturbation_level"],
                "n_states": len(rows),
                "steps": [row["step"] for row in rows],
                "winner_counts": dict(Counter(row["winner"] for row in rows)),
                "has_informative_state": bool(winners),
                "within_task_winner_flip": winners == {"continue", "fallback"},
                "continue_success_rate": continue_successes / len(rows),
                "fallback_success_rate": fallback_successes / len(rows),
                "state_oracle_success_rate": oracle_successes / len(rows),
                "task_best_fixed_success_rate": best_fixed_successes / len(rows),
                "oracle_minus_task_best_fixed": (
                    oracle_successes - best_fixed_successes
                ) / len(rows),
            }
        )

    n_tasks = len(task_rows)
    if n_tasks == 0:
        raise ValueError("no tasks in cohort")
    state_oracle = sum(
        row["continue_success"] or row["fallback_success"] for row in per_state
    ) / len(per_state)
    best_fixed = sum(
        row["task_best_fixed_success_rate"] for row in task_rows
    ) / n_tasks
    h_within = sum(row["within_task_winner_flip"] for row in task_rows) / n_tasks
    oracle_gain = state_oracle - best_fixed

    def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        task_ids = {row["task_id"] for row in rows}
        tasks = [row for row in task_rows if row["task_id"] in task_ids]
        oracle_rate = sum(
            row["continue_success"] or row["fallback_success"] for row in rows
        ) / len(rows)
        task_fixed_rate = sum(
            row["task_best_fixed_success_rate"] for row in tasks
        ) / len(tasks)
        return {
            "n_states": len(rows),
            "n_tasks": len(tasks),
            "continue_success_rate": sum(row["continue_success"] for row in rows)
            / len(rows),
            "fallback_success_rate": sum(row["fallback_success"] for row in rows)
            / len(rows),
            "state_oracle_success_rate": oracle_rate,
            "task_best_fixed_success_rate": task_fixed_rate,
            "oracle_minus_task_best_fixed": oracle_rate - task_fixed_rate,
            "within_task_heterogeneity": sum(
                row["within_task_winner_flip"] for row in tasks
            )
            / len(tasks),
            "winner_counts": dict(Counter(row["winner"] for row in rows)),
            "fallback_weakly_dominates_every_state": not any(
                row["winner"] == "continue" for row in rows
            ),
        }

    by_group = {}
    for field in ("suite", "perturbation_dimension"):
        values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in per_state:
            values[str(row[field])].append(row)
        by_group[field] = {
            value: summarize_group(rows) for value, rows in sorted(values.items())
        }
    rng = random.Random(bootstrap_seed)
    boot_h = []
    boot_gain = []
    for _ in range(bootstrap_replicates):
        sample = [task_rows[rng.randrange(n_tasks)] for _ in range(n_tasks)]
        boot_h.append(sum(row["within_task_winner_flip"] for row in sample) / n_tasks)
        boot_gain.append(
            sum(row["oracle_minus_task_best_fixed"] for row in sample) / n_tasks
        )

    return {
        "schema_version": "rase-continue-fallback-opportunity/v1",
        "status": "pass"
        if h_within >= min_heterogeneity and oracle_gain >= min_oracle_gain
        else "fail",
        "gate": {
            "min_within_task_heterogeneity": min_heterogeneity,
            "min_oracle_gain": min_oracle_gain,
            "within_task_heterogeneity_pass": h_within >= min_heterogeneity,
            "oracle_gain_pass": oracle_gain >= min_oracle_gain,
        },
        "cohort": {
            "n_states": len(per_state),
            "n_tasks": n_tasks,
            "states_per_task": dict(Counter(row["n_states"] for row in task_rows)),
            "suites": dict(Counter(row["suite"] for row in task_rows)),
            "perturbation_dimensions": dict(
                Counter(row["perturbation_dimension"] for row in task_rows)
            ),
        },
        "metrics": {
            "within_task_heterogeneity": h_within,
            "within_task_heterogeneity_ci95_task_bootstrap": [
                _percentile(boot_h, 0.025),
                _percentile(boot_h, 0.975),
            ],
            "n_tasks_with_winner_flip": sum(
                row["within_task_winner_flip"] for row in task_rows
            ),
            "n_tasks_with_informative_state": sum(
                row["has_informative_state"] for row in task_rows
            ),
            "continue_success_rate": sum(row["continue_success"] for row in per_state)
            / len(per_state),
            "fallback_success_rate": sum(row["fallback_success"] for row in per_state)
            / len(per_state),
            "state_oracle_success_rate": state_oracle,
            "task_best_fixed_success_rate": best_fixed,
            "oracle_minus_task_best_fixed": oracle_gain,
            "oracle_gain_ci95_task_bootstrap": [
                _percentile(boot_gain, 0.025),
                _percentile(boot_gain, 0.975),
            ],
            "state_winner_counts": dict(Counter(row["winner"] for row in per_state)),
            "fallback_weakly_dominates_every_state": not any(
                row["winner"] == "continue" for row in per_state
            ),
        },
        "by_group": by_group,
        "label_provenance": {
            "continue": "full restored-state strict active-chunk continuation rollout success",
            "fallback": "full restored-state direct OFT rollout success",
            "training_label_uses_cost": False,
            "proxy_consequence_label_used": False,
            "task_id_used_as_selector_feature": False,
        },
        "bootstrap": {
            "unit": "task",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "per_task": task_rows,
        "per_state": per_state,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--continue-summary", type=Path, required=True)
    parser.add_argument("--fallback-summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-heterogeneity", type=float, default=0.05)
    parser.add_argument("--min-oracle-gain", type=float, default=0.05)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")
    result = analyze(
        _read(args.state_keys_json),
        _read(args.continue_summary),
        [_read(path) for path in args.fallback_summary],
        min_heterogeneity=args.min_heterogeneity,
        min_oracle_gain=args.min_oracle_gain,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write(args.output, result)
    print(json.dumps({"status": result["status"], **result["metrics"]}), flush=True)
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
