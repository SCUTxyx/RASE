#!/usr/bin/env python3
"""Join Smol action-selection and OFT RPC timing on an exact state-key cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


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


def _stats(
    rows: list[dict[str, Any]],
    *,
    elapsed_field: str,
    calls_field: str,
    success_field: str,
    rollout_field: str,
    env_steps_field: str,
    measurement_scope: str,
) -> dict[str, Any]:
    elapsed = np.asarray([float(row[elapsed_field]) for row in rows], dtype=float)
    rollout = np.asarray([float(row[rollout_field]) for row in rows], dtype=float)
    calls = sum(int(row[calls_field]) for row in rows)
    env_steps = sum(int(row[env_steps_field]) for row in rows)
    successes = sum(bool(row[success_field]) for row in rows)
    return {
        "measurement_scope": measurement_scope,
        "n_trials": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "total_policy_time_s": float(elapsed.sum()),
        "mean_policy_time_s_per_trial": float(elapsed.mean()) if rows else None,
        "median_policy_time_s_per_trial": float(np.median(elapsed)) if rows else None,
        "p90_policy_time_s_per_trial": (
            float(np.percentile(elapsed, 90)) if rows else None
        ),
        "total_policy_calls": calls,
        "mean_ms_per_policy_call": (
            1000 * float(elapsed.sum()) / calls if calls else None
        ),
        "total_env_steps": env_steps,
        "policy_ms_per_env_step": (
            1000 * float(elapsed.sum()) / env_steps if env_steps else None
        ),
        "mean_full_rollout_s": float(rollout.mean()) if rows else None,
    }


def analyze(
    key_payload: dict[str, Any],
    smol_summary: dict[str, Any],
    oft_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = [str(value) for value in key_payload.get("state_keys") or []]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("state-key cohort must be non-empty and unique")
    expected = set(keys)
    smol_by_key = {
        str(row["state_key"]): row for row in smol_summary.get("per_pair") or []
    }
    if set(smol_by_key) != expected:
        raise ValueError("Smol timing coverage differs from state-key cohort")
    oft_by_key = {}
    for summary in oft_summaries:
        for row in summary.get("per_state") or []:
            key = str(row["state_key"])
            if key in expected:
                if key in oft_by_key:
                    raise ValueError(f"duplicate OFT timing state: {key}")
                oft_by_key[key] = row
    if set(oft_by_key) != expected:
        raise ValueError("OFT timing coverage differs from state-key cohort")

    operators = {}
    scope = (
        "wall time inside SmolVLA select_env_action; includes cached action queue "
        "access and model forward passes, excludes environment stepping"
    )
    for operator in ("continue_smol_active_chunk", "replan_smol"):
        rows = [smol_by_key[key] for key in keys]
        required = [
            f"{operator}_action_select_elapsed_s",
            f"{operator}_action_select_calls",
        ]
        if any(any(field not in row for field in required) for row in rows):
            raise ValueError(f"Smol timing instrumentation incomplete for {operator}")
        operators[operator] = _stats(
            rows,
            elapsed_field=f"{operator}_action_select_elapsed_s",
            calls_field=f"{operator}_action_select_calls",
            success_field=operator,
            rollout_field=f"{operator}_latency_seconds",
            env_steps_field=f"{operator}_env_steps",
            measurement_scope=scope,
        )

    oft_rows = []
    for key in keys:
        source = oft_by_key[key]
        result = dict(source.get("result") or {})
        if "oracle_predict_calls" not in result or "oracle_predict_elapsed_s" not in result:
            raise ValueError(f"OFT timing instrumentation incomplete for {key}")
        oft_rows.append(
            {
                **result,
                "success": bool(source["direct_oft_success"]),
            }
        )
    operators["switch_oft"] = _stats(
        oft_rows,
        elapsed_field="oracle_predict_elapsed_s",
        calls_field="oracle_predict_calls",
        success_field="success",
        rollout_field="elapsed_s",
        env_steps_field="env_steps",
        measurement_scope=(
            "OFT action-prediction RPC transfer plus server inference; excludes "
            "environment stepping and client rollout control"
        ),
    )
    reference = operators["continue_smol_active_chunk"]
    relative_to_continue = {}
    for operator, values in operators.items():
        relative_to_continue[operator] = {
            metric: values[metric] / reference[metric]
            for metric in (
                "mean_policy_time_s_per_trial",
                "policy_ms_per_env_step",
                "mean_full_rollout_s",
            )
            if values[metric] is not None and reference[metric]
        }
    records = key_payload.get("records") or []
    return {
        "schema_version": "rase-intervention-timing-analysis/v1",
        "status": "complete",
        "n_states": len(keys),
        "n_episodes": len({str(row["episode_id"]) for row in records}),
        "n_tasks": len({str(row["task_id"]) for row in records}),
        "operators": operators,
        "relative_to_continue": relative_to_continue,
        "normalization_note": (
            "Per-env-step policy time captures action-chunk amortization. Per-call "
            "latency is not cross-operator comparable because Smol selects one action "
            "through a cached queue while OFT returns an action chunk per RPC."
        ),
        "comparability_warning": (
            "Smol times a single-action API with cached-queue accesses whereas OFT "
            "times chunk-prediction RPCs. Compare total policy acquisition time and "
            "deployment latency with explicit scope; do not equate per-call values or "
            "use outcome-dependent full rollout duration as inference cost."
        ),
        "use_for": "physical-cost calibration only; not opportunity confirmation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--smol-summary", type=Path, required=True)
    parser.add_argument("--oft-summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        _read(args.state_keys_json.resolve()),
        _read(args.smol_summary.resolve()),
        [_read(path.resolve()) for path in args.oft_summary],
    )
    _write(args.output.resolve(), result)
    print(json.dumps({"output": str(args.output.resolve()), **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
