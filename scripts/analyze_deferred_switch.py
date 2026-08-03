#!/usr/bin/env python3
"""Audit and summarize immediate versus decision-suffix OFT switching."""

from __future__ import annotations

import argparse
import json
import math
import os
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


def _mcnemar_exact(a_only: int, b_only: int) -> float:
    disagreements = a_only + b_only
    if disagreements == 0:
        return 1.0
    tail = sum(
        math.comb(disagreements, value)
        for value in range(min(a_only, b_only) + 1)
    ) / (2**disagreements)
    return min(1.0, 2 * tail)


def _arm(row: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [arm for arm in row.get("arms") or [] if arm.get("arm_label") == label]
    if len(matches) != 1:
        raise ValueError(f"state {row.get('state_key')} requires exactly one {label} arm")
    return dict(matches[0])


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["classification"]) for row in rows)
    direct = sum(bool(row["direct_oft_success"]) for row in rows)
    deferred = sum(bool(row["decision_suffix_oft_success"]) for row in rows)
    oracle = sum(
        bool(row["direct_oft_success"] or row["decision_suffix_oft_success"])
        for row in rows
    )
    best = max(direct, deferred)
    return {
        "n_states": len(rows),
        "direct_oft_successes": direct,
        "decision_suffix_oft_successes": deferred,
        "same_state_oracle_successes": oracle,
        "direct_oft_success_rate": direct / len(rows),
        "decision_suffix_oft_success_rate": deferred / len(rows),
        "same_state_oracle_success_rate": oracle / len(rows),
        "oracle_minus_best_fixed": (oracle - best) / len(rows),
        "classification_counts": {
            label: int(counts.get(label, 0))
            for label in ("neither", "direct_only", "deferred_only", "both")
        },
    }


_THREE_OPERATORS = (
    "continue_smol_active_chunk",
    "direct_oft",
    "decision_suffix_oft",
)


def _group_three(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = {
        label: sum(bool(row[f"{label}_success"]) for row in rows)
        for label in _THREE_OPERATORS
    }
    oracle = sum(
        any(bool(row[f"{label}_success"]) for label in _THREE_OPERATORS)
        for row in rows
    )
    patterns = Counter(
        "C{}D{}S{}".format(
            int(bool(row["continue_smol_active_chunk_success"])),
            int(bool(row["direct_oft_success"])),
            int(bool(row["decision_suffix_oft_success"])),
        )
        for row in rows
    )
    unique = {
        label: [
            str(row["state_key"])
            for row in rows
            if bool(row[f"{label}_success"])
            and sum(
                bool(row[f"{other}_success"]) for other in _THREE_OPERATORS
            )
            == 1
        ]
        for label in _THREE_OPERATORS
    }
    best = max(successes.values())
    return {
        "n_states": len(rows),
        "successes": successes,
        "success_rates": {
            label: value / len(rows) for label, value in successes.items()
        },
        "same_state_oracle_successes": oracle,
        "same_state_oracle_success_rate": oracle / len(rows),
        "oracle_minus_best_fixed": (oracle - best) / len(rows),
        "success_pattern_counts": dict(sorted(patterns.items())),
        "unique_success_state_keys": unique,
        "unique_success_task_counts": {
            label: len(
                {
                    str(row["task_id"])
                    for row in rows
                    if str(row["state_key"]) in set(keys)
                }
            )
            for label, keys in unique.items()
        },
    }


def _three_operator_analysis(
    key_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    continue_summary: dict[str, Any],
) -> dict[str, Any]:
    if continue_summary.get("schema_version") not in {
        "rase-smol-intervention-summary/v1",
        "rase-smol-intervention-summary/v2",
    }:
        raise ValueError("unexpected strict-CONTINUE summary schema")
    if continue_summary.get("status") != "complete":
        raise ValueError("strict-CONTINUE summary is incomplete")
    keys = {str(row["state_key"]) for row in rows}
    continue_pairs = [
        pair
        for pair in (continue_summary.get("per_pair") or [])
        if str(pair.get("state_key")) in keys
    ]
    continue_by_key = {
        str(pair["state_key"]): bool(pair["continue_smol_active_chunk"])
        for pair in continue_pairs
    }
    if len(continue_by_key) != len(continue_pairs) or set(continue_by_key) != keys:
        raise ValueError("strict-CONTINUE state coverage differs from deferred-switch rows")
    metadata_by_key = {
        str(record["state_key"]): record
        for record in key_payload.get("records") or []
        if str(record.get("state_key")) in keys
    }
    if set(metadata_by_key) != keys:
        raise ValueError("frozen-key records are required for three-operator analysis")
    enriched = []
    for row in rows:
        key = str(row["state_key"])
        enriched.append(
            {
                **row,
                "task_id": str(metadata_by_key[key]["task_id"]),
                "continue_smol_active_chunk_success": continue_by_key[key],
            }
        )
    groups: dict[str, dict[str, Any]] = {"suite": {}, "dimension_level": {}}
    for field in groups:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in enriched:
            group = (
                str(row["suite"])
                if field == "suite"
                else f"{row['dim']}:L{row['level']}"
            )
            grouped[group].append(row)
        groups[field] = {
            group: _group_three(values) for group, values in sorted(grouped.items())
        }
    return {
        "operator_order": list(_THREE_OPERATORS),
        "overall": _group_three(enriched),
        "by_group": groups,
        "per_state": enriched,
    }


def analyze(
    key_payload: dict[str, Any],
    summaries: list[dict[str, Any]],
    continue_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    keys = [str(value) for value in key_payload.get("state_keys") or []]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("state-key cohort must be non-empty and unique")
    rows = []
    for summary in summaries:
        if summary.get("schema_version") != "rase-oft-decision-suffix/v1":
            raise ValueError("unexpected deferred-switch schema")
        if summary.get("status") != "complete":
            raise ValueError("deferred-switch summary is incomplete")
        rows.extend(dict(row) for row in summary.get("per_state") or [])
    observed = [str(row.get("state_key")) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != set(keys):
        raise ValueError("deferred-switch state coverage differs from frozen keys")
    order = {key: index for index, key in enumerate(keys)}
    rows.sort(key=lambda row: order[str(row["state_key"])])

    prefix_lengths = Counter()
    parity_failures = []
    arm_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        direct = _arm(row, "direct_oft")
        deferred = _arm(row, "decision_suffix_oft")
        arm_rows["direct_oft"].append(direct)
        arm_rows["decision_suffix_oft"].append(deferred)
        prefix_steps = int(deferred.get("prefix_steps", -1))
        candidate_steps = int(deferred.get("candidate_steps", -1))
        calls = int(deferred.get("oracle_predict_calls", -1))
        prefix_lengths[prefix_steps] += 1
        valid_early_terminal = (
            bool(deferred.get("success"))
            and bool(deferred.get("terminal_during_prefix"))
            and calls == 0
            and 0 <= candidate_steps < prefix_steps
        )
        valid_complete = (
            bool(deferred.get("prefix_completed"))
            and candidate_steps == prefix_steps
            and prefix_steps > 0
        )
        valid = (
            deferred.get("prefix_source")
            == "decision_context.active_action_suffix"
            and len(str(deferred.get("prefix_sha256") or "")) == 64
            and (valid_complete or valid_early_terminal)
            and int(direct.get("prefix_steps", -1)) == 0
        )
        if not valid:
            parity_failures.append(str(row["state_key"]))
    if parity_failures:
        raise ValueError(f"deferred-switch prefix parity failed: {parity_failures}")

    timing = {}
    for label, values in arm_rows.items():
        calls = sum(int(row["oracle_predict_calls"]) for row in values)
        policy_s = sum(float(row["oracle_predict_elapsed_s"]) for row in values)
        env_steps = sum(int(row["env_steps"]) for row in values)
        timing[label] = {
            "predict_calls": calls,
            "predict_elapsed_s": policy_s,
            "mean_ms_per_predict_call": 1000 * policy_s / calls if calls else None,
            "policy_ms_per_env_step": 1000 * policy_s / env_steps if env_steps else None,
            "mean_rollout_elapsed_s": sum(float(row["elapsed_s"]) for row in values)
            / len(values),
        }

    overall = _group(rows)
    groups = {"suite": {}, "dimension_level": {}}
    for field in groups:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (
                str(row["suite"])
                if field == "suite"
                else f"{row['dim']}:L{row['level']}"
            )
            grouped[key].append(row)
        groups[field] = {key: _group(value) for key, value in sorted(grouped.items())}
    direct_only = overall["classification_counts"]["direct_only"]
    deferred_only = overall["classification_counts"]["deferred_only"]
    result = {
        "schema_version": "rase-deferred-switch-analysis/v1",
        "status": "complete",
        "n_states": len(rows),
        "n_episodes": len({str(row["episode_id"]) for row in rows}),
        "overall": overall,
        "mcnemar_exact_p": _mcnemar_exact(direct_only, deferred_only),
        "prefix_parity": {
            "status": "pass",
            "source": "decision_context.active_action_suffix",
            "n_valid_states": len(rows),
            "prefix_length_counts": {
                str(key): value for key, value in sorted(prefix_lengths.items())
            },
        },
        "timing": timing,
        "by_group": groups,
        "per_state": rows,
        "use_for": "operator-semantics calibration only; not independent confirmation",
    }
    if continue_summary is not None:
        result["three_operator"] = _three_operator_analysis(
            key_payload, rows, continue_summary
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument(
        "--continue-summary",
        type=Path,
        help="optional paired strict-CONTINUE summary on the identical state keys",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        _read(args.state_keys_json.resolve()),
        [_read(path.resolve()) for path in args.summary],
        _read(args.continue_summary.resolve()) if args.continue_summary else None,
    )
    _write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "overall": result["overall"],
                "mcnemar_exact_p": result["mcnemar_exact_p"],
                "prefix_parity": result["prefix_parity"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
