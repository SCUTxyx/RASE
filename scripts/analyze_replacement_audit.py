#!/usr/bin/env python3
"""Analyze SOURCE-ONLY, OFT-ONLY, and source-to-OFT replacement outcomes."""

from __future__ import annotations

import argparse
import json
import math
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


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use SUITE=PATH")
    suite, raw = value.split("=", 1)
    return suite, Path(raw)


def _mcnemar_exact(a_only: int, b_only: int) -> float:
    disagreements = a_only + b_only
    if disagreements == 0:
        return 1.0
    tail = sum(
        math.comb(disagreements, value)
        for value in range(min(a_only, b_only) + 1)
    ) / (2**disagreements)
    return min(1.0, 2 * tail)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return float(ordered[index])


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    a: str,
    b: str,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    n_rows = len(rows)
    values = []
    for _ in range(replicates):
        sampled = [rows[rng.randrange(n_rows)] for _ in range(n_rows)]
        values.append(
            sum(int(row[a]) - int(row[b]) for row in sampled) / n_rows
        )
    return values


def _pair_summary(
    rows: list[dict[str, Any]],
    a: str,
    b: str,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    a_hits = sum(bool(row[a]) for row in rows)
    b_hits = sum(bool(row[b]) for row in rows)
    a_only = sum(bool(row[a]) and not bool(row[b]) for row in rows)
    b_only = sum(bool(row[b]) and not bool(row[a]) for row in rows)
    both = sum(bool(row[a]) and bool(row[b]) for row in rows)
    neither = len(rows) - a_only - b_only - both
    bootstrap = _paired_bootstrap(
        rows,
        a,
        b,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "a": a,
        "b": b,
        "n_tasks": len(rows),
        "a_successes": a_hits,
        "b_successes": b_hits,
        "paired_difference": (a_hits - b_hits) / len(rows),
        "paired_difference_bootstrap_95_ci": [
            _percentile(bootstrap, 0.025),
            _percentile(bootstrap, 0.975),
        ],
        "a_only": a_only,
        "b_only": b_only,
        "both": both,
        "neither": neither,
        "mcnemar_exact_p": _mcnemar_exact(a_only, b_only),
    }


def _timing_from_oft_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = [dict(row["result"]) for row in rows]
    calls = sum(int(row["oracle_predict_calls"]) for row in records)
    predict_s = sum(float(row["oracle_predict_elapsed_s"]) for row in records)
    env_steps = sum(int(row["env_steps"]) for row in records)
    return {
        "n_episodes": len(records),
        "env_steps": env_steps,
        "predict_calls": calls,
        "predict_elapsed_s": predict_s,
        "policy_ms_per_env_step": 1000 * predict_s / env_steps,
        "mean_episode_rollout_s": sum(float(row["elapsed_s"]) for row in records)
        / len(records),
    }


def _timing_from_source_summary(summary: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in summary.get("episode_metrics") or []]
    if not rows:
        raise ValueError("source collection summary lacks episode timing metrics")
    env_steps = sum(int(row["env_steps"]) for row in rows)
    policy_s = sum(float(row["policy_select_elapsed_s"]) for row in rows)
    return {
        "n_episodes": len(rows),
        "env_steps": env_steps,
        "policy_select_calls": sum(int(row["policy_select_calls"]) for row in rows),
        "policy_select_elapsed_s": policy_s,
        "policy_ms_per_env_step": 1000 * policy_s / env_steps,
        "mean_episode_rollout_s": sum(float(row["episode_wall_s"]) for row in rows)
        / len(rows),
    }


def analyze(
    keys: dict[str, Any],
    source_summary: dict[str, Any],
    oft_summaries: list[tuple[str, dict[str, Any]]],
    handoff_analysis: dict[str, Any],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if keys.get("artifact_version") != "rase-replacement-initial-keys/v1":
        raise ValueError("unexpected replacement-key schema")
    if keys.get("selection_uses_outcomes"):
        raise ValueError("replacement reset-state selection must be outcome-independent")
    key_rows = {str(row["state_key"]): dict(row) for row in keys.get("records") or []}
    episodes = {str(row["episode_id"]): dict(row) for row in key_rows.values()}
    if len(key_rows) != int(keys.get("n_states", -1)) or len(episodes) != len(key_rows):
        raise ValueError("replacement keys are duplicate or incomplete")

    oft_rows: dict[str, dict[str, Any]] = {}
    for suite, summary in oft_summaries:
        if summary.get("schema_version") != "rase-oft-direct-escalation/v1":
            raise ValueError(f"unexpected OFT-only summary schema for {suite}")
        if summary.get("status") != "complete":
            raise ValueError(f"incomplete OFT-only summary for {suite}")
        if summary.get("state_keys_sha256") != keys.get("state_keys_sha256"):
            raise ValueError(f"OFT-only state-key provenance mismatch for {suite}")
        if summary.get("suite") not in {None, suite}:
            raise ValueError(f"OFT-only suite mismatch: {summary.get('suite')} != {suite}")
        for row in summary.get("per_state") or []:
            key = str(row["state_key"])
            if key in oft_rows:
                raise ValueError(f"duplicate OFT-only state: {key}")
            oft_rows[key] = dict(row)
    if set(oft_rows) != set(key_rows):
        raise ValueError("OFT-only coverage differs from frozen reset states")

    if handoff_analysis.get("schema_version") != "rase-deferred-switch-analysis/v1":
        raise ValueError("unexpected source-to-OFT analysis schema")
    handoff_rows = {
        str(row["episode_id"]): dict(row)
        for row in (handoff_analysis.get("three_operator") or {}).get("per_state") or []
    }
    if set(handoff_rows) != set(episodes):
        raise ValueError("source-to-OFT episode coverage differs from replacement cohort")

    rows = []
    for state_key, frozen in key_rows.items():
        episode_id = str(frozen["episode_id"])
        oft = oft_rows[state_key]
        handoff = handoff_rows[episode_id]
        if str(handoff["task_id"]) != str(frozen["task_id"]):
            raise ValueError(f"handoff task identity mismatch for {episode_id}")
        rows.append(
            {
                **frozen,
                "source_only_success": bool(frozen["source_only_success"]),
                "oft_only_success": bool(oft["direct_oft_success"]),
                "source_to_oft_success": bool(handoff["direct_oft_success"]),
                "oft_only_result": dict(oft["result"]),
            }
        )

    methods = ("source_only_success", "oft_only_success", "source_to_oft_success")
    overall_successes = {
        method: sum(bool(row[method]) for row in rows) for method in methods
    }
    quadrants = Counter(
        (
            "rescue"
            if not row["source_only_success"] and row["oft_only_success"]
            else "harm"
            if row["source_only_success"] and not row["oft_only_success"]
            else "redundant"
            if row["source_only_success"] and row["oft_only_success"]
            else "unsupported"
        )
        for row in rows
    )

    grouped: dict[str, dict[str, Any]] = {"suite": {}, "dimension_level": {}}
    for group_name in grouped:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            label = (
                str(row["suite"])
                if group_name == "suite"
                else f"{row['perturbation_dimension']}:L{row['perturbation_level']}"
            )
            buckets[label].append(row)
        grouped[group_name] = {
            label: {
                "n_tasks": len(values),
                "successes": {
                    method: sum(bool(row[method]) for row in values)
                    for method in methods
                },
                "source_vs_oft_only": _pair_summary(
                    values,
                    "source_only_success",
                    "oft_only_success",
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=bootstrap_seed,
                ),
            }
            for label, values in sorted(buckets.items())
        }

    source_unique = [
        row for row in rows if row["source_only_success"] and not row["oft_only_success"]
    ]
    unique_suites = {str(row["suite"]) for row in source_unique}
    clean = [row for row in rows if row["perturbation_dimension"] == "clean"]
    clean_source = sum(bool(row["source_only_success"]) for row in clean)
    clean_oft = sum(bool(row["oft_only_success"]) for row in clean)
    handoff_hits = overall_successes["source_to_oft_success"]
    oft_hits = overall_successes["oft_only_success"]
    if len(source_unique) >= 2 and len(unique_suites) >= 2:
        gate_status = "recovery_framing_signal"
        reasons = ["SOURCE-ONLY has at least two OFT-ONLY-unique wins across two suites"]
    elif (
        not source_unique
        and oft_hits >= overall_successes["source_only_success"]
        and clean_oft >= clean_source
        and oft_hits >= handoff_hits
    ):
        gate_status = "replacement_risk_high_cost_audit_required"
        reasons = [
            "OFT-ONLY weakly dominates SOURCE-ONLY overall and on clean tasks",
            "SOURCE-ONLY contributes no unique success over OFT-ONLY",
            "source-to-OFT does not exceed OFT-ONLY in terminal success",
        ]
    else:
        gate_status = "inconclusive_scale_or_pair_change_required"
        reasons = ["success complementarity is insufficient for a framing decision"]

    return {
        "schema_version": "rase-replacement-audit/v1",
        "status": "complete",
        "use_for": "development-only pilot; excluded from flagship hidden test",
        "n_tasks": len(rows),
        "n_episodes": len(rows),
        "overall_successes": overall_successes,
        "overall_success_rates": {
            method: hits / len(rows) for method, hits in overall_successes.items()
        },
        "source_vs_oft_only_quadrants": {
            label: int(quadrants.get(label, 0))
            for label in ("rescue", "harm", "redundant", "unsupported")
        },
        "pairwise": {
            "oft_only_vs_source_only": _pair_summary(
                rows,
                "oft_only_success",
                "source_only_success",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            ),
            "source_to_oft_vs_oft_only": _pair_summary(
                rows,
                "source_to_oft_success",
                "oft_only_success",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + 1,
            ),
        },
        "by_group": grouped,
        "timing": {
            "source_only_full_episode": _timing_from_source_summary(source_summary),
            "oft_only_full_episode": _timing_from_oft_rows(list(oft_rows.values())),
            "source_to_oft": {
                "scope": "post-step-25 continuation only; source prefix cost unavailable",
                **dict((handoff_analysis.get("timing") or {}).get("direct_oft") or {}),
            },
        },
        "replacement_gate": {
            "status": gate_status,
            "reasons": reasons,
            "source_unique_tasks": len(source_unique),
            "source_unique_suites": sorted(unique_suites),
            "clean_source_successes": clean_source,
            "clean_oft_only_successes": clean_oft,
            "cost_complete": False,
            "cost_limitation": (
                "source-to-OFT total episode cost lacks the source prefix; do not make "
                "a final resource-complementarity claim from this pilot"
            ),
        },
        "bootstrap": {
            "unit": "task/episode (one unique task per episode)",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "per_task": rows,
    }


def _render(result: dict[str, Any]) -> str:
    hits = result["overall_successes"]
    pair = result["pairwise"]
    lines = [
        "# RASE Phase 1A replacement-audit pilot",
        "",
        f"Status: **{result['status']}**",
        "",
        "> Development-only Phase 0 task reuse. This is not flagship test evidence.",
        "",
        "## Overall terminal success",
        "",
        "| mode | success |",
        "|---|---:|",
        f"| SOURCE-ONLY full horizon | {hits['source_only_success']}/{result['n_tasks']} |",
        f"| OFT-ONLY from reset | {hits['oft_only_success']}/{result['n_tasks']} |",
        f"| source→OFT at env step 25 | {hits['source_to_oft_success']}/{result['n_tasks']} |",
        "",
        "## Paired comparisons",
        "",
        f"- OFT-ONLY minus SOURCE-ONLY: "
        f"{pair['oft_only_vs_source_only']['paired_difference']:.4f}, 95% CI "
        f"`{pair['oft_only_vs_source_only']['paired_difference_bootstrap_95_ci']}`, "
        f"McNemar p={pair['oft_only_vs_source_only']['mcnemar_exact_p']:.6g}.",
        f"- source→OFT minus OFT-ONLY: "
        f"{pair['source_to_oft_vs_oft_only']['paired_difference']:.4f}, 95% CI "
        f"`{pair['source_to_oft_vs_oft_only']['paired_difference_bootstrap_95_ci']}`, "
        f"McNemar p={pair['source_to_oft_vs_oft_only']['mcnemar_exact_p']:.6g}.",
        "",
        "## Replacement gate",
        "",
        f"Decision: **{result['replacement_gate']['status']}**.",
        "",
    ]
    lines.extend(f"- {reason}" for reason in result["replacement_gate"]["reasons"])
    lines.extend(
        [
            "- Final cost framing remains incomplete because the historical source→OFT "
            "artifact does not contain source-prefix compute.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--oft-summary", action="append", type=_named_path, required=True)
    parser.add_argument("--handoff-analysis", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026080201)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        _read(args.initial_keys.resolve()),
        _read(args.source_summary.resolve()),
        [(suite, _read(path.resolve())) for suite, path in args.oft_summary],
        _read(args.handoff_analysis.resolve()),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    args.output_md.write_text(_render(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "overall_successes": result["overall_successes"],
                "quadrants": result["source_vs_oft_only_quadrants"],
                "replacement_gate": result["replacement_gate"]["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
