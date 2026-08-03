#!/usr/bin/env python3
"""Summarize direct-policy action support and selector readiness/metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _optional(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _method_decision(audit: dict | None, metrics: dict | None) -> dict:
    if not audit or not audit.get("ready"):
        return {"status": "not_ready", "reason": "readiness audit did not pass"}
    test = (metrics or {}).get("test") or {}
    delta = (test.get("paired_utility_differences") or {}).get(
        "learned_minus_matched_random_actions"
    ) or {}
    mean = delta.get("mean_difference")
    lower = (delta.get("bootstrap_ci_95") or {}).get("lower")
    if mean is None or lower is None:
        return {"status": "not_evaluable", "reason": "paired task-heldout delta missing"}
    if float(mean) <= 0:
        status = "kill_method_branch"
    elif float(lower) <= 0:
        status = "inconclusive_do_not_scale"
    else:
        status = "pass_to_feature_scaleup"
    return {
        "status": status,
        "comparison": "learned_minus_action_matched_random",
        "mean_utility_difference": float(mean),
        "bootstrap_ci_95": delta.get("bootstrap_ci_95"),
        "n_pairs": delta.get("n_pairs"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    from rase.collect.policy_matrix import exact_mcnemar_p
    from rase.selector.lightweight import ABSTAIN, ACTIONS, CONTINUE_SMOL, ESCALATE_OFT

    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    cohort_counts: dict[str, Counter[str]] = {}
    optimal = Counter()
    for row in rows:
        cohort = str(row.get("cohort"))
        smol = bool(row["arms"][CONTINUE_SMOL]["success"])
        oft = bool(row["arms"][ESCALATE_OFT]["success"])
        label = "both_success" if smol and oft else (
            "smol_only" if smol else ("oft_only" if oft else "both_fail")
        )
        counts = cohort_counts.setdefault(cohort, Counter())
        counts[label] += 1
        utilities = {
            action: float(bool(row["arms"][action]["success"])) - float(row["arms"][action]["cost"])
            for action in ACTIONS
        }
        best = max(ACTIONS, key=lambda action: (utilities[action], -ACTIONS.index(action)))
        optimal[best] += 1
    labels = ("both_success", "smol_only", "oft_only", "both_fail")
    cohorts = {
        cohort: {
            "n_states": sum(counts.values()),
            **{label: counts[label] for label in labels},
            "mcnemar_exact_p_two_sided": exact_mcnemar_p(
                counts["smol_only"], counts["oft_only"]
            ),
        }
        for cohort, counts in sorted(cohort_counts.items())
    }
    episode_audit = _optional(args.episode_dir / "readiness_audit.json")
    episode_metrics = _optional(args.episode_dir / "metrics.json")
    task_audit = _optional(args.task_dir / "readiness_audit.json")
    task_metrics = _optional(args.task_dir / "metrics.json")
    result = {
        "schema_version": "rase-selector-gate-summary/v1",
        "status": "complete",
        "n_states": len(rows),
        "cohorts": cohorts,
        "optimal_action_counts": {action: optimal[action] for action in ACTIONS},
        "episode": {
            "audit": episode_audit,
            "metrics": episode_metrics,
            "method_decision": _method_decision(episode_audit, episode_metrics),
        },
        "task": {
            "audit": task_audit,
            "metrics": task_metrics,
            "method_decision": _method_decision(task_audit, task_metrics),
        },
    }
    lines = ["# W9 selector gate summary", "", "## Direct action support", ""]
    for cohort, counts in cohorts.items():
        lines.append(
            f"- {cohort}: n={counts['n_states']}, both={counts['both_success']}, "
            f"Smol-only={counts['smol_only']}, OFT-only={counts['oft_only']}, "
            f"neither={counts['both_fail']}, McNemar p={counts['mcnemar_exact_p_two_sided']}"
        )
    lines.extend(["", "## Optimal action labels", ""])
    lines.extend(f"- {action}: {optimal[action]}" for action in ACTIONS)
    for grouping in ("episode", "task"):
        audit = result[grouping]["audit"] or {}
        lines.extend(
            [
                "",
                f"## {grouping}-disjoint gate",
                "",
                f"- ready: {audit.get('ready')}",
                f"- reasons: {audit.get('reasons', [])}",
                f"- metrics available: {result[grouping]['metrics'] is not None}",
                f"- method decision: {result[grouping]['method_decision']}",
            ]
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "n_states": len(rows),
        "cohorts": cohorts,
        "optimal_action_counts": result["optimal_action_counts"],
        "episode_ready": (result["episode"]["audit"] or {}).get("ready"),
        "task_ready": (result["task"]["audit"] or {}).get("ready"),
        "task_method_decision": result["task"]["method_decision"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
