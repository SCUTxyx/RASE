#!/usr/bin/env python3
"""Describe completed R6-C.1B source-only screening without changing its gate.

The frozen go/no-go report includes a legacy ``source_success_within_16``
diagnostic.  Task completion within 16 environment steps is not a valid
definition of a hard manipulation state, so this analysis uses only final
source-rollout outcomes.  It is descriptive and cannot authorize OFT label
collection or selector training.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trajectories: list[dict] = []
    seen: set[tuple[str, str, int, str]] = set()
    pattern_data: dict[tuple[str, str, str], dict[int, bool]] = defaultdict(dict)
    paths = sorted(glob.glob(str(args.screen_root / "suite_*" / "*" / "*" /
                                  "seed_*" / "*__seed*.json")))
    for value in paths:
        path = Path(value)
        data = json.loads(path.read_text())
        if not data.get("rows"):
            continue
        row = data["rows"][0]
        policy = str(row["policy_id"])
        state = str(row["state_key"])
        seed = int(row["seed_index"])
        role = ("train_enrichment" if "train_enrichment" in path.parts
                else "natural_development_eval")
        key = (policy, role, seed, state)
        if key in seen:
            raise ValueError(f"duplicate trajectory: {key}")
        seen.add(key)
        success = bool(data.get("source_success", row["source_final_success"]))
        record = {
            "policy_id": policy,
            "role": role,
            "seed_index": seed,
            "state_key": state,
            "suite": str(row["suite"]),
            "task_id": str(row["task_id"]),
            "source_success": success,
            "source_steps": int(data.get("source_steps", row["source_total_steps"])),
        }
        trajectories.append(record)
        pattern_data[(policy, role, state)][seed] = success

    policy_results = {}
    for policy in sorted({row["policy_id"] for row in trajectories}):
        policy_rows = [row for row in trajectories if row["policy_id"] == policy]
        roles = {}
        for role in ("natural_development_eval", "train_enrichment"):
            rows = [row for row in policy_rows if row["role"] == role]
            states = sorted({row["state_key"] for row in rows})
            failed_rows = [row for row in rows if not row["source_success"]]
            failed_states = sorted({row["state_key"] for row in failed_rows})
            success_only_states = sorted(set(states) - set(failed_states))
            patterns = Counter()
            variable_states = []
            for state in states:
                outcomes = pattern_data[(policy, role, state)]
                label = "".join("S" if outcomes[seed] else "F"
                                for seed in sorted(outcomes))
                patterns[label] += 1
                if len(set(outcomes.values())) > 1:
                    variable_states.append(state)
            by_suite = {}
            for suite in sorted({row["suite"] for row in rows}):
                subset = [row for row in rows if row["suite"] == suite]
                by_suite[suite] = {
                    "trajectories": len(subset),
                    "failures": sum(not row["source_success"] for row in subset),
                    "failure_states": len({row["state_key"] for row in subset
                                           if not row["source_success"]}),
                    "failure_tasks": len({row["task_id"] for row in subset
                                          if not row["source_success"]}),
                }
            roles[role] = {
                "trajectories": len(rows),
                "states": len(states),
                "failures": len(failed_rows),
                "failure_rate": len(failed_rows) / max(1, len(rows)),
                "failure_states": len(failed_states),
                "success_only_states": len(success_only_states),
                "failure_tasks": len({row["task_id"] for row in failed_rows}),
                "failure_suites": sorted({row["suite"] for row in failed_rows}),
                "seed_outcome_patterns": dict(sorted(patterns.items())),
                "cross_seed_variable_states": len(variable_states),
                "cross_seed_variable_state_keys": variable_states,
                "by_suite": by_suite,
            }
        policy_results[policy] = {"roles": roles}

    result = {
        "schema_version": "rase-r6c1b-screening-outcome-analysis/v1",
        "status": "complete",
        "scientific_scope": (
            "descriptive source-final-outcome analysis; does not change the "
            "frozen screening gate and cannot establish OFT rescueability"
        ),
        "screen_root": str(args.screen_root.resolve()),
        "trajectories": len(trajectories),
        "policy_results": policy_results,
        "warning": (
            "Cross-seed outcome variation is deployment variability, not an "
            "exact-repeat reproducibility failure. Exact-repeat parity is audited later."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
