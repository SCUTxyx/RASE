#!/usr/bin/env python3
"""Audit the frozen R6-C.1B OFT collection plan before GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEEDS = {"pi05_libero": (2, 3), "pi0fast_libero": (1,)}
REPLICAS = (0, 1)
BOUNDARIES = (0, 8, 16)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    initial = json.loads(args.initial_keys.read_text())
    if selection.get("status") != "frozen":
        raise ValueError("selection is not frozen")
    if selection.get("initial_keys_sha256") != sha256(args.initial_keys):
        raise ValueError("selection/initial-keys mismatch")
    metadata = {row["state_key"]: row for row in initial["records"]}
    natural = set(selection["natural_state_keys"])
    expected_natural = {row["state_key"] for row in initial["records"]
                        if row["role"] == "natural_development_eval"}
    if natural != expected_natural:
        raise ValueError("selection does not contain exactly the natural cohort")

    rows = []
    total = 0
    for policy, seeds in SEEDS.items():
        selected = set(selection["policies"][policy]["selected_enrichment_state_keys"])
        failures = set(selection["policies"][policy]["failure_state_keys"])
        if not failures.issubset(selected):
            raise ValueError(f"{policy}: selected enrichment drops failure states")
        for role, states in (("natural_development_eval", natural),
                             ("train_enrichment", selected)):
            for suite in sorted({metadata[key]["suite"] for key in states}):
                state_count = sum(metadata[key]["suite"] == suite for key in states)
                trajectories = state_count * len(seeds) * len(REPLICAS)
                total += trajectories
                rows.append({
                    "policy_id": policy, "role": role, "suite": suite,
                    "states": state_count, "seeds": list(seeds),
                    "replicas": list(REPLICAS), "trajectories": trajectories,
                    "oft_counterfactual_branches": trajectories * len(BOUNDARIES),
                })
    result = {
        "schema_version": "rase-r6c1b-collection-plan/v1",
        "status": "ready",
        "selection": str(args.selection.resolve()),
        "selection_sha256": sha256(args.selection),
        "initial_keys_sha256": sha256(args.initial_keys),
        "boundaries": list(BOUNDARIES),
        "expected_trajectory_groups": total,
        "expected_oft_counterfactual_branches": total * len(BOUNDARIES),
        "rows": rows,
        "gates": {
            policy: {
                "failure_states_ge_30": selection["policies"][policy]["n_failure_states"] >= 30,
                "four_suites": len(selection["policies"][policy]["by_suite"]) == 4,
                "all_failure_states_retained": set(selection["policies"][policy]["failure_state_keys"])
                                               .issubset(set(selection["policies"][policy]["selected_enrichment_state_keys"])),
            } for policy in SEEDS
        },
    }
    result["stage_gate_passed"] = all(all(value.values()) for value in result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "stage_gate_passed": result["stage_gate_passed"],
        "expected_trajectory_groups": total,
        "expected_oft_counterfactual_branches": total * len(BOUNDARIES),
        "gates": result["gates"],
    }, indent=2, sort_keys=True))
    return 0 if result["stage_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
