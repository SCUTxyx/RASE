#!/usr/bin/env python3
"""Freeze the per-VLA OFT-label selection after source-only screening.

Natural development states are all retained.  Training enrichment includes
every state on which the target source VLA failed in at least one screened
seed, plus deterministic matched success-only controls.  Matching prioritizes
same task, perturbation type/level and episode step.  No OFT outcome is used.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path


POLICIES = ("pi05_libero", "pi0fast_libero")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_cost(failure: dict, control: dict) -> tuple:
    return (
        failure["suite"] != control["suite"],
        failure["task_id"] != control["task_id"],
        failure["perturb_dim"] != control["perturb_dim"],
        abs(int(failure["perturb_level"]) - int(control["perturb_level"])),
        abs(int(failure["step"]) - int(control["step"])),
        control["state_key"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--screen-audit", type=Path, required=True)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.screen_audit.read_text())
    if audit.get("decision") != "GO_LABEL_COLLECTION":
        raise ValueError("screening audit does not authorize label selection")
    initial = json.loads(args.initial_keys.read_text())
    metadata = {row["state_key"]: row for row in initial["records"]}
    natural = sorted(row["state_key"] for row in initial["records"]
                     if row["role"] == "natural_development_eval")
    enrichment = {row["state_key"] for row in initial["records"]
                  if row["role"] == "train_enrichment"}

    policy_results = {}
    for policy in POLICIES:
        outcomes: dict[str, list[bool]] = defaultdict(list)
        paths = glob.glob(str(args.screen_root / "suite_*" / policy /
                                  "train_enrichment" / "seed_*" / "*__seed*.json"))
        for value in paths:
            data = json.loads(Path(value).read_text())
            if not data.get("rows"):
                continue
            state = str(data["rows"][0]["state_key"])
            if state not in enrichment:
                raise ValueError(f"screening state is not enrichment: {state}")
            outcomes[state].append(bool(data["source_success"]))
        if set(outcomes) != enrichment:
            raise ValueError(f"incomplete enrichment screening for {policy}")

        failure_keys = sorted(state for state, values in outcomes.items()
                              if any(not value for value in values))
        success_keys = sorted(state for state, values in outcomes.items()
                              if all(values))
        remaining = set(success_keys)
        matched = []
        match_pairs = []
        for failure_key in sorted(failure_keys,
                                  key=lambda key: (metadata[key]["suite"],
                                                   metadata[key]["task_id"], key)):
            if not remaining:
                break
            candidates = [key for key in remaining
                          if metadata[key]["suite"] == metadata[failure_key]["suite"]]
            if not candidates:
                # A control from another suite is not a matched control.  Keep
                # the failure unmatched rather than silently changing its
                # domain distribution.
                continue
            control_key = min(candidates,
                              key=lambda key: match_cost(metadata[failure_key], metadata[key]))
            remaining.remove(control_key)
            matched.append(control_key)
            match_pairs.append({
                "failure_state_key": failure_key,
                "control_state_key": control_key,
                "failure_task_id": metadata[failure_key]["task_id"],
                "control_task_id": metadata[control_key]["task_id"],
                "same_suite": metadata[failure_key]["suite"] == metadata[control_key]["suite"],
                "same_task": metadata[failure_key]["task_id"] == metadata[control_key]["task_id"],
                "same_perturb_dim": (metadata[failure_key]["perturb_dim"]
                                     == metadata[control_key]["perturb_dim"]),
            })
        selected = sorted(set(failure_keys) | set(matched))
        by_suite = {}
        for suite in sorted({metadata[key]["suite"] for key in enrichment}):
            by_suite[suite] = {
                "failure_states": sum(metadata[key]["suite"] == suite for key in failure_keys),
                "matched_controls": sum(metadata[key]["suite"] == suite for key in matched),
                "selected_states": sum(metadata[key]["suite"] == suite for key in selected),
            }
        policy_results[policy] = {
            "failure_state_keys": failure_keys,
            "success_only_state_keys": success_keys,
            "matched_control_state_keys": sorted(matched),
            "selected_enrichment_state_keys": selected,
            "n_failure_states": len(failure_keys),
            "n_success_only_states": len(success_keys),
            "n_matched_controls": len(matched),
            "n_selected_enrichment_states": len(selected),
            "n_unmatched_failure_states": max(0, len(failure_keys) - len(matched)),
            "by_suite": by_suite,
            "match_pairs": match_pairs,
        }

    result = {
        "schema_version": "rase-r6c1b-oft-selection/v1",
        "status": "frozen",
        "scientific_scope": ("OFT labels for all natural states and per-VLA training "
                             "enrichment selected only from source screening; no OFT outcomes"),
        "selection_rule": ("all source-failure states plus one deterministic success-only "
                           "control per failure when available; prioritize same suite/task/perturbation"),
        "natural_state_keys": natural,
        "n_natural_states": len(natural),
        "policies": policy_results,
        "screen_audit": str(args.screen_audit.resolve()),
        "screen_audit_sha256": sha256(args.screen_audit),
        "initial_keys": str(args.initial_keys.resolve()),
        "initial_keys_sha256": sha256(args.initial_keys),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "n_natural_states": len(natural),
        "policies": {policy: {key: value[key] for key in
                               ("n_failure_states", "n_success_only_states",
                                "n_matched_controls", "n_selected_enrichment_states",
                                "n_unmatched_failure_states", "by_suite")}
                     for policy, value in policy_results.items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
