#!/usr/bin/env python3
"""Aggregate PRE-C0 corrective rollouts and apply the frozen Natural Gate A."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rase.collect.pre_c0 import (
    analyze_natural_headroom,
    episode_cluster_bootstrap_natural_headroom,
    horizon_decomposition,
    leave_one_task_out_natural_direction,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2_026_080_405)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.rollout_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "rase-pre-c0-corrective-rollout/v1":
            continue
        rows.append(payload)
    if not rows:
        raise SystemExit(f"no PRE-C0 rollout JSON files under {args.rollout_dir}")

    audit = analyze_natural_headroom(rows)
    audit["suite_counts"] = dict(Counter(str(row["suite"]) for row in rows))
    audit["cell_counts"] = dict(Counter(str(row["cell"]) for row in rows))
    audit["stage_counts"] = dict(Counter(str(row["stage"]) for row in rows))
    audit["family_successes"] = {
        family: sum(bool(row["family_success"][family]) for row in rows)
        for family in (
            "current_suffix",
            "strict_resample",
            "fresh_replan",
            "receding_horizon",
        )
    }
    audit["family_rollout_counts"] = {
        family: sum(
            1
            for row in rows
            for arm in row.get("arms") or []
            if str(arm.get("family")) == family
        )
        for family in audit["family_successes"]
    }
    audit["matched_compute"] = {
        "strict_resample_k": 8,
        "fresh_replan_k": 4,
        "execution_horizons": [1, 2, 4],
        "note": "Family oracles are reported separately; no learned selector.",
    }
    audit["episode_cluster_bootstrap"] = episode_cluster_bootstrap_natural_headroom(
        rows,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    audit["horizon_decomposition"] = horizon_decomposition(rows)
    audit["leave_one_task_out"] = leave_one_task_out_natural_direction(rows)
    audit["per_state"] = rows
    _write(args.output, audit)

    decision = {
        "schema_version": "rase-pre-c0-decision/v1",
        "decision": (
            "natural_candidate_critic_eligible"
            if audit["gate_pass"]
            else "run_privileged_guidance_upper_bound"
        ),
        "natural_same_policy_gate": "open" if audit["gate_pass"] else "closed",
        "candidate_critic_gate": audit["candidate_critic_gate"],
        "guided_generation_gate": "untested",
        "pre_a3_method_gate": "closed",
        "pre_b_allowed": False,
        "world_model_gate": "closed",
        "hidden_test": "sealed_not_unblinded",
        "audit": str(args.output),
        "natural_headroom_pp": audit["headroom_pp"]["natural_total"],
        "bootstrap_ci95_pp": audit["episode_cluster_bootstrap"]["ci95_pp"],
    }
    decision_path = args.decision_output or args.output.with_name("decision.json")
    _write(decision_path, decision)
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
