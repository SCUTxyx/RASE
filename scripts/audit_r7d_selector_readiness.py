#!/usr/bin/env python3
"""Decide whether risk evidence unlocks model-free selector opportunity audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-vla-stability", type=Path, action="append", required=True)
    parser.add_argument("--shared-stability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    per_vla = [json.loads(path.read_text()) for path in args.per_vla_stability]
    shared = json.loads(args.shared_stability.read_text())
    qualified = sorted({
        str(row.get("policy_id")) for row in per_vla
        if row.get("status") == "PASS" and row.get("decision") == "FULL_PASS"
    })
    shared_policies = sorted(shared.get("policies") or [])
    gate = {
        "at_least_two_per_vla_full_pass": len(qualified) >= 2,
        "shared_calibrated_model_pass": shared.get("status") == "PASS",
        "shared_policies_match_qualified": shared_policies == qualified,
    }
    ready = all(gate.values())
    result = {
        "schema_version": "rase-r7d-selector-readiness/v1",
        "status": "READY_FOR_MODEL_FREE_OPPORTUNITY_AUDIT" if ready else "LOCKED",
        "qualified_source_vlas": qualified,
        "shared_policies": shared_policies,
        "gate": gate,
        "unlocks_on_ready": [
            "fresh independent-cohort t0 persistent-fallback counterfactuals",
            "per source/fallback model-free opportunity audit",
        ],
        "still_locked_even_when_ready": [
            "selector training until each pair opportunity PASS",
            "world-model features", "independent validation", "test",
        ],
        "candidate_pairs": [
            f"{policy}+openvla_oft" for policy in qualified
            if policy in {"pi0fast_libero", "pi05_libero"}
        ],
        "risk_only_policies": [
            policy for policy in qualified
            if policy not in {"pi0fast_libero", "pi05_libero"}
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
