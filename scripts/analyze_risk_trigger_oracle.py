#!/usr/bin/env python3
"""PRE-C0 / RASE-CI risk-trigger oracle audit on natural corrective rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rase.collect.pre_c0 import analyze_risk_trigger_oracle


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.rollout_dir.glob("*.json")):
        if path.name == "run_manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "rase-pre-c0-corrective-rollout/v1":
            continue
        rows.append(payload)
    if not rows:
        raise SystemExit(f"no PRE-C0 rollout JSON under {args.rollout_dir}")

    audit = analyze_risk_trigger_oracle(rows)
    audit["per_state_count"] = len(rows)
    _write(args.output, audit)

    decision = {
        "schema_version": "rase-pre-c0-decision/v1",
        "decision": audit["decision"],
        "natural_same_policy_gate": "open",
        "candidate_critic_gate": (
            "eligible" if audit["meaningful_for_critic"] else "closed"
        ),
        "guided_generation_gate": "not_required",
        "pre_a3_method_gate": "closed",
        "pre_b_allowed": False,
        "world_model_gate": "closed",
        "hidden_test": "sealed_not_unblinded",
        "audit": str(args.output),
        "risk_trigger_headroom_pp": audit["headroom_pp"]["risk_trigger_vs_current"],
        "harm_reduction_vs_always": audit["harm_reduction_vs_always"],
    }
    decision_path = args.decision_output or args.output.with_name(
        "risk_trigger_decision.json"
    )
    _write(decision_path, decision)
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
