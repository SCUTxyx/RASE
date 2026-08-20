#!/usr/bin/env python3
"""Analyze PRE-C1.2 dual gate (same thresholds; notes discrete 9-trial nature)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rase.adapt.pre_c1 import analyze_pre_c1_recovery_gate
from rase.adapt.pre_c1_2 import load_protocol_lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    parser.add_argument("--progress-md", type=Path, required=True)
    parser.add_argument("--artifact-json", type=Path, required=True)
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    gate_cfg = dict(lock["gate"])
    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    audit = analyze_pre_c1_recovery_gate(
        recovery_rows=payload.get("recovery") or [],
        retention_rows=payload.get("retention") or [],
        recovery_gain_pp=float(gate_cfg["recovery_gain_pp"]),
        clean_retention_drop_pp=float(gate_cfg["clean_retention_drop_pp"]),
        bootstrap_replicates=int(gate_cfg.get("bootstrap_replicates", 2000)),
        bootstrap_seed=int(gate_cfg.get("bootstrap_seed", 2_026_080_405)),
    )
    audit["phase"] = "PRE-C1.2"
    audit["recovery_execution_horizon"] = payload.get("recovery_execution_horizon")
    audit["retention_n_action_steps"] = payload.get("retention_n_action_steps", 10)
    audit["comparator_recovery"] = "adapted_minus_base_same_horizon"
    audit["primary_gate_note"] = (
        "9 locked failure keys × 1 seed → 8pp threshold ≈ ≥1 extra success (discrete)"
    )
    audit["secondary_affects_gate"] = False
    if payload.get("secondary_recovery"):
        sec = payload["secondary_recovery"]
        audit["secondary_summary"] = {
            "n_trials": len(sec),
            "adapted_successes": sum(bool(r.get("adapted_success")) for r in sec),
            "base_successes": sum(bool(r.get("base_success")) for r in sec),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    decision = {
        "schema_version": "rase-pre-c1-2-decision/v1",
        "phase": "PRE-C1.2",
        "decision": audit["decision"],
        "gate_pass": audit["gate_pass"],
        "same_backbone_recovery_method": audit["same_backbone_recovery_method"],
        "abstention_track": audit["abstention_track"],
        "recovery_gain_pp": audit["recovery_gain_pp"],
        "clean_retention_drop_pp": audit["clean_retention_drop_pp"],
        "recovery_execution_horizon": payload.get("recovery_execution_horizon"),
        "world_model_gate": "closed",
        "pre_a3_method_gate": "closed",
        "hidden_test": "sealed_not_unblinded",
        "naming": "recovery LoRA / OFT action distillation",
        "not_runtime_oft": True,
        "audit": str(args.output),
    }
    args.decision_output.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.artifact_json.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md = [
        "# PRE-C1.2 recovery LoRA gate results",
        "",
        f"- decision: `{decision['decision']}`",
        f"- gate_pass: `{decision['gate_pass']}`",
        f"- recovery_gain_pp: `{decision['recovery_gain_pp']}`",
        f"- clean_retention_drop_pp: `{decision['clean_retention_drop_pp']}`",
        f"- recovery_execution_horizon: `{decision.get('recovery_execution_horizon')}`",
        f"- comparator: `adapted_minus_base_same_horizon`",
        f"- retention_n_action_steps: `10`",
        f"- same_backbone_recovery_method: `{decision['same_backbone_recovery_method']}`",
        "",
        "Primary gate uses locked 9 failure keys and C1.1 seed. "
        "8pp on 9 trials is discrete (≈ ≥1 extra success). "
        "Secondary multi-seed does not affect the decision.",
        "",
        "Naming: recovery LoRA / OFT action distillation (not runtime OFT).",
        "",
    ]
    args.progress_md.parent.mkdir(parents=True, exist_ok=True)
    args.progress_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
