#!/usr/bin/env python3
"""Analyze PRE-C1 recovery vs clean-retention dual gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rase.adapt.pre_c1 import analyze_pre_c1_recovery_gate, load_protocol_lock


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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    decision = {
        "schema_version": "rase-pre-c1-decision/v1",
        "decision": audit["decision"],
        "gate_pass": audit["gate_pass"],
        "same_backbone_recovery_method": audit["same_backbone_recovery_method"],
        "abstention_track": audit["abstention_track"],
        "recovery_gain_pp": audit["recovery_gain_pp"],
        "clean_retention_drop_pp": audit["clean_retention_drop_pp"],
        "world_model_gate": "closed",
        "pre_a3_method_gate": "closed",
        "hidden_test": "sealed_not_unblinded",
        "naming": "recovery LoRA / OFT action distillation",
        "not_runtime_oft": True,
        "not_smolvla_flow_api_guidance": True,
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
        "# PRE-C1 recovery LoRA gate results",
        "",
        f"- decision: `{decision['decision']}`",
        f"- gate_pass: `{decision['gate_pass']}`",
        f"- recovery_gain_pp: `{decision['recovery_gain_pp']}`",
        f"- clean_retention_drop_pp: `{decision['clean_retention_drop_pp']}`",
        f"- same_backbone_recovery_method: `{decision['same_backbone_recovery_method']}`",
        f"- abstention_track: `{decision['abstention_track']}`",
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
