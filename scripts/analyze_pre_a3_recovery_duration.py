#!/usr/bin/env python3
"""Analyze PRE-A3 live recovery-duration sweeps and emit preregistered gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rase.collect.pre_a3 import analyze_recovery_duration, decide_method_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-summary", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2_026_080_403)
    args = parser.parse_args()

    duration = json.loads(args.duration_summary.read_text(encoding="utf-8"))
    keys = json.loads(args.state_keys_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audits = {}
    for split in (None, "train", "val", "test"):
        try:
            audit = analyze_recovery_duration(
                duration,
                keys=keys,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
                split=split,
            )
        except ValueError as exc:
            if split is None:
                raise
            audits[split or "all"] = {"status": "empty_split", "error": str(exc)}
            continue
        name = split or "all"
        audits[name] = audit
        (args.output_dir / f"audit_{name}.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )

    gate = decide_method_gate(
        audits.get("test") or {"gate_pass": False, "status": "missing_test", "next_step": "no hidden test"},
        val_audit=audits.get("val"),
    )
    (args.output_dir / "method_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    compact = {
        "all_status": (audits.get("all") or {}).get("status"),
        "val_status": (audits.get("val") or {}).get("status"),
        "test_status": (audits.get("test") or {}).get("status"),
        "decision": gate["decision"],
        "termination_model_gate": gate["termination_model_gate"],
        "world_model_gate": gate["world_model_gate"],
    }
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
