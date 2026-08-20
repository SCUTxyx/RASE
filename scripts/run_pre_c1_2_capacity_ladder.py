#!/usr/bin/env python3
"""PRE-C1.2 Phase 4: emit single-variable capacity experiment configs (C4-A..E)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rase.adapt.pre_c1_2 import capacity_ladder_step, load_protocol_lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_protocol_lock.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pre_c1/capacity_ladder"),
    )
    parser.add_argument(
        "--step",
        choices=[
            "expand_lora_targets",
            "rank_32",
            "rank_64",
            "full_action_expert",
            "top_cross_modal_layers",
            "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Emit capacity configs even though Phase4 is paused pending R0.",
    )
    args = parser.parse_args()

    unlock_env = os.environ.get("ALLOW_LEGACY_E3_E4", "0") == "1"
    unlock_file = Path("artifacts/pre_c1/ALLOW_LEGACY_E3_E4").is_file()
    if not (args.allow_legacy or unlock_env or unlock_file):
        blocked = {
            "schema_version": "rase-pre-c1-2-capacity-block/v1",
            "blocked": True,
            "reason": "R0 pivot: capacity ladder only after TF/R(k)/data-target gates",
            "unlock": "--allow-legacy or ALLOW_LEGACY_E3_E4=1 or artifacts/pre_c1/ALLOW_LEGACY_E3_E4",
        }
        Path("runs").mkdir(parents=True, exist_ok=True)
        Path("runs/rase_pre_c1_2_capacity_blocked.json").write_text(
            json.dumps(blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("LEGACY_CAPACITY_BLOCKED", json.dumps(blocked, sort_keys=True), flush=True)
        return 42

    lock = load_protocol_lock(args.protocol_lock)
    order = list(lock["capacity"]["order"])
    steps = order if args.step == "all" else [args.step]
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in steps:
        cfg = capacity_ladder_step(name, lock)
        # Freeze shared non-capacity knobs so experiments stay single-variable.
        cfg["locked_shared"] = {
            "selected_horizon": lock["evaluation"]["recovery"].get("selected_horizon"),
            "batch_schedule": lock["batch_schedule"],
            "recovery_source_weights": lock["recovery_source_weights"],
            "loss": lock["loss"],
            "gate": {
                "recovery_gain_pp": lock["gate"]["recovery_gain_pp"],
                "clean_retention_drop_pp": lock["gate"]["clean_retention_drop_pp"],
            },
            "train_epochs": lock["train"]["epochs"],
            "train_seed": lock["train"]["seed"],
        }
        path = out_dir / f"c4_{name}.json"
        path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(str(path))
        print(json.dumps({"step": name, "path": str(path), **{k: cfg[k] for k in cfg if k != "locked_shared"}}, sort_keys=True))
    manifest = {
        "schema_version": "rase-pre-c1-2-capacity-ladder/v1",
        "order": order,
        "configs": written,
        "note": "Run only after E4 gate fail; one step at a time; do not retune lr from gate.",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PRE_C1_2_CAPACITY_LADDER_DONE output={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
