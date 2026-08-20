#!/usr/bin/env python3
"""Export the frozen PRE-A3-S same-policy opportunity-screen specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPEC = {
    "schema_version": "rase-pre-a3-s-opportunity-spec/v1",
    "status": "frozen_spec_not_executed",
    "question": (
        "When the current SmolVLA action may fail, can same-policy resample/"
        "fresh replan/local correction create unique rescues beyond OFT?"
    ),
    "n_states_target": 120,
    "families": [
        "queued_source_suffix",
        "fresh_smol_replan",
        "equal_budget_oft_prefix",
        "bounded_local_correction_a",
        "bounded_local_correction_b",
    ],
    "matched_controls": [
        "equal_prefix_length",
        "equal_continuation_horizon",
        "family_ids_and_seeds_in_provenance",
    ],
    "go_no_go": {
        "oracle_headroom_pp": 8,
        "failure_rescue_fraction": 0.20,
        "min_families_with_unique_success": 2,
        "min_tasks_with_rescue": 2,
        "held_out_direction_must_hold": True,
    },
    "on_fail": "close_candidate_critic_and_world_model; treat OFT as recovery option",
    "on_pass": "fit non-world-model candidate critic before any world-model ablation",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/rase_pre_a3_s_opportunity_spec_v1.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(SPEC, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": SPEC["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
