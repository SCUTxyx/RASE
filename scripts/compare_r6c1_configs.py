#!/usr/bin/env python3
"""R6-C.2: cross-configuration generalization comparison.

Reads the per-mode stability.json files from the R6-C.1C OOF ladder and builds
the multi-VLA generalization table that answers:

  1. per-VLA models;
  2. shared core (no policy condition);
  3. shared + VLA identity embedding;
  4. shared + deployable behavior descriptor;
  5. shared + descriptor + small per-VLA FiLM calibration;
  6. leave-one-VLA-out (descriptor from few-shot split);
  7. pure zero-shot (challenge metric only).

The output is a single ``comparison.json`` with, per mode and policy: passing
seed count, fold-correct gate metrics across seeds, and the stage-gate verdict.
It is meant to feed the final R6-C.2 progress document.  Pure zero-shot is
reported as a challenge metric and never gates the generalization claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.audit_r6c1_selector_stability as stability  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability", action="append", required=True,
                        help="path to a mode's stability.json (repeat per mode)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-passing-seeds", type=int, default=4)
    args = parser.parse_args()

    modes: dict[str, dict] = {}
    mode_paths: dict[str, str] = {}
    missing: list[str] = []
    for path_string in args.stability:
        path = Path(path_string)
        if not path.exists():
            missing.append(str(path))
            continue
        data = json.loads(path.read_text())
        mode = data["mode"]
        modes[mode] = data
        mode_paths[mode] = str(path.resolve())

    # The R6-C.2 ladder order (from the plan).
    ladder = ["per_vla", "shared", "shared_id", "shared_desc", "shared_calib",
              "loo", "zero_shot"]
    comparison: list[dict] = []
    for mode in ladder:
        data = modes.get(mode)
        if data is None:
            comparison.append({"mode": mode, "present": False})
            continue
        row: dict = {
            "mode": mode,
            "present": True,
            "stage_gate_passed": data["stage_gate_passed"],
            "required_passing_seed_count": args.required_passing_seeds,
            "policies": {},
        }
        for policy, result in sorted(data["policy_results"].items()):
            row["policies"][policy] = {
                "passing_seed_count": result["passing_seed_count"],
                "policy_gate_passed": result["policy_gate_passed"],
                "success_gap": result["metric_across_seeds"]["success_gap"],
                "false_continue_rate": result["metric_across_seeds"]["false_continue_rate"],
                "absolute_paired_harm": result["metric_across_seeds"]["absolute_paired_harm"],
                "savings": result["metric_across_seeds"]["savings"],
                "conditional_missed_rescue_rate": result["metric_across_seeds"]["conditional_missed_rescue_rate"],
            }
        comparison.append(row)

    # The generalization claim: shared + calibration vs per-VLA, and vs pure
    # zero-shot (challenge metric).  A positive claim requires shared_calib to
    # reach or beat per-VLA while using far less VLA-specific adaptation data.
    oof_gate_supported = (modes.get("shared_calib") is not None
                          and modes["shared_calib"]["stage_gate_passed"]
                          and modes.get("per_vla") is not None
                          and modes["per_vla"]["stage_gate_passed"])
    claim = {
        "method": ("shared risk core + deployable behavior descriptor + small "
                   "descriptor-conditioned FiLM calibration"),
        "oof_gate_supported": oof_gate_supported,
        "paper_claim_supported": False,
        "paper_claim_status": ("candidate_pending_independent_validation"
                               if oof_gate_supported else "not_supported_by_oof"),
        "zero_shot": {
            "role": "challenge metric only (not a gate)",
            "stage_gate_passed": modes.get("zero_shot", {}).get("stage_gate_passed", False),
        },
    }

    result = {
        "schema_version": "rase-r6c1-config-comparison/v1",
        "status": "complete",
        "scientific_scope": ("R6-C.2 multi-VLA generalization comparison: per-VLA vs "
                             "shared variants vs leave-one-out vs pure zero-shot"),
        "ladder": ladder,
        "required_passing_seed_count": args.required_passing_seeds,
        "missing_stability_files": missing,
        "comparison": comparison,
        "generalization_claim": claim,
        "mode_paths": mode_paths,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    table = [{"mode": item["mode"], "stage_gate_passed": item.get("stage_gate_passed"),
              "policies": {policy: value["passing_seed_count"]
                           for policy, value in item.get("policies", {}).items()}}
             for item in comparison if item.get("present")]
    print(json.dumps({
        "ladder": ladder,
        "comparison": table,
        "generalization_claim": claim,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
