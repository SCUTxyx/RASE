#!/usr/bin/env python3
"""PRE-C1.4-R3 Phase 0A: Restore and branch parity audit.

Verifies on calibration anchors:
  1. Consecutive snapshot restore produces identical simulator state hash.
  2. Branch order randomized A→B vs B→A parity audit.
  3. Pre-branch observation tensors, history hash, policy state, cache state identical.
  4. Action cache cleared → new action chunk is generated.
  5. Same action, same seed → continuation is repeatable (>=3 repeats).
  6. Chunk boundary alignment with policy step.
  7. H > chunk_length → teacher live closed-loop replanning.
  8. Failure/timeout/irreversible termination semantics consistent across branches.

Output: phase0_restore_pass.json or failure report.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text())


def main():
    parser = argparse.ArgumentParser(
        description="PRE-C1.4-R3 Phase 0A: Restore and branch parity audit"
    )
    parser.add_argument(
        "--manifest",
        default=str(
            ROOT
            / "runs"
            / "rase_pre_c1_4_r3_protocol"
            / "pre_c1_4_r3_identity_manifest.json"
        ),
        help="Path to identity manifest",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "runs" / "rase_pre_c1_4_r3_protocol"),
    )
    parser.add_argument(
        "--num-anchors", type=int, default=8,
        help="How many calibration anchors to audit",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use smoke mode (limited anchors, quick checks)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(Path(args.manifest))
    calib_keys = manifest["splits"]["calibration"]["state_keys"]
    n_anchors = min(args.num_anchors if not args.smoke else 2, len(calib_keys))
    anchors = calib_keys[:n_anchors]

    checklist = {
        "schema_version": "rase-pre-c1-4-r3-restore-parity-audit/v1",
        "calibration_anchors": anchors,
        "n_anchors": n_anchors,
        "tests": {
            "exact_restore_hash_match": "pending_live_run",
            "branch_order_parity": "pending_live_run",
            "pre_branch_observation_match": "pending_live_run",
            "action_cache_cleared": "pending_live_run",
            "continuation_repeatability": "pending_live_run",
            "chunk_boundary_alignment": "pending_live_run",
            "live_replan_for_long_horizon": "pending_live_run",
            "termination_semantics_consistent": "pending_live_run",
        },
        "all_passed": False,
        "requires_live_simulator": True,
    }

    gate = {
        "phase": "restore_parity",
        "passed": False,
        "status": "pending_live_run",
        "message": (
            "All 8 checks require live LIBERO simulator with full RASE"
            " environment. Run this audit in a live session before Phase 1."
            " Checks: exact restore hash, branch order parity,"
            " pre-branch observation match, action cache cleared,"
            " continuation repeatability, chunk boundary alignment,"
            " live replan for long horizon, termination semantics."
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    gate_path = output_dir / "phase0_restore_pass.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n")
    print(f"Gate file: {gate_path}")
    print("STATUS: pending_live_run — all checks require LIBERO simulator")

    detail_path = output_dir / "restore_parity_audit.json"
    detail_path.write_text(json.dumps(checklist, indent=2, sort_keys=True) + "\n")
    print(f"Audit details: {detail_path}")


if __name__ == "__main__":
    main()
