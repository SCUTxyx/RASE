#!/usr/bin/env python3
"""PRE-C1.4-R3 Phase 0B: Dense reward certification.

Lexicographic outcome (not weighted sum):
  1. terminal success has priority.
  2. irreversible event is a veto.
  3. Only when terminal outcomes are equal, use normalized dense progress/stability.
  4. Dense score used only for secondary preference and diagnostics.

On calibration anchors, verifies:
  - Same trajectory recomputed yields identical score.
  - Score direction aligns with task-defined progress.
  - Dense score is positively correlated with terminal success.
  - Per-suite normalization (no raw metric mixing across suites).
  - Reward weights and delta_G frozen before viewing variant results.

Output: phase0_reward_pass.json or failure report.
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# Default reward weights — frozen before any training
DEFAULT_REWARD_WEIGHTS = {
    "terminal_success": 1.0,
    "progress_delta": 0.3,
    "grasp_stability": 0.2,
    "irreversible_penalty": -0.5,
    "regression_penalty": -0.2,
}


# Default lexicographic outcome definitions
LEXICOGRAPHIC_DEFINITION = {
    "schema": "rase-pre-c1-4-r3-reward/v1",
    "order": [
        {"name": "terminal_success", "type": "binary", "priority": 1},
        {"name": "irreversible_event", "type": "veto", "priority": 0},
        {"name": "dense_progress", "type": "continuous", "priority": 2},
        {"name": "grasp_stability", "type": "continuous", "priority": 3},
    ],
    "veto_rule": (
        "If irreversible_event occurs, the outcome is classified as terminal failure"
        " regardless of other indicators."
    ),
}


def compute_lexicographic_outcome(rollout_metrics: dict) -> dict:
    """Apply lexicographic outcome to a rollout.

    Returns {outcome: 'success'|'failure', irreversible: bool,
             dense_score: float | None, terminal_success: bool}
    """
    terminal = rollout_metrics.get("terminal_success", False)
    irreversible = rollout_metrics.get("irreversible_event", False)

    outcome = {
        "terminal_success": terminal,
        "irreversible": irreversible,
        "dense_score": None,
        "outcome": "failure",
    }

    if irreversible:
        outcome["outcome"] = "failure"
        return outcome

    if terminal:
        outcome["outcome"] = "success"
        # Compute dense score only for diagnostic purposes
        progress = rollout_metrics.get("progress_delta", 0)
        grasp = rollout_metrics.get("grasp_stability", 0)
        outcome["dense_score"] = (
            DEFAULT_REWARD_WEIGHTS["progress_delta"] * progress
            + DEFAULT_REWARD_WEIGHTS["grasp_stability"] * grasp
        )
    else:
        outcome["outcome"] = "failure"
        # Even failures can have partial progress
        progress = rollout_metrics.get("progress_delta", 0)
        regression = rollout_metrics.get("regression", 0)
        outcome["dense_score"] = (
            DEFAULT_REWARD_WEIGHTS["progress_delta"] * progress
            + DEFAULT_REWARD_WEIGHTS["regression_penalty"] * regression
        )

    return outcome


def main():
    parser = argparse.ArgumentParser(
        description="PRE-C1.4-R3 Phase 0B: Dense reward certification"
    )
    parser.add_argument(
        "--manifest",
        default=str(
            ROOT
            / "runs"
            / "rase_pre_c1_4_r3_protocol"
            / "pre_c1_4_r3_identity_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "runs" / "rase_pre_c1_4_r3_protocol"),
    )
    parser.add_argument(
        "--delta-g", type=float, default=0.1,
        help="Minimum teacher advantage for teacher-preferred label",
    )
    parser.add_argument(
        "--delta-equiv", type=float, default=0.05,
        help="Maximum absolute difference for equivalent label",
    )
    parser.add_argument(
        "--reward-weights-json",
        default="",
        help="Path to custom reward weights JSON (otherwise use defaults)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or use default weights
    if args.reward_weights_json and Path(args.reward_weights_json).exists():
        weights = json.loads(Path(args.reward_weights_json).read_text())
    else:
        weights = DEFAULT_REWARD_WEIGHTS

    # Build certification report
    certification = {
        "schema_version": "rase-pre-c1-4-r3-reward-certification/v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "lexicographic_definition": LEXICOGRAPHIC_DEFINITION,
        "frozen_weights": weights,
        "frozen_thresholds": {
            "delta_G": args.delta_g,
            "delta_equiv": args.delta_equiv,
        },
        "checks": {
            "repeatability": "pending_live_run",
            "progress_direction_alignment": "pending_live_run",
            "correlation_with_terminal_success": "pending_live_run",
            "per_suite_normalization": "pending_live_run",
            "weights_frozen_before_variant_view": True,
        },
        "all_passed": False,
        "requires_live_simulator": True,
    }

    # Write gate
    gate = {
        "phase": "reward_certification",
        "passed": False,
        "status": "pending_live_run",
        "message": (
            "Dense reward certification requires live LIBERO simulator."
            " Weights and thresholds are frozen. Live certification must"
            " verify: repeatability, direction alignment, correlation"
            " with terminal success, per-suite normalization."
        ),
        "frozen_weights": weights,
        "frozen_delta_G": args.delta_g,
        "frozen_delta_equiv": args.delta_equiv,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    gate_path = output_dir / "phase0_reward_pass.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n")
    print(f"Gate file: {gate_path}")
    print(f"Frozen delta_G={args.delta_g}, delta_equiv={args.delta_equiv}")
    print("STATUS: pending_live_run — certification requires LIBERO simulator")

    cert_path = output_dir / "reward_certification.json"
    cert_path.write_text(json.dumps(certification, indent=2, sort_keys=True) + "\n")
    print(f"Certification details: {cert_path}")

    # Also freeze the weights for later use by training scripts
    frozen_path = output_dir / "frozen_reward_config.json"
    frozen_path.write_text(
        json.dumps(
            {
                "weights": weights,
                "delta_G": args.delta_g,
                "delta_equiv": args.delta_equiv,
                "lexicographic": LEXICOGRAPHIC_DEFINITION,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Frozen reward config: {frozen_path}")


if __name__ == "__main__":
    main()
