#!/usr/bin/env python3
"""PRE-C1.4-R3 Phase 1: Paired counterfactual collection.

On train_collection anchors (16 anchors), collects paired (student, teacher)
counterfactual rollouts using the unified branch protocol:

  s_t → snapshot → query student a_S → branch S execute & evaluate
                 → restore exact snapshot
                 → query teacher a_T → branch T execute & evaluate
                 → compare outcomes

Labels: teacher_preferred, equivalent, both_fail, ambiguous.
Uses J_screen=3 for candidate screening; J_verify=5 (separate seeds) for label
freezing.

Output: runs/rase_pre_c1_4_counterfactual/{labeled_pairs, branch_rollouts,
        snapshot_manifest, restore_audit, pair_statistics}.jsonl/json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCREEN_SEEDS = 3
VERIFY_SEEDS = 5
DEFAULT_H_STAR = 4


def _build_empty_manifest(anchors: list, max_per_anchor: int) -> dict:
    return {
        "schema_version": "rase-pre-c1-4-counterfactual-collection/v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "anchors": anchors,
        "n_anchors": len(anchors),
        "max_per_anchor": max_per_anchor,
        "screen_seeds": SCREEN_SEEDS,
        "verify_seeds": VERIFY_SEEDS,
        "h_star": None,
        "collected_snapshots": 0,
        "teacher_preferred_pairs": 0,
        "equivalent_pairs": 0,
        "both_fail_pairs": 0,
        "ambiguous_pairs": 0,
        "status": "pending_live_run",
        "requires_live_simulator": True,
        "notes": (
            "Full paired counterfactual collection requires"
            " live LIBERO simulator with frozen C1.1 student and OFT teacher."
            " Run with --live to execute actual collection."
        ),
    }


def _generate_counterfactual_pairs(
    state_key: str,
    h_star: int,
    screen_seeds: int,
    verify_seeds: int,
) -> dict:
    """Simulate paired branch collection for one state key.

    In live mode, would:
      1. restore_pool_state(state_key)
      2. snapshot → student action → execute → evaluate
      3. restore snapshot → teacher action → execute → evaluate
      4. compare outcomes over screen_seeds + verify_seeds
    """
    import random

    rng = random.Random(hash(state_key) % (2**31))

    # Simulate outcomes
    teacher_wins_screen = sum(1 for _ in range(screen_seeds) if rng.random() < 0.3)
    teacher_wins_verify = sum(1 for _ in range(verify_seeds) if rng.random() < 0.3)
    mean_advantage = rng.uniform(-0.2, 0.5)

    if mean_advantage >= 0.15 and teacher_wins_verify >= 4:
        pair_type = "teacher_preferred"
    elif abs(mean_advantage) <= 0.05:
        pair_type = "equivalent"
    elif mean_advantage < -0.05:
        pair_type = "both_fail"
    else:
        pair_type = "ambiguous"

    return {
        "state_key": state_key,
        "h_star": h_star,
        "screen_seeds": {
            "n": screen_seeds,
            "teacher_wins": teacher_wins_screen,
            "student_wins": screen_seeds - teacher_wins_screen,
        },
        "verify_seeds": {
            "n": verify_seeds,
            "teacher_wins": teacher_wins_verify,
            "student_wins": verify_seeds - teacher_wins_verify,
        },
        "mean_advantage": round(mean_advantage, 4),
        "pair_type": pair_type,
        "student_irreversible": rng.random() < 0.05,
        "teacher_irreversible": rng.random() < 0.02,
    }


def _check_data_gate(stats: dict) -> dict:
    """Check if data gate requirements are met."""
    pref = stats.get("teacher_preferred_pairs", 0)
    anchors_with_pref = stats.get("anchors_with_preferred", 0)
    total_anchors = stats.get("total_anchors", 1)
    max_anchor_fraction = stats.get("max_anchor_fraction", 0)

    passed = True
    reasons = []

    if pref < 80:
        passed = False
        reasons.append(f"preferred_pairs={pref} < 80")
    if anchors_with_pref < 10:
        passed = False
        reasons.append(f"anchors_with_pref={anchors_with_pref} < 10")
    if max_anchor_fraction > 0.125:
        passed = False
        reasons.append(f"max_anchor_fraction={max_anchor_fraction:.3f} > 0.125")

    return {"passed": passed, "reasons": reasons if reasons else ["all gates passed"]}


def main():
    parser = argparse.ArgumentParser(
        description="PRE-C1.4-R3 Phase 1: Paired counterfactual collection"
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
        "--causal-unit-gate",
        default=str(
            ROOT
            / "runs"
            / "rase_pre_c1_4_r3_protocol"
            / "phase0_causal_unit_pass.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "runs" / "rase_pre_c1_4_counterfactual"),
    )
    parser.add_argument(
        "--h-star", type=int, default=None,
        help="Override H_star from causal-unit pilot",
    )
    parser.add_argument(
        "--max-per-anchor", type=int, default=10,
    )
    parser.add_argument(
        "--min-offset-gap", type=int, default=8,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate data collection for structure preview",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifests
    manifest = json.loads(Path(args.manifest).read_text())
    train_keys = manifest["splits"]["train_collection"]["state_keys"]

    # Load H_star from gate if available
    h_star = args.h_star
    if h_star is None:
        gate_path = Path(args.causal_unit_gate)
        if gate_path.exists():
            gate = json.loads(gate_path.read_text())
            h_star = gate.get("H_star", DEFAULT_H_STAR)
        else:
            h_star = DEFAULT_H_STAR
    print(f"H_star = {h_star}")

    # ---- Build collection manifest ----
    collection_manifest = _build_empty_manifest(train_keys, args.max_per_anchor)
    collection_manifest["h_star"] = h_star

    # ---- Collect pairs (dry run or live) ----
    pairs = []
    stats = defaultdict(int)
    anchor_counts = defaultdict(int)

    for state_key in train_keys[: min(args.max_per_anchor, 3) if args.dry_run else len(train_keys)]:
        if not args.dry_run:
            # Placeholder for live collection
            pair = {
                "state_key": state_key,
                "status": "pending_live_run",
                "message": "Requires live LIBERO simulator",
            }
            pairs.append(pair)
        else:
            pair = _generate_counterfactual_pairs(
                state_key, h_star, SCREEN_SEEDS, VERIFY_SEEDS
            )
            pairs.append(pair)
            anchor_counts[state_key] += 1
            stats[pair["pair_type"] + "_pairs"] = (
                stats.get(pair["pair_type"] + "_pairs", 0) + 1
            )
            stats["total_pairs"] += 1

    # ---- Gate check ----
    pref_anchors = set()
    for p in pairs:
        if p.get("pair_type") == "teacher_preferred":
            pref_anchors.add(p["state_key"])

    stats["anchors_with_preferred"] = len(pref_anchors)
    stats["total_anchors"] = len(train_keys)
    stats["max_anchor_fraction"] = (
        max(anchor_counts.values()) / max(1, sum(anchor_counts.values()))
        if anchor_counts else 0.0
    )

    gate_result = _check_data_gate(stats)
    collection_manifest.update({
        "teacher_preferred_pairs": stats.get("teacher_preferred_pairs", 0),
        "equivalent_pairs": stats.get("equivalent_pairs", 0),
        "both_fail_pairs": stats.get("both_fail_pairs", 0),
        "ambiguous_pairs": stats.get("ambiguous_pairs", 0),
        "total_pairs": stats.get("total_pairs", 0),
        "anchors_with_preferred": stats.get("anchors_with_preferred", 0),
        "max_anchor_fraction": stats.get("max_anchor_fraction", 0),
        "data_gate": gate_result,
    })

    # ---- Write outputs ----
    # labeled_pairs.jsonl
    pairs_path = output_dir / "labeled_pairs.jsonl"
    with open(pairs_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Labeled pairs: {pairs_path} ({len(pairs)} pairs)")

    # Collection manifest
    manifest_path = output_dir / "collection_manifest.json"
    manifest_path.write_text(json.dumps(collection_manifest, indent=2) + "\n")
    print(f"Collection manifest: {manifest_path}")

    # Branch rollouts (placeholder)
    rollouts_path = output_dir / "branch_rollouts.jsonl"
    rollouts_path.write_text("")
    print(f"Branch rollouts: {rollouts_path} (placeholder)")

    # Data gate
    gate_path = output_dir / "data_gate_pass.json"
    gate_path.write_text(json.dumps(gate_result, indent=2) + "\n")
    print(f"Data gate: {gate_path}")
    print(f"  Passed: {gate_result['passed']}")

    # Statistics
    stats_path = output_dir / "pair_statistics.json"
    stats_path.write_text(json.dumps(dict(stats), indent=2) + "\n")
    print(f"Statistics: {stats_path}")


if __name__ == "__main__":
    main()
