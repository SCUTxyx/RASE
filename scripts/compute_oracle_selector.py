#!/usr/bin/env python3
"""PRE-C0-R0 Step 2.5: Oracle Selector calculation.

Reads the counterfactual repair matrix (counterfactual_matrix.jsonl)
and computes:
  - S_base, S_F0, S_F2 (per-arm success rates)
  - S_best_fixed = max(S_base, S_F0, S_F2)
  - S_oracle = (1/N) * sum(max_k success(s_i, R_k))
  - H_selector = S_oracle - S_best_fixed
  - Winner diversity W_k = P(k == argmax_j success(s, R_j))
  - Cross-type analysis (Type A/B/C/D classification)

Outputs: runs/pre_c0_r0/oracle_selector_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_matrix(matrix_path: Path) -> list[dict]:
    rows = []
    with open(matrix_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_oracle_metrics(rows: list[dict], utility_key: str = "success") -> dict:
    """Compute S_best_fixed, S_oracle, H_selector, winner diversity."""

    arms = ["Base", "F0", "F2", "Guided"]
    # Check which arms actually have data
    available_arms = []
    for r in rows:
        for arm in arms:
            if arm in r.get("outcomes", {}):
                if arm not in available_arms:
                    available_arms.append(arm)

    # Per-arm success rates
    per_arm = {}
    for arm in available_arms:
        outcomes = [
            r["outcomes"][arm].get(utility_key, False)
            for r in rows if arm in r.get("outcomes", {})
        ]
        if outcomes:
            per_arm[arm] = {
                "N": len(outcomes),
                "success": sum(1 for o in outcomes if o),
                "rate": sum(1 for o in outcomes if o) / len(outcomes),
            }

    # S_best_fixed
    rates = {arm: per_arm[arm]["rate"] for arm in available_arms if arm != "Guided"}
    best_fixed_arm = max(rates, key=rates.get)
    S_best_fixed = rates[best_fixed_arm]

    # S_oracle: for each snapshot, what's the max success among non-Guided arms
    oracle_count = 0
    free_count = 0
    for r in rows:
        arm_results = []
        for arm in available_arms:
            if arm == "Guided":
                continue
            if arm in r.get("outcomes", {}):
                arm_results.append(r["outcomes"][arm].get(utility_key, False))
        if arm_results:
            oracle_count += max(arm_results)
            free_count += 1

    S_oracle = oracle_count / free_count if free_count > 0 else 0
    H_selector = S_oracle - S_best_fixed

    # Winner diversity: for each snapshot (with ties), count how often each arm is best
    winner_counts = {arm: 0 for arm in available_arms if arm != "Guided"}
    total_non_guided = 0
    for r in rows:
        best_val = -1
        best_arms = []
        for arm in available_arms:
            if arm == "Guided":
                continue
            if arm in r.get("outcomes", {}):
                val = r["outcomes"][arm].get(utility_key, False)
                if val > best_val:
                    best_val = val
                    best_arms = [arm]
                elif val == best_val and val > 0:
                    best_arms.append(arm)
        if best_arms:
            total_non_guided += 1
            for arm in best_arms:
                winner_counts[arm] += 1

    winner_diversity = {}
    for arm in available_arms:
        if arm != "Guided":
            winner_diversity[arm] = {
                "count": winner_counts.get(arm, 0),
                "pct": winner_counts.get(arm, 0) / max(total_non_guided, 1),
            }

    # Cross-type analysis
    cross_types = {"Type_A_F0_specific": 0, "Type_B_F2_specific": 0,
                   "Type_C_do_not_repair": 0, "Type_D_neutral": 0,
                   "Type_E_both_repair": 0, "unknown": 0}
    for r in rows:
        outcomes = r.get("outcomes", {})
        base_s = outcomes.get("Base", {}).get(utility_key, False)
        f0_s = outcomes.get("F0", {}).get(utility_key, False)
        f2_s = outcomes.get("F2", {}).get(utility_key, False)

        if not base_s and f0_s and not f2_s:
            cross_types["Type_A_F0_specific"] += 1
        elif not base_s and not f0_s and f2_s:
            cross_types["Type_B_F2_specific"] += 1
        elif base_s and not f0_s and not f2_s:
            cross_types["Type_C_do_not_repair"] += 1
        elif base_s and f0_s and f2_s:
            cross_types["Type_D_neutral"] += 1
        elif not base_s and f0_s and f2_s:
            cross_types["Type_E_both_repair"] += 1
        else:
            cross_types["unknown"] += 1

    # Additional per-arm statistics
    arm_details = {}
    for arm in available_arms:
        steps = []
        drops = []
        collisions = []
        for r in rows:
            if arm in r.get("outcomes", {}):
                outcome = r["outcomes"][arm]
                steps.append(outcome.get("steps", 0))
                drops.append(outcome.get("drop", False))
                collisions.append(outcome.get("collision", False))
        if steps:
            arm_details[arm] = {
                "mean_steps": sum(steps) / len(steps),
                "drop_count": sum(1 for d in drops if d),
                "collision_count": sum(1 for c in collisions if c),
            }

    # H_selector by snapshot type
    type_headroom = {}
    for stype in sorted(set(r["snapshot_type"] for r in rows)):
        type_rows = [r for r in rows if r["snapshot_type"] == stype]
        if len(type_rows) < 3:
            continue
        type_rates = {}
        type_oracle = 0
        for arm in available_arms:
            if arm == "Guided":
                continue
            outcomes = [r["outcomes"][arm].get(utility_key, False)
                        for r in type_rows if arm in r.get("outcomes", {})]
            if outcomes:
                type_rates[arm] = sum(o for o in outcomes) / len(outcomes)
        best_fixed_type = max(type_rates.values()) if type_rates else 0
        for r in type_rows:
            vals = [r["outcomes"][arm].get(utility_key, False)
                    for arm in available_arms if arm != "Guided"
                    and arm in r.get("outcomes", {})]
            if vals:
                type_oracle += max(vals)
        type_oracle_rate = type_oracle / max(len(type_rows), 1)
        type_headroom[stype] = {
            "N": len(type_rows),
            "S_best_fixed": best_fixed_type,
            "S_oracle": type_oracle_rate,
            "H_selector": type_oracle_rate - best_fixed_type,
        }

    return {
        "N_total_snapshots": len(rows),
        "available_arms": available_arms,
        "per_arm_success": per_arm,
        "S_best_fixed": S_best_fixed,
        "S_best_fixed_arm": best_fixed_arm,
        "S_oracle": S_oracle,
        "H_selector": round(H_selector, 6),
        "H_selector_pp": round(H_selector * 100, 2),
        "winner_diversity": winner_diversity,
        "cross_types": cross_types,
        "arm_details": arm_details,
        "headroom_by_snapshot_type": type_headroom,
        "gating_decision": _gating_decision(H_selector, winner_diversity,
                                              cross_types, len(rows)),
    }


def _gating_decision(H: float, winner_diversity: dict, cross_types: dict,
                      N: int) -> dict:
    """Determine gate status based on H_selector and winner diversity."""
    # Check winner diversity
    diverse = sum(1 for v in winner_diversity.values()
                  if v.get("pct", 0) >= 0.10) >= 2

    # Check cross-type patterns
    has_type_a = cross_types.get("Type_A_F0_specific", 0) >= 2
    has_type_b = cross_types.get("Type_B_F2_specific", 0) >= 2
    has_type_c = cross_types.get("Type_C_do_not_repair", 0) >= 2

    H_pp = round(H * 100, 2)

    if H_pp < 5:
        decision = "NO-GO"
        rationale = (f"H_selector = {H_pp}pp < 5pp — Selector training not worth it. "
                     f"Default to best_fixed repair.")
    elif H_pp < 8:
        decision = "EXPLORE"
        rationale = (f"H_selector = {H_pp}pp — modest headroom. "
                     f"Selector exploration possible but not main track.")
    elif H_pp < 10:
        decision = "GO"
        rationale = (f"H_selector = {H_pp}pp — sufficient headroom for Selector training.")
    else:
        decision = "STRONG-SIGNAL"
        rationale = (f"H_selector = {H_pp}pp — very strong Selector signal.")

    if not diverse and H_pp >= 5:
        rationale += (f" BUT winner diversity insufficient: "
                      f"{winner_diversity}. Consider fixed repair anyway.")

    return {
        "decision": decision,
        "rationale": rationale,
        "N_snapshots": N,
        "H_selector_pp": H_pp,
        "diverse": diverse,
        "has_type_A": has_type_a,
        "has_type_B": has_type_b,
        "has_type_C": has_type_c,
    }


def format_report(report: dict) -> str:
    """Generate human-readable markdown report."""
    lines = []
    lines.append("# Oracle Discrete Selector Audit Report\n")

    lines.append(f"**Total snapshots**: {report['N_total_snapshots']}")
    lines.append(f"**Available arms**: {report['available_arms']}\n")

    lines.append("## Per-Arm Success Rates\n")
    lines.append("| Arm | N | Success | Rate |")
    lines.append("|-----|---|---------|------|")
    for arm, stats in report["per_arm_success"].items():
        lines.append(f"| {arm} | {stats['N']} | {stats['success']} | {stats['rate']:.1%} |")
    lines.append("")

    lines.append("## Core Metrics\n")
    lines.append(f"- **S_best_fixed**: {report['S_best_fixed']:.1%} (arm: {report['S_best_fixed_arm']})")
    lines.append(f"- **S_oracle**: {report['S_oracle']:.1%}")
    lines.append(f"- **H_selector**: {report['H_selector_pp']} pp\n")

    lines.append("## Winner Diversity\n")
    lines.append("| Arm | Winner Count | % of Snapshots |")
    lines.append("|-----|-------------|----------------|")
    for arm, wd in report["winner_diversity"].items():
        lines.append(f"| {arm} | {wd['count']} | {wd['pct']:.1%} |")
    lines.append("")

    lines.append("## Cross-Type Analysis\n")
    lines.append("| Type | Description | Count |")
    lines.append("|------|-------------|-------|")
    type_descriptions = {
        "Type_A_F0_specific": "Base=fail, F0=success, F2=fail",
        "Type_B_F2_specific": "Base=fail, F0=fail, F2=success",
        "Type_C_do_not_repair": "Base=success, F0=fail, F2=fail",
        "Type_D_neutral": "All succeed",
        "Type_E_both_repair": "Base=fail, both repairs succeed",
        "unknown": "Other",
    }
    for ttype, count in report["cross_types"].items():
        desc = type_descriptions.get(ttype, "?")
        lines.append(f"| {ttype} | {desc} | {count} |")
    lines.append("")

    if report.get("headroom_by_snapshot_type"):
        lines.append("## Headroom by Snapshot Type\n")
        lines.append("| Type | N | S_best_fixed | S_oracle | H_selector |")
        lines.append("|------|---|-------------|----------|------------|")
        for stype, hr in report["headroom_by_snapshot_type"].items():
            lines.append(f"| {stype} | {hr['N']} | {hr['S_best_fixed']:.1%} | "
                         f"{hr['S_oracle']:.1%} | {hr['H_selector']*100:.1f}pp |")
        lines.append("")

    lines.append("## Gating Decision\n")
    gd = report["gating_decision"]
    lines.append(f"**Decision**: **{gd['decision']}**")
    lines.append(f"**Rationale**: {gd['rationale']}")
    lines.append(f"- H_selector = {gd['H_selector_pp']}pp (N={gd['N_snapshots']})")
    lines.append(f"- Winner diversity: {'YES' if gd['diverse'] else 'NO'}")
    lines.append(f"- Has Type A (F0-specific): {'YES' if gd['has_type_A'] else 'NO'}")
    lines.append(f"- Has Type B (F2-specific): {'YES' if gd['has_type_B'] else 'NO'}")
    lines.append(f"- Has Type C (do-not-repair): {'YES' if gd['has_type_C'] else 'NO'}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-path", type=Path,
                        default=ROOT / "runs/pre_c0_r0/counterfactual_matrix.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/pre_c0_r0")
    parser.add_argument("--utility-key", type=str, default="success",
                        help="Key to use for utility (default: success)")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.matrix_path.is_file():
        print(f"ERROR: Matrix file not found: {args.matrix_path}")
        print("Run audit_repair_matrix.py first to generate the matrix.")
        return 1

    rows = load_matrix(args.matrix_path)
    print(f"Loaded {len(rows)} snapshot rows from {args.matrix_path}")

    report = compute_oracle_metrics(rows, utility_key=args.utility_key)

    # Save JSON
    json_path = output_dir / "oracle_selector_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"JSON report saved to: {json_path}")

    # Save markdown
    md = format_report(report)
    md_path = output_dir / "oracle_selector_report.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"MD report saved to: {md_path}")

    print("\n" + "=" * 60)
    print("GATING DECISION")
    print("=" * 60)
    print(report["gating_decision"]["rationale"])
    print(f"Decision: {report['gating_decision']['decision']}")
    print(f"H_selector: {report['gating_decision']['H_selector_pp']}pp")
    print(f"S_best_fixed: {report['S_best_fixed']:.1%} ({report['S_best_fixed_arm']})")
    print(f"S_oracle:     {report['S_oracle']:.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
