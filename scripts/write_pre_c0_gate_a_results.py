#!/usr/bin/env python3
"""Write Gate A progress markdown + artifacts symlink/copy from frozen audit/decision."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument(
        "--progress",
        type=Path,
        default=Path("progress/2026-08-04_pre_c0_gate_a_results.md"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/pre_c0/gate_a_results.json"),
    )
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    headroom = audit.get("headroom_pp") or {}
    bootstrap = audit.get("episode_cluster_bootstrap") or {}
    leave = audit.get("leave_one_task_out") or {}
    horizon = audit.get("horizon_decomposition") or {}
    nested = audit.get("nested_successes") or {}

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.audit, args.artifact)

    lines = [
        "# PRE-C0 Gate A results (48-state Natural Same-Policy)",
        "",
        "Date: 2026-08-04",
        "",
        "## Decision (frozen)",
        "",
        f"- decision: `{decision.get('decision')}`",
        f"- natural_same_policy_gate: `{decision.get('natural_same_policy_gate')}`",
        f"- candidate_critic_gate: `{decision.get('candidate_critic_gate')}`",
        f"- natural_headroom_pp: `{decision.get('natural_headroom_pp')}`",
        f"- bootstrap_ci95_pp: `{decision.get('bootstrap_ci95_pp')}`",
        f"- audit: `{args.audit}`",
        f"- artifact copy: `{args.artifact}`",
        "",
        "## Nested oracle",
        "",
        f"- n_states: {audit.get('n_states')}",
        f"- S0_current: {nested.get('S0_current')}",
        f"- S1_resample: {nested.get('S1_resample')}",
        f"- S2_replan: {nested.get('S2_replan')}",
        f"- S3_closed_loop: {nested.get('S3_closed_loop')}",
        "",
        "## Headroom (pp)",
        "",
        f"- sampling: {headroom.get('sampling')}",
        f"- reconditioning: {headroom.get('reconditioning')}",
        f"- closed_loop: {headroom.get('closed_loop')}",
        f"- natural_total: {headroom.get('natural_total')}",
        "",
        "## Pass conditions",
        "",
    ]
    for key, value in sorted((audit.get("pass_conditions") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Bootstrap / sensitivity",
            "",
            f"- episode-cluster CI95 pp: `{bootstrap.get('ci95_pp')}`",
            f"- ci95_lower_positive: `{bootstrap.get('ci95_lower_positive')}`",
            f"- leave-one-task all nonnegative: `{leave.get('all_folds_nonnegative')}`",
            f"- H_adaptive_horizon_pp: `{horizon.get('H_adaptive_horizon_pp')}`",
            f"- control_harm_rate: `{audit.get('control_harm_rate')}`",
            "",
            "## Protocol locks",
            "",
            "- Thresholds unchanged from protocol lock (5pp / K / arms / cohort).",
            "- PRE-A3 method gate remains closed; hidden test24 sealed.",
            "- World model gate remains closed.",
            "",
        ]
    )
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    args.progress.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {args.progress}")
    print(f"WROTE {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
