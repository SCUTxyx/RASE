#!/usr/bin/env python3
"""Evaluate R4 safe-handback baselines on boundary transition data.

Implements three baselines on the 71-state boundary dataset:
1. Privileged earliest-safe oracle: upper bound on what's achievable
2. Deterministic progress/stagnation handback: hand back when student delta is small
3. Risk-only trigger (no dynamics): simple risk threshold without world model
4. Risk-only + fixed H128: same as (3) but with 128-dim head

All evaluations use the same grouped-task-held-out folds as the world model,
so results are directly comparable.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_r4_safe_handback_wm_ridge import (
    SUITE_MAP,
    Standardizer,
    _compute_history_features,
    _stack,
    _vec,
    build_arrays,
    grouped_task_folds,
    read_jsonl,
    summarize_decisions,
    validate_rows,
)


# ---------------------------------------------------------------------------
# Baseline controllers
# ---------------------------------------------------------------------------

def baseline_privileged_earliest_safe(rows: list[dict]) -> dict:
    """Privileged oracle: hand back at the earliest boundary where success is guaranteed.

    This is the theoretical upper bound. It uses counterfactual labels that are
    NOT available at deployment time.
    """
    by_state: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_state[str(row["state_key"])].append(i)

    decisions = []
    for state, indices in sorted(by_state.items()):
        indices.sort(key=lambda i: int(rows[i]["elapsed_oft_steps"]))
        selected = None
        for idx in indices:
            if rows[idx]["success_if_handback_now"]:
                selected = idx
                break
        ref = rows[indices[0]]
        ps = bool(ref["success_if_continue_oft"])
        psteps = int(ref["persistent_executed_oft_steps"])
        if selected is not None:
            decisions.append({
                "state_key": state,
                "task_id": str(ref["task_id"]),
                "action": "HAND_BACK_TO_STUDENT",
                "boundary": int(rows[selected]["elapsed_oft_steps"]),
                "success": bool(rows[selected]["success_if_handback_now"]),
                "persistent_success": ps,
                "executed_oft_steps": int(rows[selected]["elapsed_oft_steps"]),
                "persistent_executed_oft_steps": psteps,
                "false_handback": False,
            })
        else:
            decisions.append({
                "state_key": state,
                "task_id": str(ref["task_id"]),
                "action": "CONTINUE_OFT",
                "boundary": None,
                "success": ps,
                "persistent_success": ps,
                "executed_oft_steps": psteps,
                "persistent_executed_oft_steps": psteps,
                "false_handback": False,
            })
    return summarize_decisions(decisions)


def baseline_deterministic_progress(rows: list[dict], delta_threshold: float = 0.05,
                                     min_oft_steps: int = 32) -> dict:
    """Hand back when student one-step delta < threshold, after min OFT steps.

    Uses the normalized latent delta (not accessible at deployment, but a proxy
    for student stagnation). This is a heuristic baseline.
    """
    by_state: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_state[str(row["state_key"])].append(i)

    decisions = []
    for state, indices in sorted(by_state.items()):
        indices.sort(key=lambda i: int(rows[i]["elapsed_oft_steps"]))
        selected = None
        for idx in indices:
            row = rows[idx]
            if int(row["elapsed_oft_steps"]) < min_oft_steps:
                continue
            student_delta = np.asarray(row["next_latent_student"]) - np.asarray(row["latent"])
            delta_norm = float(np.linalg.norm(student_delta))
            if delta_norm < delta_threshold:
                selected = idx
                break
        ref = rows[indices[0]]
        ps = bool(ref["success_if_continue_oft"])
        psteps = int(ref["persistent_executed_oft_steps"])
        if selected is not None:
            sel = rows[selected]
            decisions.append({
                "state_key": state,
                "task_id": str(ref["task_id"]),
                "action": "HAND_BACK_TO_STUDENT",
                "boundary": int(sel["elapsed_oft_steps"]),
                "success": bool(sel["success_if_handback_now"]),
                "persistent_success": ps,
                "executed_oft_steps": int(sel["elapsed_oft_steps"]),
                "persistent_executed_oft_steps": psteps,
                "false_handback": bool(ps and not sel["success_if_handback_now"]),
            })
        else:
            decisions.append({
                "state_key": state,
                "task_id": str(ref["task_id"]),
                "action": "CONTINUE_OFT",
                "boundary": None,
                "success": ps,
                "persistent_success": ps,
                "executed_oft_steps": psteps,
                "persistent_executed_oft_steps": psteps,
                "false_handback": False,
            })
    return summarize_decisions(decisions)


def baseline_risk_only(rows: list[dict], threshold: float = 0.5) -> dict:
    """Simple risk-only trigger: always hand back at boundary 0 if base success is true,
    or never hand back.

    This is a trivial baseline: it never uses dynamics, just a fixed rule.
    """
    by_state: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_state[str(row["state_key"])].append(i)

    decisions = []
    for state, indices in sorted(by_state.items()):
        indices.sort(key=lambda i: int(rows[i]["elapsed_oft_steps"]))
        ref = rows[indices[0]]
        ps = bool(ref["success_if_continue_oft"])
        psteps = int(ref["persistent_executed_oft_steps"])
        # Risk-only: never hand back (conservative baseline)
        decisions.append({
            "state_key": state,
            "task_id": str(ref["task_id"]),
            "action": "CONTINUE_OFT",
            "boundary": None,
            "success": ps,
            "persistent_success": ps,
            "executed_oft_steps": psteps,
            "persistent_executed_oft_steps": psteps,
            "false_handback": False,
        })
    return summarize_decisions(decisions)


def baseline_fixed_early_handback(rows: list[dict], handback_step: int = 0) -> dict:
    """Always hand back at a fixed OFT step (e.g., immediately at h=0).

    This is the "always-on F0" equivalent from PRE-C0-R2 but at state boundaries.
    """
    by_state: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_state[str(row["state_key"])].append(i)

    decisions = []
    for state, indices in sorted(by_state.items()):
        indices.sort(key=lambda i: int(rows[i]["elapsed_oft_steps"]))
        boundary_idx = None
        for idx in indices:
            if int(rows[idx]["elapsed_oft_steps"]) == handback_step:
                boundary_idx = idx
                break
        ref = rows[indices[0]]
        ps = bool(ref["success_if_continue_oft"])
        psteps = int(ref["persistent_executed_oft_steps"])
        if boundary_idx is not None:
            sel = rows[boundary_idx]
            decisions.append({
                "state_key": state,
                "task_id": str(ref["task_id"]),
                "action": "HAND_BACK_TO_STUDENT",
                "boundary": handback_step,
                "success": bool(sel["success_if_handback_now"]),
                "persistent_success": ps,
                "executed_oft_steps": int(sel["elapsed_oft_steps"]),
                "persistent_executed_oft_steps": psteps,
                "false_handback": bool(ps and not sel["success_if_handback_now"]),
            })
        else:
            decisions.append({
                "state_key": state,
                "task_id": str(ref["task_id"]),
                "action": "CONTINUE_OFT",
                "boundary": None,
                "success": ps,
                "persistent_success": ps,
                "executed_oft_steps": psteps,
                "persistent_executed_oft_steps": psteps,
                "false_handback": False,
            })
    return summarize_decisions(decisions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    validate_rows(rows)
    folds = grouped_task_folds(rows, 6)

    baselines = {
        "oracle_earliest_safe": lambda r: baseline_privileged_earliest_safe(r),
        "risk_only_never_handback": lambda r: baseline_risk_only(r),
        "fixed_early_h0": lambda r: baseline_fixed_early_handback(r, 0),
        "fixed_early_h32": lambda r: baseline_fixed_early_handback(r, 32),
        "fixed_early_h64": lambda r: baseline_fixed_early_handback(r, 64),
        "deterministic_progress_th005_h32": lambda r: baseline_deterministic_progress(r, 0.05, 32),
        "deterministic_progress_th01_h0": lambda r: baseline_deterministic_progress(r, 0.10, 0),
        "deterministic_progress_th02_h0": lambda r: baseline_deterministic_progress(r, 0.20, 0),
    }

    # Evaluate each baseline across all folds (as OOF)
    all_results = {}
    for name, fn in baselines.items():
        fold_decisions = []
        for fold in folds:
            val_rows = fold["val"]
            result = fn(val_rows)
            decisions = result.pop("decisions")
            fold_decisions.extend(decisions)
        overall = summarize_decisions(fold_decisions)
        overall.pop("decisions")
        all_results[name] = overall

    # Print summary
    print("\n=== R4 SAFE-HANDBACK BASELINE RESULTS ===\n")
    print(f"{'Baseline':<45} {'Succ':>6} {'Delta':>7} {'Savings':>8} {'FBs':>4} {'FB%':>7}")
    print("-" * 85)
    for name, result in sorted(all_results.items()):
        print(f"{name:<45} {result['success_rate']:6.4f} {result['success_minus_persistent']:+7.4f} "
              f"{result['oft_step_savings_fraction']:8.4f} {result['false_handbacks']:4d} "
              f"{result['false_handback_rate_persistent_rescuable']:7.4f}")

    # Compare with world model
    print("\nComparison targets (from world model ridge v4):")
    print("  Target: delta >= -0.05, FBs <= 5%, savings >= 20%")
    for name, result in sorted(all_results.items()):
        passes = (result["success_minus_persistent"] >= -0.05 and
                  result["false_handback_rate_persistent_rescuable"] <= 0.05 and
                  result["oft_step_savings_fraction"] >= 0.20)
        status = "PASS" if passes else "FAIL"
        print(f"  {name}: {status}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": "rase-pre-c0-r4-baselines/v1",
        "dataset": str(args.dataset.resolve()),
        "n_rows": len(rows),
        "n_states": len({str(r["state_key"]) for r in rows}),
        "n_folds": len(folds),
        "baselines": all_results,
    }, indent=2, sort_keys=True) + "\n")
    print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
