#!/usr/bin/env python3
"""Recompute the R4-v3 merged opportunity gate with all-state cost accounting."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    if report.get("schema_version") != "rase-pre-c0-r4-boundary-merged/v3":
        raise SystemExit("expected a merged R4-v3 report")
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if len(rows) != int(report.get("n_rows", -1)):
        raise SystemExit("dataset/report row-count mismatch")
    states = [
        state
        for suite in report.get("suite_reports", [])
        for state in suite.get("state_summaries", [])
    ]
    if len(states) != int(report.get("n_states", -1)):
        raise SystemExit("suite summaries/report state-count mismatch")
    rows_by_state: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_state[str(row["state_key"])].append(row)

    persistent_success = [state for state in states if state.get("persistent_replay_success")]
    finite_safe = [state for state in states if state.get("finite_safe")]
    finite_tasks = {str(state["task_id"]) for state in finite_safe}
    bins: dict[str, int] = {}
    for state in finite_safe:
        boundary = int(state["minimum_successful_handback_boundary"])
        bins[str(boundary)] = bins.get(str(boundary), 0) + 1
    persistent_steps = sum(int(state["executed_oft_steps"]) for state in states)
    oracle_steps = 0
    for state in states:
        successful = [
            int(row["elapsed_oft_steps"])
            for row in rows_by_state[str(state["state_key"])]
            if bool(row["success_if_handback_now"])
        ]
        oracle_steps += min(successful, default=int(state["executed_oft_steps"]))
    savings = 1.0 - oracle_steps / max(1, persistent_steps)
    populated = [key for key, count in bins.items() if int(key) > 0 and count >= 3]
    reasons = []
    if len(finite_safe) < 20:
        reasons.append(f"only {len(finite_safe)} live finite-safe states (<20)")
    if len(finite_tasks) < 3:
        reasons.append(f"only {len(finite_tasks)} true tasks have live finite-safe states (<3)")
    if savings < 0.20:
        reasons.append(f"live oracle OFT-step savings {savings:.4f} (<0.20)")
    if len(populated) < 2:
        reasons.append(f"only {len(populated)} populated live finite stopping bins (<2)")
    report.update({
        "cost_accounting_scope": "all_audited_states",
        "persistent_success_states": len(persistent_success),
        "live_finite_safe_states": len(finite_safe),
        "live_finite_safe_task_count": len(finite_tasks),
        "live_finite_safe_tasks": sorted(finite_tasks),
        "live_minimum_successful_boundary_counts": bins,
        "persistent_total_executed_oft_steps": persistent_steps,
        "live_oracle_minimum_total_executed_oft_steps": oracle_steps,
        "live_oracle_oft_step_savings_fraction": savings,
        "safe_handback_status": "ready" if not reasons else "not_ready",
        "safe_handback_reasons": reasons,
    })
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: report[key]
        for key in (
            "cost_accounting_scope",
            "persistent_success_states",
            "live_finite_safe_states",
            "live_finite_safe_task_count",
            "live_minimum_successful_boundary_counts",
            "persistent_total_executed_oft_steps",
            "live_oracle_minimum_total_executed_oft_steps",
            "live_oracle_oft_step_savings_fraction",
            "safe_handback_status",
            "safe_handback_reasons",
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
