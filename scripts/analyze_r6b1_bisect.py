#!/usr/bin/env python3
"""Summarize the R6-B1 bisection and locate the first divergent source action."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


EPS = 1e-6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs: list[dict] = []
    for report_path in sorted(glob.glob(str(args.input_root / "**" / "report.json"), recursive=True)):
        report = json.loads(Path(report_path).read_text())
        rel = Path(report_path).relative_to(args.input_root)
        parts = rel.parts  # (label, mode, use_server, repN, report.json)
        if len(parts) != 5:
            continue
        label, mode, use_server, rep, _ = parts
        step_data = {}
        for traj in report.get("trajectories", []):
            state = traj["state_key"]
            npz = np.load(traj["npz"])
            trace = npz.get("source_action_trace")
            step_data[state] = {
                "steps": int(traj["source_steps"]),
                "success": bool(traj["source_success"]),
                "trace": np.asarray(trace, dtype=np.float32),
            }
        runs.append({
            "label": label, "mode": mode, "use_server": int(use_server), "rep": int(rep[3:]),
            "policy_id": report.get("policy_id"), "bookkeeping_mode": report.get("bookkeeping_mode"),
            "skip_oft": bool(report.get("skip_oft")), "steps_by_state": step_data,
        })

    if not runs:
        raise SystemExit("no bisection runs found")

    by_key = {}
    for run in runs:
        for state, data in run["steps_by_state"].items():
            key = (run["label"], state, run["mode"], run["use_server"])
            by_key.setdefault(key, []).append((run["rep"], data))

    table = []
    for (label, state, mode, use_server), entries in sorted(by_key.items()):
        steps = [entry["steps"] for _, entry in entries]
        successes = [entry["success"] for _, entry in entries]
        traces = [entry["trace"] for _, entry in entries]
        min_len = min((t.shape[0] for t in traces), default=0)
        first_div = None
        if len(traces) > 1 and min_len > 0:
            common = np.stack([t[:min_len] for t in traces])
            for idx in range(min_len):
                spread = np.max(np.abs(common[:, idx] - common[0, idx]))
                if spread > EPS:
                    first_div = int(idx)
                    break
        table.append({
            "label": label, "state": state[:16], "mode": mode, "use_server": use_server,
            "steps": steps, "successes": successes,
            "steps_consistent": len(set(steps)) == 1,
            "first_divergent_step": first_div,
        })

    # Reference: none mode (server=1), first rep per label.
    references = {}
    for (label, state, mode, use_server), entries in sorted(by_key.items()):
        if mode == "none" and use_server == 1:
            entries.sort(key=lambda item: item[0])
            references.setdefault(label, {})[state] = entries[0][1]["trace"]
    divergences = []
    for run in runs:
        if run["mode"] == "none":
            continue
        for state, data in run["steps_by_state"].items():
            ref = references.get(run["label"], {}).get(state)
            if ref is None:
                continue
            trace = data["trace"]
            common = min(ref.shape[0], trace.shape[0])
            first = None
            for idx in range(common):
                if np.max(np.abs(ref[idx] - trace[idx])) > EPS:
                    first = int(idx)
                    break
            divergences.append({
                "label": run["label"], "state": state[:16], "mode": run["mode"],
                "use_server": run["use_server"], "rep": run["rep"],
                "ref_steps": int(ref.shape[0]), "steps": int(trace.shape[0]),
                "first_divergent_step_vs_none": first,
            })

    result = {
        "schema_version": "rase-r6b1-bisect-summary/v1",
        "runs": table, "divergence_vs_none": divergences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for row in table:
        print(f"{row['label']:10s} {row['state']:16s} mode={row['mode']:13s} "
              f"server={row['use_server']} steps={row['steps']} consistent={row['steps_consistent']} "
              f"first_div={row['first_divergent_step']}")
    print("\n--- divergence vs none ---")
    for row in divergences:
        print(f"{row['label']:10s} {row['state']:16s} mode={row['mode']:13s} "
              f"server={row['use_server']} rep={row['rep']} "
              f"ref={row['ref_steps']} steps={row['steps']} first_div={row['first_divergent_step_vs_none']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
