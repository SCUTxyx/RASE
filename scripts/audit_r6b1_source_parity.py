#!/usr/bin/env python3
"""Hard gate: verify collector source trajectories match frozen R6-A references.

Compares every collected boundary row against the per-state R6-A reference
(outcome records under ``policy_pair_atlas_v1/<policy>/seed_<k>/summary.json``)
and verifies every saved feature array is finite.  Exits non-zero on any
parity or finiteness violation, so it can be wired into the parity/pilot
runners as a hard gate.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np


def load_references(atlas_root: Path, policy_id: str, seed_index: int) -> dict[str, dict]:
    summary = atlas_root / policy_id / f"seed_{seed_index}" / "summary.json"
    if not summary.exists():
        return {}
    data = json.loads(summary.read_text())
    return {
        rec["state_key"]: {
            "rollout_seed": int(rec["rollout_seed"]),
            "success": bool(rec.get("source_success", rec["result"]["success"])),
            "env_steps": int(rec["result"]["env_steps"]),
            "suite": rec["suite"],
            "task_id": rec["task_id"],
        }
        for rec in data.get("per_state", [])
    }


def load_exclusions(path: Path) -> set[tuple[str, int, str]]:
    """Load a frozen exclusion manifest into {(policy_id, seed_index, state_key)}.

    Each entry must be a [policy_id, seed_index, state_key] triple.  Excluded
    trajectories are skipped by the audit (they are pre-recorded known
    nondeterministic states) but still reported under ``excluded`` so the gate
    decision remains auditable.
    """
    if path is None:
        return set()
    data = json.loads(path.read_text())
    excluded = set()
    for entry in data["excluded"]:
        policy, seed, state = entry
        excluded.add((str(policy), int(seed), str(state)))
    return excluded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-id", action="append", default=[])
    parser.add_argument("--seed-index", action="append", default=[])
    parser.add_argument("--exclude", type=Path, default=None,
                        help="frozen exclusion manifest (see load_exclusions)")
    args = parser.parse_args(argv)
    atlas = json.loads(args.atlas.read_text())
    atlas_root = Path(atlas.get("atlas_root", str(args.atlas.resolve().parent / "policy_pair_atlas_v1")))
    policies = args.policy_id or sorted(atlas.get("pairs", {}))
    seeds = [int(value) for value in args.seed_index] or None
    excluded = load_exclusions(args.exclude)

    metadata_paths = sorted(glob.glob(str(args.input_root / "**" / "*__seed*.json"), recursive=True))
    metadata_paths = [path for path in metadata_paths if Path(path).name != "report.json"]
    if not metadata_paths:
        report_path = args.input_root / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            for traj in report.get("trajectories", []):
                stem = f"{traj['state_key']}__seed{report['seed_index']}"
                candidate = args.input_root / f"{stem}.json"
                if candidate.exists():
                    metadata_paths.append(str(candidate))
    rows = []
    parity_failures = []
    finite_failures = []
    missing_reference = []
    mode_violations = []
    excluded_trajectories = []
    seen = set()
    for path_string in metadata_paths:
        path = Path(path_string)
        data = json.loads(path.read_text())
        policy_id = data["rows"][0]["policy_id"] if data["rows"] else None
        seed_index = int(data["rows"][0]["seed_index"]) if data["rows"] else None
        if policy_id not in policies:
            continue
        if seeds is not None and seed_index not in seeds:
            continue
        state = data["rows"][0]["state_key"] if data["rows"] else None
        if state is not None and (policy_id, seed_index, state) in excluded:
            excluded_trajectories.append({
                "key": [policy_id, seed_index, state],
                "detail": "pre-recorded known nondeterministic state; excluded from parity gate",
            })
            continue
        sibling_report = path.parent / "report.json"
        if sibling_report.exists():
            sibling = json.loads(sibling_report.read_text())
            mode = sibling.get("bookkeeping_mode")
            if mode not in (None, "full"):
                mode_violations.append({"path": str(path), "bookkeeping_mode": mode})
        references = load_references(atlas_root, policy_id, seed_index)
        npz = np.load(data["npz"])
        if not all(np.isfinite(npz[key]).all() for key in npz.files):
            finite_failures.append(str(path))
        for boundary in data["rows"]:
            state = boundary["state_key"]
            key = (policy_id, seed_index, state)
            seen.add(key)
            ref = references.get(state)
            if ref is None:
                missing_reference.append({"key": list(key), "detail": "state not in atlas reference"})
                continue
            observed = {
                "rollout_seed": int(boundary["rollout_seed"]),
                "source_final_success": bool(boundary["source_final_success"]),
                "source_total_steps": int(boundary["source_total_steps"]),
            }
            expected = {
                "rollout_seed": ref["rollout_seed"],
                "source_final_success": bool(ref["success"]),
                "source_total_steps": ref["env_steps"],
            }
            if observed != expected:
                parity_failures.append({
                    "key": list(key), "boundary": boundary["elapsed_source_steps"],
                    "observed": observed, "expected": expected,
                })
            rows.append(boundary)

    report = json.loads((args.input_root / "report.json").read_text()) if (args.input_root / "report.json").exists() else {}

    reasons = []
    if missing_reference: reasons.append(f"{len(missing_reference)} trajectories missing atlas reference")
    if parity_failures: reasons.append(f"{len(parity_failures)} source-parity failures")
    if finite_failures: reasons.append(f"{len(finite_failures)} nonfinite npz files")
    if mode_violations: reasons.append(f"{len(mode_violations)} collected with non-full bookkeeping mode")
    if not rows: reasons.append("no collector boundary rows found")
    if report and report.get("collector_sha256") is None:
        reasons.append("collector report missing collector_sha256")

    result = {
        "schema_version": "rase-r6b1-source-parity-audit/v1",
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "n_seen_trajectories": len(seen),
        "n_excluded_trajectories": len(excluded_trajectories),
        "excluded": excluded_trajectories,
        "n_rows": len(rows),
        "n_parity_checked": len(rows),
        "missing_reference": missing_reference,
        "parity_failures": parity_failures,
        "nonfinite_files": finite_failures,
        "mode_violations": mode_violations,
        "collector_report": {
            key: report.get(key) for key in ("collector_sha256", "bookkeeping_mode", "boundaries", "policy_id", "seed_index")
        } if report else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
