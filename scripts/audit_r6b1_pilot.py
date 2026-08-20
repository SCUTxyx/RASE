#!/usr/bin/env python3
"""Validate R6-B1 pilot provenance, exact source parity, and label support."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected = {(row["policy_id"], row["suite"], row["state_key"], seed): row
                for row in manifest["records"] for seed in row["seed_indices"]}
    metadata_paths = sorted(glob.glob(str(args.input_root / "suite_*" / "*" / "seed_*" / "*.json")))
    rows = []
    parity_failures = []
    finite_failures = []
    for path_string in metadata_paths:
        path = Path(path_string)
        if path.name == "report.json":
            continue
        data = json.loads(path.read_text())
        for boundary in data["rows"]:
            key = (boundary["policy_id"], boundary["suite"], boundary["state_key"], int(boundary["seed_index"]))
            record = expected.get(key)
            if record is None:
                parity_failures.append({"reason": "unexpected_trajectory", "key": key})
                continue
            reference = record[f"r6a_seed{key[-1]}"]
            if (int(boundary["rollout_seed"]) != int(reference["rollout_seed"])
                    or bool(boundary["source_final_success"]) != bool(reference["success"])
                    or int(boundary["source_total_steps"]) != int(reference["env_steps"])):
                parity_failures.append({"key": key, "boundary": boundary["elapsed_source_steps"],
                                        "observed": {name: boundary.get(name) for name in ["rollout_seed", "source_final_success", "source_total_steps"]},
                                        "expected": reference})
            rows.append(boundary)
        npz = np.load(data["npz"])
        if not all(np.isfinite(npz[key]).all() for key in npz.files):
            finite_failures.append(str(path))
    seen = {(row["policy_id"], row["suite"], row["state_key"], int(row["seed_index"])) for row in rows}
    missing = sorted(set(expected) - seen)
    by_policy = {}
    for policy in sorted({row["policy_id"] for row in rows}):
        subset = [row for row in rows if row["policy_id"] == policy]
        by_policy[policy] = {
            "rows": len(subset),
            "trajectory_groups": len({row["group_id"] for row in subset}),
            "source_failures": sum(not row["source_final_success"] for row in subset),
            "source_successes": sum(bool(row["source_final_success"]) for row in subset),
            "persistent_successes": sum(bool(row["persistent_success_if_enter_now"]) for row in subset),
            "later_boundaries": sum(int(row["elapsed_source_steps"]) > 0 for row in subset),
        }
    reasons = []
    if missing: reasons.append(f"missing {len(missing)} expected trajectories")
    if parity_failures: reasons.append(f"{len(parity_failures)} source-parity failures")
    if finite_failures: reasons.append(f"{len(finite_failures)} nonfinite npz files")
    if len(rows) < len(expected) * 2: reasons.append("too few reached boundaries")
    if any(value["source_failures"] == 0 or value["source_successes"] == 0 for value in by_policy.values()):
        reasons.append("at least one policy lacks both source label classes")
    if any(value["later_boundaries"] == 0 for value in by_policy.values()):
        reasons.append("at least one policy lacks later boundaries")
    result = {
        "schema_version": "rase-r6b1-pilot-audit/v1", "status": "pass" if not reasons else "fail",
        "reasons": reasons, "n_expected_trajectories": len(expected), "n_seen_trajectories": len(seen),
        "n_rows": len(rows), "missing": missing, "parity_failures": parity_failures,
        "nonfinite_files": finite_failures, "by_policy": by_policy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
