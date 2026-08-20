#!/usr/bin/env python3
"""Audit R9-B temporal replicas and freeze the collector provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(left: np.ndarray, right: np.ndarray, atol: float = 1e-6) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.allclose(left, right, rtol=0.0, atol=atol))
    return bool(np.array_equal(left, right))


def valid_boundary_set(rows: set[int], source_success: bool) -> bool:
    """Return whether a trajectory exposes a valid causal boundary prefix."""
    requested = (0, 4, 8, 12, 16)
    ordered = tuple(sorted(rows))
    if not ordered or ordered != requested[:len(ordered)]:
        return False
    # Early success may end the rollout before the next planned boundary;
    # failure/horizon trajectories must retain the complete planned window.
    return bool(source_success) or ordered == requested


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "frozen" or manifest.get("expected_records") != 24:
        raise ValueError("unexpected R9-B manifest")
    errors, records = [], []
    expected_trials = 0
    for item in manifest["records"]:
        for policy in manifest["source_policies"]:
            seed = 4 if policy == "pi05_libero" else 2
            paths = []
            for replica in range(args.replicas):
                stem = f"{item['state_key']}__seed{seed}"
                if replica:
                    stem += f"__rep{replica}"
                path = (args.collect_root / f"suite_{item['suite'].lower()}" / policy
                        / f"rep{replica}" / f"{stem}.json")
                paths.append(path)
            if not all(path.is_file() for path in paths):
                errors.append({"state_key": item["state_key"], "policy_id": policy,
                               "reason": "missing_replica",
                               "paths": [str(path) for path in paths]})
                continue
            payloads = [json.loads(path.read_text()) for path in paths]
            npzs = [Path(payload["npz"]) for payload in payloads]
            if any((not path.is_file()
                    or sha256(path) != payload.get("npz_sha256"))
                   for path, payload in zip(npzs, payloads, strict=True)):
                errors.append({"state_key": item["state_key"], "policy_id": policy,
                               "reason": "npz_hash_failure"})
                continue
            row_sets = []
            arrays = []
            for replica, payload in enumerate(payloads):
                rows = {int(row["elapsed_source_steps"]): row
                        for row in payload.get("rows", [])}
                expected_trials += 1
                # A successful episode can terminate before the next planned
                # boundary.  Such a trajectory legitimately contains the
                # prefix {0} (or {0,4,...}) rather than all five boundaries;
                # requiring {0,4,8,12,16} would reject valid short successes.
                # Failed/horizon episodes must expose the complete window so
                # that every causal transition is auditable.
                metadata_ok = (
                    int(payload.get("rollout_index", -1)) == replica
                    and valid_boundary_set(set(rows), bool(payload.get("source_success")))
                    and all(str(row["state_key"]) == item["state_key"]
                            and str(row["policy_id"]) == policy
                            and int(row["rollout_seed"]) == int(payloads[0]["rows"][0]["rollout_seed"])
                            for row in rows.values())
                )
                if not metadata_ok:
                    errors.append({"state_key": item["state_key"], "policy_id": policy,
                                   "replica": replica, "reason": "metadata_failure"})
                row_sets.append(rows)
                with np.load(npzs[replica], allow_pickle=False) as loaded:
                    arrays.append({key: loaded[key] for key in loaded.files})
            parity_keys = ("image", "proprio", "source_action", "source_action_summary",
                           "temporal_image_history", "temporal_proprio_history",
                           "temporal_action_history")
            parity = all(equal(arrays[0][key][0], arrays[replica][key][0])
                         for replica in range(1, args.replicas) for key in parity_keys)
            if not parity:
                errors.append({"state_key": item["state_key"], "policy_id": policy,
                               "reason": "t0_feature_parity_failure"})
            label_values = {
                "source": [int(bool(payload["source_success"])) for payload in payloads],
                "persistent_t0": [int(bool(rows[0]["persistent_success_if_enter_now"]))
                                  for rows in row_sets],
            }
            records.append({
                "state_key": item["state_key"], "task_id": item["task_id"],
                "suite": item["suite"], "policy_id": policy,
                "perturb_dim": item["perturb_dim"], "perturb_level": item["perturb_level"],
                "replica_count": args.replicas, "t0_feature_parity": parity,
                "rollout_seeds": [int(payload["rows"][0]["rollout_seed"]) for payload in payloads],
                "source_successes": label_values["source"],
                "persistent_t0_successes": label_values["persistent_t0"],
                "metadata_sha256": [sha256(path) for path in paths],
            })
    source_counts = Counter(value for row in records for value in row["source_successes"])
    persistent_counts = Counter(value for row in records for value in row["persistent_t0_successes"])
    status = "PASS" if not errors and len(records) == 48 else "FAIL"
    result = {
        "schema_version": "rase-r9b-temporal-repro-audit/v1",
        "status": status,
        "decision": "UNLOCK_R9B_DATASET_BUILD" if status == "PASS" else "STOP_TEMPORAL_PROTOCOL",
        "scientific_scope": "R9-B temporal development collection reproducibility audit",
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "expected_state_policy_groups": 48, "audited_state_policy_groups": len(records),
        "expected_replicas_per_group": args.replicas, "errors": errors,
        "source_success_counts": dict(source_counts),
        "persistent_t0_success_counts": dict(persistent_counts),
        "records": records,
        "full_late_action_trace_parity_required": False,
        "short_success_prefixes_allowed": True,
        "t0_feature_parity_required": True,
        "remains_locked": ["risk_model", "selector", "world_model", "validation", "test"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "decision", "audited_state_policy_groups", "source_success_counts",
        "persistent_t0_success_counts", "errors",
    )}, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
