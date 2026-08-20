#!/usr/bin/env python3
"""Audit R10-B K=3 labels, K2 agreement, and t8 causal feature parity."""

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


def metadata_path(root: Path, row: dict, replica: int) -> Path:
    stem = f"{row['state_key']}__seed{row['seed_index']}"
    if replica:
        stem += f"__rep{replica}"
    return (root / f"suite_{row['suite'].lower()}" / row["policy_id"]
            / f"seed_{row['seed_index']}" / f"rep{replica}" / f"{stem}.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "frozen" or manifest.get("replicas") != 3:
        raise ValueError("unexpected R10-B manifest")
    errors, records = [], []
    for item in manifest["records"]:
        paths = [metadata_path(args.collect_root, item, replica) for replica in range(3)]
        if not all(path.is_file() for path in paths):
            errors.append({"group_id": item["group_id"], "reason": "missing_replica",
                           "paths": [str(path) for path in paths]})
            continue
        payloads = [json.loads(path.read_text()) for path in paths]
        arrays, rows_by_replica = [], []
        for replica, (path, payload) in enumerate(zip(paths, payloads, strict=True)):
            npz = Path(payload["npz"])
            if not npz.is_file() or sha256(npz) != payload.get("npz_sha256"):
                errors.append({"group_id": item["group_id"], "replica": replica,
                               "reason": "npz_hash_failure"})
                continue
            rows = {int(row["elapsed_source_steps"]): row for row in payload.get("rows", [])}
            if (int(payload.get("rollout_index", -1)) != replica
                    or set(rows) != {0, 4, 8, 12, 16}
                    or any(str(row["state_key"]) != item["state_key"]
                           or str(row["policy_id"]) != item["policy_id"]
                           for row in rows.values())):
                errors.append({"group_id": item["group_id"], "replica": replica,
                               "reason": "metadata_failure",
                               "boundaries": sorted(rows)})
            rows_by_replica.append(rows)
            with np.load(npz, allow_pickle=False) as loaded:
                arrays.append({key: loaded[key] for key in loaded.files})
        if len(arrays) != 3:
            continue
        seeds = [int(rows[0]["rollout_seed"]) for rows in rows_by_replica]
        if len(set(seeds)) != 1:
            errors.append({"group_id": item["group_id"], "reason": "rollout_seed_mismatch",
                           "values": seeds})
        parity_keys = ("image", "proprio", "source_action", "source_action_summary",
                       "temporal_image_history", "temporal_proprio_history",
                       "temporal_action_history")
        t8_indices = []
        for replica, rows in enumerate(rows_by_replica):
            elapsed = [int(row["elapsed_source_steps"]) for row in payloads[replica]["rows"]]
            t8_indices.append(elapsed.index(8))
        t8_parity = all(equal(arrays[0][key][t8_indices[0]],
                              arrays[replica][key][t8_indices[replica]])
                        for replica in (1, 2) for key in parity_keys)
        if not t8_parity:
            errors.append({"group_id": item["group_id"], "reason": "t8_feature_parity_failure"})
        t8 = [int(bool(rows[8]["persistent_success_if_enter_now"]))
              for rows in rows_by_replica]
        t16 = [int(bool(rows[16]["persistent_success_if_enter_now"]))
               for rows in rows_by_replica]
        k3_stable = len(set(t8)) == 1 and len(set(t16)) == 1
        label_k3 = int(t8[0] == 1 and t16[0] == 0) if k3_stable else -1
        label_match = k3_stable and all(t8) and label_k3 == int(item["hazard_label_k2"])
        # Controls must remain safe at t16; cases must become unsafe.
        if not k3_stable:
            errors.append({"group_id": item["group_id"], "reason": "k3_label_instability",
                           "t8": t8, "t16": t16})
        elif not label_match:
            errors.append({"group_id": item["group_id"], "reason": "k3_k2_label_mismatch",
                           "label_k2": item["hazard_label_k2"], "label_k3": label_k3,
                           "t8": t8, "t16": t16})
        records.append({
            "group_id": item["group_id"], "state_key": item["state_key"],
            "task_id": item["task_id"], "suite": item["suite"],
            "policy_id": item["policy_id"], "seed_index": item["seed_index"],
            "outer_fold": item["outer_fold"], "label_k2": item["hazard_label_k2"],
            "label_k3": label_k3, "t8_labels_k3": t8, "t16_labels_k3": t16,
            "k3_stable": k3_stable, "k3_matches_k2": label_match,
            "t8_feature_parity": t8_parity,
            "metadata_sha256": [sha256(path) for path in paths],
        })
    status = "PASS" if not errors and len(records) == manifest["expected_groups"] else "FAIL"
    result = {
        "schema_version": "rase-r10b-case-control-repro-audit/v1",
        "status": status,
        "decision": "UNLOCK_R10B_DATASET" if status == "PASS" else "STOP_R10B_PROTOCOL",
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "expected_groups": manifest["expected_groups"], "audited_groups": len(records),
        "labels_k3": dict(sorted(Counter(row["label_k3"] for row in records).items())),
        "errors": errors, "records": records,
        "remains_locked": ["risk_model", "selector", "world_model", "validation", "test"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "decision", "expected_groups", "audited_groups", "labels_k3", "errors")},
        indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
