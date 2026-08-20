#!/usr/bin/env python3
"""Finalize R8-A1 after pre-registered K=5 expansion of mixed K=3 groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(left: np.ndarray, right: np.ndarray, atol: float) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.allclose(left, right, rtol=0.0, atol=atol))
    return bool(np.array_equal(left, right))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rep3-audit", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--float-atol", type=float, default=1e-6)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    first_audit = json.loads(args.rep3_audit.read_text())
    if (manifest.get("status") != "frozen"
            or first_audit.get("status") != "INCOMPLETE"
            or first_audit.get("decision") != "EXPAND_MIXED_GROUPS_TO_K5"
            or first_audit.get("manifest_sha256") != sha256(args.manifest)
            or first_audit.get("boundary_disagreement_wilson_upper95", 1.0)
            > first_audit.get("maximum_boundary_disagreement_upper95", 0.0)):
        raise ValueError("R8-A1 K=3 audit does not authorize K=5 expansion")
    manifest_by_group = {row["group_id"]: row for row in manifest["records"]}
    errors, records = [], []
    for expansion in first_audit["expansion_records"]:
        item = manifest_by_group[expansion["group_id"]]
        canonical_npz = Path(item["canonical_npz"])
        if (not canonical_npz.is_file()
                or sha256(canonical_npz) != item["canonical_npz_sha256"]):
            errors.append({"group_id": item["group_id"], "reason": "canonical_changed"})
            continue
        role = ("natural_development_eval" if item["cohort_role"] == "natural"
                else "train_enrichment")
        labels = {elapsed: int(item["persistent_successes_k2"][str(elapsed)])
                  for elapsed in (0, 8, 16)}
        source_successes = int(item["source_successes_k2"])
        replica_records = []
        valid = True
        with np.load(canonical_npz, allow_pickle=False) as canonical:
            for replica in (2, 3, 4):
                path = (args.repeat_root / f"suite_{item['suite'].lower()}"
                        / item["policy_id"] / role / f"seed_{item['seed_index']}"
                        / f"rep{replica}"
                        / f"{item['state_key']}__seed{item['seed_index']}__rep{replica}.json")
                if not path.is_file():
                    errors.append({"group_id": item["group_id"],
                                   "reason": f"rep{replica}_missing", "path": str(path)})
                    valid = False
                    break
                payload = json.loads(path.read_text())
                npz = Path(payload["npz"])
                if (int(payload.get("rollout_index", -1)) != replica
                        or not npz.is_file() or sha256(npz) != payload.get("npz_sha256")):
                    errors.append({"group_id": item["group_id"],
                                   "reason": f"rep{replica}_contract_failure"})
                    valid = False
                    break
                rows = {int(row["elapsed_source_steps"]): row
                        for row in payload.get("rows", [])}
                metadata_ok = (
                    set(rows) == {0, 8, 16}
                    and int(payload["rows"][0]["rollout_seed"]) == int(item["rollout_seed"])
                    and str(payload["rows"][0]["state_key"]) == item["state_key"]
                    and str(payload["rows"][0]["policy_id"]) == item["policy_id"]
                    and int(payload["rows"][0]["seed_index"]) == int(item["seed_index"])
                )
                with np.load(npz, allow_pickle=False) as observed:
                    parity = all(equal(canonical[key][0], observed[key][0], args.float_atol)
                                 for key in ("image", "proprio", "source_action",
                                             "source_action_summary"))
                if not metadata_ok or not parity:
                    errors.append({"group_id": item["group_id"],
                                   "reason": f"rep{replica}_parity_failure",
                                   "metadata_ok": metadata_ok, "t0_feature_parity": parity})
                    valid = False
                    break
                replica_labels = {
                    elapsed: int(bool(rows[elapsed]["persistent_success_if_enter_now"]))
                    for elapsed in (0, 8, 16)
                }
                for elapsed, value in replica_labels.items():
                    labels[elapsed] += value
                source_value = int(bool(payload["source_success"]))
                source_successes += source_value
                replica_records.append({
                    "replica": replica, "persistent_labels": replica_labels,
                    "source_success": source_value, "metadata": str(path.resolve()),
                    "metadata_sha256": sha256(path),
                })
        if valid:
            records.append({
                "group_id": item["group_id"], "state_key": item["state_key"],
                "policy_id": item["policy_id"], "suite": item["suite"],
                "persistent_successes_k5": {str(key): value for key, value in labels.items()},
                "persistent_trials": 5, "source_successes_k5": source_successes,
                "source_trials": 5, "replicas": replica_records,
            })
    expected = len(first_audit["expansion_records"])
    status = "PASS" if not errors and len(records) == expected else "FAIL"
    ambiguous_boundaries = sum(
        value not in (0, 5)
        for record in records for value in record["persistent_successes_k5"].values()
    )
    result = {
        "schema_version": "rase-r8a1-label-stability-final/v1",
        "status": status,
        "decision": "FREEZE_PROBABILISTIC_K5_LABELS" if status == "PASS"
                    else "STOP_PROTOCOL_ERROR",
        "scientific_scope": "label reproducibility final; no model result",
        "manifest_sha256": sha256(args.manifest),
        "rep3_audit_sha256": sha256(args.rep3_audit),
        "expected_expansion_groups": expected, "audited_expansion_groups": len(records),
        "boundary_third_disagreements": first_audit["boundary_third_disagreements"],
        "boundary_third_trials": first_audit["boundary_third_trials"],
        "boundary_disagreement_wilson_upper95": first_audit["boundary_disagreement_wilson_upper95"],
        "ambiguous_boundaries_after_k5": ambiguous_boundaries,
        "expanded_boundary_trials": 3 * 5 * expected,
        "records": records, "errors": errors,
        "unlocks_on_pass": ["R8-B no-world-model hazard probe"],
        "remains_locked": ["selector", "world-model", "validation", "test"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "decision", "expected_expansion_groups",
        "audited_expansion_groups", "ambiguous_boundaries_after_k5", "errors",
    )}, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
