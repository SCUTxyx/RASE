#!/usr/bin/env python3
"""Freeze a stratified, hash-selected exact-repeat audit for R7-A labels.

The audit is deliberately a *stability* check, not a second source of training
labels.  It selects two source successes and two source failures per suite from
the already frozen natural cohort, using only a public hash of state_key.  Each
selected state is rerun with exactly the same rollout seed and compared before
any source-risk training is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SELECTION_SALT = "rase-r7a-exact-repeat/v1/20260812"
PER_SUITE_AND_CLASS = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(key: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{key}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path)
    args = parser.parse_args()

    audit = json.loads(args.label_audit.read_text())
    if audit.get("status") != "PASS":
        raise ValueError("R7 exact-repeat manifest requires a PASS label-support audit")
    policy_id = str(audit.get("policy_id") or "pi0fast_libero")
    excluded: set[str] = set()
    exclusion_sha = None
    if args.exclusion_manifest is not None:
        exclusion = json.loads(args.exclusion_manifest.read_text())
        if exclusion.get("status") != "frozen":
            raise ValueError("R7 exclusion manifest is not frozen")
        excluded = {str(key) for key in exclusion.get("excluded_state_keys", [])}
        exclusion_sha = sha256(args.exclusion_manifest)
        if audit.get("exclusion_manifest_sha256") != exclusion_sha:
            raise ValueError("label audit / exclusion-manifest hash mismatch")
    groups: dict[tuple[str, bool], list[dict]] = {}
    for metadata_path in sorted(args.input_root.glob("suite_*/seed_0/*__seed0.json")):
        payload = json.loads(metadata_path.read_text())
        rows = payload.get("rows") or []
        if len(rows) != 1:
            raise ValueError(f"expected exactly one t0 row in {metadata_path}")
        row = rows[0]
        key = str(row["state_key"])
        if key in excluded:
            continue
        item = {
            "state_key": key,
            "suite": str(row["suite"]),
            "task_id": str(row["task_id"]),
            "source_success": bool(payload["source_success"]),
            "source_steps": int(payload["source_steps"]),
            "rollout_seed": int(row["rollout_seed"]),
            "policy_id": str(row["policy_id"]),
            "canonical_metadata": str(metadata_path.resolve()),
            "canonical_metadata_sha256": sha256(metadata_path),
            "selection_rank": rank(key),
        }
        if item["policy_id"] != policy_id:
            raise ValueError(
                f"label audit policy {policy_id} != row policy {item['policy_id']}"
            )
        groups.setdefault((item["suite"], item["source_success"]), []).append(item)
    records = []
    for suite in ("Spatial", "Object", "Goal", "Long"):
        for success in (False, True):
            candidates = sorted(groups.get((suite, success), []), key=lambda row: row["selection_rank"])
            if len(candidates) < PER_SUITE_AND_CLASS:
                label = "success" if success else "failure"
                raise ValueError(f"{suite} has only {len(candidates)} {label} candidates")
            records.extend(candidates[:PER_SUITE_AND_CLASS])
    records.sort(key=lambda row: (row["suite"], row["source_success"], row["selection_rank"]))
    payload = {
        "schema_version": "rase-r7a-exact-repeat-manifest/v1",
        "status": "frozen",
        "scientific_scope": "development exact-repeat stability audit; not training data",
        "selection_salt": SELECTION_SALT,
        "per_suite_and_class": PER_SUITE_AND_CLASS,
        "label_audit": str(args.label_audit.resolve()),
        "label_audit_sha256": sha256(args.label_audit),
        "input_root": str(args.input_root.resolve()),
        "exclusion_manifest": (str(args.exclusion_manifest.resolve())
                               if args.exclusion_manifest is not None else None),
        "exclusion_manifest_sha256": exclusion_sha,
        "excluded_state_keys": sorted(excluded),
        "records": records,
        "expected_records": 4 * 2 * PER_SUITE_AND_CLASS,
        "rerun": {
            "policy_id": policy_id, "seed_index": 0,
            "rollout_index": 1, "boundary": [0], "no_oracle": True,
            "bookkeeping_mode": "full",
            "require_same_rollout_seed": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"], "records": len(records),
        "by_suite_and_outcome": {
            f"{suite}|{'success' if success else 'failure'}": len(groups[(suite, success)])
            for suite in ("Spatial", "Object", "Goal", "Long") for success in (False, True)
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
