#!/usr/bin/env python3
"""Freeze post-R10B root-cause groups: all unstable plus matched stable controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SALT = "rase-r10b-oft-trace-root-cause/v1/20260813"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(group_id: str) -> str:
    return hashlib.sha256(f"{SALT}|{group_id}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--repro-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parent = json.loads(args.parent_manifest.read_text())
    audit = json.loads(args.repro_audit.read_text())
    if parent.get("status") != "frozen" or audit.get("status") != "FAIL":
        raise ValueError("expected frozen parent and failed R10-B audit")
    if audit.get("manifest_sha256") != sha256(args.parent_manifest):
        raise ValueError("audit/parent-manifest hash mismatch")
    parent_rows = {row["group_id"]: row for row in parent["records"]}
    unstable = [row for row in audit["records"] if not row["k3_stable"]]
    stable_by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in audit["records"]:
        if row["k3_stable"]:
            stable_by_cell[(row["suite"], row["policy_id"])].append(row)
    controls = []
    used: set[str] = set()
    for row in sorted(unstable, key=lambda value: rank(value["group_id"])):
        cell = (row["suite"], row["policy_id"])
        eligible = [
            value
            for value in stable_by_cell[cell]
            if value["group_id"] not in used
        ]
        if not eligible:
            raise ValueError(f"no unused stable control for {cell}")
        chosen = min(eligible, key=lambda value: rank(value["group_id"]))
        used.add(chosen["group_id"])
        controls.append(chosen)

    roles = {row["group_id"]: "unstable" for row in unstable}
    roles.update({row["group_id"]: "stable_control" for row in controls})
    selected = []
    for group_id in sorted(roles, key=rank):
        row = dict(parent_rows[group_id])
        row["diagnostic_role"] = roles[group_id]
        row["prior_t8_labels_k3"] = next(
            value["t8_labels_k3"] for value in audit["records"]
            if value["group_id"] == group_id
        )
        row["prior_t16_labels_k3"] = next(
            value["t16_labels_k3"] for value in audit["records"]
            if value["group_id"] == group_id
        )
        selected.append(row)

    result = {
        "schema_version": "rase-r10b-oft-trace-diagnostic-manifest/v1",
        "status": "frozen_diagnostic",
        "scientific_scope": "post-R10B-failure root-cause diagnostic only",
        "not_valid_for": ["training", "model selection", "selector", "validation", "test"],
        "parent_manifest_sha256": sha256(args.parent_manifest),
        "repro_audit_sha256": sha256(args.repro_audit),
        "salt": SALT,
        "pool": parent["pool"],
        "boundaries": [0, 8, 16],
        "temporal_history": 8,
        "replicas": 3,
        "expected_groups": len(selected),
        "expected_trajectories": 3 * len(selected),
        "roles": dict(sorted(Counter(roles.values()).items())),
        "cells": dict(
            sorted(Counter(f"{row['suite']}|{row['policy_id']}" for row in selected).items())
        ),
        "records": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
