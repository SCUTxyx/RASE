#!/usr/bin/env python3
"""Merge qualified, same-state R7 source-risk cohorts for multi-VLA OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--dataset-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.dataset) != len(args.dataset_report) or len(args.dataset) < 2:
        raise ValueError("provide matching dataset/report pairs for at least two VLAs")

    cohorts: list[tuple[str, dict[str, np.ndarray], dict, Path, Path]] = []
    seen: set[str] = set()
    for dataset, report_path in zip(args.dataset, args.dataset_report, strict=True):
        report = json.loads(report_path.read_text())
        if report.get("dataset_sha256") != sha256(dataset):
            raise ValueError(f"dataset/report hash mismatch: {dataset}")
        if int(report.get("rows", -1)) != 192 or int(report.get("tasks", -1)) != 48:
            raise ValueError(f"unqualified cohort shape: {dataset}")
        with np.load(dataset, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        policies = sorted(set(data["policy_id"].tolist()))
        if len(policies) != 1:
            raise ValueError(f"cohort must contain exactly one policy: {dataset}")
        policy = str(policies[0])
        if policy in seen:
            raise ValueError(f"duplicate policy cohort: {policy}")
        if str(report.get("policy_id") or policy) != policy:
            raise ValueError(f"report policy mismatch: {dataset}")
        seen.add(policy)
        cohorts.append((policy, data, report, dataset, report_path))

    cohorts.sort(key=lambda row: row[0])
    reference = cohorts[0][1]
    state_keys = reference["state_key"].astype(str)
    if len(set(state_keys.tolist())) != 192:
        raise ValueError("reference cohort does not have 192 unique state keys")
    order_by_policy: dict[str, np.ndarray] = {}
    alignment = {}
    for policy, data, _, _, _ in cohorts:
        positions = {str(key): index for index, key in enumerate(data["state_key"])}
        if set(positions) != set(state_keys.tolist()):
            raise ValueError(f"{policy} does not use the exact frozen 192-state cohort")
        order = np.asarray([positions[key] for key in state_keys], dtype=np.int64)
        order_by_policy[policy] = order
        checks = {}
        for key in ("task_id", "suite", "init_state_id", "instruction",
                    "language_hash", "image", "proprio"):
            if key not in data or key not in reference:
                raise ValueError(f"missing aligned field {key}")
            checks[key] = bool(np.array_equal(data[key][order], reference[key]))
        if not all(checks.values()):
            raise ValueError(f"same-state feature alignment failed for {policy}: {checks}")
        alignment[policy] = checks

    policy_names = [row[0] for row in cohorts]
    policy_to_index = {name: index for index, name in enumerate(policy_names)}
    merge_keys = (
        "image", "proprio", "action_summary", "action_summary_single_step",
        "language_hash", "instruction", "source_failure", "source_success",
        "source_steps", "state_key", "task_id", "suite", "perturb_dim",
        "init_state_id", "policy_id",
    )
    merged = {}
    for key in merge_keys:
        merged[key] = np.concatenate([
            data[key][order_by_policy[policy]] for policy, data, _, _, _ in cohorts
        ], axis=0)
    merged["policy_index"] = np.concatenate([
        np.full(192, policy_to_index[policy], dtype=np.int64)
        for policy in policy_names
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)
    result = {
        "schema_version": "rase-r7c-multivla-source-risk-dataset/v1",
        "status": "frozen",
        "dataset": str(args.output.resolve()),
        "dataset_sha256": sha256(args.output),
        "rows": int(len(merged["source_failure"])),
        "states_per_policy": 192,
        "tasks": 48,
        "policies": policy_names,
        "policy_to_index": policy_to_index,
        "same_state_alignment": alignment,
        "sources": [{
            "policy_id": policy,
            "dataset": str(dataset.resolve()),
            "dataset_sha256": sha256(dataset),
            "report_sha256": sha256(report_path),
        } for policy, _, _, dataset, report_path in cohorts],
        "split_rule": "task-held-out; all policies and init states of a task share a fold",
        "forbidden": [
            "OFT labels/actions/cost", "outcome-derived policy descriptor",
            "validation/test states", "policy success-rate feature",
        ],
    }
    report_path = args.output.with_suffix(".npz.report.json")
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
