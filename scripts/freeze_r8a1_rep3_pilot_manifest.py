#!/usr/bin/env python3
"""Freeze a hash-selected third-replica pilot for stable R8 hazard labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SALT = "rase-r8a1-rep3-pilot/v1/20260813"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(group_id: str) -> str:
    return hashlib.sha256(f"{SALT}:{group_id}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--hazard-audit", type=Path, required=True)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=2)
    args = parser.parse_args()

    report = json.loads(args.dataset_report.read_text())
    hazard = json.loads(args.hazard_audit.read_text())
    initial = json.loads(args.initial_keys.read_text())
    if report.get("status") != "complete" or report.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("dataset report is incomplete or unbound")
    if hazard.get("status") != "PASS" or hazard.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("R8-A0 hazard gate did not pass on this dataset")
    initial_keys = {str(row["state_key"]) for row in initial.get("records", [])}
    with np.load(args.dataset, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(data["group_id"]):
        grouped[str(value)].append(index)
    strata: dict[tuple[str, str, bool], list[dict]] = defaultdict(list)
    for group_id, indices in sorted(grouped.items()):
        by_elapsed = {int(data["elapsed_source_steps"][i]): i for i in indices}
        if set(by_elapsed) != {0, 8, 16}:
            continue
        selected = [by_elapsed[value] for value in (0, 8, 16)]
        trials = [int(data["persistent_trials"][i]) for i in selected]
        successes = [int(data["persistent_successes"][i]) for i in selected]
        replica_counts = {int(data["replica_count"][i]) for i in selected}
        if trials != [2, 2, 2] or replica_counts != {2}:
            continue
        if any(value not in (0, 2) for value in successes):
            continue
        safe = [value == 2 for value in successes]
        hazard_positive = (safe[0] and not safe[1]) or (safe[1] and not safe[2])
        first = selected[0]
        state_key = str(data["state_key"][first])
        if state_key not in initial_keys:
            raise ValueError(f"selected dataset state absent from initial keys: {state_key}")
        policy = str(data["policy_id"][first])
        suite = str(data["suite"][first])
        try:
            seed = int(group_id.rsplit(":seed", 1)[1])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"cannot parse seed from {group_id}") from exc
        item = {
            "group_id": group_id, "state_key": state_key,
            "task_id": str(data["task_id"][first]), "suite": suite,
            "policy_id": policy, "seed_index": seed,
            "cohort_role": str(data["cohort_role"][first]),
            "hazard_positive": hazard_positive,
            "persistent_successes_k2": {str(t): int(v) for t, v in zip((0, 8, 16), successes)},
            "persistent_trials_k2": 2,
            "source_successes_k2": int(data["source_successes"][first]),
            "source_trials_k2": int(data["source_trials"][first]),
            "selection_rank": rank(group_id),
        }
        strata[(policy, suite, hazard_positive)].append(item)

    records = []
    stratum_counts = {}
    for policy in ("pi05_libero", "pi0fast_libero"):
        for suite in ("Spatial", "Object", "Goal", "Long"):
            for hazard_positive in (False, True):
                key = (policy, suite, hazard_positive)
                candidates = sorted(strata.get(key, []), key=lambda row: row["selection_rank"])
                label = f"{policy}|{suite}|{'hazard' if hazard_positive else 'control'}"
                stratum_counts[label] = len(candidates)
                if len(candidates) < args.per_stratum:
                    raise ValueError(f"pilot stratum {label} has only {len(candidates)} candidates")
                records.extend(candidates[:args.per_stratum])
    records.sort(key=lambda row: (row["suite"], row["policy_id"],
                                  row["hazard_positive"], row["selection_rank"]))
    for row in records:
        role_dir = ("natural_development_eval" if row["cohort_role"] == "natural"
                    else "train_enrichment")
        metadata = (args.raw_root / f"suite_{row['suite'].lower()}" / row["policy_id"]
                    / role_dir / f"seed_{row['seed_index']}" / "rep0"
                    / f"{row['state_key']}__seed{row['seed_index']}.json")
        if not metadata.is_file():
            raise ValueError(f"canonical rep0 metadata missing: {metadata}")
        payload = json.loads(metadata.read_text())
        npz = Path(str(payload["npz"]))
        if not npz.is_file() or sha256(npz) != payload.get("npz_sha256"):
            raise ValueError(f"canonical rep0 NPZ missing or changed: {metadata}")
        rows = {int(item["elapsed_source_steps"]): item
                for item in payload.get("rows", [])}
        if set(rows) != {0, 8, 16}:
            raise ValueError(f"canonical rep0 boundaries changed: {metadata}")
        if any(
            int(bool(rows[elapsed]["persistent_success_if_enter_now"]))
            != int(row["persistent_successes_k2"][str(elapsed)]) // 2
            for elapsed in (0, 8, 16)
        ):
            raise ValueError(f"canonical rep0 labels disagree with frozen K=2 aggregate: {metadata}")
        if int(bool(payload["source_success"])) != int(row["source_successes_k2"]) // 2:
            raise ValueError(f"canonical source label disagrees with frozen K=2 aggregate: {metadata}")
        first_row = payload["rows"][0]
        expected = {
            "state_key": row["state_key"], "policy_id": row["policy_id"],
            "seed_index": row["seed_index"],
        }
        if any(str(first_row[key]) != str(value) for key, value in expected.items()):
            raise ValueError(f"canonical rep0 identity mismatch: {metadata}")
        row.update({
            "canonical_metadata": str(metadata.resolve()),
            "canonical_metadata_sha256": sha256(metadata),
            "canonical_npz": str(npz.resolve()),
            "canonical_npz_sha256": sha256(npz),
            "rollout_seed": int(first_row["rollout_seed"]),
        })
    payload = {
        "schema_version": "rase-r8a1-rep3-pilot-manifest/v1",
        "status": "frozen",
        "scientific_scope": "third-replica stability pilot; balanced, not prevalence evaluation",
        "selection_salt": SALT, "per_stratum": args.per_stratum,
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
        "dataset_report_sha256": sha256(args.dataset_report),
        "hazard_audit": str(args.hazard_audit.resolve()),
        "hazard_audit_sha256": sha256(args.hazard_audit),
        "initial_keys": str(args.initial_keys.resolve()),
        "initial_keys_sha256": sha256(args.initial_keys),
        "raw_root": str(args.raw_root.resolve()),
        "expected_records": 2 * 4 * 2 * args.per_stratum,
        "stratum_candidate_counts": stratum_counts,
        "records": records,
        "rerun": {
            "rollout_index": 2, "boundaries": [0, 8, 16],
            "same_seed_and_checkpoint": True, "persistent_oft": True,
            "t0_feature_parity_required": True,
            "full_action_trace_parity_required": False,
        },
        "adaptive_rule": {
            "if_any_k3_mixed_label": "collect rollout_index 3 and 4 only for mixed groups",
            "otherwise": "freeze K=3 pilot",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "frozen", "records": len(records),
                      "stratum_candidate_counts": stratum_counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
