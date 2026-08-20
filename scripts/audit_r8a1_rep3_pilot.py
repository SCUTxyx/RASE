#!/usr/bin/env python3
"""Audit the R8-A1 third-replica pilot and decide PASS or K=5 expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_upper(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return float("nan")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return (center + radius) / denominator


def array_equal(left: np.ndarray, right: np.ndarray, atol: float) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.allclose(left, right, rtol=0.0, atol=atol))
    return bool(np.array_equal(left, right))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--float-atol", type=float, default=1e-6)
    parser.add_argument("--max-boundary-disagreement-upper95", type=float, default=0.10)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "frozen" or manifest.get("expected_records") != 32:
        raise ValueError("unexpected or unfrozen R8-A1 pilot manifest")
    records, errors, expansion = [], [], []
    boundary_disagreements = 0
    source_disagreements = 0
    for item in manifest["records"]:
        canonical_path = Path(item["canonical_metadata"])
        if (not canonical_path.is_file()
                or sha256(canonical_path) != item["canonical_metadata_sha256"]):
            errors.append({"group_id": item["group_id"], "reason": "canonical_changed"})
            continue
        role = ("natural_development_eval" if item["cohort_role"] == "natural"
                else "train_enrichment")
        repeat_path = (args.repeat_root / f"suite_{item['suite'].lower()}"
                       / item["policy_id"] / role / f"seed_{item['seed_index']}" / "rep2"
                       / f"{item['state_key']}__seed{item['seed_index']}__rep2.json")
        if not repeat_path.is_file():
            errors.append({"group_id": item["group_id"], "reason": "rep2_missing",
                           "path": str(repeat_path)})
            continue
        canonical = json.loads(canonical_path.read_text())
        repeat = json.loads(repeat_path.read_text())
        if int(repeat.get("rollout_index", -1)) != 2:
            errors.append({"group_id": item["group_id"], "reason": "wrong_rollout_index"})
            continue
        canonical_npz = Path(canonical["npz"])
        repeat_npz = Path(repeat["npz"])
        if (not canonical_npz.is_file()
                or sha256(canonical_npz) != item["canonical_npz_sha256"]
                or sha256(canonical_npz) != canonical.get("npz_sha256")):
            errors.append({"group_id": item["group_id"],
                           "reason": "canonical_npz_missing_or_changed"})
            continue
        if (not repeat_npz.is_file() or sha256(repeat_npz) != repeat.get("npz_sha256")):
            errors.append({"group_id": item["group_id"], "reason": "rep2_npz_missing_or_changed"})
            continue
        with np.load(canonical_npz, allow_pickle=False) as first, \
             np.load(repeat_npz, allow_pickle=False) as third:
            t0_parity = all(array_equal(first[key][0], third[key][0], args.float_atol)
                            for key in ("image", "proprio", "source_action",
                                        "source_action_summary"))
        by_elapsed = {int(row["elapsed_source_steps"]): row for row in repeat.get("rows", [])}
        metadata_ok = (
            int(repeat["rows"][0]["rollout_seed"]) == int(item["rollout_seed"])
            and all(value in by_elapsed for value in (0, 8, 16))
            and str(repeat["rows"][0]["state_key"]) == item["state_key"]
            and str(repeat["rows"][0]["policy_id"]) == item["policy_id"]
            and int(repeat["rows"][0]["seed_index"]) == int(item["seed_index"])
        )
        if not t0_parity or not metadata_ok:
            errors.append({"group_id": item["group_id"], "reason": "parity_or_metadata_failure",
                           "t0_parity": t0_parity, "metadata_ok": metadata_ok})
            continue
        third_labels = {
            str(elapsed): int(bool(by_elapsed[elapsed]["persistent_success_if_enter_now"]))
            for elapsed in (0, 8, 16)
        }
        mixed_boundaries = []
        for elapsed in (0, 8, 16):
            previous_successes = int(item["persistent_successes_k2"][str(elapsed)])
            previous_label = 1 if previous_successes == 2 else 0
            disagreement = third_labels[str(elapsed)] != previous_label
            boundary_disagreements += int(disagreement)
            if disagreement:
                mixed_boundaries.append(elapsed)
        source_previous = int(item["source_successes_k2"]) // 2
        source_third = int(bool(repeat["source_success"]))
        source_disagreements += int(source_third != source_previous)
        record = {
            "group_id": item["group_id"], "state_key": item["state_key"],
            "policy_id": item["policy_id"], "suite": item["suite"],
            "hazard_positive_k2": item["hazard_positive"],
            "third_persistent_labels": third_labels,
            "mixed_boundaries_after_k3": mixed_boundaries,
            "source_third": source_third,
            "t0_feature_parity": t0_parity,
            "same_rollout_seed": True,
            "repeat_metadata": str(repeat_path.resolve()),
            "repeat_metadata_sha256": sha256(repeat_path),
        }
        records.append(record)
        if mixed_boundaries or source_third != source_previous:
            expansion.append({
                "group_id": item["group_id"], "state_key": item["state_key"],
                "policy_id": item["policy_id"], "suite": item["suite"],
                "seed_index": item["seed_index"], "cohort_role": item["cohort_role"],
                "mixed_boundaries": mixed_boundaries,
                "source_mixed": source_third != source_previous,
            })
    boundary_trials = 3 * int(manifest["expected_records"])
    upper = wilson_upper(boundary_disagreements, boundary_trials)
    stable_enough = upper <= args.max_boundary_disagreement_upper95
    if errors:
        status, decision, exit_code = "FAIL", "STOP_PROTOCOL_ERROR", 2
    elif not stable_enough:
        status, decision, exit_code = "FAIL", "STOP_LABEL_INSTABILITY", 2
    elif expansion:
        status, decision, exit_code = "INCOMPLETE", "EXPAND_MIXED_GROUPS_TO_K5", 3
    else:
        status, decision, exit_code = "PASS", "FREEZE_K3_PILOT", 0
    result = {
        "schema_version": "rase-r8a1-rep3-pilot-audit/v1",
        "status": status, "decision": decision,
        "scientific_scope": "label reproducibility pilot; no model result",
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "expected_records": manifest["expected_records"],
        "audited_records": len(records), "records": records, "errors": errors,
        "boundary_third_disagreements": boundary_disagreements,
        "boundary_third_trials": boundary_trials,
        "boundary_disagreement_rate": boundary_disagreements / boundary_trials,
        "boundary_disagreement_wilson_upper95": upper,
        "maximum_boundary_disagreement_upper95": args.max_boundary_disagreement_upper95,
        "source_third_disagreements": source_disagreements,
        "source_third_trials": int(manifest["expected_records"]),
        "expansion_records": expansion,
        "unlocks_on_pass": ["R8-B no-world-model hazard probe on frozen existing data"],
        "remains_locked": ["selector", "world-model", "validation", "test"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "decision", "audited_records", "boundary_third_disagreements",
        "boundary_disagreement_rate", "boundary_disagreement_wilson_upper95",
        "source_third_disagreements", "expansion_records", "errors",
    )}, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
