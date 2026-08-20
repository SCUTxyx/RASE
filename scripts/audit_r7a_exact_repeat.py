#!/usr/bin/env python3
"""Fail-closed exact-repeat audit for R7-A source-only source-risk labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.dtype != right.dtype:
        return float("inf")
    if not np.issubdtype(left.dtype, np.floating):
        return 0.0 if np.array_equal(left, right) else float("inf")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--float-atol", type=float, default=1e-6)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "frozen":
        raise ValueError("exact-repeat manifest is not frozen")
    records, errors = [], []
    for item in manifest.get("records") or []:
        canonical_path = Path(item["canonical_metadata"])
        repeat_path = args.repeat_root / f"suite_{item['suite'].lower()}" / "seed_0" / (
            f"{item['state_key']}__seed0__rep1.json"
        )
        if not canonical_path.is_file() or sha256(canonical_path) != item["canonical_metadata_sha256"]:
            errors.append({"state_key": item["state_key"], "reason": "canonical_metadata_changed"})
            continue
        if not repeat_path.is_file():
            errors.append({"state_key": item["state_key"], "reason": "repeat_missing"})
            continue
        canonical = json.loads(canonical_path.read_text())
        repeat = json.loads(repeat_path.read_text())
        canonical_row, repeat_row = canonical["rows"][0], repeat["rows"][0]
        checks = {
            "same_rollout_seed": int(canonical_row["rollout_seed"]) == int(repeat_row["rollout_seed"]),
            "same_source_success": bool(canonical["source_success"]) == bool(repeat["source_success"]),
            "same_source_steps": int(canonical["source_steps"]) == int(repeat["source_steps"]),
            "same_stop_reason": canonical.get("stop_reason") == repeat.get("stop_reason"),
            "same_provenance": all(canonical_row[key] == repeat_row[key] for key in (
                "state_key", "task_id", "suite", "policy_id", "seed_index", "elapsed_source_steps",
            )),
            "one_t0_row_each": (len(canonical["rows"]) == len(repeat["rows"]) == 1
                                and int(canonical_row["elapsed_source_steps"]) == 0
                                and int(repeat_row["elapsed_source_steps"]) == 0),
            "no_oft_each": (canonical_row["persistent_success_if_enter_now"] is None
                            and repeat_row["persistent_success_if_enter_now"] is None),
        }
        npz_diff: dict[str, float] = {}
        try:
            with np.load(Path(canonical["npz"]), allow_pickle=False) as first, \
                 np.load(Path(repeat["npz"]), allow_pickle=False) as second:
                expected = {
                    "image", "proprio", "source_action", "source_action_summary",
                    "source_action_trace", "oft_action", "oft_action_summary",
                }
                checks["same_npz_keys"] = set(first.files) == expected == set(second.files)
                for key in sorted(expected):
                    npz_diff[key] = array_difference(first[key], second[key])
                checks["same_t0_features"] = all(npz_diff[key] <= args.float_atol for key in (
                    "image", "proprio", "source_action", "source_action_summary",
                ))
                checks["same_action_trace"] = npz_diff["source_action_trace"] <= args.float_atol
                checks["no_oft_arrays"] = (first["oft_action"].size == second["oft_action"].size == 0
                                            and first["oft_action_summary"].size
                                            == second["oft_action_summary"].size == 0)
        except Exception as exc:  # noqa: BLE001
            checks["npz_readable"] = False
            errors.append({"state_key": item["state_key"], "reason": "npz_read_failure", "detail": repr(exc)})
            continue
        failed = sorted(name for name, value in checks.items() if not value)
        record = {
            "state_key": item["state_key"], "suite": item["suite"],
            "source_success": bool(item["source_success"]),
            "canonical_metadata": str(canonical_path), "repeat_metadata": str(repeat_path),
            "checks": checks, "max_abs_difference": npz_diff,
        }
        records.append(record)
        if failed:
            errors.append({"state_key": item["state_key"], "reason": "repeat_mismatch",
                           "failed_checks": failed, "max_abs_difference": npz_diff})
    expected = int(manifest.get("expected_records", -1))
    status = "PASS" if len(records) == expected and not errors else "FAIL"
    result = {
        "schema_version": "rase-r7a-exact-repeat-audit/v1",
        "status": status, "scientific_scope": "development reproducibility gate",
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "label_audit_sha256": manifest.get("label_audit_sha256"),
        "exclusion_manifest_sha256": manifest.get("exclusion_manifest_sha256"),
        "repeat_root": str(args.repeat_root.resolve()), "expected_records": expected,
        "audited_records": len(records), "float_atol": args.float_atol,
        "records": records, "errors": errors,
        "unlocks_on_pass": ["canonical source-risk OOF only"],
        "remains_locked": ["OFT", "selector", "world-model", "validation", "test"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "audited_records": len(records),
                      "errors": errors}, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
