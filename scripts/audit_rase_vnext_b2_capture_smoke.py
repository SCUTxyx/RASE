#!/usr/bin/env python3
"""Audit synchronous candidate capture and Phase-B parity for the B2 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.adapter_parity import audit_action_roundtrip, audit_motion_trace_conversion
from rase.vnext.candidate_capture import audit_candidate_capture
from rase.vnext.libero import LIBERO_MOTION_SEMANTIC_MAP, LiberoPolicyAdapter


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report_path = args.output_dir / "collection_report.json"
    report = json.loads(report_path.read_text())
    metadata_paths = sorted(args.capture_dir.glob("*.json"))
    rows = [json.loads(line) for line in (args.output_dir / "branches.jsonl").read_text().splitlines()]
    expected_groups = len(manifest["jobs"]) // 5
    capture_audits = {path.name: audit_candidate_capture(path) for path in metadata_paths}
    captures = [json.loads(path.read_text()) for path in metadata_paths]
    capture_by_path = {str(path.resolve()): data for path, data in zip(metadata_paths, captures)}
    join_failures: list[str] = []
    for row in rows:
        if row.get("available") is not True:
            continue
        meta_path = str(row.get("candidate_capture_metadata_path", ""))
        data = capture_by_path.get(meta_path)
        if data is None:
            join_failures.append(f"{row['job_id']}:missing_capture")
            continue
        if row.get("candidate_capture_arrays_sha256") != data.get("arrays_sha256"):
            join_failures.append(f"{row['job_id']}:arrays_hash")

    action_samples: list[np.ndarray] = []
    for data in captures:
        with np.load(data["arrays_path"], allow_pickle=False) as arrays:
            for index in range(len(data["operator_order"])):
                action_samples.append(arrays["actions"][index][arrays["action_step_mask"][index]])
    adapter = LiberoPolicyAdapter(
        policy_id="pi0fast.libero", family="pi0fast",
        supports_requery=True, supports_resample=False, stochastic_sampling=False,
    )
    roundtrip = audit_action_roundtrip(adapter, action_samples)
    tokens = [adapter.raw_to_canonical(sample) for sample in action_samples]
    motion = audit_motion_trace_conversion(tokens, semantic_map=LIBERO_MOTION_SEMANTIC_MAP)
    checks = {
        "scope_is_parity_only": manifest.get("scientific_scope") == "B2_PARITY_ONLY_NOT_AN_EFFECT_COHORT",
        "collection_complete": report.get("status") == "COMPLETE",
        "expected_capture_count": len(metadata_paths) == expected_groups,
        "capture_integrity": all(value["status"] == "PASS" for value in capture_audits.values()),
        "branch_capture_join": not join_failures,
        "all_three_executable_operators_captured": all(
            {"continue.source", "requery.source", "fallback.persistent"}.issubset(
                data["operator_order"]
            ) for data in captures
        ),
        "raw_canonical_raw_roundtrip": roundtrip.status == "PASS",
        "motion_trace_conversion": motion.status == "PASS",
    }
    verdict = "B2_CAPTURE_PASS" if all(checks.values()) else "B2_CAPTURE_FAIL"
    result = {
        "schema_version": "rase-vnext-b2-capture-audit/v1",
        "status": verdict,
        "scientific_scope": "PARITY_ONLY_NOT_AN_EFFECT_RESULT",
        "checks": checks,
        "expected_groups": expected_groups,
        "capture_groups": len(metadata_paths),
        "captured_action_chunks": len(action_samples),
        "capture_audits": capture_audits,
        "join_failures": join_failures,
        "roundtrip": roundtrip.details,
        "motion_trace": motion.details,
        "artifacts": {
            "manifest_sha256": sha256(args.manifest),
            "collection_report_sha256": sha256(report_path),
        },
        "next_action": (
            "permit_single_policy_semantic_pilot_only"
            if verdict == "B2_CAPTURE_PASS" else "repair_capture_contract"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if verdict == "B2_CAPTURE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
