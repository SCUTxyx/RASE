#!/usr/bin/env python3
"""Audit full OFT action-trace hashes in the frozen R10-B root-cause pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata_path(root: Path, row: dict, replica: int) -> Path:
    stem = f"{row['state_key']}__seed{row['seed_index']}"
    if replica:
        stem += f"__rep{replica}"
    return (
        root
        / f"suite_{row['suite'].lower()}"
        / row["policy_id"]
        / f"seed_{row['seed_index']}"
        / f"rep{replica}"
        / f"{stem}.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "frozen_diagnostic":
        raise ValueError("trace diagnostic manifest is not frozen")

    errors, records = [], []
    for item in manifest["records"]:
        paths = [metadata_path(args.collect_root, item, replica) for replica in range(3)]
        if not all(path.is_file() for path in paths):
            errors.append({"group_id": item["group_id"], "reason": "missing_replica"})
            continue
        payloads = [json.loads(path.read_text()) for path in paths]
        rows = [
            {int(row["elapsed_source_steps"]): row for row in payload["rows"]}
            for payload in payloads
        ]
        boundary_records = {}
        for boundary in (8, 16):
            hashes = [row[boundary].get("persistent_action_trace_sha256") for row in rows]
            shapes = [row[boundary].get("persistent_action_trace_shape") for row in rows]
            outcomes = [int(bool(row[boundary]["persistent_success_if_enter_now"])) for row in rows]
            if any(value is None for value in hashes + shapes):
                errors.append({
                    "group_id": item["group_id"], "boundary": boundary,
                    "reason": "missing_trace_hash",
                })
            boundary_records[str(boundary)] = {
                "trace_hashes": hashes,
                "trace_shapes": shapes,
                "outcomes": outcomes,
                "trace_exact": len(set(hashes)) == 1 and len({tuple(x) for x in shapes}) == 1,
                "outcome_stable": len(set(outcomes)) == 1,
            }
        records.append({
            "group_id": item["group_id"], "task_id": item["task_id"],
            "suite": item["suite"], "policy_id": item["policy_id"],
            "diagnostic_role": item["diagnostic_role"],
            "boundaries": boundary_records,
        })

    categories = Counter()
    for record in records:
        for boundary, values in record["boundaries"].items():
            categories[(
                int(boundary),
                "trace_same" if values["trace_exact"] else "trace_diff",
                "outcome_stable" if values["outcome_stable"] else "outcome_flip",
            )] += 1
    if errors or len(records) != manifest["expected_groups"]:
        status, decision = "FAIL_CONTRACT", "STOP_ROOT_CAUSE_DIAGNOSTIC"
    elif any(key[1] == "trace_same" and key[2] == "outcome_flip" for key in categories):
        status, decision = "ENVIRONMENT_VARIANCE_EVIDENCE", "RUN_FIXED_ACTION_REPLAY"
    elif any(key[1] == "trace_diff" and key[2] == "outcome_flip" for key in categories):
        status, decision = "CLOSED_LOOP_TRACE_DIVERGENCE", "AUDIT_CHUNK_INPUT_DIVERGENCE"
    else:
        status, decision = "NO_REPRODUCED_FLIPS", "STOP_OR_REPEAT_WITHOUT_TRAINING"
    result = {
        "schema_version": "rase-r10b-oft-trace-diagnostic-audit/v1",
        "status": status, "decision": decision,
        "scientific_scope": "post-R10B-failure root-cause diagnostic only",
        "manifest_sha256": sha256(args.manifest),
        "groups": len(records), "errors": errors,
        "categories": {
            f"t{boundary}|{trace}|{outcome}": count
            for (boundary, trace, outcome), count in sorted(categories.items())
        },
        "remains_locked": ["risk_model", "selector", "world_model", "validation", "test"],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))
    return 2 if status == "FAIL_CONTRACT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
