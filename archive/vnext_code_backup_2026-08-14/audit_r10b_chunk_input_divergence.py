#!/usr/bin/env python3
"""Classify the first per-chunk OFT input/output divergence in R10-B repeats."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


INPUT_FIELDS = (
    "agentview_sha256", "agentview_shape", "wrist_sha256", "wrist_shape",
    "proprio_sha256", "proprio_shape",
)
ACTION_FIELDS = ("action_chunk_sha256", "action_chunk_shape")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata_path(root: Path, row: dict, replica: int) -> Path:
    stem = f"{row['state_key']}__seed{row['seed_index']}"
    if replica:
        stem += f"__rep{replica}"
    return root / f"suite_{row['suite'].lower()}" / row["policy_id"] / f"seed_{row['seed_index']}" / f"rep{replica}" / f"{stem}.json"


def first_difference(left: list[dict], right: list[dict]) -> dict | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            changed = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
            return {
                "query_index": index,
                "action_offset": a.get("action_offset", b.get("action_offset")),
                "input_diff": any(field in changed for field in INPUT_FIELDS),
                "action_diff": any(field in changed for field in ACTION_FIELDS),
                "changed_fields": changed,
            }
    if len(left) != len(right):
        return {
            "query_index": min(len(left), len(right)), "action_offset": None,
            "input_diff": False, "action_diff": False, "changed_fields": ["query_count"],
        }
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "frozen_diagnostic":
        raise ValueError("chunk-input audit requires frozen diagnostic manifest")

    errors, records = [], []
    for item in manifest["records"]:
        paths = [metadata_path(args.collect_root, item, replica) for replica in range(3)]
        if not all(path.is_file() for path in paths):
            errors.append({"group_id": item["group_id"], "reason": "missing_replica"})
            continue
        rows = [{int(row["elapsed_source_steps"]): row for row in json.loads(path.read_text())["rows"]} for path in paths]
        boundaries = {}
        for boundary in (8, 16):
            try:
                traces = [row[boundary]["persistent_chunk_query_records"] for row in rows]
            except KeyError:
                errors.append({"group_id": item["group_id"], "boundary": boundary, "reason": "missing_chunk_trace"})
                continue
            if any(not trace for trace in traces):
                errors.append({"group_id": item["group_id"], "boundary": boundary, "reason": "empty_chunk_trace"})
                continue
            differences = [first_difference(traces[0], trace) for trace in traces[1:]]
            first = min((value for value in differences if value is not None), key=lambda value: value["query_index"], default=None)
            boundaries[str(boundary)] = {
                "query_counts": [len(trace) for trace in traces],
                "first_query_input_exact": len({json.dumps(trace[0], sort_keys=True) for trace in traces}) == 1,
                "first_difference": first,
            }
        records.append({"group_id": item["group_id"], "boundaries": boundaries})

    first_differences = [
        values["first_difference"] for record in records for values in record["boundaries"].values()
        if values["first_difference"] is not None
    ]
    if errors or len(records) != manifest["expected_groups"]:
        status, decision = "FAIL_CONTRACT", "STOP_ROOT_CAUSE_DIAGNOSTIC"
    elif any(value["input_diff"] and value["query_index"] == 0 for value in first_differences):
        status, decision = "INITIAL_OBSERVATION_DIVERGENCE", "AUDIT_RESTORE_AND_OBSERVABLE_STATE"
    elif any(value["input_diff"] for value in first_differences):
        status, decision = "CLOSED_LOOP_OBSERVATION_DIVERGENCE", "STOP_MODEL_ESCALATION_AND_RECORD"
    elif any(value["action_diff"] for value in first_differences):
        status, decision = "OFT_OUTPUT_DIVERGENCE_WITH_MATCHED_INPUT", "AUDIT_OFT_INFERENCE_VARIANCE"
    else:
        status, decision = "NO_REPRODUCED_CHUNK_DIVERGENCE", "STOP_OR_REPEAT_WITHOUT_TRAINING"
    result = {
        "schema_version": "rase-r10b-chunk-input-divergence-audit/v1",
        "status": status, "decision": decision,
        "scientific_scope": "post-R10B-failure root-cause diagnostic only",
        "manifest_sha256": sha256(args.manifest), "groups": len(records), "errors": errors,
        "records": records,
        "remains_locked": ["risk_model", "selector", "world_model", "validation", "test"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))
    return 2 if status == "FAIL_CONTRACT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
