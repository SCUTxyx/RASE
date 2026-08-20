#!/usr/bin/env python3
"""Classify the first per-chunk OFT input/output divergence in R10-B repeats."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


INPUT_FIELDS = (
    "agentview_sha256", "agentview_shape", "wrist_sha256", "wrist_shape",
    "proprio_sha256", "proprio_shape",
)
ACTION_FIELDS = ("action_chunk_sha256", "action_chunk_shape")
CATEGORY_DECISIONS = {
    "A_INITIAL_INPUT_DIVERGENCE": "AUDIT_RESTORE_AND_OBSERVABLE_STATE",
    "B_CLOSED_LOOP_INPUT_DIVERGENCE": "FREEZE_CLOSED_LOOP_AMPLIFICATION_EVIDENCE",
    "C_MATCHED_INPUT_OUTPUT_DIVERGENCE": "AUDIT_OFT_INFERENCE_VARIANCE",
    "D_NO_REPRODUCED_CHUNK_DIVERGENCE": "STOP_OR_REPEAT_WITHOUT_TRAINING",
}


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


def _first_field_difference(
    left: list[dict], right: list[dict], fields: tuple[str, ...]
) -> dict | None:
    """Return the first aligned query where one of ``fields`` differs."""
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        changed = [field for field in fields if a.get(field) != b.get(field)]
        if changed:
            return {
                "query_index": index,
                "action_offset": a.get("action_offset", b.get("action_offset")),
                "changed_fields": changed,
            }
    return None


def compare_replica_pair(left: list[dict], right: list[dict]) -> dict:
    """Compare one replica pair without collapsing independent root causes."""
    first_input = _first_field_difference(left, right, INPUT_FIELDS)
    matched_input_output = None
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        inputs_match = all(a.get(field) == b.get(field) for field in INPUT_FIELDS)
        changed = [field for field in ACTION_FIELDS if a.get(field) != b.get(field)]
        if inputs_match and changed:
            matched_input_output = {
                "query_index": index,
                "action_offset": a.get("action_offset", b.get("action_offset")),
                "changed_fields": changed,
            }
            break
    categories: list[str] = []
    if first_input is not None and first_input["query_index"] == 0:
        categories.append("A_INITIAL_INPUT_DIVERGENCE")
    if first_input is not None and first_input["query_index"] > 0:
        categories.append("B_CLOSED_LOOP_INPUT_DIVERGENCE")
    if len(left) != len(right):
        categories.append("B_CLOSED_LOOP_INPUT_DIVERGENCE")
    if matched_input_output is not None:
        categories.append("C_MATCHED_INPUT_OUTPUT_DIVERGENCE")
    if not categories:
        categories.append("D_NO_REPRODUCED_CHUNK_DIVERGENCE")
    return {
        "query_counts": [len(left), len(right)],
        "first_input_difference": first_input,
        "first_matched_input_output_difference": matched_input_output,
        "query_count_difference": len(left) != len(right),
        "categories": sorted(set(categories)),
    }


def classify_cell(traces: list[list[dict]]) -> dict:
    if len(traces) != 3:
        raise ValueError(f"R10-B chunk audit requires exactly three replicas, got {len(traces)}")
    pairwise = {}
    categories: set[str] = set()
    for left_index, right_index in combinations(range(3), 2):
        comparison = compare_replica_pair(traces[left_index], traces[right_index])
        pairwise[f"rep{left_index}_vs_rep{right_index}"] = comparison
        categories.update(comparison["categories"])
    if len(categories) > 1:
        categories.discard("D_NO_REPRODUCED_CHUNK_DIVERGENCE")
    return {
        "query_counts": [len(trace) for trace in traces],
        "first_query_input_exact": len({
            json.dumps({field: trace[0].get(field) for field in INPUT_FIELDS}, sort_keys=True)
            for trace in traces
        }) == 1,
        "categories": sorted(categories),
        "pairwise": pairwise,
    }


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
        boundaries: dict[str, dict] = {}
        for boundary in (8, 16):
            try:
                traces = [row[boundary]["persistent_chunk_query_records"] for row in rows]
            except KeyError:
                errors.append({"group_id": item["group_id"], "boundary": boundary, "reason": "missing_chunk_trace"})
                continue
            if any(not trace for trace in traces):
                errors.append({"group_id": item["group_id"], "boundary": boundary, "reason": "empty_chunk_trace"})
                continue
            boundaries[str(boundary)] = classify_cell(traces)
        records.append({
            "group_id": item["group_id"],
            "task_id": item["task_id"],
            "suite": item["suite"],
            "policy_id": item["policy_id"],
            "diagnostic_role": item["diagnostic_role"],
            "boundaries": boundaries,
        })

    category_counts = Counter(
        category
        for record in records
        for values in record["boundaries"].values()
        for category in values["categories"]
    )
    active_categories = sorted(category for category in category_counts if not category.startswith("D_"))
    if errors or len(records) != manifest["expected_groups"]:
        status, decisions = "FAIL_CONTRACT", ["STOP_ROOT_CAUSE_DIAGNOSTIC"]
    elif len(active_categories) > 1:
        status = "MIXED_ROOT_CAUSES"
        decisions = [CATEGORY_DECISIONS[category] for category in active_categories]
    elif active_categories:
        status = active_categories[0]
        decisions = [CATEGORY_DECISIONS[active_categories[0]]]
    else:
        status = "D_NO_REPRODUCED_CHUNK_DIVERGENCE"
        decisions = [CATEGORY_DECISIONS[status]]
    result = {
        "schema_version": "rase-r10b-chunk-input-divergence-audit/v2",
        "status": status,
        "decision": decisions[0] if len(decisions) == 1 else "FOLLOW_ROOT_CAUSE_MATRIX",
        "decisions": decisions,
        "scientific_scope": "post-R10B-failure root-cause diagnostic only",
        "manifest_sha256": sha256(args.manifest), "groups": len(records), "errors": errors,
        "category_counts": dict(sorted(category_counts.items())),
        "matrix_unit": "group_id x boundary",
        "records": records,
        "remains_locked": ["risk_model", "selector", "world_model", "validation", "test"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))
    return 2 if status == "FAIL_CONTRACT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
