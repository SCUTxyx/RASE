#!/usr/bin/env python3
"""Build an outcome-free vNext root catalog from a frozen StatePool manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rase.vnext.discovery import validate_root_catalog


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-roots-per-task", type=int, default=2)
    args = parser.parse_args()

    pool = args.pool.resolve()
    manifest_path = pool / "manifest.json"
    manifest = read_object(manifest_path)
    states = manifest.get("states")
    if not isinstance(states, dict) or not states:
        raise SystemExit("StatePool manifest must contain a non-empty states object")

    records: list[dict[str, Any]] = []
    task_roots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state_key, source_row in sorted(states.items()):
        if not isinstance(source_row, dict):
            raise SystemExit(f"invalid StatePool row: {state_key}")
        state_dir = pool / str(source_row["path"])
        metadata = read_object(state_dir / "meta.json")
        if int(source_row.get("step", metadata.get("step", -1))) != 0:
            continue
        task_id = str(metadata["task_id"])
        episode_id = str(metadata["episode_id"])
        init_state_id = int(metadata["init_state_id"])
        row = {
            "root_id": f"root.{state_key}",
            "state_key": str(state_key),
            "task_id": task_id,
            "suite": str(metadata["suite"]),
            "init_state_id": init_state_id,
            "environment_seed": int(metadata["seed"]),
            "restore_state_ref": f"state_pool:{pool}#{state_key}",
            "episode_id": episode_id,
            "state_bundle_sha256": str(source_row["bundle_sha256"]),
        }
        records.append(row)
        task_roots[task_id].append(row)

    validate_root_catalog(records)
    errors: list[str] = []
    for task_id, rows in sorted(task_roots.items()):
        episodes = {str(row["episode_id"]) for row in rows}
        init_states = {int(row["init_state_id"]) for row in rows}
        if len(episodes) < args.minimum_roots_per_task:
            errors.append(f"{task_id}: only {len(episodes)} independent episodes")
        if len(init_states) < args.minimum_roots_per_task:
            errors.append(f"{task_id}: only {len(init_states)} distinct init states")
    if errors:
        raise SystemExit("root independence contract failed: " + "; ".join(errors))

    payload = {
        "schema_version": "rase-vnext-root-catalog/v1",
        "status": "frozen_outcome_independent",
        "selection_eligible_fields": [
            "root_id", "state_key", "task_id", "suite", "init_state_id",
            "environment_seed", "restore_state_ref", "episode_id",
            "state_bundle_sha256",
        ],
        "forbidden_source_fields_removed": [
            "outcome", "success", "reward", "return", "failure", "label",
        ],
        "pool": str(pool),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "n_records": len(records),
        "n_tasks": len(task_roots),
        "suite_counts": {
            suite: sum(str(row["suite"]) == suite for row in records)
            for suite in sorted({str(row["suite"]) for row in records})
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
