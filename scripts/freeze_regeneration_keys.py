#!/usr/bin/env python3
"""Freeze an outcome-independent metadata slice for regeneration screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _checksum(keys: list[str]) -> str:
    payload = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def freeze(
    source: dict[str, Any],
    *,
    suites: set[str],
    dimensions: set[str],
    steps: set[int],
) -> dict[str, Any]:
    records = [dict(row) for row in source.get("records") or []]
    source_keys = [str(value) for value in source.get("state_keys") or []]
    if not records or not source_keys:
        raise ValueError("source must contain non-empty records and state_keys")
    by_key = {str(row["state_key"]): row for row in records}
    if len(by_key) != len(records) or set(by_key) != set(source_keys):
        raise ValueError("source records and state_keys must be unique and identical")

    selected = [
        by_key[key]
        for key in source_keys
        if str(by_key[key].get("suite")) in suites
        and str(by_key[key].get("perturbation_dimension")) in dimensions
        and int(by_key[key].get("step", -1)) in steps
    ]
    if not selected:
        raise ValueError("metadata filters selected no states")
    keys = [str(row["state_key"]) for row in selected]
    task_counts: dict[str, int] = {}
    for row in selected:
        task = str(row["task_id"])
        task_counts[task] = task_counts.get(task, 0) + 1
    return {
        "artifact_version": "rase-regeneration-state-keys/v1",
        "selection_uses_outcomes": False,
        "n_states": len(keys),
        "n_tasks": len(task_counts),
        "pool": source.get("pool"),
        "state_keys": keys,
        "state_keys_sha256": _checksum(keys),
        "records": selected,
        "selection": {
            "source_artifact_version": source.get("artifact_version"),
            "source_state_keys_sha256": source.get("state_keys_sha256"),
            "suites": sorted(suites),
            "perturbation_dimensions": sorted(dimensions),
            "steps": sorted(steps),
            "order": "source_artifact_order",
            "task_counts": task_counts,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", action="append", required=True)
    parser.add_argument("--dimension", action="append", required=True)
    parser.add_argument("--step", action="append", required=True, type=int)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    result = freeze(
        source,
        suites=set(args.suite),
        dimensions=set(args.dimension),
        steps=set(args.step),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "n_states": result["n_states"],
        "n_tasks": result["n_tasks"],
        "state_keys_sha256": result["state_keys_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
