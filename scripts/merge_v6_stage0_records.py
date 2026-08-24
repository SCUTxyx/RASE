#!/usr/bin/env python3
"""Merge independently collected, outcome-blind V6 Stage-0 lanes safely."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object record in {path}")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    root_ids: set[str] = set()
    source_files: list[str] = []
    for path in args.input:
        records = path / "stage0_records.jsonl" if path.is_dir() else path
        for row in read_rows(records.resolve()):
            root_id = str(row.get("root_id") or "")
            if not root_id:
                raise ValueError(f"missing root_id in {records}")
            # Six records constitute one root; repeats must therefore be
            # detected after grouping, not rejected row-by-row.
            rows.append(row)
        source_files.append(str(records.resolve()))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["root_id"]), []).append(row)
    for root_id, group in grouped.items():
        if root_id in root_ids:
            raise ValueError(f"duplicate root id across lanes: {root_id}")
        root_ids.add(root_id)
        if len(group) != 6:
            raise ValueError(f"root {root_id} has {len(group)} branch rows; expected 6")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for root_id in sorted(grouped):
            for row in sorted(grouped[root_id], key=lambda item: (str(item["arm"]), -1 if item.get("arm_index") is None else int(item["arm_index"]))):
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, output)
    manifest = {
        "schema_version": "rase-v6-stage0-merged-records/v1",
        "inputs": source_files,
        "n_roots": len(grouped),
        "n_branch_rows": len(rows),
        "selection_outcomes_used": False,
    }
    manifest_path = output.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
