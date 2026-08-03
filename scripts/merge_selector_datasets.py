#!/usr/bin/env python3
"""Merge selector JSONL cohorts while rejecting duplicate state keys."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    rows = {}
    sources = []
    for path in args.dataset:
        raw = path.read_bytes()
        sources.append({
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["state_key"])
            if key in rows:
                raise SystemExit(f"duplicate state key {key} at {path}:{number}")
            rows[key] = row
    ordered = [rows[key] for key in sorted(rows)]
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": "rase-selector-dataset-merge/v1",
        "n_rows": len(ordered),
        "cohort_counts": {
            cohort: sum(str(row.get("cohort")) == cohort for row in ordered)
            for cohort in sorted({str(row.get("cohort")) for row in ordered})
        },
        "sources": sources,
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
