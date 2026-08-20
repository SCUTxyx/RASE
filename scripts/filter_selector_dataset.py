#!/usr/bin/env python3
"""Filter selector JSONL rows by suite and cohort without modifying rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def filter_dataset(
    dataset: Path,
    output: Path,
    manifest_output: Path,
    *,
    suites: set[str],
    cohort: str,
) -> dict[str, Any]:
    resolved_paths = {dataset.resolve(), output.resolve(), manifest_output.resolve()}
    if len(resolved_paths) != 3:
        raise ValueError("dataset, output, and manifest-output must be distinct paths")
    if not suites:
        raise ValueError("at least one suite filter is required")
    raw = dataset.read_bytes()
    seen: dict[str, int] = {}
    selected_lines: list[bytes] = []
    input_suites: Counter[str] = Counter()
    input_cohorts: Counter[str] = Counter()
    output_suites: Counter[str] = Counter()
    output_cohorts: Counter[str] = Counter()
    n_rows = 0

    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON row at {dataset}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"selector row must be an object at {dataset}:{number}")
        key = str(row.get("state_key") or "")
        if not key:
            raise ValueError(f"missing state_key at {dataset}:{number}")
        if key in seen:
            raise ValueError(
                f"duplicate state_key {key!r} at {dataset}:{number}; first seen at line {seen[key]}"
            )
        seen[key] = number
        n_rows += 1
        suite = str(row.get("suite") or "")
        row_cohort = str(row.get("cohort") or "")
        input_suites[suite] += 1
        input_cohorts[row_cohort] += 1
        if suite in suites and row_cohort == cohort:
            selected_lines.append(line if line.endswith((b"\n", b"\r")) else line + b"\n")
            output_suites[suite] += 1
            output_cohorts[row_cohort] += 1

    output_raw = b"".join(selected_lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(output_raw)
    manifest = {
        "schema_version": "rase-selector-dataset-filter/v1",
        "filters": {"suites": sorted(suites), "cohort": cohort},
        "source": {
            "path": str(dataset.resolve()),
            "sha256": _sha256(raw),
            "n_rows": n_rows,
            "suite_counts": dict(sorted(input_suites.items())),
            "cohort_counts": dict(sorted(input_cohorts.items())),
        },
        "output": {
            "path": str(output.resolve()),
            "sha256": _sha256(output_raw),
            "n_rows": len(selected_lines),
            "suite_counts": dict(sorted(output_suites.items())),
            "cohort_counts": dict(sorted(output_cohorts.items())),
        },
        "rows_preserved_verbatim": True,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--suite", action="append", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = filter_dataset(
            args.dataset,
            args.output,
            args.manifest_output,
            suites={str(value) for value in args.suite},
            cohort=str(args.cohort),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
