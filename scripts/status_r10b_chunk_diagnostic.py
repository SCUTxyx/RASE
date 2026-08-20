#!/usr/bin/env python3
"""Show exact, resumable progress for the 18-group × K3 R10-B diagnostic."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def metadata_path(root: Path, row: dict, replica: int) -> Path:
    stem = f"{row['state_key']}__seed{row['seed_index']}"
    if replica:
        stem += f"__rep{replica}"
    return (
        root / f"suite_{row['suite'].lower()}" / row["policy_id"]
        / f"seed_{row['seed_index']}" / f"rep{replica}" / f"{stem}.json"
    )


def validate_record(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False, "invalid_json"
    rows = {int(row.get("elapsed_source_steps", -1)): row for row in payload.get("rows", [])}
    for boundary in (8, 16):
        trace = rows.get(boundary, {}).get("persistent_chunk_query_records")
        if not isinstance(trace, list) or not trace:
            return False, f"missing_trace_t{boundary}"
    return True, "complete"


def snapshot(manifest_path: Path, root: Path, audit: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "frozen_diagnostic":
        raise ValueError("status monitor requires a frozen_diagnostic manifest")
    expected = int(manifest.get("expected_trajectories", len(manifest["records"]) * 3))
    reasons: Counter[str] = Counter()
    cells: Counter[str] = Counter()
    pending: list[str] = []
    complete = 0
    for row in manifest["records"]:
        for replica in range(3):
            path = metadata_path(root, row, replica)
            valid, reason = validate_record(path)
            reasons[reason] += 1
            key = f"{row['suite']}|{row['policy_id']}"
            cells[key] += int(valid)
            complete += int(valid)
            if not valid:
                pending.append(f"{row['group_id']}|rep{replica}|{reason}")
    return {
        "schema_version": "rase-r10b-chunk-diagnostic-progress/v1",
        "status": "AUDITED" if audit.is_file() else (
            "COLLECTED_NOT_AUDITED" if complete == expected and (root / "COMPLETE").is_file()
            else "IN_PROGRESS"
        ),
        "complete": complete,
        "expected": expected,
        "percent": round(100.0 * complete / expected, 1) if expected else 100.0,
        "complete_marker": (root / "COMPLETE").is_file(),
        "audit_exists": audit.is_file(),
        "reason_counts": dict(sorted(reasons.items())),
        "complete_by_suite_policy": dict(sorted(cells.items())),
        "pending": pending,
    }


def render(result: dict) -> str:
    width = 30
    filled = round(width * result["complete"] / result["expected"])
    bar = "#" * filled + "-" * (width - filled)
    rows = [
        f"R10-B chunk diagnostic [{bar}] {result['complete']}/{result['expected']} ({result['percent']:.1f}%)",
        f"status={result['status']} COMPLETE={result['complete_marker']} audit={result['audit_exists']}",
        "cells: " + ", ".join(f"{key}={value}" for key, value in result["complete_by_suite_policy"].items()),
        "records: " + ", ".join(f"{key}={value}" for key, value in result["reason_counts"].items()),
    ]
    if result["pending"]:
        rows.append("next: " + result["pending"][0])
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("runs/pre_c0_r10/r10b_oft_trace_diagnostic_manifest_v1.json"))
    parser.add_argument("--collect-root", type=Path, default=Path("runs/pre_c0_r10/r10b_chunk_input_diagnostic_collect_v1"))
    parser.add_argument("--audit", type=Path, default=Path("runs/pre_c0_r10/r10b_chunk_input_divergence_audit_v1.json"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least one second")
    while True:
        result = snapshot(args.manifest, args.collect_root, args.audit)
        output = json.dumps(result, indent=2, sort_keys=True) if args.json else render(result)
        if args.watch:
            print("\033[2J\033[H" + output, flush=True)
            time.sleep(args.interval)
        else:
            print(output)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
