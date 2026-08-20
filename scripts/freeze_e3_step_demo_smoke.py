#!/usr/bin/env python3
"""Freeze a tiny outcome-selected development smoke for step-demo collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    keys = json.loads(args.keys.read_text())
    source = json.loads(args.source_summary.read_text())
    audit = json.loads(args.audit.read_text())
    source_rows = {str(row["state_key"]): row for row in source.get("per_pair") or []}
    correction = sorted(
        (row for row in audit.get("per_root") or [] if row.get("success")),
        key=lambda row: (int(row.get("continuation_steps", 10**9)), str(row["state_key"])),
    )[:2]
    identity = sorted(
        (row for row in source_rows.values() if row.get("continue_smol_active_chunk")),
        key=lambda row: str(row["state_key"]),
    )[:1]
    selected_keys = [str(row["state_key"]) for row in correction + identity]
    if len(selected_keys) != 3 or len(set(selected_keys)) != 3:
        raise ValueError("need two correction and one identity smoke roots")
    records_by_key = {str(row["state_key"]): row for row in keys.get("records") or []}
    payload = {
        "schema_version": "rase-e3-step-demo-smoke/v1",
        "status": "frozen",
        "scientific_scope": "outcome_selected_development_collection_smoke_only",
        "selection_uses_outcomes": True,
        "pool": keys["pool"],
        "state_keys": selected_keys,
        "records": [records_by_key.get(key, {"state_key": key}) for key in selected_keys],
    }
    payload["state_keys_sha256"] = hashlib.sha256(
        json.dumps(selected_keys, separators=(",", ":")).encode()
    ).hexdigest()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"state_keys": selected_keys}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
