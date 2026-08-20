#!/usr/bin/env python3
"""Freeze the successful-reference subset for development prefix-length search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    audit = json.loads(args.audit.read_text())
    successes = {str(row["state_key"]) for row in audit.get("per_root") or [] if row.get("success")}
    records = [dict(row) for row in protocol.get("records") or [] if str(row["state_key"]) in successes]
    if not records or len(records) != len(successes):
        raise ValueError("protocol/audit successful-root join is incomplete")
    payload = {
        "schema_version": "rase-e3v-reference-success-subset/v1",
        "status": "frozen",
        "scientific_scope": "outcome_selected_development_prefix_horizon_search_only",
        "selection_uses_outcomes": True,
        "selection_rule": "live exact-root OFT recovery success",
        "parent_protocol_sha256": protocol.get("protocol_sha256"),
        "parent_audit": str(args.audit.resolve()),
        "pool": protocol["pool"],
        "n_states": len(records),
        "n_tasks": len({row["task_id"] for row in records}),
        "state_keys": [row["state_key"] for row in records],
        "records": records,
    }
    payload["protocol_sha256"] = canonical_sha(payload)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_states": len(records), "n_tasks": payload["n_tasks"], "protocol_sha256": payload["protocol_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
