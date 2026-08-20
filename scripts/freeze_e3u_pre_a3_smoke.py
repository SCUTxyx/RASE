#!/usr/bin/env python3
"""Freeze an outcome-independent balanced PRE-A3 train smoke for E3-U."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def checksum(keys: list[str]) -> str:
    return hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--per-suite-dimension", type=int, default=1)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in source.get("records") or []:
        if row.get("split") == args.split:
            cells[(str(row["suite"]), str(row["dimension"]))].append(dict(row))
    selected = []
    for cell in sorted(cells):
        rows = sorted(cells[cell], key=lambda row: str(row["state_key"]))
        selected.extend(rows[: args.per_suite_dimension])
    expected = 4 * 3 * args.per_suite_dimension
    if len(selected) != expected:
        raise ValueError(f"expected {expected} balanced roots, got {len(selected)}")
    keys = [str(row["state_key"]) for row in selected]
    payload = {
        "schema_version": "rase-e3u-pre-a3-smoke/v1",
        "status": "frozen",
        "scientific_scope": "outcome_independent_PRE_A3_train_smoke; never hidden test",
        "selection_uses_outcomes": False,
        "source": str(args.source.resolve()),
        "pool": str(source["pool"]),
        "split": args.split,
        "n_states": len(keys),
        "state_keys": keys,
        "state_keys_sha256": checksum(keys),
        "records": selected,
    }
    payload["protocol_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_states": len(keys), "protocol_sha256": payload["protocol_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
