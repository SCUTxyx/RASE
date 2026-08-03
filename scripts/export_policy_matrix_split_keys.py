#!/usr/bin/env python3
"""Freeze keys selected by a W6 policy-matrix state-pair label."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def select_keys(matrix: dict, label: str) -> list[str]:
    rows = matrix.get("per_state") or []
    available = sorted({str(row.get("state_pair_label")) for row in rows})
    keys = sorted(
        str(row["state_key"])
        for row in rows
        if str(row.get("state_pair_label")) == label
    )
    if not keys:
        raise ValueError(
            f"no states with label {label!r}; available: {', '.join(available) or '(none)'}"
        )
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.matrix.resolve()
    matrix = json.loads(source.read_text(encoding="utf-8"))
    if matrix.get("status") != "complete":
        raise SystemExit("policy matrix is not complete")
    try:
        keys = select_keys(matrix, args.label)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    encoded = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    payload = {
        "artifact_version": "rase-state-keys/v2",
        "selection": f"policy_matrix_state_pair:{args.label}",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "label": args.label,
        "n_states": len(keys),
        "state_keys": keys,
        "state_keys_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in (
        "label", "n_states", "state_keys", "state_keys_sha256"
    )}, indent=2), flush=True)
    print(f"WROTE {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
