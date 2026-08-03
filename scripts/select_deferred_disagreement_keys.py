#!/usr/bin/env python3
"""Freeze the immediate/deferred disagreement subset for replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def select(keys: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis.get("schema_version") != "rase-deferred-switch-analysis/v1":
        raise ValueError("unexpected deferred-switch analysis schema")
    disagreements = [
        str(row["state_key"])
        for row in analysis.get("per_state") or []
        if row.get("classification") in {"direct_only", "deferred_only"}
    ]
    if not disagreements or len(disagreements) != len(set(disagreements)):
        raise ValueError("analysis must contain unique immediate/deferred disagreements")
    records_by_key = {
        str(record["state_key"]): record for record in keys.get("records") or []
    }
    missing = set(disagreements) - set(records_by_key)
    if missing:
        raise ValueError(f"disagreement keys missing from frozen cohort: {sorted(missing)}")
    digest = hashlib.sha256(
        json.dumps(
            disagreements, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {
        "artifact_version": "rase-deferred-disagreement-keys/v1",
        "source_artifact_version": keys.get("artifact_version"),
        "selection": {
            "classifications": ["direct_only", "deferred_only"],
            "purpose": "deterministic outcome replay; not independent confirmation",
        },
        "n_states": len(disagreements),
        "state_keys": disagreements,
        "state_keys_sha256": digest,
        "records": [records_by_key[key] for key in disagreements],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(_read(args.state_keys_json), _read(args.analysis))
    _write(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()), "n_states": result["n_states"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
