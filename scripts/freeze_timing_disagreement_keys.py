#!/usr/bin/env python3
"""Freeze the outcome-selected timing-disagreement cohort for mechanism audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keys_sha256(keys: list[str]) -> str:
    raw = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def freeze_cohort(
    analysis: dict[str, Any],
    *,
    source_path: Path,
    expected_direct_only: int,
    expected_deferred_only: int,
) -> dict[str, Any]:
    if analysis.get("schema_version") != "rase-deferred-switch-analysis/v1":
        raise ValueError("unexpected source-analysis schema")
    if analysis.get("status") != "complete":
        raise ValueError("source analysis is incomplete")
    rows = [
        dict(row)
        for row in analysis.get("per_state") or []
        if row.get("classification") in {"direct_only", "deferred_only"}
    ]
    counts = Counter(str(row["classification"]) for row in rows)
    expected = {
        "direct_only": expected_direct_only,
        "deferred_only": expected_deferred_only,
    }
    if dict(counts) != expected:
        raise ValueError(f"disagreement counts changed: {dict(counts)} != {expected}")
    keys = [str(row["state_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate state keys in disagreement cohort")
    suite_counts = Counter(str(row.get("suite")) for row in rows)
    cells = Counter(
        f"{row.get('dim')}:L{row.get('level')}" for row in rows
    )
    return {
        "schema_version": "rase-timing-disagreement-cohort/v1",
        "status": "frozen",
        "purpose": "exploratory selected-subset mechanism audit; no population claim",
        "selection_outcome_conditioned": True,
        "source_analysis": str(source_path),
        "source_analysis_sha256": _file_sha256(source_path),
        "source_state_keys_sha256": analysis.get("state_keys_sha256"),
        "n_states": len(keys),
        "state_keys": keys,
        "state_keys_sha256": _keys_sha256(keys),
        "classification_counts": dict(sorted(counts.items())),
        "suite_counts": dict(sorted(suite_counts.items())),
        "cell_counts": dict(sorted(cells.items())),
        "expected_source_classification": {
            str(row["state_key"]): str(row["classification"]) for row in rows
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-direct-only", type=int, default=4)
    parser.add_argument("--expected-deferred-only", type=int, default=2)
    args = parser.parse_args()

    source_path = args.analysis.resolve()
    result = freeze_cohort(
        _load(source_path),
        source_path=source_path,
        expected_direct_only=args.expected_direct_only,
        expected_deferred_only=args.expected_deferred_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_states": result["n_states"],
                "classification_counts": result["classification_counts"],
                "state_keys_sha256": result["state_keys_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
