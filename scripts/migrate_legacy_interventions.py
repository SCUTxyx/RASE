#!/usr/bin/env python3
"""Migrate legacy three-arm selector rows into explicit RASE-UI records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--direct-smol-family",
        choices=("replan", "continue"),
        default="replan",
        help=(
            "Default is replan because legacy restore calls policy.reset(). "
            "Use continue only with a separately proven active-suffix protocol."
        ),
    )
    parser.add_argument("--allow-proxy", action="store_true")
    args = parser.parse_args()

    from rase.interventions.dataset import migrate_legacy_escalation_rows, registry_payload
    from rase.interventions.schema import OperatorFamily

    family = (
        OperatorFamily.REPLAN
        if args.direct_smol_family == "replan"
        else OperatorFamily.CONTINUE
    )
    specs, snapshots, outcomes = migrate_legacy_escalation_rows(
        _read_jsonl(args.input.resolve()),
        direct_smol_family=family,
        allow_proxy=args.allow_proxy,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "operators.json").write_text(
        json.dumps(registry_payload(specs), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output / "snapshots.jsonl", [row.to_dict() for row in snapshots])
    _write_jsonl(output / "outcomes.jsonl", [row.to_dict() for row in outcomes])
    manifest = {
        "schema_version": "rase-intervention-migration-manifest/v1",
        "source": str(args.input.resolve()),
        "direct_smol_family": family.value,
        "allow_proxy": bool(args.allow_proxy),
        "n_snapshots": len(snapshots),
        "n_outcomes": len(outcomes),
        "warning": (
            "Legacy direct Smol is mapped to REPLAN because policy.reset() is called."
            if family is OperatorFamily.REPLAN
            else "CONTINUE alias is unverified unless active action suffix provenance exists."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
