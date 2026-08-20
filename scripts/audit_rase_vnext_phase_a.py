#!/usr/bin/env python3
"""Run the strict read-only Phase-A audit on a frozen confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.phase_a_audit import audit_phase_a, sha256_file


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=202708)
    parser.add_argument(
        "--tie-margin", type=float, default=0.0,
        help="Pre-registered practical-equivalence margin in utility units.",
    )
    parser.add_argument("--abort-operator", default="abort.safe")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    protocol = json.loads(args.protocol.read_text())
    rows = _read_jsonl(args.branches)
    result = audit_phase_a(
        rows,
        manifest=manifest,
        repeats=int(protocol["collection"]["confirmation_repeats"]),
        weights=protocol["utility"],
        gate=protocol["gates"]["opportunity"],
        abort_operator=args.abort_operator,
        tie_margin=args.tie_margin,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["input_files"] = {
        "branches": str(args.branches),
        "branches_sha256": sha256_file(args.branches),
        "manifest": str(args.manifest),
        "manifest_file_sha256": sha256_file(args.manifest),
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "A_PASS":
        return 0
    if result["status"] == "INTEGRITY_FAIL":
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
