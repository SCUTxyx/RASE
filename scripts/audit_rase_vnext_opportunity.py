#!/usr/bin/env python3
"""Audit a frozen K5 confirmation branch JSONL without fitting a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.opportunity import audit_opportunity
from scripts.validate_rase_vnext_protocol import validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=202708)
    parser.add_argument("--exclude-operator", action="append", default=[])
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    errors = validate(protocol, allow_draft=False)
    if errors:
        raise SystemExit("protocol is not frozen and valid: " + "; ".join(errors))
    rows = [json.loads(line) for line in args.branches.read_text().splitlines() if line.strip()]
    excluded = set(args.exclude_operator)
    rows = [row for row in rows if str(row.get("operator_id")) not in excluded]
    result = audit_opportunity(
        rows, repeats=int(protocol["collection"]["confirmation_repeats"]),
        weights=protocol["utility"], gate=protocol["gates"]["opportunity"],
        bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
    )
    result["excluded_operators"] = sorted(excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
