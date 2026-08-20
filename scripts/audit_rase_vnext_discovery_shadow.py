#!/usr/bin/env python3
"""Run the non-abort scientific shadow audit after fixed-K discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.shadow import audit_discovery_shadow
from scripts.validate_rase_vnext_protocol import validate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=202708)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    errors = validate(protocol, allow_draft=False)
    if errors:
        raise SystemExit("protocol is not frozen and valid: " + "; ".join(errors))
    rows = [json.loads(line) for line in args.branches.read_text().splitlines() if line.strip()]
    result = audit_discovery_shadow(
        rows,
        repeats=int(protocol["collection"]["discovery_repeats"]),
        weights=protocol["utility"],
        opportunity_gate=protocol["gates"]["opportunity"],
        minimum_nondegenerate_fraction=float(
            protocol["gates"]["feasibility"]["minimum_nondegenerate_outcome_fraction"]
        ),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["branches_sha256"] = sha256(args.branches)
    result["protocol_sha256"] = sha256(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "GO_CONFIRMATION" else 3


if __name__ == "__main__":
    raise SystemExit(main())

