#!/usr/bin/env python3
"""Audit fixed-K discovery completeness, masks, and raw label density."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.feasibility import audit_discovery_feasibility
from scripts.validate_rase_vnext_protocol import validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    errors = validate(protocol, allow_draft=False)
    if errors:
        raise SystemExit("invalid frozen protocol: " + "; ".join(errors))
    manifest = json.loads(args.manifest.read_text())
    rows = [json.loads(line) for line in args.branches.read_text().splitlines() if line.strip()]
    result = audit_discovery_feasibility(
        rows, manifest=manifest, gate=protocol["gates"]["feasibility"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
