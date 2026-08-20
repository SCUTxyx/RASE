#!/usr/bin/env python3
"""Amend PRE-A3 design v1 → v1.1 (clean IDs 1-10) before first collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rase.collect.pre_a3_schedule import amend_design_v1_to_v1_1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("runs/rase_pre_a3_design120_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/rase_pre_a3_design120_v1.1.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen design: {args.output}")
    design = json.loads(args.input.read_text(encoding="utf-8"))
    amended = amend_design_v1_to_v1_1(design)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(amended, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "design_sha256": amended["design_sha256"],
                "parent_design_sha256": amended.get("parent_design_sha256"),
                "clean_ids_rewritten": amended["design_amendment"]["clean_ids_rewritten"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
