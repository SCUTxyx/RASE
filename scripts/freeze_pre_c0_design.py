#!/usr/bin/env python3
"""Freeze the outcome-independent PRE-C0 pilot design from PRE-A3 train tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rase.collect.pre_c0 import build_pre_c0_design


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-a3-design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2_026_080_405)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen design: {args.output}")
    source = json.loads(args.pre_a3_design.read_text(encoding="utf-8"))
    design = build_pre_c0_design(source, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "design_sha256": design["design_sha256"],
                "n_episodes": design["n_episodes"],
                "suite_counts": design["suite_counts"],
                "cell_counts": design["cell_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
