#!/usr/bin/env python3
"""Recompute or freeze the PRE-A3 method-gate decision artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rase.collect.pre_a3 import decide_method_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-audit", type=Path, required=True)
    parser.add_argument("--val-audit", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hidden = json.loads(args.hidden_audit.read_text(encoding="utf-8"))
    val = (
        json.loads(args.val_audit.read_text(encoding="utf-8"))
        if args.val_audit is not None
        else None
    )
    gate = decide_method_gate(hidden, val_audit=val)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
