#!/usr/bin/env python3
"""Merge per-suite live duration summaries into one PRE-A3 duration artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lengths: list[int] | None = None
    per_state: list[dict[str, Any]] = []
    for path in args.input:
        payload = json.loads(path.read_text(encoding="utf-8"))
        current = [int(value) for value in payload["prefix_lengths"]]
        if lengths is None:
            lengths = current
        elif lengths != current:
            raise ValueError(f"prefix_lengths mismatch in {path}")
        per_state.extend(payload.get("per_state") or [])

    keys = [row["state_key"] for row in per_state]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate state_key after suite merge")

    summary = {
        "schema_version": "rase-live-oft-duration-to-smol/v1",
        "status": "complete",
        "n_states": len(per_state),
        "prefix_lengths": lengths,
        "execution_mode": "live_closed_loop",
        "per_state": per_state,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_states": len(per_state), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
