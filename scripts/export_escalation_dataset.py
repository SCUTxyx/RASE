#!/usr/bin/env python3
"""Export diagnostic state-level selector rows from a paired policy matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smol-cost", type=float, default=0.02)
    parser.add_argument("--oft-cost", type=float, default=0.1)
    parser.add_argument("--abstain-cost", type=float, default=0.0)
    args = parser.parse_args()

    from rase.collect.state_pool import StatePool
    from rase.selector.lightweight import build_policy_matrix_proxy_rows

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    pool = StatePool(args.pool)
    keys = [str(row["state_key"]) for row in matrix.get("per_state") or []]
    metadata = {
        key: pool.read_state(key, load_observations=False).metadata.to_dict()
        for key in keys
    }
    rows = build_policy_matrix_proxy_rows(
        matrix,
        metadata_by_state=metadata,
        smol_cost=args.smol_cost,
        oft_cost=args.oft_cost,
        abstain_cost=args.abstain_cost,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_rows": len(rows),
                "output": str(args.output),
                "trainable": False,
                "reason": "portfolio outcomes are exported as explicit proxies",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
