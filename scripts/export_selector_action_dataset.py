#!/usr/bin/env python3
"""Export direct-Smol/direct-OFT/abstain selector rows with deployable features."""

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
    parser.add_argument("--smol-direct-summary", type=Path, required=True)
    parser.add_argument("--oft-direct-summary", type=Path, action="append", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smol-cost", type=float, default=0.02)
    parser.add_argument("--oft-cost", type=float, default=0.10)
    parser.add_argument("--abstain-cost", type=float, default=0.0)
    args = parser.parse_args()

    from rase.collect.state_pool import StatePool
    from rase.selector.lightweight import build_direct_policy_rows

    smol = json.loads(args.smol_direct_summary.read_text(encoding="utf-8"))
    oft = [json.loads(path.read_text(encoding="utf-8")) for path in args.oft_direct_summary]
    feature_payload = json.loads(args.features.read_text(encoding="utf-8"))
    features = dict(feature_payload.get("features_by_state") or {})
    keys = [str(row["state_key"]) for row in smol.get("per_state") or []]
    pool = StatePool(args.pool)
    metadata = {
        key: pool.read_state(key, load_observations=False).metadata.to_dict()
        for key in keys
    }
    rows = build_direct_policy_rows(
        smol,
        oft,
        metadata_by_state=metadata,
        features_by_state=features,
        smol_cost=args.smol_cost,
        oft_cost=args.oft_cost,
        abstain_cost=args.abstain_cost,
        cohort=args.cohort,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"n_rows": len(rows), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
