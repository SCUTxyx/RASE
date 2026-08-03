#!/usr/bin/env python3
"""Extract low-cost observation features for frozen selector states."""

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
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--state-keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from rase.collect.state_pool import StatePool
    from rase.selector.features import extract_deployable_features, feature_artifact

    payload = json.loads(args.state_keys.read_text(encoding="utf-8"))
    keys = payload if isinstance(payload, list) else payload.get("state_keys") or []
    pool = StatePool(args.pool)
    rows = {}
    for raw_key in keys:
        key = str(raw_key)
        state = pool.read_state(key, load_observations=True)
        rows[key] = extract_deployable_features(
            observations=state.observations,
            proprio=state.proprio,
            t0=state.metadata.step,
        )
    result = feature_artifact(rows, source_pool=str(args.pool.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n_states": len(rows), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
