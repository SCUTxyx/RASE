#!/usr/bin/env python3
"""Probe a running RASE oracle server (health + model-info + tiny predict)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--expect-suite", default=None)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    from rase.oracle.client import OracleClient

    with OracleClient(args.endpoint, timeout_ms=args.timeout_ms) as client:
        health = client.health()
        info = client.model_info()
        print("HEALTH", json.dumps(health, sort_keys=True))
        print("MODEL", json.dumps(info, sort_keys=True))
        if args.expect_suite and info.get("suite") != args.expect_suite:
            print(
                f"SUITE_MISMATCH expected={args.expect_suite} got={info.get('suite')}",
                file=sys.stderr,
            )
            return 2
        if not args.skip_predict:
            out = client.predict(
                {
                    "agentview": np.zeros((1, 256, 256, 3), dtype=np.uint8),
                    "wrist": np.zeros((1, 256, 256, 3), dtype=np.uint8),
                    "proprio": np.zeros((1, 8), dtype=np.float32),
                },
                payload={
                    "instructions": ["pick up the object"],
                    "return_mode": "chunk",
                    "proprio_format": "policy_state",
                    "images_already_flipped": True,
                },
            )
            print("PREDICT", {key: list(value.shape) for key, value in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
