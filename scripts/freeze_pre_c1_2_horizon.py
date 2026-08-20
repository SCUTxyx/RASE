#!/usr/bin/env python3
"""Freeze selected_horizon into PRE-C1.2 protocol from a horizon-sweep JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rase.adapt.pre_c1_2 import freeze_selected_horizon, select_recovery_horizon


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_protocol_lock.yaml"),
    )
    parser.add_argument("--sweep-json", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=None, help="Override selection")
    args = parser.parse_args()

    payload = json.loads(args.sweep_json.read_text(encoding="utf-8"))
    if not payload.get("sweep_valid", True):
        raise SystemExit("sweep_valid=false; refusing to freeze")
    if args.horizon is not None:
        h = int(args.horizon)
    else:
        selection = payload.get("selection") or select_recovery_horizon(payload.get("rows") or [])
        h = int(selection["selected_horizon"])
    frozen = freeze_selected_horizon(args.protocol_lock, selected_horizon=h)
    print(json.dumps(frozen, sort_keys=True, default=str)[:500])
    print(
        f"PRE_C1_2_HORIZON_FROZEN H={frozen['selected_horizon']} "
        f"sha256={frozen['protocol_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
