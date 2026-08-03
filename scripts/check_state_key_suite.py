#!/usr/bin/env python3
"""Check whether a frozen state-key artifact contains a requested LIBERO suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()

    cfg = _load(args.config.resolve())
    pool_path = Path(cfg.get("pool") or "pool/ngc_step1_scale200")
    if not pool_path.is_absolute():
        pool_path = (ROOT / pool_path).resolve()
    payload = _load(args.state_keys_json.resolve())
    values = payload if isinstance(payload, list) else payload.get("state_keys") or []
    keys = [str(key) for key in values]
    if not keys:
        raise SystemExit("state-key artifact is empty")

    from rase.collect.libero_env_factory import parse_pool_task_id
    from rase.collect.state_pool import StatePool

    pool = StatePool(pool_path)
    matched = [
        key
        for key in keys
        if parse_pool_task_id(
            pool.read_state(key, load_observations=False).metadata.task_id
        ).suite
        == args.suite
    ]
    print(
        json.dumps(
            {"suite": args.suite, "matched": len(matched), "total": len(keys)},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if matched else 4


if __name__ == "__main__":
    raise SystemExit(main())
