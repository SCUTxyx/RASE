#!/usr/bin/env python3
"""Repeat a pure persistent-OFT replay with no Student/model feature queries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict:
    if path.suffix in {".yaml", ".yml"}:
        import yaml
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def _expand(value, env_name=None):
    if value in {None, ""}:
        return os.environ.get(env_name) if env_name else None
    return str(Path(os.path.expandvars(str(value))).expanduser())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-key", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = _load(args.config.resolve())
    adapter = dict(cfg.get("adapter") or {})
    libero_plus_root = _expand(adapter.get("libero_plus_root"), "LIBERO_PLUS_ROOT")
    pool_path = Path(_expand(cfg.get("pool"), "RASE_POOL_ROOT") or "pool")
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import evaluate_candidate, restore_pool_state
    from rase.collect.oracle_continuation import OracleChunkContinuation
    from rase.collect.state_pool import StatePool
    from rase.oracle.client import OracleClient

    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_path.resolve())
    loaded = pool.read_state(args.state_key, load_observations=False)
    client = OracleClient(args.endpoint, timeout_ms=60_000)
    model_info = client.model_info()
    if model_info.get("suite") not in {None, args.suite}:
        raise ValueError(f"oracle suite mismatch: {model_info.get('suite')} != {args.suite}")

    results = []
    for repeat in range(args.repeats):
        restored = restore_pool_state(
            pool,
            args.state_key,
            libero_plus_root=libero_plus_root,
            observation_height=int(adapter.get("observation_height", 360)),
            observation_width=int(adapter.get("observation_width", 360)),
        )
        try:
            continuation = OracleChunkContinuation(
                client, instruction=loaded.metadata.instruction
            )
            result = evaluate_candidate(
                restored,
                np.zeros((0, 7), dtype=np.float32),
                continuation,
            )
            row = {"repeat": repeat, **result.to_dict()}
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        finally:
            restored.close()
    report = {
        "schema_version": "rase-pre-c0-r4-persistent-parity/v1",
        "state_key": args.state_key,
        "suite": args.suite,
        "n_repeats": args.repeats,
        "successes": sum(row["success"] for row in results),
        "all_outcomes_identical": len({row["success"] for row in results}) == 1,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
