#!/usr/bin/env python3
"""Generate one auditable OFT action chunk at each frozen pool state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _expand(value: object | None, env_name: str | None = None) -> str | None:
    if value in {None, ""}:
        return os.environ.get(env_name) if env_name else None
    return str(Path(os.path.expandvars(str(value))).expanduser())


def _suite(task_id: str) -> str:
    for prefix, name in (
        ("libero_spatial", "libero_spatial"),
        ("libero_object", "libero_object"),
        ("libero_goal", "libero_goal"),
        ("libero_10", "libero_10"),
    ):
        if task_id.startswith(prefix):
            return name
    raise ValueError(f"unknown task suite: {task_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()

    cfg = _load(args.config.resolve())
    payload = _load(args.state_keys_json.resolve())
    keys = [str(value) for value in payload.get("state_keys") or []]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("frozen state keys must be non-empty and unique")
    pool_path = Path(_expand(cfg.get("pool"), "RASE_POOL_ROOT") or "pool")
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.candidates import load_artifact, make_artifact, save_artifact
    from rase.collect.forked_rollout import restore_pool_state
    from rase.collect.oracle_continuation import raw_libero_to_oracle_arrays
    from rase.collect.state_pool import StatePool
    from rase.oracle.client import OracleClient

    libero_plus_root = _expand(
        (cfg.get("adapter") or {}).get("libero_plus_root"), "LIBERO_PLUS_ROOT"
    )
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_path.resolve())
    selected = [
        key
        for key in keys
        if _suite(pool.read_state(key, load_observations=False).metadata.task_id)
        == args.suite
    ]
    if not selected:
        raise SystemExit(f"no states match suite {args.suite}")

    client = OracleClient(args.endpoint, timeout_ms=60_000)
    model_info = client.model_info()
    if model_info.get("suite") not in {None, args.suite}:
        raise ValueError(f"oracle suite mismatch: {model_info.get('suite')} != {args.suite}")
    policy_hash = hashlib.sha256(
        json.dumps(model_info, sort_keys=True, default=str).encode()
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh_run and args.summary_output.exists():
        raise SystemExit(f"fresh summary exists: {args.summary_output}")

    records = []
    try:
        for state_key in selected:
            target = args.output_dir / f"{state_key}.npz"
            if target.exists() and not args.fresh_run:
                artifact = load_artifact(target)
                records.append(
                    {"state_key": state_key, "shape": list(artifact.actions.shape), "skipped": True}
                )
                continue
            if target.exists():
                raise SystemExit(f"fresh candidate exists: {target}")
            restored = restore_pool_state(
                pool,
                state_key,
                libero_plus_root=libero_plus_root,
                observation_height=int((cfg.get("adapter") or {}).get("observation_height", 360)),
                observation_width=int((cfg.get("adapter") or {}).get("observation_width", 360)),
            )
            try:
                restored.forkable.restore(
                    restored.snapshot,
                    check_task_fingerprint=restored.check_task_fingerprint,
                )
                agentview, wrist, proprio = raw_libero_to_oracle_arrays(
                    restored.handle.control_env
                )
                meta = restored.loaded.metadata
                predicted = client.predict(
                    {
                        "agentview": agentview[None, ...],
                        "wrist": wrist[None, ...],
                        "proprio": proprio[None, ...],
                    },
                    payload={
                        "instructions": [meta.instruction],
                        "return_mode": "chunk",
                        "proprio_format": "policy_state",
                        "images_already_flipped": False,
                        "env_id": [0],
                    },
                )
                actions = np.asarray(predicted["actions"], dtype=np.float32)
                if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 7:
                    raise ValueError(f"unexpected OFT chunk shape: {actions.shape}")
                artifact = make_artifact(
                    actions,
                    seeds=[0],
                    temperature=0.0,
                    policy_hash=policy_hash,
                )
                save_artifact(target, artifact)
                records.append(
                    {"state_key": state_key, "shape": list(actions.shape), "skipped": False}
                )
                print(f"OFT_CHUNK state={state_key} shape={actions.shape}", flush=True)
            finally:
                restored.close()
    finally:
        client.close()

    result = {
        "schema_version": "rase-oft-state-chunks/v1",
        "status": "complete",
        "suite": args.suite,
        "model_info": model_info,
        "policy_hash": policy_hash,
        "n_states": len(records),
        "records": records,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"OFT_CHUNKS_DONE suite={args.suite} n={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
