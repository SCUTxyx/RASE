#!/usr/bin/env python3
"""Roll out persistent OFT and save its closed-loop action trajectory per state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_oft_pool_candidates import _expand, _load, _suite  # noqa: E402


class RecordingContinuation:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.actions: list[np.ndarray] = []

    def bind_control_env(self, control_env: Any) -> None:
        self.inner.bind_control_env(control_env)

    def reset(self) -> None:
        self.actions.clear()
        self.inner.reset()

    def act(self, observation: Any, *, task: str) -> np.ndarray:
        action = np.asarray(self.inner.act(observation, task=task), dtype=np.float32)
        self.actions.append(action.copy())
        return action


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
    key_payload = _load(args.state_keys_json.resolve())
    keys = [str(value) for value in key_payload.get("state_keys") or []]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("frozen state keys must be non-empty and unique")
    pool_path = Path(_expand(cfg.get("pool"), "RASE_POOL_ROOT") or "pool")
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.candidates import load_artifact, make_artifact, save_artifact
    from rase.collect.forked_rollout import evaluate_candidate, restore_pool_state
    from rase.collect.oracle_continuation import OracleChunkContinuation
    from rase.collect.state_pool import StatePool
    from rase.oracle.client import OracleClient

    adapter = dict(cfg.get("adapter") or {})
    libero_plus_root = _expand(adapter.get("libero_plus_root"), "LIBERO_PLUS_ROOT")
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
                    {
                        "state_key": state_key,
                        "shape": list(artifact.actions.shape),
                        "skipped": True,
                    }
                )
                continue
            if target.exists():
                raise SystemExit(f"fresh trajectory exists: {target}")
            restored = restore_pool_state(
                pool,
                state_key,
                libero_plus_root=libero_plus_root,
                observation_height=int(adapter.get("observation_height", 360)),
                observation_width=int(adapter.get("observation_width", 360)),
            )
            try:
                meta = restored.loaded.metadata
                recording = RecordingContinuation(
                    OracleChunkContinuation(client, instruction=meta.instruction)
                )
                result = evaluate_candidate(
                    restored,
                    np.empty((0, 7), dtype=np.float32),
                    recording,
                )
                actions = np.asarray(recording.actions, dtype=np.float32)[None, ...]
                if actions.ndim != 3 or actions.shape[2] != 7:
                    raise ValueError(f"unexpected trajectory shape: {actions.shape}")
                artifact = make_artifact(
                    actions,
                    seeds=[0],
                    temperature=0.0,
                    policy_hash=policy_hash,
                )
                save_artifact(target, artifact)
                records.append(
                    {
                        "state_key": state_key,
                        "shape": list(actions.shape),
                        "skipped": False,
                        "direct_oft_result": result.to_dict(),
                    }
                )
                print(
                    f"OFT_TRAJECTORY state={state_key} shape={actions.shape} "
                    f"success={result.success}",
                    flush=True,
                )
            finally:
                restored.close()
    finally:
        client.close()

    summary = {
        "schema_version": "rase-oft-recovery-trajectories/v1",
        "status": "complete",
        "suite": args.suite,
        "model_info": model_info,
        "policy_hash": policy_hash,
        "n_states": len(records),
        "records": records,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"OFT_TRAJECTORIES_DONE suite={args.suite} n={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
