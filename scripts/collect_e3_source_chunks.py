#!/usr/bin/env python3
"""Capture deterministic frozen-Smol native chunks at exact frozen roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()
    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    cfg = read_json(args.config.resolve())
    keys_payload = read_json(args.state_keys_json.resolve())
    keys = [str(key) for key in keys_payload.get("state_keys") or []]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("state keys must be non-empty and unique")
    explicit = cfg.get("adapter_config")
    adapter = dict(explicit) if isinstance(explicit, Mapping) else {}
    pool_path = Path(cfg.get("pool") or keys_payload.get("pool") or "")
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    policy_path = Path(adapter.get("policy_path") or "ckpts/smolvla_libero")
    tokenizer_path = Path(adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct")
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    if not tokenizer_path.is_absolute():
        tokenizer_path = ROOT / tokenizer_path

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.candidates import make_artifact, save_artifact, seed_everything
    from rase.collect.forked_rollout import load_smolvla_policy_bundle, restore_pool_state, rollout_seed
    from rase.collect.policy_step import capture_inference_event, clear_policy_queues
    from rase.collect.pool_candidates import observation_from_libero_env
    from rase.collect.state_pool import StatePool

    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT") or adapter.get("libero_plus_root")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_path.resolve())
    bundle = load_smolvla_policy_bundle(
        policy_path.resolve(), device=str(adapter.get("device", "cuda")),
        num_steps=int(adapter.get("num_steps", 10)),
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=tokenizer_path.resolve(),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    output = args.output_dir.resolve()
    if args.fresh_run and output.exists():
        raise SystemExit(f"fresh output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    policy_hash = hashlib.sha256(str(policy_path.resolve()).encode()).hexdigest()
    rows = []
    for index, state_key in enumerate(keys):
        target = output / f"{state_key}.npz"
        if target.is_file():
            skipped = True
            seed = rollout_seed(state_key, 0, 0, salt=0x52415345)
        else:
            seed = rollout_seed(state_key, 0, 0, salt=0x52415345)
            restored = restore_pool_state(
                pool, state_key, libero_plus_root=libero_plus_root,
                observation_height=int(adapter.get("observation_height", 360)),
                observation_width=int(adapter.get("observation_width", 360)),
            )
            try:
                single = restored.handle.vector_env.envs[0]
                seed_everything(seed)
                bundle["policy"].reset()
                clear_policy_queues(bundle["policy"])
                _first, event = capture_inference_event(
                    bundle, observation_from_libero_env(single),
                    task=str(getattr(single, "task_description", "") or restored.loaded.metadata.instruction),
                    boundary_step=int(restored.loaded.metadata.step),
                    generation_seed=seed, horizon=args.horizon,
                )
                chunk = np.asarray(event.env_chunk, dtype=np.float32)
                if chunk.shape != (args.horizon, 7):
                    raise ValueError(f"{state_key}: captured {chunk.shape}")
                save_artifact(
                    target,
                    make_artifact(chunk[None, ...], seeds=[seed], temperature=0.0, policy_hash=policy_hash),
                )
            finally:
                restored.close()
            skipped = False
        rows.append({"state_key": state_key, "seed": seed, "path": str(target), "skipped": skipped})
        print(f"E3_SOURCE_CHUNK state={index+1}/{len(keys)} key={state_key} skipped={skipped}", flush=True)
    summary = {
        "schema_version": "rase-e3-source-chunks/v1",
        "status": "complete",
        "source_chunk_mode": "exact_root_frozen_smol_greedy_replan",
        "horizon": args.horizon,
        "n_states": len(rows),
        "state_keys_sha256": keys_payload.get("state_keys_sha256"),
        "records": rows,
    }
    write_json(output / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
