#!/usr/bin/env python3
"""Rewrite PRE-C1.1 success chunk npz files with full gym obs (incl. robot_state).

Replays saved OFT action chunks from the pool fork — no OFT server required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from collect_pre_c1_1_oft_success_chunks import _pack_observation  # noqa: E402


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = _load(args.config.resolve())
    adapter = dict(cfg.get("adapter_config") or {})
    pool_root = Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve()
    libero_plus_root = adapter.get("libero_plus_root")

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import restore_pool_state
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env
    from rase.collect.state_pool import StatePool

    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_root)

    trajs = []
    for path in sorted(args.teacher_dir.resolve().glob("suite_*/*.json")):
        payload = _load(path)
        if payload.get("schema_version") != "rase-pre-c1-1-oft-success-traj/v1":
            continue
        if not bool(payload.get("rollout_success")):
            continue
        if not payload.get("chunks"):
            continue
        trajs.append((path, payload))
    if args.limit:
        trajs = trajs[: args.limit]

    rewritten = 0
    for ordinal, (path, payload) in enumerate(trajs):
        state_key = str(payload["state_key"])
        chunks = sorted(payload["chunks"], key=lambda c: int(c.get("chunk_index", 0)))
        # Skip if first chunk already has robot_state.
        first = Path(chunks[0]["chunk_path"])
        if first.is_file():
            keys = set(np.load(first, allow_pickle=False).files)
            if any(k.startswith("rs_") for k in keys):
                print(f"SKIP_OK {state_key}", flush=True)
                continue

        restored = restore_pool_state(
            pool,
            state_key,
            libero_plus_root=libero_plus_root,
            observation_height=int(adapter.get("observation_height", 360)),
            observation_width=int(adapter.get("observation_width", 360)),
        )
        try:
            single = restored.handle.vector_env.envs[0]
            vector_env = restored.handle.vector_env
            task = str(getattr(single, "task_description", "") or "")
            for chunk_meta in chunks:
                chunk_path = Path(chunk_meta["chunk_path"])
                old = dict(np.load(chunk_path, allow_pickle=False))
                actions = np.asarray(old["oft_action_chunk"], dtype=np.float32)
                gym_obs = observation_from_libero_env(single)
                packed = _pack_observation(gym_obs, task=task)
                timestep = int(current_timestep(restored.handle.control_env))
                np.savez_compressed(
                    chunk_path,
                    oft_action_chunk=actions,
                    timestep=np.asarray(timestep, dtype=np.int32),
                    **{k: v for k, v in packed.items() if k != "task"},
                    task=np.asarray(packed["task"]),
                )
                for action in actions:
                    vector_env.step(as_batched_action(action))
            rewritten += 1
            print(
                f"REWRITE_OK ordinal={ordinal} state={state_key} chunks={len(chunks)}",
                flush=True,
            )
        finally:
            restored.close()

    print(f"PRE_C1_1_OBS_REWRITE_DONE rewritten={rewritten} scanned={len(trajs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
