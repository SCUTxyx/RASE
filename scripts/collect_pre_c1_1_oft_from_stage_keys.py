#!/usr/bin/env python3
"""Collect successful OFT multi-chunk teachers from PRE-C0 stage-key snapshots.

Used to expand PRE-C1.1 beyond the 48-state T1/T3 audit set (T0/T2/T4) after
hard-stops on successful-recovery count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from collect_pre_c1_1_oft_success_chunks import MultiChunkOracleRecorder  # noqa: E402


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-keys", type=Path, required=True)
    parser.add_argument("--stages", default="T0,T2,T4", help="Comma-separated stages")
    parser.add_argument("--suite", required=True, help="Spatial|Object|Goal|Long")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon-steps", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-keys-file",
        type=Path,
        default=None,
        help="Optional JSON list/set of state_keys already collected elsewhere",
    )
    args = parser.parse_args()

    cfg = _load(args.config.resolve())
    adapter = dict(cfg.get("adapter_config") or cfg.get("adapter") or {})
    pool_root = Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve()
    libero_plus_root = adapter.get("libero_plus_root")
    want_stages = {s.strip() for s in str(args.stages).split(",") if s.strip()}

    stage_payload = _load(args.stage_keys.resolve())
    selected = list(stage_payload.get("selected_states") or [])
    skip: set[str] = set()
    if args.skip_keys_file and args.skip_keys_file.is_file():
        raw = _load(args.skip_keys_file.resolve())
        if isinstance(raw, list):
            skip = {str(x) for x in raw}
        elif isinstance(raw, dict):
            skip = {str(x) for x in (raw.get("state_keys") or raw.get("keys") or [])}

    rows = []
    for row in selected:
        if str(row.get("suite")) != str(args.suite):
            continue
        if str(row.get("stage")) not in want_stages:
            continue
        sk = str(row["state_key"])
        if sk in skip:
            continue
        rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"PRE_C1_1_STAGE_EMPTY suite={args.suite} stages={sorted(want_stages)}", flush=True)
        return 0

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import (
        RolloutConfig,
        evaluate_candidate,
        restore_pool_state,
    )
    from rase.collect.policy_step import current_timestep
    from rase.collect.state_pool import StatePool
    from rase.oracle.client import OracleClient

    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    client = OracleClient(args.endpoint)
    try:
        empty = np.empty((0, 7), dtype=np.float32)
        rollout_cfg = RolloutConfig(
            n_action_steps=int(adapter.get("n_action_steps", 10)),
            num_steps=int(adapter.get("num_steps", 10)),
            observation_height=int(adapter.get("observation_height", 360)),
            observation_width=int(adapter.get("observation_width", 360)),
        )
        for ordinal, row in enumerate(rows):
            state_key = str(row["state_key"])
            target = output_dir / f"{state_key}.json"
            chunk_dir = output_dir / f"{state_key}_chunks"
            if args.resume and target.exists():
                print(f"SKIP {state_key}", flush=True)
                continue
            restored = restore_pool_state(
                pool,
                state_key,
                libero_plus_root=libero_plus_root,
                observation_height=rollout_cfg.observation_height,
                observation_width=rollout_cfg.observation_width,
            )
            try:
                instruction = str(restored.loaded.metadata.instruction)
                now_t = current_timestep(restored.handle.control_env)
                single = restored.handle.vector_env.envs[0]
                episode_max = int(getattr(single, "_max_episode_steps", 600))
                remaining = max(0, episode_max - int(now_t))
                if int(args.horizon_steps) <= 0:
                    max_steps = max(128, remaining)
                    max_episode_steps = int(now_t + max_steps)
                    horizon_mode = "persistent_min128_from_fork"
                else:
                    max_steps = int(args.horizon_steps)
                    max_episode_steps = int(now_t + args.horizon_steps)
                    horizon_mode = "fixed_from_fork"
                meta = restored.loaded.metadata
                cell = f"{meta.perturb_dim}:L{meta.level}" if getattr(meta, "perturb_dim", None) else None
                recorder = MultiChunkOracleRecorder(
                    client,
                    instruction=instruction,
                    max_steps=max_steps,
                    chunk_dir=chunk_dir,
                )
                recorder.bind_libero_env(restored.handle.vector_env.envs[0])
                result = evaluate_candidate(
                    restored,
                    empty,
                    recorder,
                    max_episode_steps=max_episode_steps,
                )
            finally:
                restored.close()

            success = bool(result.success)
            n_chunks_recorded = int(len(recorder.chunk_records))
            payload = {
                "schema_version": "rase-pre-c1-1-oft-success-traj/v1",
                "state_key": state_key,
                "episode_id": row.get("episode_id"),
                "task_id": row.get("task_id") or row.get("concrete_task_id"),
                "suite": row.get("suite"),
                "cell": cell,
                "stage": row.get("stage"),
                "teacher_source": "oft",
                "teacher_horizon_mode": horizon_mode,
                "teacher_horizon_steps": int(args.horizon_steps),
                "teacher_max_steps_from_fork": int(max_steps),
                "teacher_steps": int(len(recorder.actions)),
                "n_chunks": n_chunks_recorded,
                "n_chunks_recorded": n_chunks_recorded,
                "chunks": list(recorder.chunk_records),
                "rollout_success": success,
                "stop_reason": result.stop_reason,
                "env_steps": int(result.env_steps),
                "fork_timestep": int(now_t),
                "source": "pre_c0_stage_keys_expand",
                "naming": "offline long-horizon OFT recovery teacher",
                "not_runtime_oft": True,
                "kept_for_training": success,
            }
            if not success:
                if chunk_dir.is_dir():
                    for path in chunk_dir.glob("*.npz"):
                        path.unlink(missing_ok=True)
                    try:
                        chunk_dir.rmdir()
                    except OSError:
                        pass
                payload["chunks"] = []
                payload["n_chunks"] = 0
            _atomic_json(target, payload)
            print(
                f"PRE_C1_1_OFT_DONE ordinal={ordinal} state={state_key} stage={row.get('stage')} "
                f"steps={len(recorder.actions)} chunks_recorded={n_chunks_recorded} "
                f"kept_chunks={payload['n_chunks']} success={success} stop={result.stop_reason}",
                flush=True,
            )
    finally:
        client.close()
    print(f"PRE_C1_1_OFT_STAGE_SUITE_DONE suite={args.suite} output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
