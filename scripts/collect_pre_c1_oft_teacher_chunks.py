#!/usr/bin/env python3
"""Collect short OFT teacher action chunks on PRE-C0 failure snapshots.

OFT is used offline as a recovery teacher only. Runtime SmolVLA never calls OFT.
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


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _suite_to_libero(suite: str) -> str:
    mapping = {
        "Spatial": "libero_spatial",
        "Object": "libero_object",
        "Goal": "libero_goal",
        "Long": "libero_10",
    }
    if suite not in mapping:
        raise ValueError(f"unknown suite {suite}")
    return mapping[suite]


class RecordingContinuation:
    def __init__(self, inner: Any, *, max_steps: int) -> None:
        self.inner = inner
        self.max_steps = int(max_steps)
        self.actions: list[np.ndarray] = []

    def bind_control_env(self, control_env: Any) -> None:
        self.inner.bind_control_env(control_env)

    def reset(self) -> None:
        self.actions.clear()
        self.inner.reset()

    def act(self, observation: Any, *, task: str) -> np.ndarray:
        if len(self.actions) >= self.max_steps:
            return np.zeros(7, dtype=np.float32)
        action = np.asarray(self.inner.act(observation, task=task), dtype=np.float32)
        self.actions.append(action.copy())
        return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--suite", required=True, help="Spatial|Object|Goal|Long")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-steps", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = _load(args.config.resolve())
    adapter = dict(cfg.get("adapter_config") or cfg.get("adapter") or {})
    pool_root = Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve()
    libero_plus_root = adapter.get("libero_plus_root")

    rows = []
    for path in sorted(args.rollout_dir.glob("*.json")):
        if path.name == "run_manifest.json":
            continue
        payload = _load(path)
        if payload.get("schema_version") != "rase-pre-c0-corrective-rollout/v1":
            continue
        if str(payload.get("suite")) != str(args.suite):
            continue
        if bool(payload.get("family_success", {}).get("current_suffix")):
            continue
        rows.append(payload)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no current-failure PRE-C0 rows for suite={args.suite}")

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import (
        RolloutConfig,
        evaluate_candidate,
        restore_pool_state,
    )
    from rase.collect.oracle_continuation import OracleChunkContinuation
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
                cont_base = OracleChunkContinuation(client, instruction=instruction)
                recorder = RecordingContinuation(cont_base, max_steps=int(args.chunk_steps))
                # Snapshots are mid-episode; absolute horizon must be current_t + chunk.
                from rase.collect.policy_step import current_timestep

                now_t = current_timestep(restored.handle.control_env)
                result = evaluate_candidate(
                    restored,
                    empty,
                    recorder,
                    max_episode_steps=int(now_t + args.chunk_steps),
                )
            finally:
                restored.close()
            actions = np.asarray(recorder.actions, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[0] < 1:
                raise RuntimeError(
                    f"OFT produced no actions for {state_key} "
                    f"(recorded={len(recorder.actions)} now_t={now_t})"
                )
            payload = {
                "schema_version": "rase-pre-c1-oft-teacher-chunk/v1",
                "state_key": state_key,
                "episode_id": row.get("episode_id"),
                "task_id": row.get("task_id"),
                "suite": row.get("suite"),
                "cell": row.get("cell"),
                "stage": row.get("stage"),
                "teacher_source": "oft",
                "teacher_actions": actions.tolist(),
                "teacher_steps": int(actions.shape[0]),
                "rollout_success": bool(result.success),
                "stop_reason": result.stop_reason,
                "naming": "offline OFT recovery teacher chunk",
                "not_runtime_oft": True,
            }
            _atomic_json(target, payload)
            print(
                f"PRE_C1_OFT_TEACHER_DONE ordinal={ordinal} state={state_key} "
                f"steps={actions.shape[0]} success={bool(result.success)}",
                flush=True,
            )
    finally:
        client.close()
    print(f"PRE_C1_OFT_TEACHER_SUITE_DONE suite={args.suite} output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
