#!/usr/bin/env python3
"""Collect long-horizon successful OFT teacher chunks for PRE-C1.1.

From each PRE-C0 current_suffix failure, run OFT closed-loop up to
``--horizon-steps`` (default 128). Every OFT predict records (obs, action chunk).
Only trajectories with ``rollout_success=true`` enter the training corpus;
failures are QC-counted only.
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


def _squeeze_batch(value: Any) -> Any:
    array = np.asarray(value)
    while array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array


def _pack_observation(observation: dict[str, Any], task: str) -> dict[str, Any]:
    """Serialize gym-style observation for later BC batch rebuild."""
    pixels = observation.get("pixels") or {}
    packed: dict[str, Any] = {"task": str(task)}
    if isinstance(pixels, dict):
        if "image" in pixels:
            packed["pixels_image"] = np.asarray(_squeeze_batch(pixels["image"]), dtype=np.uint8)
        if "image2" in pixels:
            packed["pixels_image2"] = np.asarray(_squeeze_batch(pixels["image2"]), dtype=np.uint8)
    # Flatten robot_state leaves (SmolVLA obs uses robot_state, not agent_pos).
    robot_state = observation.get("robot_state")
    if isinstance(robot_state, dict):

        def _walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    _walk(value, f"{prefix}{key}.")
            else:
                packed[f"rs_{prefix[:-1]}"] = np.asarray(_squeeze_batch(node))

        _walk(robot_state, "")
    # Legacy fallback.
    agent_pos = observation.get("agent_pos")
    if agent_pos is not None:
        packed["agent_pos"] = np.asarray(_squeeze_batch(agent_pos), dtype=np.float32)
    return packed


def _unpack_observation(packed: dict[str, Any]) -> dict[str, Any]:
    pixels: dict[str, Any] = {}
    if "pixels_image" in packed:
        pixels["image"] = np.asarray(packed["pixels_image"])[None, ...]
    if "pixels_image2" in packed:
        pixels["image2"] = np.asarray(packed["pixels_image2"])[None, ...]
    observation: dict[str, Any] = {"pixels": pixels}
    robot_state: dict[str, Any] = {}
    for key, value in packed.items():
        if not str(key).startswith("rs_"):
            continue
        parts = str(key)[3:].split(".")
        cursor: dict[str, Any] = robot_state
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = np.asarray(value)[None, ...]
    if robot_state:
        observation["robot_state"] = robot_state
    if "agent_pos" in packed:
        observation["agent_pos"] = np.asarray(packed["agent_pos"], dtype=np.float32)[None, ...]
    task = packed.get("task")
    if task is not None:
        observation["task"] = str(np.asarray(task).item() if hasattr(task, "item") else task)
    return observation


class MultiChunkOracleRecorder:
    """OFT chunk continuation that logs every predict (obs + full chunk)."""

    def __init__(
        self,
        client: Any,
        *,
        instruction: str,
        env_id: int = 0,
        max_steps: int,
        chunk_dir: Path,
    ) -> None:
        from collections import deque

        from rase.collect.oracle_continuation import OracleChunkContinuation

        self.inner = OracleChunkContinuation(client, instruction=instruction, env_id=env_id)
        self.max_steps = int(max_steps)
        self.chunk_dir = Path(chunk_dir)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.actions: list[np.ndarray] = []
        self.chunk_records: list[dict[str, Any]] = []
        self._queue: deque[np.ndarray] = self.inner._queue
        self._libero_env: Any | None = None

    def bind_control_env(self, control_env: Any) -> None:
        self.inner.bind_control_env(control_env)

    def bind_libero_env(self, libero_env: Any) -> None:
        self._libero_env = libero_env

    def reset(self) -> None:
        self.actions.clear()
        self.chunk_records.clear()
        self.inner.reset()

    def act(self, observation: Any, *, task: str) -> np.ndarray:
        from rase.collect.oracle_continuation import raw_libero_to_oracle_arrays
        from rase.collect.pool_candidates import observation_from_libero_env
        from rase.collect.policy_step import current_timestep

        if len(self.actions) >= self.max_steps:
            return np.zeros(7, dtype=np.float32)

        if not self._queue:
            if self.inner.control_env is None:
                raise RuntimeError("MultiChunkOracleRecorder has no bound control_env")
            if self._libero_env is None:
                raise RuntimeError("MultiChunkOracleRecorder has no bound libero_env")

            gym_obs = observation_from_libero_env(self._libero_env)
            packed = _pack_observation(gym_obs, task=task or self.inner.instruction)
            timestep = int(current_timestep(self.inner.control_env))

            agentview, wrist, proprio = raw_libero_to_oracle_arrays(self.inner.control_env)
            outputs = self.inner.client.predict(
                {
                    "agentview": agentview[None, ...],
                    "wrist": wrist[None, ...],
                    "proprio": proprio[None, ...],
                },
                payload={
                    "instructions": [task or self.inner.instruction],
                    "return_mode": "chunk",
                    "proprio_format": "policy_state",
                    "images_already_flipped": False,
                    "env_id": [self.inner.env_id],
                },
            )
            actions = np.asarray(outputs["actions"], dtype=np.float32)
            if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 7:
                raise ValueError(f"unexpected oracle actions shape {actions.shape}")
            chunk = np.asarray(actions[0], dtype=np.float32)
            chunk_index = len(self.chunk_records)
            chunk_path = self.chunk_dir / f"chunk_{chunk_index:04d}.npz"
            np.savez_compressed(
                chunk_path,
                oft_action_chunk=chunk,
                timestep=np.asarray(timestep, dtype=np.int32),
                **{k: v for k, v in packed.items() if k != "task"},
                task=np.asarray(packed["task"]),
            )
            self.chunk_records.append(
                {
                    "chunk_index": chunk_index,
                    "timestep": timestep,
                    "chunk_steps": int(chunk.shape[0]),
                    "chunk_path": str(chunk_path),
                }
            )
            for step in chunk:
                self._queue.append(np.asarray(step, dtype=np.float32))

        action = np.asarray(self._queue.popleft(), dtype=np.float32)
        self.actions.append(action.copy())
        return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--suite", required=True, help="Spatial|Object|Goal|Long")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=128,
        help="Max OFT steps from fork. 0 = persistent until episode end / success.",
    )
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
                    # Persistent OFT, but never shorter than 128 steps from fork.
                    # Late mid-episode snapshots otherwise hit episode_max too soon
                    # (e.g. fork@231 with max@280 → only 49 steps).
                    max_steps = max(128, remaining)
                    max_episode_steps = int(now_t + max_steps)
                    horizon_mode = "persistent_min128_from_fork"
                else:
                    max_steps = int(args.horizon_steps)
                    max_episode_steps = int(now_t + args.horizon_steps)
                    horizon_mode = "fixed_from_fork"
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
                "task_id": row.get("task_id"),
                "suite": row.get("suite"),
                "cell": row.get("cell"),
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
                "naming": "offline long-horizon OFT recovery teacher",
                "not_runtime_oft": True,
                "kept_for_training": success,
            }
            if not success:
                # Drop chunk files for failed trajs to save disk; keep QC JSON.
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
                f"PRE_C1_1_OFT_DONE ordinal={ordinal} state={state_key} "
                f"steps={len(recorder.actions)} chunks_recorded={n_chunks_recorded} "
                f"kept_chunks={payload['n_chunks']} success={success} "
                f"stop={result.stop_reason}",
                flush=True,
            )
    finally:
        client.close()
    print(f"PRE_C1_1_OFT_SUITE_DONE suite={args.suite} output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
