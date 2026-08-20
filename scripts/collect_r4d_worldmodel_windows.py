#!/usr/bin/env python3
"""R4-D trajectory-window collection for world-model teacher training.

For each train state, restores the pool snapshot and runs a continuous OFT
trajectory while recording agent-view frames, proprio, and per-step actions.
Then extracts non-overlapping trajectory windows:

    (frames[t-L:t], proprio[t], action[t:t+K], frames[t+1:t+K])

These windows feed the offline V-JEPA 2-AC teacher and the LightRiskStudent
distillation dataset.  Labels are per-VLA continuation outcomes.

Usage (mirrors collect_r4_boundary_transitions.py):
    python scripts/collect_r4d_worldmodel_windows.py \
        --config configs/pre_a3_recovery_duration120.yaml \
        --design runs/pre_c0_r4/r4d_train_design.json \
        --suite libero_object --endpoint tcp://127.0.0.1:5555 \
        --output-dir runs/pre_c0_r4/worldmodel_windows/suite_object \
        --window 8 --stride 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        value = yaml.safe_load(text)
        return value if isinstance(value, dict) else {}
    return json.loads(text)


def _suite_of_task(task_id: str) -> str:
    """Map a task id (e.g. `pre_a3_libero_spatial_task00`) to its suite."""
    for name in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        if name in str(task_id):
            return name
    raise ValueError(f"unknown task suite: {task_id}")


def _expand(value: object | None, env_name: str | None = None) -> str | None:
    import os

    if value is None:
        return None
    text = str(value)
    if text.startswith("${") and text.endswith("}"):
        return os.environ.get(text[2:-1])
    if env_name:
        return os.environ.get(env_name, text)
    return text


def save_window_record(
    *,
    state_key: str,
    task_id: str,
    suite: str,
    window_index: int,
    frames: list[np.ndarray],
    proprio: np.ndarray,
    action_chunk: np.ndarray,
    vla_name: str,
    window_start: int,
    outcome_label: int | None = None,
) -> dict[str, Any]:
    """Serialize one trajectory window into a JSONL row.

    Frames are saved as uint8 PNG-encoded base64 to keep the row self-contained.
    """
    import base64
    import io

    from PIL import Image

    frame_encodings = []
    for frame in frames:
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="PNG")
        frame_encodings.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    return {
        "schema_version": "rase-pre-c0-r4d-worldmodel-window/v1",
        "state_key": state_key,
        "task_id": task_id,
        "suite": suite,
        "vla_name": vla_name,
        "window_index": window_index,
        "window_start": window_start,
        "frames_b64": frame_encodings,
        "proprio": np.asarray(proprio, np.float32).tolist(),
        "action_chunk": np.asarray(action_chunk, np.float32).tolist(),
        "outcome_label": outcome_label,
        "n_frames": len(frame_encodings),
    }


def collect_windows_for_state(
    *,
    pool: Any,
    state_key: str,
    design_state: dict[str, Any],
    bundle: Any,
    capture: Any,
    client: Any,
    adapter: dict[str, Any],
    libero_plus_root: str | None,
    window: int,
    stride: int,
    max_windows: int,
    vla_name: str,
) -> dict[str, Any]:
    """Collect trajectory windows for one state by replaying OFT persistently."""
    from rase.collect.forked_rollout import restore_pool_state, rollout_seed
    from rase.collect.oracle_continuation import (
        OracleChunkContinuation,
        raw_libero_to_oracle_arrays,
    )
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env

    restored = restore_pool_state(
        pool,
        state_key,
        libero_plus_root=libero_plus_root,
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    windows: list[dict[str, Any]] = []
    frames_hist: list[np.ndarray] = []
    proprio_hist: list[np.ndarray] = []
    action_hist: list[np.ndarray] = []
    try:
        restored.forkable.restore(
            restored.snapshot, check_task_fingerprint=restored.check_task_fingerprint
        )
        single = restored.handle.vector_env.envs[0]
        vector_env = restored.handle.vector_env
        task = str(
            getattr(single, "task_description", "")
            or restored.loaded.metadata.instruction
        )
        horizon = int(getattr(single, "_max_episode_steps", 600))
        shared_seed = rollout_seed(state_key, 0, 0, salt=2_026_080_403)
        oft = OracleChunkContinuation(client, instruction=task)
        oft.bind_control_env(restored.handle.control_env)
        oft.reset()

        observation = observation_from_libero_env(single)
        # Extract agent-view frame + proprio from the control env
        raw_obs = observation_from_libero_env(single)
        frame = _frame_from_obs(raw_obs)
        _, _, proprio = raw_libero_to_oracle_arrays(restored.handle.control_env)

        elapsed = 0
        terminal = False
        trajectory_success = False
        stop_reason = "horizon"
        window_counter = 0

        while not terminal:
            timestep = current_timestep(restored.handle.control_env)
            if timestep >= horizon:
                stop_reason = "horizon"
                break

            # Advance with OFT
            oft_action = np.asarray(oft.act(observation, task=task), dtype=np.float32)
            next_obs, _, term, trunc, info = vector_env.step(as_batched_action(oft_action))
            terminated, truncated, success = _success_from_step(term, trunc, info)
            terminal = terminated or truncated
            trajectory_success = success if terminal else False
            if terminal:
                stop_reason = "success" if success else ("terminated" if terminated else "truncated")

            # Record history
            frames_hist.append(frame)
            proprio_hist.append(proprio)
            action_hist.append(oft_action)

            # Extract a window every `stride` steps once we have `window` frames
            if len(frames_hist) >= window and (len(frames_hist) - window) % stride == 0:
                if window_counter < max_windows:
                    w_frames = frames_hist[-window:]
                    w_proprio = proprio_hist[-1]
                    w_action = np.stack(action_hist[-window:], axis=0)
                    row = save_window_record(
                        state_key=state_key,
                        task_id=design_state.get("task_id", task),
                        suite=design_state.get("suite", "unknown"),
                        window_index=window_counter,
                        frames=w_frames,
                        proprio=w_proprio,
                        action_chunk=w_action,
                        vla_name=vla_name,
                        window_start=elapsed - window + 1,
                        outcome_label=int(trajectory_success) if terminal else None,
                    )
                    windows.append(row)
                    window_counter += 1

            observation = next_obs
            raw_obs = observation_from_libero_env(single)
            frame = _frame_from_obs(raw_obs)
            _, _, proprio = raw_libero_to_oracle_arrays(restored.handle.control_env)
            elapsed += 1

        return {
            "state_key": state_key,
            "task_id": str(design_state.get("task_id", task)),
            "suite": str(design_state.get("suite", "unknown")),
            "n_windows": len(windows),
            "trajectory_success": trajectory_success,
            "stop_reason": stop_reason,
            "executed_oft_steps": elapsed,
            "windows": windows,
        }
    finally:
        restored.close()


def _frame_from_obs(observation: dict[str, Any]) -> np.ndarray:
    """Extract agent-view RGB frame from a batched observation dict."""
    raw = (
        observation.get("agentview_image")
        or observation.get("image")
        or observation.get("rgb")
        or observation.get("observation_view_image")
    )
    if raw is None and isinstance(observation.get("pixels"), dict):
        # pixels nests the image (e.g. {"pixels": {"agentview_image": ...}})
        raw = observation["pixels"].get("agentview_image") or observation["pixels"].get("image")
    if raw is None:
        raise ValueError("no agent-view image key in observation; keys: " + str(list(observation.keys())))
    arr = np.asarray(raw, dtype=np.uint8)
    if arr.ndim == 4:  # (B, H, W, C)
        arr = arr[0]
    return arr


def _success_from_step(term: Any, trunc: Any, info: Any) -> tuple[bool, bool, bool]:
    terminated = bool(term)
    truncated = bool(trunc)
    success = bool(info.get("success", False)) if isinstance(info, dict) else False
    if not success and isinstance(info, dict):
        success = bool(info.get("final_success", False))
    return terminated, truncated, success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=20)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--vla-name", default="smolvla")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = _load(args.config.resolve())
    design = _load(args.design.resolve())
    adapter = dict(cfg.get("adapter") or {})
    libero_plus_root = _expand(adapter.get("libero_plus_root"), "LIBERO_PLUS_ROOT")

    from rase.collect.forked_rollout import (
        load_smolvla_policy_bundle,
        restore_pool_state,
        rollout_seed,
    )
    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.state_pool import StatePool
    from rase.oracle.client import OracleClient

    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool_path = Path(_expand(cfg.get("pool"), "RASE_POOL_ROOT") or "pool")
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    pool = StatePool(pool_path.resolve())

    # Build state list from design: all train tasks for this suite
    train_tasks = design.get("train_tasks_by_suite", {}).get(args.suite, [])
    if not train_tasks:
        # Fallback: use all train tasks
        train_tasks = design.get("train_tasks", [])
    suite_states = []
    all_records = _load_any_state_keys(adapter, pool)
    for state_key, meta in all_records.items():
        if meta.get("split") != "train":
            continue
        if _suite_of_task(meta.get("task_id", "")) != args.suite:
            continue
        suite_states.append(state_key)
    suite_states.sort()
    if args.max_states > 0:
        suite_states = suite_states[: args.max_states]
    print(f"Selected {len(suite_states)} train states for suite {args.suite}")

    client = OracleClient(args.endpoint, timeout_ms=60_000)
    model_info = client.model_info()
    if model_info.get("suite") not in {None, args.suite}:
        raise ValueError(f"oracle suite mismatch: {model_info.get('suite')} != {args.suite}")

    policy_path = Path(
        _expand(adapter.get("policy_path"), "RASE_POLICY_PATH")
        or "ckpts/smolvla_libero"
    )
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    bundle = load_smolvla_policy_bundle(
        policy_path.resolve(),
        device=str(adapter.get("device", "cuda")),
        num_steps=int(adapter.get("num_steps", 10)),
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=_expand(adapter.get("tokenizer_path"), "RASE_TOKENIZER_PATH"),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = args.output_dir / "states"
    state_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for index, state_key in enumerate(suite_states, start=1):
        target = state_dir / f"{state_key}.json"
        if target.is_file() and not args.force:
            result = json.loads(target.read_text(encoding="utf-8"))
        else:
            result = collect_windows_for_state(
                pool=pool,
                state_key=state_key,
                design_state={"task_id": "unknown", "suite": args.suite},
                bundle=bundle,
                capture=None,
                client=client,
                adapter=adapter,
                libero_plus_root=libero_plus_root,
                window=args.window,
                stride=args.stride,
                max_windows=args.max_windows,
                vla_name=args.vla_name,
            )
            target.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append(result)
        print(
            f"WM_WINDOW suite={args.suite} state={state_key} "
            f"index={index}/{len(suite_states)} n_windows={result['n_windows']} "
            f"success={result['trajectory_success']}",
            flush=True,
        )

    # Merge into one jsonl
    all_windows = [w for out in outputs for w in out.get("windows", [])]
    jsonl_path = args.output_dir / f"worldmodel_windows_{args.suite}.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_windows),
        encoding="utf-8",
    )

    report = {
        "schema_version": "rase-pre-c0-r4d-worldmodel-collection/v1",
        "suite": args.suite,
        "n_states": len(outputs),
        "n_windows": len(all_windows),
        "window": args.window,
        "stride": args.stride,
        "vla_name": args.vla_name,
        "output": str(jsonl_path.resolve()),
        "source_design": str(args.design.resolve()),
        "design_sha256": hashlib.sha256(args.design.resolve().read_bytes()).hexdigest(),
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_summaries": [
            {k: out.get(k) for k in (
                "state_key", "task_id", "n_windows", "trajectory_success",
                "stop_reason", "executed_oft_steps",
            )}
            for out in outputs
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _load_any_state_keys(adapter: dict[str, Any], pool: Any) -> dict[str, Any]:
    """Read state-key records from the canonical keys file or pool."""
    keys_path = Path(_expand(adapter.get("state_keys_json"), "RASE_STATE_KEYS")
                     or ROOT / "runs/rase_pre_a3_keys120_v1.json")
    if keys_path.is_file():
        payload = _load(keys_path)
        return {str(r["state_key"]): r for r in payload.get("records", [])}
    # Fallback: scan pool
    return {str(k): {"state_key": str(k), "split": "train"} for k in pool.keys()}


if __name__ == "__main__":
    raise SystemExit(main())
