#!/usr/bin/env python3
"""Replay exact-root successful recovery traces and record stepwise residual demos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

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


def resize_rgb(value: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(value, dtype=np.uint8)).convert("RGB").resize(
            (size, size), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )


def canonical_action(value: np.ndarray, *, action_dim: int = 7) -> np.ndarray:
    """Return one environment action as ``(action_dim,)`` without broadcasting.

    LeRobot continuations currently return ``(1, action_dim)`` while captured
    chunks and recorded teacher actions return ``(action_dim,)``.  Keeping the
    batch axis here would make ``teacher - source`` broadcast across time when
    step arrays are stacked.
    """
    action = np.asarray(value, dtype=np.float32)
    if action.shape == (1, action_dim):
        action = action[0]
    if action.shape != (action_dim,):
        raise ValueError(f"expected one action with shape ({action_dim},), got {action.shape}")
    return np.ascontiguousarray(action)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--viability-audit", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--source-action-mode",
        choices=("chunked", "step-requery"),
        default="chunked",
    )
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()
    cfg = read_json(args.config.resolve())
    keys_payload = read_json(args.state_keys_json.resolve())
    source_summary = read_json(args.source_summary.resolve())
    audit = read_json(args.viability_audit.resolve())
    keys = [str(key) for key in keys_payload.get("state_keys") or []]
    source_outcomes = {
        str(row["state_key"]): bool(row["continue_smol_active_chunk"])
        for row in source_summary.get("per_pair") or []
    }
    recovery_success = {
        str(row["state_key"]): bool(row["success"])
        for row in audit.get("per_root") or []
    }
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
    from rase.collect.candidates import load_artifact, seed_everything
    from rase.collect.forked_rollout import (
        InProcessSmolVLAContinuation,
        load_smolvla_policy_bundle,
        restore_pool_state,
        rollout_seed,
    )
    from rase.collect.oracle_continuation import raw_libero_to_oracle_arrays
    from rase.collect.policy_step import (
        as_batched_action,
        capture_inference_event,
        clear_policy_queues,
        success_from_info,
    )
    from rase.collect.pool_candidates import observation_from_libero_env
    from rase.collect.state_pool import StatePool

    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT") or adapter.get("libero_plus_root")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_path.resolve())
    bundle = load_smolvla_policy_bundle(
        policy_path.resolve(), device=str(adapter.get("device", "cuda")),
        num_steps=int(adapter.get("num_steps", 10)), n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=tokenizer_path.resolve(),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    output = args.output_dir.resolve()
    if args.fresh_run and output.exists():
        raise SystemExit(f"fresh output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for root_index, state_key in enumerate(keys):
        target_json = output / f"{state_key}.json"
        target_npz = output / f"{state_key}.npz"
        if target_json.is_file():
            rows.append(read_json(target_json))
            print(f"E3_STEP_DEMO root={root_index+1}/{len(keys)} key={state_key} skipped=True", flush=True)
            continue
        source_success = source_outcomes.get(state_key)
        if source_success is None:
            raise ValueError(f"missing source outcome for {state_key}")
        if source_success:
            mode = "identity_source_success"
            teacher_actions = None
        elif recovery_success.get(state_key, False):
            mode = "successful_recovery_replay"
            artifact = load_artifact(args.trajectory_dir.resolve() / f"{state_key}.npz")
            teacher_actions = np.asarray(artifact.actions, dtype=np.float32)[0]
        else:
            row = {
                "state_key": state_key,
                "status": "excluded",
                "reason": "source_and_reference_failure",
                "n_steps": 0,
            }
            write_json(target_json, row)
            rows.append(row)
            continue

        restored = restore_pool_state(
            pool, state_key, libero_plus_root=libero_plus_root,
            observation_height=int(adapter.get("observation_height", 360)),
            observation_width=int(adapter.get("observation_width", 360)),
        )
        agentviews = []
        wrists = []
        proprios = []
        source_actions = []
        target_actions = []
        success = False
        stop_reason = "max_steps"
        try:
            vector_env = restored.handle.vector_env
            single = vector_env.envs[0]
            task = str(getattr(single, "task_description", "") or restored.loaded.metadata.instruction)
            horizon = min(
                args.max_steps,
                len(teacher_actions) if teacher_actions is not None else int(getattr(single, "_max_episode_steps", args.max_steps)),
            )
            source_policy = InProcessSmolVLAContinuation(
                bundle,
                temperature=float(adapter.get("continuation_temperature", 0.5)),
                seed=rollout_seed(state_key, 0, 0, salt=0x45335354),
            )
            source_policy.reset()
            for step in range(horizon):
                observation = observation_from_libero_env(single)
                agentview, wrist, proprio = raw_libero_to_oracle_arrays(restored.handle.control_env)
                if args.source_action_mode == "chunked":
                    source_action = source_policy.act(observation, task=task)
                else:
                    seed = rollout_seed(state_key, 0, step, salt=0x45335354)
                    seed_everything(seed)
                    bundle["policy"].reset()
                    clear_policy_queues(bundle["policy"])
                    source_action, _event = capture_inference_event(
                        bundle, observation, task=task,
                        boundary_step=int(restored.loaded.metadata.step) + step,
                        generation_seed=seed, horizon=1,
                    )
                source_action = canonical_action(source_action)
                target_action = canonical_action(
                    source_action if teacher_actions is None else teacher_actions[step]
                )
                agentviews.append(resize_rgb(agentview, args.image_size))
                wrists.append(resize_rgb(wrist, args.image_size))
                proprios.append(np.asarray(proprio, dtype=np.float32))
                source_actions.append(np.asarray(source_action, dtype=np.float32))
                target_actions.append(np.asarray(target_action, dtype=np.float32))
                _obs, _reward, term, trunc, info = vector_env.step(as_batched_action(target_action))
                terminated = bool(np.asarray(term).reshape(-1)[0])
                truncated = bool(np.asarray(trunc).reshape(-1)[0])
                if terminated or truncated:
                    success = success_from_info(info)
                    stop_reason = "success" if success else "terminal_failure"
                    break
        finally:
            restored.close()

        usable = bool(success)
        row = {
            "schema_version": "rase-e3-step-demo-root/v1",
            "state_key": state_key,
            "task_id": restored.loaded.metadata.task_id,
            "suite": restored.loaded.metadata.suite,
            "instruction": restored.loaded.metadata.instruction,
            "mode": mode,
            "source_action_mode": args.source_action_mode,
            "status": "complete" if usable else "excluded",
            "reason": "" if usable else f"replay_not_successful:{stop_reason}",
            "n_steps": len(source_actions) if usable else 0,
            "rollout_steps_observed": len(source_actions),
            "success": success,
            "stop_reason": stop_reason,
            "artifact": str(target_npz.resolve()) if usable else None,
        }
        if usable:
            save_npz(
                target_npz,
                agentview=np.stack(agentviews), wrist=np.stack(wrists),
                proprio=np.stack(proprios).astype(np.float32),
                source_action=np.stack(source_actions).astype(np.float32),
                target_action=np.stack(target_actions).astype(np.float32),
                delta_target=(np.stack(target_actions) - np.stack(source_actions)).astype(np.float32),
            )
        write_json(target_json, row)
        rows.append(row)
        print(
            f"E3_STEP_DEMO root={root_index+1}/{len(keys)} key={state_key} mode={mode} "
            f"success={success} steps={len(source_actions)} usable={usable}", flush=True,
        )

    usable_rows = [row for row in rows if row.get("status") == "complete"]
    correction_rows = [row for row in usable_rows if row.get("mode") == "successful_recovery_replay"]
    identity_rows = [row for row in usable_rows if row.get("mode") == "identity_source_success"]
    checks = {
        "correction_roots_at_least_20": len(correction_rows) >= 20,
        "correction_steps_at_least_1000": sum(row["n_steps"] for row in correction_rows) >= 1000,
        "identity_roots_at_least_4": len(identity_rows) >= 4,
        "identity_steps_at_least_200": sum(row["n_steps"] for row in identity_rows) >= 200,
    }
    summary = {
        "schema_version": "rase-e3-step-demos/v1",
        "status": "complete",
        "scientific_scope": "development_only_exact_root_stepwise_residual_supervision",
        "source_action_mode": args.source_action_mode,
        "n_requested_roots": len(keys),
        "n_correction_roots": len(correction_rows),
        "n_identity_roots": len(identity_rows),
        "n_correction_steps": sum(row["n_steps"] for row in correction_rows),
        "n_identity_steps": sum(row["n_steps"] for row in identity_rows),
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "records": rows,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("decision", "n_correction_roots", "n_identity_roots", "n_correction_steps", "n_identity_steps")}, sort_keys=True))
    return 0 if summary["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
