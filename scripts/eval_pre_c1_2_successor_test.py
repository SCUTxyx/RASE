#!/usr/bin/env python3
"""PRE-C1.2 Phase 0: one-step successor + restore-repeatability controls."""

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

from rase.adapt.pre_c1_2 import interface_mismatch_decision, load_protocol_lock
from rase.adapt.pre_c1_2_eval import (
    action_space_report,
    load_pre_c0_failure_keys,
    successor_distance,
)
from rase.adapt.recovery_lora import load_lora_onto_policy, set_adapter_enabled
from rase.collect.forked_rollout import (
    InProcessSmolVLAContinuation,
    load_smolvla_policy_bundle,
    restore_pool_state,
)
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.policy_step import as_batched_action
from rase.collect.state_pool import StatePool


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _step_once(restored: Any, action: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    env = restored.handle.vector_env
    obs, _reward, _term, _trunc, info = env.step(as_batched_action(action))
    # observation_from_libero_env expects the single env.
    single = env.envs[0]
    observation = observation_from_libero_env(single)
    return observation, dict(info) if isinstance(info, dict) else {}


def _predict_student_env_action(
    continuation: InProcessSmolVLAContinuation,
    observation: dict[str, Any],
    *,
    task: str,
) -> np.ndarray:
    continuation.reset_metrics()
    continuation.reset()
    action = np.asarray(continuation.act(observation, task=task), dtype=np.float32)
    return action.reshape(-1)


def _predict_oft_env_action(
    client: Any,
    restored: Any,
    *,
    instruction: str,
    env_id: int = 0,
) -> np.ndarray:
    from rase.collect.oracle_continuation import OracleChunkContinuation

    cont = OracleChunkContinuation(client, instruction=instruction, env_id=env_id)
    cont.bind_control_env(restored.handle.control_env)
    cont.reset()
    observation = observation_from_libero_env(restored.handle.vector_env.envs[0])
    return np.asarray(cont.act(observation, task=instruction), dtype=np.float32).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_protocol_lock.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collect_pre_c0_deviation_pilot24.json"),
    )
    parser.add_argument(
        "--failure-rollout-dir",
        type=Path,
        default=Path("runs/rase_pre_c0_same_policy_pilot48_v1"),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("runs/rase_pre_c1_1_lora_train_v1/adapter_final"),
    )
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--output", type=Path, default=Path("runs/rase_pre_c1_2_successor_v1.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--suite", default=None, help="Optional suite filter: Spatial|Object|Goal|Long")
    parser.add_argument(
        "--state-keys-json",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json"),
    )
    parser.add_argument(
        "--skip-oft",
        action="store_true",
        help="Skip OFT arms (student-repeat only); cannot clear interface gate.",
    )
    parser.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="Merge suite results into an existing successor JSON.",
    )
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    succ_cfg = dict(lock.get("successor_test") or {})
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    adapter = dict(cfg.get("adapter_config") or {})
    pool = StatePool(Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve())
    failures = load_pre_c0_failure_keys(args.failure_rollout_dir.resolve())
    if args.state_keys_json and args.state_keys_json.is_file():
        allowed = set(json.loads(args.state_keys_json.read_text(encoding="utf-8"))["state_keys"])
        failures = [row for row in failures if str(row["state_key"]) in allowed]
    if args.suite:
        failures = [row for row in failures if str(row.get("suite")) == str(args.suite)]
    if args.limit:
        failures = failures[: args.limit]
    if not failures:
        raise SystemExit("no PRE-C0 failure keys found")

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths

    ensure_libero_plus_paths(adapter.get("libero_plus_root"))
    _patch_lerobot_init_states()

    bundle = load_smolvla_policy_bundle(
        Path(adapter.get("policy_path") or "ckpts/smolvla_libero"),
        device=str(adapter.get("device", "cuda")),
        num_steps=int(adapter.get("num_steps", 10)),
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=Path(adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct"),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    handle = load_lora_onto_policy(bundle["policy"], str(args.adapter_dir.resolve()))
    set_adapter_enabled(handle, True)
    bundle["policy"] = handle.policy

    client = None
    if not args.skip_oft:
        from rase.oracle.client import OracleClient

        client = OracleClient(args.endpoint)

    results = []
    try:
        for row in failures:
            state_key = str(row["state_key"])
            record: dict[str, Any] = {
                "state_key": state_key,
                "episode_id": row.get("episode_id"),
                "suite": row.get("suite"),
                "stage": row.get("stage"),
            }
            # --- Student action (fixed seed) ---
            restored = restore_pool_state(
                pool,
                state_key,
                libero_plus_root=adapter.get("libero_plus_root"),
                observation_height=int(adapter.get("observation_height", 360)),
                observation_width=int(adapter.get("observation_width", 360)),
            )
            try:
                task = str(restored.loaded.metadata.instruction)
                obs0 = observation_from_libero_env(restored.handle.vector_env.envs[0])
                student_cont = InProcessSmolVLAContinuation(
                    bundle,
                    temperature=float(adapter.get("continuation_temperature", 0.5)),
                    seed=2026080405,
                )
                student_action = _predict_student_env_action(
                    student_cont, obs0, task=task
                )
            finally:
                restored.close()

            # B. student-repeat
            succ_b = []
            for _ in range(2):
                restored = restore_pool_state(
                    pool,
                    state_key,
                    libero_plus_root=adapter.get("libero_plus_root"),
                    observation_height=int(adapter.get("observation_height", 360)),
                    observation_width=int(adapter.get("observation_width", 360)),
                )
                try:
                    obs_next, _ = _step_once(restored, student_action)
                    succ_b.append(obs_next)
                finally:
                    restored.close()
            student_repeat = successor_distance(succ_b[0], succ_b[1])
            record["student_repeat"] = student_repeat

            if client is None:
                record["skipped_oft"] = True
                results.append(record)
                print(f"SUCC_STUDENT_ONLY state={state_key}", flush=True)
                continue

            # Teacher action from same restore obs
            restored = restore_pool_state(
                pool,
                state_key,
                libero_plus_root=adapter.get("libero_plus_root"),
                observation_height=int(adapter.get("observation_height", 360)),
                observation_width=int(adapter.get("observation_width", 360)),
            )
            try:
                task = str(restored.loaded.metadata.instruction)
                teacher_action = _predict_oft_env_action(
                    client, restored, instruction=task
                )
            finally:
                restored.close()

            # A. sim-floor: same OFT action twice
            succ_a = []
            for _ in range(2):
                restored = restore_pool_state(
                    pool,
                    state_key,
                    libero_plus_root=adapter.get("libero_plus_root"),
                    observation_height=int(adapter.get("observation_height", 360)),
                    observation_width=int(adapter.get("observation_width", 360)),
                )
                try:
                    obs_next, _ = _step_once(restored, teacher_action)
                    succ_a.append(obs_next)
                finally:
                    restored.close()
            sim_floor = successor_distance(succ_a[0], succ_a[1])
            record["sim_floor"] = sim_floor

            # C. cross: OFT vs student successors
            restored = restore_pool_state(
                pool,
                state_key,
                libero_plus_root=adapter.get("libero_plus_root"),
                observation_height=int(adapter.get("observation_height", 360)),
                observation_width=int(adapter.get("observation_width", 360)),
            )
            try:
                obs_oft, _ = _step_once(restored, teacher_action)
            finally:
                restored.close()
            restored = restore_pool_state(
                pool,
                state_key,
                libero_plus_root=adapter.get("libero_plus_root"),
                observation_height=int(adapter.get("observation_height", 360)),
                observation_width=int(adapter.get("observation_width", 360)),
            )
            try:
                obs_stu, _ = _step_once(restored, student_action)
            finally:
                restored.close()
            cross = successor_distance(obs_oft, obs_stu)
            record["cross"] = cross
            record["actions"] = action_space_report(
                student_normalized=None,
                teacher_normalized=None,
                student_denormalized=None,
                teacher_denormalized=None,
                student_env=student_action,
                teacher_env=teacher_action,
            )
            decision = interface_mismatch_decision(
                env_action_mae=float(record["actions"]["env_action_mae"]),
                cross_successor_error=float(cross.get("aggregate_l2") or 0.0),
                sim_floor_error=float(sim_floor.get("aggregate_l2") or 0.0),
                student_repeat_error=float(student_repeat.get("aggregate_l2") or 0.0),
                cross_over_sim_floor_ratio=float(
                    succ_cfg.get("cross_over_sim_floor_ratio", 5.0)
                ),
            )
            record["interface_decision"] = decision
            results.append(record)
            print(
                f"SUCC state={state_key} env_mae={decision['env_action_mae']:.4f} "
                f"cross/floor={decision['cross_over_sim_floor_ratio']:.2f} "
                f"mismatch={decision['interface_mismatch']}",
                flush=True,
            )
    finally:
        if client is not None:
            client.close()

    if args.merge_into and args.merge_into.is_file():
        prev = json.loads(args.merge_into.read_text(encoding="utf-8"))
        prev_results = list(prev.get("results") or [])
        by_key = {str(r["state_key"]): r for r in prev_results}
        for row in results:
            by_key[str(row["state_key"])] = row
        results = list(by_key.values())

    mismatches = [
        r for r in results if bool((r.get("interface_decision") or {}).get("interface_mismatch"))
    ]
    payload = {
        "schema_version": "rase-pre-c1-2-successor/v1",
        "n_states": len(results),
        "n_interface_mismatch": len(mismatches),
        "block_training": bool(mismatches) and not args.skip_oft,
        "decision": "fix_interface" if mismatches and not args.skip_oft else "proceed",
        "adapter_dir": str(args.adapter_dir),
        "results": results,
        "not_runtime_oft": True,
    }
    _write(args.output.resolve(), payload)
    if args.merge_into:
        _write(args.merge_into.resolve(), payload)
    print(json.dumps({k: payload[k] for k in payload if k != "results"}, sort_keys=True))
    print(f"PRE_C1_2_SUCCESSOR_DONE output={args.output} decision={payload['decision']}", flush=True)
    return 1 if payload["block_training"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
