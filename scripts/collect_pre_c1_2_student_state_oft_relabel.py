#!/usr/bin/env python3
"""PRE-C1.2 Phase 2: student-query OFT relabel (forked teacher, query vs suffix)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rase.adapt.pre_c1_2 import dagger_qc_report, file_sha256, load_protocol_lock
from rase.adapt.pre_c1_2_eval import load_pre_c0_failure_keys
from rase.adapt.recovery_lora import load_lora_onto_policy, set_adapter_enabled
from rase.collect.forked_rollout import (
    RolloutConfig,
    load_smolvla_policy_bundle,
    restore_pool_state,
)
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.policy_step import as_batched_action, current_timestep, success_from_info
from rase.collect.same_policy_corrective import RecedingHorizonSmolVLAContinuation
from rase.collect.state_pool import StatePool


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _pack_observation(observation: dict[str, Any], task: str) -> dict[str, Any]:
    pixels = observation.get("pixels") or {}
    packed: dict[str, Any] = {"task": str(task)}

    def _squeeze(value: Any) -> Any:
        array = np.asarray(value)
        while array.ndim > 0 and array.shape[0] == 1:
            array = array[0]
        return array

    if isinstance(pixels, dict):
        if "image" in pixels:
            packed["pixels_image"] = np.asarray(_squeeze(pixels["image"]), dtype=np.uint8)
        if "image2" in pixels:
            packed["pixels_image2"] = np.asarray(_squeeze(pixels["image2"]), dtype=np.uint8)
    robot_state = observation.get("robot_state")
    if isinstance(robot_state, dict):

        def _walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    _walk(value, f"{prefix}{key}.")
            else:
                packed[f"rs_{prefix[:-1]}"] = np.asarray(_squeeze(node))

        _walk(robot_state, "")
    if observation.get("agent_pos") is not None:
        packed["agent_pos"] = np.asarray(_squeeze(observation["agent_pos"]), dtype=np.float32)
    return packed


class MultiChunkOracleRecorder:
    """OFT recorder that keeps only query + max_suffix chunks for training."""

    def __init__(
        self,
        client: Any,
        *,
        instruction: str,
        max_steps: int,
        chunk_dir: Path,
        max_suffix_chunks: int = 2,
    ) -> None:
        from collections import deque

        from rase.collect.oracle_continuation import OracleChunkContinuation

        self.inner = OracleChunkContinuation(client, instruction=instruction)
        self.max_steps = int(max_steps)
        self.chunk_dir = Path(chunk_dir)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.max_kept = 1 + int(max_suffix_chunks)
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

        if len(self.actions) >= self.max_steps:
            return np.zeros(7, dtype=np.float32)
        if not self._queue:
            if self.inner.control_env is None or self._libero_env is None:
                raise RuntimeError("OFT recorder missing env bindings")
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
            chunk = np.asarray(actions[0], dtype=np.float32)
            chunk_index = len(self.chunk_records)
            chunk_path = self.chunk_dir / f"chunk_{chunk_index:04d}.npz"
            if chunk_index < self.max_kept:
                np.savez_compressed(
                    chunk_path,
                    oft_action_chunk=chunk,
                    timestep=np.asarray(timestep, dtype=np.int32),
                    **{k: v for k, v in packed.items() if k != "task"},
                    task=np.asarray(packed["task"]),
                )
                source = (
                    "student_query_state"
                    if chunk_index == 0
                    else "teacher_suffix_after_student_query"
                )
                self.chunk_records.append(
                    {
                        "chunk_index": chunk_index,
                        "offset_from_student_state": chunk_index,
                        "source": source,
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
    parser.add_argument("--round-id", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rase_pre_c1_2_dagger_r1_v1"))
    parser.add_argument("--limit-anchors", type=int, default=0)
    parser.add_argument("--suite", default=None, help="Optional suite filter")
    parser.add_argument(
        "--state-keys-json",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json"),
    )
    parser.add_argument("--seeds-per-anchor", type=int, default=None)
    parser.add_argument("--max-student-steps", type=int, default=80)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    if not bool(lock.get("sealed", {}).get("selected_horizon_frozen")):
        print(
            "WARN: selected_horizon not frozen yet; using fallback_horizon",
            flush=True,
        )
    h = int(
        lock["evaluation"]["recovery"].get("selected_horizon")
        or lock["horizon_selection"]["fallback_horizon"]
    )
    dagger = dict(lock["dagger"])
    betas = list(dagger["beta_by_round"])
    beta = float(betas[min(max(args.round_id - 1, 0), len(betas) - 1)])
    period = int(dagger["query_triggers"]["periodic_every_student_steps"])
    max_suffix = int(lock["dagger_sources"]["max_suffix_chunks_per_query"])
    seeds_n = int(args.seeds_per_anchor or lock["dagger_round_1_minimum"]["seeds_per_anchor"])

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    adapter = dict(cfg.get("adapter_config") or {})
    pool = StatePool(Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve())
    failures = load_pre_c0_failure_keys(args.failure_rollout_dir.resolve())
    if args.state_keys_json and args.state_keys_json.is_file():
        allowed = set(json.loads(args.state_keys_json.read_text(encoding="utf-8"))["state_keys"])
        failures = [row for row in failures if str(row["state_key"]) in allowed]
    if args.suite:
        failures = [row for row in failures if str(row.get("suite")) == str(args.suite)]
    if args.limit_anchors:
        failures = failures[: args.limit_anchors]

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.oracle.client import OracleClient

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
    adapter_hash = file_sha256(Path(args.adapter_dir) / "adapter_model.safetensors")
    policy_cfg_hash = hashlib.sha256(
        json.dumps(
            {
                "n_action_steps": adapter.get("n_action_steps", 10),
                "num_steps": adapter.get("num_steps", 10),
                "execution_horizon": h,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    client = OracleClient(args.endpoint)
    all_rows: list[dict[str, Any]] = []
    rollout_cfg = RolloutConfig(
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        num_steps=int(adapter.get("num_steps", 10)),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
        continuation_temperature=float(adapter.get("continuation_temperature", 0.5)),
    )

    try:
        for failure in failures:
            state_key = str(failure["state_key"])
            for seed_i in range(seeds_n):
                seed = 2026080405 + 1000 * args.round_id + seed_i
                run_id = f"{state_key}__r{args.round_id}__s{seed_i}"
                out_json = output_dir / f"{run_id}.json"
                if args.resume and out_json.exists():
                    payload = json.loads(out_json.read_text(encoding="utf-8"))
                    all_rows.extend(payload.get("accepted_rows") or [])
                    continue
                rng = random.Random(seed)
                student = RecedingHorizonSmolVLAContinuation(
                    bundle,
                    execution_horizon=h,
                    temperature=float(adapter.get("continuation_temperature", 0.5)),
                    seed=seed,
                )
                restored = restore_pool_state(
                    pool,
                    state_key,
                    libero_plus_root=adapter.get("libero_plus_root"),
                    observation_height=rollout_cfg.observation_height,
                    observation_width=rollout_cfg.observation_width,
                )
                accepted: list[dict[str, Any]] = []
                queries = 0
                successes = 0
                try:
                    task = str(restored.loaded.metadata.instruction)
                    student.reset()
                    if hasattr(student, "continuation") and hasattr(
                        student.continuation, "bind_control_env"
                    ):
                        pass
                    vector_env = restored.handle.vector_env
                    single = vector_env.envs[0]
                    student_steps = 0
                    progress_hist: list[float] = []
                    while student_steps < int(args.max_student_steps):
                        # β mixing at replan boundary only.
                        at_boundary = student_steps % h == 0
                        use_teacher_chunk = at_boundary and rng.random() < beta
                        trigger = None
                        if student_steps == 0:
                            trigger = "anchor_start"
                        elif student_steps % period == 0:
                            trigger = "periodic"
                        if len(progress_hist) >= 4 and all(
                            abs(progress_hist[-1] - p) < 1e-6 for p in progress_hist[-4:]
                        ):
                            trigger = trigger or "progress_stall"

                        if trigger is not None:
                            queries += 1
                            live_snap = restored.forkable.snapshot()
                            query_id = f"{run_id}__q{queries}"
                            query_dir = output_dir / "chunks" / query_id
                            # Forked teacher query: independent env, same mid-state.
                            teacher_env = restore_pool_state(
                                pool,
                                state_key,
                                libero_plus_root=adapter.get("libero_plus_root"),
                                observation_height=rollout_cfg.observation_height,
                                observation_width=rollout_cfg.observation_width,
                            )
                            try:
                                # Forked query: restore the *student mid-rollout*
                                # snapshot. Do not call evaluate_candidate (it
                                # would re-restore the original pool snapshot).
                                teacher_env.forkable.restore(
                                    live_snap, check_task_fingerprint=True
                                )
                                now_t = current_timestep(teacher_env.handle.control_env)
                                episode_max = int(
                                    getattr(
                                        teacher_env.handle.vector_env.envs[0],
                                        "_max_episode_steps",
                                        600,
                                    )
                                )
                                max_steps = max(128, max(0, episode_max - int(now_t)))
                                recorder = MultiChunkOracleRecorder(
                                    client,
                                    instruction=task,
                                    max_steps=max_steps,
                                    chunk_dir=query_dir,
                                    max_suffix_chunks=max_suffix,
                                )
                                recorder.bind_control_env(teacher_env.handle.control_env)
                                recorder.bind_libero_env(
                                    teacher_env.handle.vector_env.envs[0]
                                )
                                recorder.reset()
                                ok = False
                                for _step_i in range(max_steps):
                                    obs_t = observation_from_libero_env(
                                        teacher_env.handle.vector_env.envs[0]
                                    )
                                    action_t = recorder.act(obs_t, task=task)
                                    _o, _r, term, trunc, info = (
                                        teacher_env.handle.vector_env.step(
                                            as_batched_action(action_t)
                                        )
                                    )
                                    if success_from_info(info):
                                        ok = True
                                        break
                                    if bool(np.asarray(term).reshape(-1)[0]) or bool(
                                        np.asarray(trunc).reshape(-1)[0]
                                    ):
                                        break
                                result_env_steps = int(len(recorder.actions))
                                if ok:
                                    successes += 1
                                    for chunk_rec in recorder.chunk_records:
                                        row = {
                                            "schema_version": "rase-pre-c1-2-dagger-row/v1",
                                            "anchor_id": state_key,
                                            "failure_key": state_key,
                                            "episode_id": failure.get("episode_id"),
                                            "suite": failure.get("suite"),
                                            "stage": failure.get("stage"),
                                            "round_id": int(args.round_id),
                                            "seed": int(seed),
                                            "query_id": query_id,
                                            "query_state_id": f"{query_id}_state",
                                            "query_trigger": trigger,
                                            "source": chunk_rec["source"],
                                            "offset_from_student_state": chunk_rec[
                                                "offset_from_student_state"
                                            ],
                                            "chunk_path": chunk_rec["chunk_path"],
                                            "chunk_index": chunk_rec["chunk_index"],
                                            "teacher_rollout_success": True,
                                            "teacher_recovery_length": int(len(recorder.actions)),
                                            "execution_horizon": h,
                                            "temperature": float(
                                                adapter.get("continuation_temperature", 0.5)
                                            ),
                                            "flow_noise_seed": int(seed),
                                            "env_seed": int(seed),
                                            "student_checkpoint": str(
                                                adapter.get("policy_path") or "ckpts/smolvla_libero"
                                            ),
                                            "adapter_hash": adapter_hash,
                                            "policy_config_hash": policy_cfg_hash,
                                            "clean_flag": False,
                                            "sample_id": f"{query_id}__c{chunk_rec['chunk_index']}",
                                            "state_key": state_key,
                                            "not_runtime_oft": True,
                                            "teacher_query_mode": "forked_environment",
                                        }
                                        accepted.append(row)
                                        all_rows.append(row)
                                else:
                                    fail_path = output_dir / "failed_teacher" / f"{query_id}.json"
                                    _atomic_json(
                                        fail_path,
                                        {
                                            "query_id": query_id,
                                            "anchor_id": state_key,
                                            "success": False,
                                            "env_steps": int(result_env_steps),
                                            "enter_bc": False,
                                        },
                                    )
                            finally:
                                teacher_env.close()

                        # Student (or teacher-mixed) execution for H steps.
                        if use_teacher_chunk and client is not None:
                            # Execute H OFT actions without mutating label store.
                            from rase.collect.oracle_continuation import OracleChunkContinuation

                            oft = OracleChunkContinuation(client, instruction=task)
                            oft.bind_control_env(restored.handle.control_env)
                            oft.reset()
                            actor = oft
                        else:
                            actor = student

                        for _ in range(h):
                            obs = observation_from_libero_env(single)
                            action = actor.act(obs, task=task)
                            _obs, _r, term, trunc, info = vector_env.step(
                                as_batched_action(action)
                            )
                            student_steps += 1
                            progress_hist.append(1.0 if success_from_info(info) else 0.0)
                            if bool(np.asarray(term).reshape(-1)[0]) or bool(
                                np.asarray(trunc).reshape(-1)[0]
                            ):
                                student_steps = int(args.max_student_steps)
                                break
                finally:
                    restored.close()

                payload = {
                    "schema_version": "rase-pre-c1-2-dagger-run/v1",
                    "run_id": run_id,
                    "anchor_id": state_key,
                    "round_id": args.round_id,
                    "beta": beta,
                    "beta_unit": "replan_boundary",
                    "execution_horizon": h,
                    "n_queries": queries,
                    "n_successful_teacher": successes,
                    "accepted_rows": accepted,
                }
                _atomic_json(out_json, payload)
                print(
                    f"DAGGER run={run_id} queries={queries} success={successes} "
                    f"rows={len(accepted)}",
                    flush=True,
                )
    finally:
        client.close()

    qc = dagger_qc_report(all_rows)
    mins = dict(lock["dagger_round_1_minimum"])
    qc["meets_round1_minimum"] = (
        qc["anchors_covered"] >= len(failures)
        and qc["unique_student_query_states"]
        >= int(mins["unique_student_query_states_per_anchor"]) * max(1, len(failures) // 2)
    )
    _atomic_json(output_dir / "dagger_qc.json", qc)
    print(json.dumps(qc, sort_keys=True))
    print(f"PRE_C1_2_DAGGER_DONE output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
