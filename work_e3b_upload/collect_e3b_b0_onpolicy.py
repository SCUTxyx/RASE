#!/usr/bin/env python3
"""Collect B0 one-shot/persistent residual rollouts and exact-boundary OFT chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_e3_residual_dataset import language_hash  # noqa: E402
from scripts.collect_e3_step_demos import canonical_action, resize_rgb  # noqa: E402
from scripts.rollout_e3_step_residual import corrected_action, route_c_history  # noqa: E402
from scripts.train_e3_step_residual import build_features, predict  # noqa: E402


ARMS = ("source_h8", "one_shot_h8", "persistent_h8")


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


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def suite_from_task(task_id: str) -> str:
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        if task_id.startswith(suite):
            return suite
    raise ValueError(f"unknown suite for {task_id}")


def collect_chunk(policy: Any, observation: Mapping[str, Any], task: str, horizon: int) -> np.ndarray:
    actions = [canonical_action(policy.act(observation, task=task)) for _ in range(horizon)]
    result = np.asarray(actions, dtype=np.float32)
    if result.shape != (horizon, 7):
        raise ValueError(f"invalid action chunk shape {result.shape}")
    return result


def should_correct(arm: str, boundary_index: int) -> bool:
    if arm == "source_h8":
        return False
    if arm == "one_shot_h8":
        return boundary_index == 0
    if arm == "persistent_h8":
        return True
    raise ValueError(arm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--model", type=Path, help="legacy stepwise E3 residual")
    candidate.add_argument("--chunk-model", type=Path, help="E3-B task-conditioned H=8 chunk residual")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()
    if args.horizon != 8:
        raise ValueError("E3-B v1 freezes the common horizon at 8")
    if not 0.0 < args.residual_scale <= 1.0:
        raise ValueError("residual scale must be in (0,1]")

    cfg = read_json(args.config.resolve())
    key_payload = read_json(args.state_keys_json.resolve())
    adapter = dict(cfg.get("adapter_config") or {})
    keys = [str(key) for key in key_payload.get("state_keys") or []]
    pool_path = Path(cfg.get("pool") or key_payload.get("pool") or "")
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    policy_path = Path(adapter.get("policy_path") or "ckpts/smolvla_libero")
    tokenizer_path = Path(adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct")
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    if not tokenizer_path.is_absolute():
        tokenizer_path = ROOT / tokenizer_path
    model = None
    chunk_model = None
    if args.model:
        with np.load(args.model.resolve(), allow_pickle=False) as archive:
            model = {key: archive[key] for key in archive.files}
        variant = str(model["feature_variant"])
        image_size = int(model["image_size"])
        language_dim = int(model["language_dim"])
    else:
        from rase.recovery.e3b_chunk_residual import load_ensemble

        chunk_model = load_ensemble(str(args.chunk_model.resolve()), device="cuda")
        variant = "e3b_chunk"
        image_size = 24
        language_dim = 64

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import (
        InProcessSmolVLAContinuation,
        load_smolvla_policy_bundle,
        restore_pool_state,
        rollout_seed,
    )
    from rase.collect.oracle_continuation import OracleChunkContinuation, raw_libero_to_oracle_arrays
    from rase.collect.policy_step import as_batched_action, success_from_info
    from rase.collect.pool_candidates import observation_from_libero_env
    from rase.collect.state_pool import StatePool
    from rase.oracle.client import OracleClient

    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT") or adapter.get("libero_plus_root")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_path.resolve())
    selected = [key for key in keys if suite_from_task(pool.read_state(key, load_observations=False).metadata.task_id) == args.suite]
    if not selected:
        raise SystemExit(f"no roots for {args.suite}")
    bundle = load_smolvla_policy_bundle(
        policy_path.resolve(),
        device=str(adapter.get("device", "cuda")),
        num_steps=int(adapter.get("num_steps", 10)),
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=tokenizer_path.resolve(),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    client = OracleClient(args.endpoint, timeout_ms=60_000)
    info = client.model_info()
    if info.get("suite") not in {None, args.suite}:
        raise ValueError(f"teacher suite {info.get('suite')} != {args.suite}")

    output = args.output_dir.resolve()
    if args.fresh_run and output.exists():
        raise SystemExit(f"fresh output exists: {output}")
    episode_dir = output / "episodes"
    sample_dir = output / "samples"
    episode_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    try:
        for root_index, state_key in enumerate(selected):
            for arm in ARMS:
                target = episode_dir / f"{state_key}__{arm}.json"
                sample_target = sample_dir / f"{state_key}__{arm}.npz"
                if target.is_file() and (arm == "source_h8" or sample_target.is_file()):
                    row = read_json(target)
                    rows.append(row)
                    print(f"E3B_B0 skip root={root_index+1}/{len(selected)} arm={arm}", flush=True)
                    continue
                restored = restore_pool_state(
                    pool,
                    state_key,
                    libero_plus_root=libero_plus_root,
                    observation_height=int(adapter.get("observation_height", 360)),
                    observation_width=int(adapter.get("observation_width", 360)),
                )
                success = False
                stop_reason = "horizon"
                steps = 0
                correction_steps = 0
                activated_boundaries = 0
                gate_scores: list[float] = []
                boundary_index = 0
                history: list[dict[str, Any]] = []
                action_digest = hashlib.sha256()
                samples: dict[str, list[np.ndarray]] = {
                    "proprio": [], "agentview": [], "wrist": [], "history": [],
                    "source_chunk": [], "teacher_chunk": [], "executed_chunk": [],
                    "executed_mask": [], "boundary_step": [],
                }
                try:
                    vector_env = restored.handle.vector_env
                    single = vector_env.envs[0]
                    task = str(getattr(single, "task_description", "") or restored.loaded.metadata.instruction)
                    language = language_hash(task, language_dim)
                    env_horizon = min(args.max_steps, int(getattr(single, "_max_episode_steps", args.max_steps)))
                    while steps < env_horizon:
                        observation = observation_from_libero_env(single)
                        agentview, wrist, proprio = raw_libero_to_oracle_arrays(restored.handle.control_env)
                        source_policy = InProcessSmolVLAContinuation(
                            bundle,
                            temperature=float(adapter.get("continuation_temperature", 0.5)),
                            seed=rollout_seed(state_key, 0, boundary_index, salt=0x45334230),
                        )
                        source_policy.reset()
                        source_chunk = collect_chunk(source_policy, observation, task, args.horizon)
                        correction_eligible = should_correct(arm, boundary_index)
                        chunk_delta = None
                        gate_score = None
                        if chunk_model is not None:
                            from rase.recovery.e3b_chunk_residual import predict_ensemble

                            chunk_delta, gate_score = predict_ensemble(
                                chunk_model,
                                proprio=proprio,
                                source_chunk=source_chunk,
                                history=route_c_history(history, 8),
                                language_hash=language,
                                agentview=resize_rgb(agentview, image_size),
                                wrist=resize_rgb(wrist, image_size),
                            )
                            correction_active = correction_eligible and gate_score >= float(chunk_model["gate_threshold"])
                        else:
                            correction_active = correction_eligible
                        if gate_score is not None:
                            gate_scores.append(float(gate_score))
                        activated_boundaries += int(correction_active)
                        teacher_chunk = None
                        if arm != "source_h8":
                            teacher = OracleChunkContinuation(
                                client,
                                instruction=task,
                                control_env=restored.handle.control_env,
                                record_chunk_trace=True,
                            )
                            teacher_chunk = collect_chunk(teacher, observation, task, args.horizon)
                            if teacher.chunk_query_records[0]["action_chunk_shape"] != [args.horizon, 7]:
                                raise ValueError(f"teacher native chunk mismatch: {teacher.chunk_query_records}")
                            samples["proprio"].append(np.asarray(proprio, dtype=np.float32))
                            samples["agentview"].append(resize_rgb(agentview, image_size))
                            samples["wrist"].append(resize_rgb(wrist, image_size))
                            samples["history"].append(route_c_history(history, 8))
                            samples["source_chunk"].append(source_chunk.copy())
                            samples["teacher_chunk"].append(teacher_chunk.copy())
                            samples["boundary_step"].append(np.asarray(steps, dtype=np.int32))

                        executed = np.zeros((args.horizon, 7), dtype=np.float32)
                        executed_mask = np.zeros(args.horizon, dtype=np.bool_)
                        for offset, source_action in enumerate(source_chunk):
                            if steps >= env_horizon:
                                break
                            current_observation = observation_from_libero_env(single)
                            current_agentview, current_wrist, current_proprio = raw_libero_to_oracle_arrays(restored.handle.control_env)
                            if correction_active and chunk_delta is not None:
                                action = corrected_action(source_action, chunk_delta[offset], args.residual_scale)
                                correction_steps += 1
                            elif correction_active:
                                one: dict[str, np.ndarray] = {
                                    "proprio": np.asarray(current_proprio, dtype=np.float32)[None, ...],
                                    "source_action": source_action[None, ...],
                                    "language_hash": language[None, ...],
                                }
                                if variant == "state_vision":
                                    one["agentview"] = resize_rgb(current_agentview, image_size)[None, ...]
                                    one["wrist"] = resize_rgb(current_wrist, image_size)[None, ...]
                                delta = predict(model, build_features(one, variant))[0]
                                action = corrected_action(source_action, delta, args.residual_scale)
                                correction_steps += 1
                            else:
                                action = np.asarray(source_action, dtype=np.float32)
                            proprio_array = np.asarray(current_proprio, dtype=np.float32).reshape(-1)
                            history.append(
                                {
                                    "proprio": proprio_array,
                                    "source_action": source_action,
                                    "progress": float(np.linalg.norm(proprio_array[:3])),
                                    "executed_action": action,
                                }
                            )
                            action_digest.update(np.ascontiguousarray(action).tobytes())
                            executed[offset] = action
                            executed_mask[offset] = True
                            _obs, _reward, term, trunc, info_step = vector_env.step(as_batched_action(action))
                            steps += 1
                            terminated = bool(np.asarray(term).reshape(-1)[0])
                            truncated = bool(np.asarray(trunc).reshape(-1)[0])
                            if terminated or truncated:
                                success = success_from_info(info_step)
                                stop_reason = "success" if success else "terminal_failure"
                                break
                        if arm != "source_h8":
                            samples["executed_chunk"].append(executed)
                            samples["executed_mask"].append(executed_mask)
                        boundary_index += 1
                        if stop_reason != "horizon":
                            break
                finally:
                    restored.close()

                if arm != "source_h8":
                    arrays = {key: np.stack(value) for key, value in samples.items()}
                    arrays["language_hash"] = np.repeat(language[None, :], len(samples["source_chunk"]), axis=0)
                    write_npz(sample_target, **arrays)
                row = {
                    "schema_version": "rase-e3b-b0-rollout/v1",
                    "state_key": state_key,
                    "task_id": restored.loaded.metadata.task_id,
                    "suite": restored.loaded.metadata.suite,
                    "arm": arm,
                    "success": success,
                    "steps": steps,
                    "correction_steps": correction_steps,
                    "activated_boundaries": activated_boundaries,
                    "gate_score_mean": None if not gate_scores else float(np.mean(gate_scores)),
                    "gate_score_min": None if not gate_scores else float(np.min(gate_scores)),
                    "gate_score_max": None if not gate_scores else float(np.max(gate_scores)),
                    "boundaries": boundary_index,
                    "teacher_samples": 0 if arm == "source_h8" else len(samples["source_chunk"]),
                    "teacher_native_chunk_shape": None if arm == "source_h8" else [args.horizon, 7],
                    "common_horizon": args.horizon,
                    "smolvla_generated_horizon": int(adapter.get("n_action_steps", 10)),
                    "smolvla_tail_policy": "discard_after_index_7_and_requery",
                    "residual_scale": args.residual_scale,
                    "candidate_mode": "e3b_chunk" if chunk_model is not None else "legacy_stepwise",
                    "stop_reason": stop_reason,
                    "action_trace_sha256": action_digest.hexdigest(),
                    "sample_artifact": None if arm == "source_h8" else str(sample_target),
                }
                write_json(target, row)
                rows.append(row)
                print(
                    f"E3B_B0 root={root_index+1}/{len(selected)} arm={arm} "
                    f"success={success} steps={steps} corrections={correction_steps} boundaries={boundary_index}",
                    flush=True,
                )
    finally:
        client.close()

    counts = Counter((row["arm"], bool(row["success"])) for row in rows)
    summary = {
        "schema_version": "rase-e3b-b0-suite-summary/v1",
        "status": "complete",
        "suite": args.suite,
        "n_roots": len(selected),
        "arms": list(ARMS),
        "common_horizon": args.horizon,
        "success": {
            arm: {"hits": counts[(arm, True)], "trials": len(selected)} for arm in ARMS
        },
        "correction_steps": sum(int(row["correction_steps"]) for row in rows),
        "teacher_samples": sum(int(row["teacher_samples"]) for row in rows),
        "teacher_native_chunk_shapes": sorted(
            {tuple(row["teacher_native_chunk_shape"]) for row in rows if row["teacher_native_chunk_shape"]}
        ),
        "elapsed_wall_s": time.perf_counter() - started,
        "per_state_arm": rows,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary["success"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
