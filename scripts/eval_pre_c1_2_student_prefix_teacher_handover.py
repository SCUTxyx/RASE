#!/usr/bin/env python3
"""R0-B/C: student prefix k steps → OFT teacher handover recoverability grid."""

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

from rase.adapt.pre_c1_2 import load_protocol_lock
from rase.adapt.pre_c1_2_eval import load_pre_c0_failure_keys, successor_distance
from rase.adapt.recovery_lora import load_lora_onto_policy, set_adapter_enabled
from rase.collect.forked_rollout import (
    RolloutConfig,
    load_smolvla_policy_bundle,
    restore_pool_state,
)
from rase.collect.oracle_continuation import OracleChunkContinuation
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.policy_step import as_batched_action, current_timestep, success_from_info
from rase.collect.same_policy_corrective import RecedingHorizonSmolVLAContinuation
from rase.collect.state_pool import StatePool
from rase.oracle.client import OracleClient


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _episode_max(single: Any) -> int:
    return int(getattr(single, "_max_episode_steps", 600))


def _run_teacher_from_live(
    *,
    client: OracleClient,
    restored: Any,
    task: str,
    max_steps: int,
) -> dict[str, Any]:
    cont = OracleChunkContinuation(client, instruction=task)
    cont.bind_control_env(restored.handle.control_env)
    cont.reset()  # fresh-forward / clear queue at handover
    vector_env = restored.handle.vector_env
    single = vector_env.envs[0]
    ok = False
    steps = 0
    stop_reason = "horizon"
    for _ in range(max(0, int(max_steps))):
        obs = observation_from_libero_env(single)
        action = cont.act(obs, task=task)
        _o, _r, term, trunc, info = vector_env.step(as_batched_action(action))
        steps += 1
        if success_from_info(info):
            ok = True
            stop_reason = "success"
            break
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            stop_reason = "terminated" if bool(np.asarray(term).reshape(-1)[0]) else "truncated"
            break
    return {
        "teacher_success": bool(ok),
        "teacher_env_steps_after_handover": int(steps),
        "stop_reason": stop_reason,
    }


def _execute_student_prefix(
    *,
    restored: Any,
    continuation: Any,
    task: str,
    k: int,
) -> dict[str, Any]:
    vector_env = restored.handle.vector_env
    single = vector_env.envs[0]
    if hasattr(continuation, "bind_control_env"):
        continuation.bind_control_env(restored.handle.control_env)
    continuation.reset()
    obs0 = observation_from_libero_env(single)
    first_action = None
    obs_after_one = None
    completed = 0
    early_success = False
    early_fail = False
    for step_i in range(int(k)):
        obs = observation_from_libero_env(single)
        action = np.asarray(continuation.act(obs, task=task), dtype=np.float32).reshape(-1)
        if step_i == 0:
            first_action = action.copy()
        _o, _r, term, trunc, info = vector_env.step(as_batched_action(action))
        completed += 1
        if step_i == 0:
            obs_after_one = observation_from_libero_env(single)
        if success_from_info(info):
            early_success = True
            break
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            early_fail = True
            break
    succ = None
    if obs0 is not None and obs_after_one is not None:
        try:
            succ = successor_distance(obs0, obs_after_one)
        except Exception:  # noqa: BLE001
            succ = None
    return {
        "student_prefix_completed": int(completed),
        "student_prefix_requested": int(k),
        "student_prefix_early_success": bool(early_success),
        "student_prefix_early_failure": bool(early_fail),
        "first_action": None if first_action is None else first_action.astype(float).tolist(),
        "one_step_successor": succ,
    }


def _trial_path(out_dir: Path, state_key: str, arm: str, k: int, seed: int) -> Path:
    safe = state_key.replace("/", "_")[:48]
    return out_dir / "trials" / f"{safe}__{arm}__k{k}__s{seed}.json"


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
        "--state-keys-json",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json"),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("runs/rase_pre_c1_1_lora_train_v1/adapter_final"),
    )
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--suite", default=None, help="Spatial|Object|Goal|Long; default all present")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/rase_pre_c1_2_r0_recoverability_v1"),
    )
    parser.add_argument("--ks", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="1 anchor/suite and ks={0,1,4,16}")
    parser.add_argument("--limit-anchors", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    r0 = dict(lock.get("r0") or {})
    ks = list(args.ks or r0.get("k_env_steps") or [0, 1, 2, 4, 8, 16])
    if args.smoke:
        ks = [0, 1, 4, 16]
    seed = int(args.seed if args.seed is not None else r0.get("seed", 2026080405))
    h = int(lock["evaluation"]["recovery"].get("selected_horizon") or 2)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    adapter = dict(cfg.get("adapter_config") or {})
    pool = StatePool(Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve())
    failures = load_pre_c0_failure_keys(args.failure_rollout_dir.resolve())
    allowed = set(json.loads(args.state_keys_json.read_text(encoding="utf-8"))["state_keys"])
    failures = [row for row in failures if str(row["state_key"]) in allowed]
    if args.suite:
        failures = [row for row in failures if str(row.get("suite")) == args.suite]
    if args.smoke:
        # one per suite present
        seen = set()
        picked = []
        for row in failures:
            suite = str(row.get("suite"))
            if suite in seen:
                continue
            seen.add(suite)
            picked.append(row)
        failures = picked
    if args.limit_anchors:
        failures = failures[: args.limit_anchors]
    if not failures:
        raise SystemExit("no failure anchors selected")

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
    bundle["policy"] = handle.policy
    rollout_cfg = RolloutConfig(
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        num_steps=int(adapter.get("num_steps", 10)),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
        continuation_temperature=float(adapter.get("continuation_temperature", 0.5)),
    )

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OracleClient(args.endpoint)
    trials: list[dict[str, Any]] = []

    try:
        for failure in failures:
            state_key = str(failure["state_key"])
            task = str(failure.get("instruction") or "")
            # k=0 shared OFT baseline once
            cells: list[tuple[str, int]] = [("oft_k0", 0)]
            for k in ks:
                if int(k) == 0:
                    continue
                cells.append(("base", int(k)))
                cells.append(("adapted", int(k)))
            # OFT one-step replan control
            cells.append(("oft_replan_k1", 1))

            for arm, k in cells:
                path = _trial_path(out_dir, state_key, arm, k, seed)
                if args.resume and path.is_file():
                    trials.append(json.loads(path.read_text(encoding="utf-8")))
                    continue

                restored = restore_pool_state(
                    pool,
                    state_key,
                    libero_plus_root=adapter.get("libero_plus_root"),
                    observation_height=rollout_cfg.observation_height,
                    observation_width=rollout_cfg.observation_width,
                )
                try:
                    single = restored.handle.vector_env.envs[0]
                    if not task:
                        task = str(getattr(single, "task_description", "") or "")
                    # Restore original snapshot explicitly for independence.
                    restored.forkable.restore(
                        restored.snapshot, check_task_fingerprint=True
                    )
                    t0 = current_timestep(restored.handle.control_env)
                    episode_max = _episode_max(single)
                    prefix_metrics: dict[str, Any] = {
                        "student_prefix_completed": 0,
                        "student_prefix_requested": int(k),
                        "student_prefix_early_success": False,
                        "student_prefix_early_failure": False,
                        "first_action": None,
                        "one_step_successor": None,
                    }

                    if arm == "oft_k0":
                        # Direct teacher from anchor; no student steps.
                        remaining = max(0, episode_max - int(t0))
                        teacher = _run_teacher_from_live(
                            client=client,
                            restored=restored,
                            task=task,
                            max_steps=remaining,
                        )
                    elif arm == "oft_replan_k1":
                        # OFT executes 1 step, then fresh OFT replan continuation.
                        oft = OracleChunkContinuation(client, instruction=task)
                        oft.bind_control_env(restored.handle.control_env)
                        oft.reset()
                        obs = observation_from_libero_env(single)
                        action = np.asarray(oft.act(obs, task=task), dtype=np.float32).reshape(-1)
                        _o, _r, term, trunc, info = restored.handle.vector_env.step(
                            as_batched_action(action)
                        )
                        prefix_metrics["student_prefix_completed"] = 1
                        prefix_metrics["first_action"] = action.astype(float).tolist()
                        if success_from_info(info):
                            teacher = {
                                "teacher_success": True,
                                "teacher_env_steps_after_handover": 0,
                                "stop_reason": "success_during_prefix",
                            }
                        elif bool(np.asarray(term).reshape(-1)[0]) or bool(
                            np.asarray(trunc).reshape(-1)[0]
                        ):
                            teacher = {
                                "teacher_success": False,
                                "teacher_env_steps_after_handover": 0,
                                "stop_reason": "prefix_terminated",
                            }
                        else:
                            now_t = current_timestep(restored.handle.control_env)
                            remaining = max(0, episode_max - int(now_t))
                            teacher = _run_teacher_from_live(
                                client=client,
                                restored=restored,
                                task=task,
                                max_steps=remaining,
                            )
                    else:
                        set_adapter_enabled(handle, arm == "adapted")
                        bundle["policy"] = handle.policy
                        student = RecedingHorizonSmolVLAContinuation(
                            bundle,
                            execution_horizon=h,
                            temperature=float(adapter.get("continuation_temperature", 0.5)),
                            seed=seed,
                        )
                        prefix_metrics = _execute_student_prefix(
                            restored=restored,
                            continuation=student,
                            task=task,
                            k=k,
                        )
                        if prefix_metrics["student_prefix_early_success"]:
                            teacher = {
                                "teacher_success": True,
                                "teacher_env_steps_after_handover": 0,
                                "stop_reason": "success_during_prefix",
                            }
                        elif prefix_metrics["student_prefix_early_failure"]:
                            teacher = {
                                "teacher_success": False,
                                "teacher_env_steps_after_handover": 0,
                                "stop_reason": "prefix_terminated",
                            }
                        else:
                            now_t = current_timestep(restored.handle.control_env)
                            remaining = max(0, episode_max - int(now_t))
                            teacher = _run_teacher_from_live(
                                client=client,
                                restored=restored,
                                task=task,
                                max_steps=remaining,
                            )

                    trial = {
                        "schema_version": "rase-pre-c1-2-r0-handover-trial/v1",
                        "state_key": state_key,
                        "suite": failure.get("suite"),
                        "stage": failure.get("stage"),
                        "episode_id": failure.get("episode_id"),
                        "arm": arm,
                        "k": int(k),
                        "seed": int(seed),
                        "execution_horizon": h,
                        "handover_timestep": int(
                            current_timestep(restored.handle.control_env)
                        ),
                        "episode_max_steps": episode_max,
                        "remaining_budget_enforced": True,
                        "independent_restore": True,
                        "fresh_forward_on_handover": True,
                        **prefix_metrics,
                        **teacher,
                        "total_env_steps": int(
                            prefix_metrics.get("student_prefix_completed", 0)
                            + teacher.get("teacher_env_steps_after_handover", 0)
                        ),
                        "adapter_dir": str(args.adapter_dir),
                        "not_runtime_oft_in_deploy": True,
                    }
                finally:
                    restored.close()

                _atomic_json(path, trial)
                trials.append(trial)
                print(
                    f"R0 trial state={state_key[:12]} arm={arm} k={k} "
                    f"success={trial['teacher_success']} prefix={trial['student_prefix_completed']}",
                    flush=True,
                )
    finally:
        client.close()

    # Re-aggregate from ALL trial files on disk so suite-serial runs don't
    # overwrite summary.json with suite-only results.
    disk_trials: list[dict[str, Any]] = []
    trials_dir = out_dir / "trials"
    if trials_dir.is_dir():
        for path in sorted(trials_dir.glob("*.json")):
            disk_trials.append(json.loads(path.read_text(encoding="utf-8")))
    if not disk_trials:
        disk_trials = trials

    def _rate(arm_name: str, k_val: int) -> float:
        subset = [
            t
            for t in disk_trials
            if t.get("arm") == arm_name and int(t.get("k", -1)) == int(k_val)
        ]
        if not subset:
            return float("nan")
        return float(np.mean([1.0 if t.get("teacher_success") else 0.0 for t in subset]))

    # Discover k values actually present, preferring the requested grid.
    present_ks = sorted({int(t.get("k", 0)) for t in disk_trials})
    use_ks = sorted(set(int(k) for k in ks) | set(present_ks))

    curves = {
        "oft_k0": _rate("oft_k0", 0),
        "oft_replan_k1": _rate("oft_replan_k1", 1),
        "base": {int(k): _rate("base", k) for k in use_ks if int(k) > 0},
        "adapted": {int(k): _rate("adapted", k) for k in use_ks if int(k) > 0},
    }
    # Include k=0 for student arms as shared OFT baseline.
    curves["base"][0] = curves["oft_k0"]
    curves["adapted"][0] = curves["oft_k0"]

    summary = {
        "schema_version": "rase-pre-c1-2-r0-recoverability/v1",
        "n_anchors": len({t["state_key"] for t in disk_trials}),
        "n_trials": len(disk_trials),
        "ks": use_ks,
        "seed": seed,
        "curves": curves,
        "R_oft_k0": curves["oft_k0"],
        "R_oft_replan_1": curves["oft_replan_k1"],
        "coverage": {
            "anchors": sorted({t["state_key"] for t in disk_trials}),
            "arms": sorted({t["arm"] for t in disk_trials}),
        },
        "protocol_revision": dict(lock.get("revision") or {}),
        "suite_filter": args.suite,
    }
    # Adjacent decay
    decay = {}
    for arm_name in ("base", "adapted"):
        series = curves[arm_name]
        ordered = sorted(series)
        decay[arm_name] = {
            f"{ordered[i-1]}->{ordered[i]}": float(series[ordered[i]] - series[ordered[i - 1]])
            for i in range(1, len(ordered))
            if ordered[i] in series and ordered[i - 1] in series
        }
    summary["adjacent_delta"] = decay
    # AUC-like mean over available k
    for arm_name in ("base", "adapted"):
        vals = [float(v) for v in curves[arm_name].values() if v == v]
        summary[f"auc_like_{arm_name}"] = float(np.mean(vals)) if vals else None

    _atomic_json(out_dir / "summary.json", summary)
    rows_path = out_dir / "trials.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle_out:
        for trial in disk_trials:
            handle_out.write(json.dumps(trial, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    print(f"PRE_C1_2_R0_RECOVERABILITY_DONE output={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
