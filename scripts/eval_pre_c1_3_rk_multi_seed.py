#!/usr/bin/env python3
"""PRE-C1.3: Two-layer R(k) evaluation with same-trajectory snapshots and multi-seed OFT.

For each anchor, runs a single student prefix trajectory. At each k in {1,2,4,8,16},
saves an exact simulator snapshot from that trajectory. Then for each saved state:

  R_teacher(k) = P(OFT succeeds | s_k)   — multi-seed OFT continuation
  R_self(k)    = P(same arm succeeds | s_k) — same-arm continuation

Outputs long-format per-row trial JSONL + aggregate summary with cluster-bootstrap CIs.
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

from rase.adapt.pre_c1_2_eval import load_pre_c0_failure_keys
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
    client: OracleClient, restored: Any, task: str, max_steps: int,
) -> dict[str, Any]:
    cont = OracleChunkContinuation(client, instruction=task)
    cont.bind_control_env(restored.handle.control_env)
    cont.reset()
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


def _run_self_continuation(
    restored: Any, continuation: Any, task: str, max_steps: int,
) -> dict[str, Any]:
    """Run same-arm student continuation from the current state."""
    vector_env = restored.handle.vector_env
    single = vector_env.envs[0]
    cont = continuation
    if hasattr(cont, "bind_control_env"):
        cont.bind_control_env(restored.handle.control_env)
    cont.reset()
    ok = False
    steps = 0
    stop_reason = "horizon"
    for _ in range(max(0, int(max_steps))):
        obs = observation_from_libero_env(single)
        action = np.asarray(cont.act(obs, task=task), dtype=np.float32).reshape(-1)
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
        "self_success": bool(ok),
        "self_env_steps": int(steps),
        "stop_reason": stop_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/collect_pre_c0_deviation_pilot24.json"))
    parser.add_argument("--failure-rollout-dir", type=Path,
                        default=Path("runs/rase_pre_c0_same_policy_pilot48_v1"))
    parser.add_argument("--state-keys-json", type=Path,
                        default=Path("artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json"))
    parser.add_argument("--adapter-dir", type=Path, required=True,
                        help="Path to adapter_final for this arm/seed.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", type=str, required=True,
                        help="Arm label (e.g., arm_a, arm_ap, arm_b, arm_c).")
    parser.add_argument("--training-seed", type=int, default=0,
                        help="Training seed index (0-based).")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--suite", default=None,
                        help="Spatial|Object|Goal|Long; default all present")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--teacher-seeds", type=int, default=5,
                        help="Number of OFT seeds per saved state.")
    parser.add_argument("--execution-horizon", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-anchors", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ks = sorted(args.ks)
    n_teacher_seeds = args.teacher_seeds
    h = args.execution_horizon
    base_seed = 2026080405 + args.training_seed * 1000

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    adapter_cfg = dict(cfg.get("adapter_config") or {})
    pool = StatePool(Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve())

    failures = load_pre_c0_failure_keys(args.failure_rollout_dir.resolve())
    allowed = set(json.loads(args.state_keys_json.read_text(encoding="utf-8"))["state_keys"])
    failures = [row for row in failures if str(row["state_key"]) in allowed]
    if args.suite:
        failures = [row for row in failures if str(row.get("suite")) == args.suite]
    if args.smoke:
        seen: set = set()
        picked: list[dict[str, Any]] = []
        for row in failures:
            suite_name = str(row.get("suite"))
            if suite_name in seen:
                continue
            seen.add(suite_name)
            picked.append(row)
        failures = picked
    if args.limit_anchors:
        failures = failures[: args.limit_anchors]
    if not failures:
        raise SystemExit("no failure anchors selected")

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    ensure_libero_plus_paths(adapter_cfg.get("libero_plus_root"))
    _patch_lerobot_init_states()

    # --- Load student policy bundle ---
    policy_path = Path(adapter_cfg.get("policy_path") or "ckpts/smolvla_libero")
    tokenizer_path = Path(adapter_cfg.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct")
    device = str(adapter_cfg.get("device", "cuda"))
    bundle = load_smolvla_policy_bundle(
        policy_path, device=device,
        num_steps=int(adapter_cfg.get("num_steps", 10)),
        n_action_steps=int(adapter_cfg.get("n_action_steps", 10)),
        tokenizer_path=tokenizer_path,
        observation_height=int(adapter_cfg.get("observation_height", 360)),
        observation_width=int(adapter_cfg.get("observation_width", 360)),
    )
    handle = load_lora_onto_policy(bundle["policy"], str(args.adapter_dir.resolve()))
    bundle["policy"] = handle.policy
    rollout_cfg = RolloutConfig(
        n_action_steps=int(adapter_cfg.get("n_action_steps", 10)),
        num_steps=int(adapter_cfg.get("num_steps", 10)),
        observation_height=int(adapter_cfg.get("observation_height", 360)),
        observation_width=int(adapter_cfg.get("observation_width", 360)),
        continuation_temperature=float(adapter_cfg.get("continuation_temperature", 0.5)),
    )

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OracleClient(args.endpoint)
    all_trials: list[dict[str, Any]] = []

    try:
        for anchor_idx, failure in enumerate(failures):
            state_key = str(failure["state_key"])
            task = str(failure.get("instruction") or "")

            # --- Restore anchor state ---
            restored = restore_pool_state(
                pool, state_key,
                libero_plus_root=adapter_cfg.get("libero_plus_root"),
                observation_height=rollout_cfg.observation_height,
                observation_width=rollout_cfg.observation_width,
            )
            try:
                single = restored.handle.vector_env.envs[0]
                if not task:
                    task = str(getattr(single, "task_description", "") or "")
                episode_max = _episode_max(single)
                t0 = current_timestep(restored.handle.control_env)

                # --- Run student prefix trajectory, saving snapshots at each k ---
                restored.forkable.restore(restored.snapshot, check_task_fingerprint=True)
                set_adapter_enabled(handle, True)
                student = RecedingHorizonSmolVLAContinuation(
                    bundle, execution_horizon=h,
                    temperature=float(adapter_cfg.get("continuation_temperature", 0.5)),
                    seed=base_seed,
                )
                if hasattr(student, "bind_control_env"):
                    student.bind_control_env(restored.handle.control_env)
                student.reset()

                snapshots: dict[int, Any] = {}
                early_stop: dict[str, Any] | None = None  # early termination info
                steps_run = 0
                for k_val in ks:
                    if k_val == 0:
                        snapshots[0] = restored.forkable.snapshot()
                        continue
                    # Run from current state up to k_val steps
                    needed = k_val - steps_run
                    for _ in range(needed):
                        obs = observation_from_libero_env(single)
                        action = np.asarray(student.act(obs, task=task), dtype=np.float32).reshape(-1)
                        _o, _r, term, trunc, info = restored.handle.vector_env.step(
                            as_batched_action(action)
                        )
                        steps_run += 1
                        if success_from_info(info):
                            early_stop = {"kind": "student_success", "at_step": steps_run}
                            break
                        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                            early_stop = {
                                "kind": "terminated" if bool(np.asarray(term).reshape(-1)[0]) else "truncated",
                                "at_step": steps_run,
                            }
                            break
                    if early_stop:
                        break
                    snapshots[k_val] = restored.forkable.snapshot()

                # --- For each snapshot, run multi-seed OFT + self-continuation ---
                for k_val, snapshot in snapshots.items():
                    # OFT continuation with multiple seeds
                    for offt_seed in range(n_teacher_seeds):
                        restored.forkable.restore(snapshot, check_task_fingerprint=True)
                        # Vary RNG for different OFT outcomes
                        np.random.seed(base_seed + k_val * 100 + offt_seed * 10)
                        now_t = current_timestep(restored.handle.control_env)
                        remaining = max(0, episode_max - int(now_t))
                        teacher_result = _run_teacher_from_live(
                            client, restored, task, max_steps=remaining,
                        )
                        trial = {
                            "schema_version": "rase-pre-c1-3-rk-multi-seed-trial/v1",
                            "arm": args.arm,
                            "training_seed": args.training_seed,
                            "state_key": state_key,
                            "suite": failure.get("suite"),
                            "episode_id": failure.get("episode_id"),
                            "task": task,
                            "k": int(k_val),
                            "teacher_seed": offt_seed,
                            "layer": "teacher",
                            "anchor_steps_run_before_k": steps_run if early_stop and k_val > early_stop["at_step"] else k_val,
                            "student_prefix_completed": min(k_val, steps_run),
                            "student_prefix_requested": int(k_val),
                            "early_stop": early_stop,
                            "teacher_success": bool(teacher_result["teacher_success"]),
                            "teacher_env_steps": int(teacher_result["teacher_env_steps_after_handover"]),
                            "stop_reason": teacher_result.get("stop_reason", "unknown"),
                            "handover_timestep": int(now_t),
                            "episode_max_steps": episode_max,
                            "adapter_dir": str(args.adapter_dir),
                        }
                        _atomic_json(
                            out_dir / "trials" / (
                                f"{state_key[:12]}__{args.arm}__ts{args.training_seed}"
                                f"__k{k_val}__oft{offt_seed}.json"
                            ), trial,
                        )
                        all_trials.append(trial)

                    # Self-continuation (one seed)
                    restored.forkable.restore(snapshot, check_task_fingerprint=True)
                    set_adapter_enabled(handle, True)
                    self_student = RecedingHorizonSmolVLAContinuation(
                        bundle, execution_horizon=h,
                        temperature=float(adapter_cfg.get("continuation_temperature", 0.5)),
                        seed=base_seed + k_val,
                    )
                    now_t = current_timestep(restored.handle.control_env)
                    remaining = max(0, episode_max - int(now_t))
                    self_result = _run_self_continuation(
                        restored, self_student, task, max_steps=remaining,
                    )
                    self_trial = {
                        "schema_version": "rase-pre-c1-3-rk-multi-seed-trial/v1",
                        "arm": args.arm,
                        "training_seed": args.training_seed,
                        "state_key": state_key,
                        "suite": failure.get("suite"),
                        "episode_id": failure.get("episode_id"),
                        "task": task,
                        "k": int(k_val),
                        "teacher_seed": -1,
                        "layer": "self",
                        "anchor_steps_run_before_k": steps_run if early_stop and k_val > early_stop["at_step"] else k_val,
                        "student_prefix_completed": min(k_val, steps_run),
                        "student_prefix_requested": int(k_val),
                        "early_stop": early_stop,
                        "self_success": bool(self_result["self_success"]),
                        "self_env_steps": int(self_result["self_env_steps"]),
                        "stop_reason": self_result.get("stop_reason", "unknown"),
                        "handover_timestep": int(now_t),
                        "episode_max_steps": episode_max,
                        "adapter_dir": str(args.adapter_dir),
                    }
                    _atomic_json(
                        out_dir / "trials" / (
                            f"{state_key[:12]}__{args.arm}__ts{args.training_seed}"
                            f"__k{k_val}__self.json"
                        ), self_trial,
                    )
                    all_trials.append(self_trial)

                if early_stop:
                    # For ks beyond early stop, mark as irreversible
                    remaining_ks = [k for k in ks if k > early_stop["at_step"]]
                    for k_val in remaining_ks:
                        for offt_seed in range(n_teacher_seeds):
                            trial = {
                                "schema_version": "rase-pre-c1-3-rk-multi-seed-trial/v1",
                                "arm": args.arm,
                                "training_seed": args.training_seed,
                                "state_key": state_key,
                                "suite": failure.get("suite"),
                                "episode_id": failure.get("episode_id"),
                                "task": task,
                                "k": int(k_val),
                                "teacher_seed": offt_seed,
                                "layer": "teacher",
                                "anchor_steps_run_before_k": steps_run,
                                "student_prefix_completed": steps_run,
                                "student_prefix_requested": int(k_val),
                                "early_stop": early_stop,
                                "teacher_success": False,
                                "teacher_env_steps": 0,
                                "stop_reason": f"irreversible_before_k_{early_stop['kind']}",
                                "handover_timestep": current_timestep(restored.handle.control_env),
                                "episode_max_steps": episode_max,
                                "adapter_dir": str(args.adapter_dir),
                            }
                            _atomic_json(
                                out_dir / "trials" / (
                                    f"{state_key[:12]}__{args.arm}__ts{args.training_seed}"
                                    f"__k{k_val}__oft{offt_seed}__irrev.json"
                                ), trial,
                            )
                            all_trials.append(trial)
                        self_trial = {
                            "schema_version": "rase-pre-c1-3-rk-multi-seed-trial/v1",
                            "arm": args.arm,
                            "training_seed": args.training_seed,
                            "state_key": state_key,
                            "suite": failure.get("suite"),
                            "episode_id": failure.get("episode_id"),
                            "task": task,
                            "k": int(k_val),
                            "teacher_seed": -1,
                            "layer": "self",
                            "anchor_steps_run_before_k": steps_run,
                            "student_prefix_completed": steps_run,
                            "student_prefix_requested": int(k_val),
                            "early_stop": early_stop,
                            "self_success": False,
                            "self_env_steps": 0,
                            "stop_reason": f"irreversible_before_k_{early_stop['kind']}",
                            "handover_timestep": current_timestep(restored.handle.control_env),
                            "episode_max_steps": episode_max,
                            "adapter_dir": str(args.adapter_dir),
                        }
                        _atomic_json(
                            out_dir / "trials" / (
                                f"{state_key[:12]}__{args.arm}__ts{args.training_seed}"
                                f"__k{k_val}__self__irrev.json"
                            ), self_trial,
                        )
                        all_trials.append(self_trial)

            finally:
                restored.close()

            print(
                f"C1_3_RK anchor={state_key[:12]} arm={args.arm} ts={args.training_seed} "
                f"snapshots={sorted(snapshots.keys())} early_stop={early_stop}",
                flush=True,
            )

    finally:
        client.close()

    # --- Aggregate ---
    def rate(layer: str, k_val: int) -> float:
        subset = [
            t for t in all_trials
            if t.get("layer") == layer and int(t.get("k", -1)) == int(k_val)
        ]
        if not subset:
            return float("nan")
        key = "teacher_success" if layer == "teacher" else "self_success"
        return float(np.mean([1.0 if t.get(key) else 0.0 for t in subset]))

    curves = {
        "R_teacher": {int(k): rate("teacher", k) for k in ks},
        "R_self": {int(k): rate("self", k) for k in ks},
        "G_exec": {int(k): rate("teacher", k) - rate("self", k) for k in ks},
    }

    # Irreversible before k rate
    irrev = {}
    for k_val in ks:
        subset = [t for t in all_trials if int(t.get("k", -1)) == int(k_val) and t.get("layer") == "teacher"]
        irrev[k_val] = sum(1 for t in subset if "irreversible" in str(t.get("stop_reason", ""))) / max(len(subset), 1)

    # Count early student success before k
    early_succ = {}
    for k_val in ks:
        subset = [t for t in all_trials if int(t.get("k", -1)) == int(k_val) and t.get("layer") == "teacher"]
        early_succ[k_val] = sum(1 for t in subset if t.get("stop_reason") == "success_during_prefix") / max(len(subset), 1)

    summary = {
        "schema_version": "rase-pre-c1-3-rk-summary/v1",
        "arm": args.arm,
        "training_seed": args.training_seed,
        "adapter_dir": str(args.adapter_dir),
        "n_anchors": len({t["state_key"] for t in all_trials}),
        "n_trials": len(all_trials),
        "ks": ks,
        "n_teacher_seeds": n_teacher_seeds,
        "execution_horizon": h,
        "curves": curves,
        "irreversible_before_k_rate": irrev,
        "student_success_before_k_rate": early_succ,
        "suite_filter": args.suite,
        "coverage": {
            "anchors": sorted({t["state_key"] for t in all_trials}),
            "n_layers": 2,
            "layers": ["teacher", "self"],
        },
    }
    _atomic_json(out_dir / "summary.json", summary)

    # Write long-format trials JSONL
    with (out_dir / "trials.jsonl").open("w", encoding="utf-8") as f:
        for trial in all_trials:
            f.write(json.dumps(trial, sort_keys=True) + "\n")

    print(json.dumps(summary, sort_keys=True, default=str))
    print(f"PRE_C1_3_RK_DONE output={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
