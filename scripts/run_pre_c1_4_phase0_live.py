#!/usr/bin/env python3
"""PRE-C1.4-R3 Phase 0 Live (FAST): Restore parity + Causal-unit pilot.

Efficient version: 1 calibration anchor, 2-3 seeds, capped continuation (200 steps).

On calibration anchor (filtered to one suite), runs:
  - Exact restore hash verification
  - Paired student/teacher branching for H={4, 64}
  - Capped continuation after prefix (max 200 steps)
  - Causal-unit gate check with minimal runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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
from rase.collect.policy_step import as_batched_action, success_from_info
from rase.collect.same_policy_corrective import RecedingHorizonSmolVLAContinuation
from rase.collect.state_pool import StatePool
from rase.oracle.client import OracleClient

CANDIDATE_H = [4, 64]
MAX_CONT_STEPS = 200


def _hash_sim_state(restored: Any) -> str:
    sim_state = restored.handle.control_env.sim.get_state()
    return hashlib.sha256(sim_state.flatten().tobytes()).hexdigest()[:16]


def _run_branch(continuation, restored, task, max_steps, seed):
    """Run a policy from current env state, capped at max_steps."""
    vector_env = restored.handle.vector_env
    single = vector_env.envs[0]
    if hasattr(continuation, "bind_control_env"):
        continuation.bind_control_env(restored.handle.control_env)
    continuation.reset()
    np.random.seed(seed)
    ok = False
    steps = 0
    for _ in range(max_steps):
        obs = observation_from_libero_env(single)
        action = np.asarray(continuation.act(obs, task=task), dtype=np.float32).reshape(-1)
        _o, _r, term, trunc, info = vector_env.step(as_batched_action(action))
        steps += 1
        if success_from_info(info):
            ok = True
            break
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            break
    return {"success": ok, "steps": steps}


def _run_teacher_prefix(client, restored, task, h_steps):
    """Run OFT teacher for exactly h_steps."""
    cont = OracleChunkContinuation(client, instruction=task)
    cont.bind_control_env(restored.handle.control_env)
    cont.reset()
    single = restored.handle.vector_env.envs[0]
    for _ in range(h_steps):
        obs = observation_from_libero_env(single)
        action = cont.act(obs, task=task)
        _o, _r, term, trunc, info = restored.handle.vector_env.step(as_batched_action(action))
        if success_from_info(info):
            return True
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            return False
    return False


def _run_student_prefix(bundle, restored, task, h_steps, seed):
    """Run C1.1 student for exactly h_steps."""
    student = RecedingHorizonSmolVLAContinuation(bundle, execution_horizon=2, temperature=0.5, seed=seed)
    if hasattr(student, "bind_control_env"):
        student.bind_control_env(restored.handle.control_env)
    student.reset()
    single = restored.handle.vector_env.envs[0]
    np.random.seed(seed)
    for _ in range(h_steps):
        obs = observation_from_libero_env(single)
        action = np.asarray(student.act(obs, task=task), dtype=np.float32).reshape(-1)
        _o, _r, term, trunc, info = restored.handle.vector_env.step(as_batched_action(action))
        if success_from_info(info):
            return True
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            return False
    return False


def _make_continuation(bundle, seed):
    """Create a shared continuation policy (frozen C1.1)."""
    return RecedingHorizonSmolVLAContinuation(bundle, execution_horizon=2, temperature=0.5, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="PRE-C1.4-R3 Phase 0 Live (Fast)")
    parser.add_argument("--config", default="configs/collect_pre_c0_deviation_pilot24.json")
    parser.add_argument("--failure-rollout-dir", default="runs/rase_pre_c0_same_policy_pilot48_v1")
    parser.add_argument("--manifest", default="runs/rase_pre_c1_4_r3_protocol/pre_c1_4_r3_identity_manifest.json")
    parser.add_argument("--adapter-dir", default="runs/rase_pre_c1_1_lora_train_v1/adapter_final")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--suite", default="Object")
    parser.add_argument("--output-dir", default="runs/rase_pre_c1_4_r3_protocol")
    parser.add_argument("--screen-seeds", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text())
    calib_by_suite = manifest["splits"]["calibration"]["by_suite"]
    if args.suite not in calib_by_suite:
        print(f"Suite {args.suite} not available. Options: {list(calib_by_suite.keys())}")
        return 1

    calib_key = calib_by_suite[args.suite][0]
    print(f"Calibration anchor: {calib_key[:20]}... ({args.suite})")

    cfg = json.loads(Path(args.config).read_text())
    adapter_cfg = dict(cfg.get("adapter_config") or {})
    pool = StatePool(Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve())
    failures = load_pre_c0_failure_keys(Path(args.failure_rollout_dir).resolve())
    failures = [r for r in failures if str(r["state_key"]) == calib_key]
    if not failures:
        print(f"ERROR: No failure entry for {calib_key}")
        return 1

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    ensure_libero_plus_paths(adapter_cfg.get("libero_plus_root"))
    _patch_lerobot_init_states()

    policy_path = Path(adapter_cfg.get("policy_path") or "ckpts/smolvla_libero")
    tokenizer_path = Path(adapter_cfg.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct")
    device = str(adapter_cfg.get("device", "cuda"))
    print(f"Loading SmolVLA...")
    bundle = load_smolvla_policy_bundle(
        policy_path, device=device,
        num_steps=int(adapter_cfg.get("num_steps", 10)),
        n_action_steps=int(adapter_cfg.get("n_action_steps", 10)),
        tokenizer_path=tokenizer_path,
        observation_height=int(adapter_cfg.get("observation_height", 360)),
        observation_width=int(adapter_cfg.get("observation_width", 360)),
    )
    handle = load_lora_onto_policy(bundle["policy"], str(Path(args.adapter_dir).resolve()))
    bundle["policy"] = handle.policy
    set_adapter_enabled(handle, True)

    rollout_cfg = RolloutConfig(
        n_action_steps=int(adapter_cfg.get("n_action_steps", 10)),
        num_steps=int(adapter_cfg.get("num_steps", 10)),
        observation_height=int(adapter_cfg.get("observation_height", 360)),
        observation_width=int(adapter_cfg.get("observation_width", 360)),
        continuation_temperature=float(adapter_cfg.get("continuation_temperature", 0.5)),
    )

    client = OracleClient(args.endpoint)
    print(f"Connected to OFT oracle")

    # Phase 0A: Restore parity
    print("\n=== Phase 0A: Restore Parity ===")
    failure = failures[0]
    restored = restore_pool_state(
        pool, calib_key,
        libero_plus_root=adapter_cfg.get("libero_plus_root"),
        observation_height=rollout_cfg.observation_height,
        observation_width=rollout_cfg.observation_width,
    )
    h1 = _hash_sim_state(restored)
    restored.forkable.restore(restored.snapshot, check_task_fingerprint=False)
    h2 = _hash_sim_state(restored)
    restore_ok = (h1 == h2)
    print(f"  hash1={h1} hash2={h2} -> {'PASS' if restore_ok else 'FAIL'}")
    if not restore_ok:
        (out_dir / "phase0_restore_pass.json").write_text(json.dumps({"passed": False, "message": "Hash mismatch"}) + "\n")
        return 1
    (out_dir / "phase0_restore_pass.json").write_text(json.dumps({"passed": True, "message": f"hash={h1}"}) + "\n")

    # Phase 0C: Causal-unit pilot
    print("\n=== Phase 0C: Causal-Unit Pilot ===")
    task = failure.get("instruction", "")
    if not task:
        task = str(getattr(restored.handle.vector_env.envs[0], "task_description", ""))

    restored.forkable.restore(restored.snapshot, check_task_fingerprint=False)
    baseline = restored.forkable.snapshot()

    base_seed = 2026080605
    results = {}

    for h in CANDIDATE_H:
        print(f"\n  H={h}:")
        teacher_wins = 0
        student_wins = 0
        total_tests = 0

        for seed_idx in range(args.screen_seeds):
            seed = base_seed + h * 100 + seed_idx

            # Student branch: prefix + short continuation
            restored.forkable.restore(baseline, check_task_fingerprint=False)
            s_prefix_success = _run_student_prefix(bundle, restored, task, h, seed)
            s_cont = _make_continuation(bundle, seed + 10000)
            s_result = _run_branch(s_cont, restored, task, MAX_CONT_STEPS, seed + 20000)
            student_success = s_prefix_success or s_result["success"]

            # Teacher branch: prefix + short continuation
            restored.forkable.restore(baseline, check_task_fingerprint=False)
            t_prefix_success = _run_teacher_prefix(client, restored, task, h)
            t_cont = _make_continuation(bundle, seed + 10000)
            t_result = _run_branch(t_cont, restored, task, MAX_CONT_STEPS, seed + 20000)
            teacher_success = t_prefix_success or t_result["success"]

            total_tests += 1
            if teacher_success and not student_success:
                teacher_wins += 1
            elif student_success and not teacher_success:
                student_wins += 1

            t_label = "+" if teacher_success else "-"
            s_label = "+" if student_success else "-"
            print(f"    seed {seed_idx}: T{pfx(t_prefix_success)}{t_label}({t_result['steps']}c) vs S{pfx(s_prefix_success)}{s_label}({s_result['steps']}c)")

        results[h] = {
            "teacher_wins": teacher_wins,
            "student_wins": student_wins,
            "total_tests": total_tests,
            "fraction": teacher_wins / max(total_tests, 1),
        }
        print(f"    Summary: T_wins={teacher_wins}/{total_tests} S_wins={student_wins}/{total_tests}")

    # Gate check
    print("\n=== Causal-Unit Gate ===")
    h_star = None
    for h in CANDIDATE_H:
        r = results[h]
        passed = r["teacher_wins"] >= 2 and r["fraction"] >= 0.25
        print(f"  H={h}: T_wins={r['teacher_wins']} frac={r['fraction']:.2f} -> {'PASS' if passed else 'FAIL'}")
        if passed and h_star is None:
            h_star = h

    if h_star:
        route = "action_level" if h_star <= 8 else "option_level"
        print(f"\n  H_star = {h_star}  route = {route}")
    else:
        print("\n  No H passed causal unit gate")

    gate = {
        "phase": "causal_unit_pilot",
        "passed": h_star is not None,
        "H_star": h_star,
        "route": "action_level" if (h_star is not None and h_star <= 8) else ("option_level" if h_star else "NONE"),
        "message": f"H_star={h_star}" if h_star else "No H passed",
        "results": {str(h): results[h] for h in CANDIDATE_H},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "paired_prefix+short_continuation",
    }
    (out_dir / "phase0_causal_unit_pass.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(f"\nGate written to phase0_causal_unit_pass.json")
    return 0 if h_star else 1


def pfx(x):
    return ("+" if x else "-")


if __name__ == "__main__":
    raise SystemExit(main())
