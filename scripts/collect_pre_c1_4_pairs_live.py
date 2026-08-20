#!/usr/bin/env python3
"""PRE-C1.4-R3 Phase 1 Live: Paired counterfactual collection.

On train_collection anchors, runs paired student/teacher branches from the
exact same snapshot state. Collects action-chunk + outcome pairs for
contrastive distillation.

Uses H_star from Phase 0 gate file. For each anchor × J_screen seeds:
  1. Restore pool state → snapshot S0
  2. Student branch: frozen C1.1 executes prefix of H_star steps,
     then continuation to terminal. Record outcome + all (s_t, a_student) pairs.
  3. Restore S0
  4. Teacher branch: OFT oracle executes prefix of H_star steps,
     then continuation to terminal. Record outcome + all teacher actions.
  5. Emit paired chunk: (state, teacher_chunk, student_chunk, outcome_pair)

Output: per-anchor JSONL of paired counterfactual episodes.
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
from rase.collect.policy_step import as_batched_action, current_timestep, success_from_info
from rase.collect.same_policy_corrective import RecedingHorizonSmolVLAContinuation
from rase.collect.state_pool import StatePool
from rase.oracle.client import OracleClient


def _episode_max(single: Any) -> int:
    return int(getattr(single, "_max_episode_steps", 600))


def _collect_chunk(
    policy_fn, restored: Any, task: str, max_steps: int, seed: int, chunk_size: int,
) -> dict:
    """Run policy collecting action chunks + trajectory data."""
    vector_env = restored.handle.vector_env
    single = vector_env.envs[0]
    np.random.seed(seed)

    actions: list[np.ndarray] = []
    success = False
    steps = 0
    stop_reason = "horizon"

    for _ in range(max(0, int(max_steps))):
        obs = observation_from_libero_env(single)
        action = np.asarray(policy_fn(obs, task=task), dtype=np.float32).reshape(-1)
        actions.append(action.copy())
        _o, _r, term, trunc, info = vector_env.step(as_batched_action(action))
        steps += 1
        if success_from_info(info):
            success = True
            stop_reason = "success"
            break
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            stop_reason = "terminated" if bool(np.asarray(term).reshape(-1)[0]) else "truncated"
            break

    return {
        "success": success,
        "steps": steps,
        "stop_reason": stop_reason,
        "actions": actions,
    }


def main():
    parser = argparse.ArgumentParser(description="PRE-C1.4-R3 Phase 1: Counterfactual Collection")
    parser.add_argument("--config", default="configs/collect_pre_c0_deviation_pilot24.json")
    parser.add_argument("--failure-rollout-dir", default="runs/rase_pre_c0_same_policy_pilot48_v1")
    parser.add_argument("--manifest", default="runs/rase_pre_c1_4_r3_protocol/pre_c1_4_r3_identity_manifest.json")
    parser.add_argument("--adapter-dir", default="runs/rase_pre_c1_1_lora_train_v1/adapter_final")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--suite", default="Object")
    parser.add_argument("--output-dir", default="runs/rase_pre_c1_4_counterfactual")
    parser.add_argument("--screen-seeds", type=int, default=5, help="J_screen seeds per anchor")
    parser.add_argument("--limit-anchors", type=int, default=5, help="Max train anchors to process")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load H_star from Phase 0 gate
    gate_path = Path("runs/rase_pre_c1_4_r3_protocol/phase0_causal_unit_pass.json")
    if gate_path.exists():
        gate = json.loads(gate_path.read_text())
        H_star = gate.get("H_star", 64)
        print(f"H_star from gate: {H_star}")
    else:
        print("WARNING: No Phase 0 gate found, defaulting H_star=64")
        H_star = 64

    # Load train collection anchors
    manifest = json.loads(Path(args.manifest).read_text())
    train_by_suite = manifest["splits"]["train_collection"]["by_suite"]
    if args.suite not in train_by_suite:
        print(f"Suite {args.suite} not in train_collection. Available: {list(train_by_suite.keys())}")
        return 1
    train_keys = train_by_suite[args.suite][:args.limit_anchors]
    print(f"Train collection anchors ({args.suite}): {len(train_keys)}")

    # Config and environment
    cfg = json.loads(Path(args.config).read_text())
    adapter_cfg = dict(cfg.get("adapter_config") or {})
    pool = StatePool(Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve())
    failures = load_pre_c0_failure_keys(Path(args.failure_rollout_dir).resolve())
    failures = [r for r in failures if str(r["state_key"]) in set(train_keys)]
    print(f"Matched failure entries: {len(failures)}")

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    ensure_libero_plus_paths(adapter_cfg.get("libero_plus_root"))
    _patch_lerobot_init_states()

    # Load student policy
    policy_path = Path(adapter_cfg.get("policy_path") or "ckpts/smolvla_libero")
    tokenizer_path = Path(adapter_cfg.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct")
    device = str(adapter_cfg.get("device", "cuda"))
    print(f"Loading SmolVLA from {policy_path}...")
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
    print(f"Connected to OFT oracle at {args.endpoint}")

    base_seed = 2026080605
    chunks_file = out_dir / f"paired_chunks_{args.suite}_{H_star}.jsonl"
    all_pairs = 0

    with open(chunks_file, "w", encoding="utf-8") as f_out:
        for anchor_idx, failure in enumerate(failures):
            sk = failure["state_key"]
            task = failure.get("instruction", "")
            suite_name = failure.get("suite", "unknown")
            print(f"\n  Anchor {anchor_idx+1}/{len(failures)}: {sk[:20]}... (H={H_star})")

            restored = restore_pool_state(
                pool, sk,
                libero_plus_root=adapter_cfg.get("libero_plus_root"),
                observation_height=rollout_cfg.observation_height,
                observation_width=rollout_cfg.observation_width,
            )
            try:
                single = restored.handle.vector_env.envs[0]
                if not task:
                    task = str(getattr(single, "task_description", "") or "")
                ep_max = _episode_max(single)

                restored.forkable.restore(restored.snapshot, check_task_fingerprint=False)
                baseline_snapshot = restored.forkable.snapshot()

                for seed_idx in range(args.screen_seeds):
                    seed = base_seed + anchor_idx * 1000 + seed_idx * 10

                    # --- Student branch ---
                    restored.forkable.restore(baseline_snapshot, check_task_fingerprint=False)
                    set_adapter_enabled(handle, True)
                    student = RecedingHorizonSmolVLAContinuation(
                        bundle, execution_horizon=2, temperature=0.5, seed=seed,
                    )
                    if hasattr(student, "bind_control_env"):
                        student.bind_control_env(restored.handle.control_env)
                    student.reset()
                    student_predict_fn = lambda obs, task: np.asarray(
                        student.act(obs, task=task), dtype=np.float32
                    ).reshape(-1)

                    # Collect student prefix + continuation
                    student_chunk = _collect_chunk(
                        student_predict_fn, restored, task, ep_max, seed, H_star,
                    )
                    student_outcome = {
                        "success": student_chunk["success"],
                        "steps": student_chunk["steps"],
                        "stop_reason": student_chunk["stop_reason"],
                        "actions": [a.tolist() for a in student_chunk["actions"]],
                    }

                    # --- Teacher branch ---
                    restored.forkable.restore(baseline_snapshot, check_task_fingerprint=False)
                    teacher_cont = OracleChunkContinuation(client, instruction=task)
                    teacher_cont.bind_control_env(restored.handle.control_env)
                    teacher_cont.reset()
                    teacher_predict_fn = lambda obs, task: teacher_cont.act(obs, task=task)

                    teacher_chunk = _collect_chunk(
                        teacher_predict_fn, restored, task, ep_max, seed + 50000, H_star,
                    )
                    teacher_outcome = {
                        "success": teacher_chunk["success"],
                        "steps": teacher_chunk["steps"],
                        "stop_reason": teacher_chunk["stop_reason"],
                        "actions": [a.tolist() for a in teacher_chunk["actions"]],
                    }

                    # Determine pair label
                    if teacher_outcome["success"] and not student_outcome["success"]:
                        label = "teacher_preferred"
                    elif student_outcome["success"] and not teacher_outcome["success"]:
                        label = "student_preferred"
                    elif student_outcome["success"] and teacher_outcome["success"]:
                        label = "both_succeed"
                    else:
                        label = "both_fail"

                    # Emit paired chunk
                    pair_record = {
                        "schema": "rase-pre-c1-4-counterfactual-pair/v1",
                        "anchor_idx": anchor_idx,
                        "seed_idx": seed_idx,
                        "state_key": sk,
                        "suite": suite_name,
                        "task": task,
                        "H_star": H_star,
                        "label": label,
                        "teacher_outcome": teacher_outcome,
                        "student_outcome": student_outcome,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                    f_out.write(json.dumps(pair_record) + "\n")
                    f_out.flush()
                    all_pairs += 1

                    t_succ = "+" if teacher_outcome["success"] else "-"
                    s_succ = "+" if student_outcome["success"] else "-"
                    print(f"    seed {seed_idx}: T={t_succ}({teacher_outcome['steps']}s) S={s_succ}({student_outcome['steps']}s) -> {label}")

            finally:
                pass

    # Statistics
    print(f"\n=== Collection Complete ===")
    print(f"  Total pairs: {all_pairs}")
    print(f"  Output: {chunks_file}")

    # Write summary stats
    with open(chunks_file, "r") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    counts = {}
    for p in pairs:
        label = p["label"]
        counts[label] = counts.get(label, 0) + 1
    print(f"  Distribution: {counts}")

    summary = {
        "phase": "counterfactual_collection",
        "H_star": H_star,
        "suite": args.suite,
        "total_pairs": all_pairs,
        "distribution": counts,
        "screen_seeds": args.screen_seeds,
        "output_file": str(chunks_file.absolute()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    summary_file = out_dir / f"collection_summary_{args.suite}_{H_star}.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
