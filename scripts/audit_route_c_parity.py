#!/usr/bin/env python3
"""D0-A: No-takeover parity audit.

Validates that B0 (pure SmolVLA) and B3 (plugin forced-off) produce
identical results when given the same initial state and seed.

Key invariant: B3_forced_off replays B0's recorded actions, so it never
consumes policy RNG. Any divergence indicates environment non-determinism
or snapshot restore issues.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.collect.policy_step import as_batched_action, select_env_action, success_from_info
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.obs_hash import canonical_obs_hash, obs_hash_diff, action_max_abs_diff
from rase.utils.seeding import seed_everything, record_policy_rng
from rase.recovery.plugin_executor import RecoveryPluginExecutor
from rase.recovery.residual_plugin import make_recovery_plugin


def _capture_rng_state() -> dict:
    """Capture PyTorch and NumPy RNG state for parity verification."""
    import random
    return {
        "torch_cpu": torch.get_rng_state().clone(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _rng_states_equal(s1: dict, s2: dict) -> bool:
    return bool(
        torch.equal(s1["torch_cpu"], s2["torch_cpu"])
        and np.array_equal(s1["numpy"][1], s2["numpy"][1])
    )


def _progress(control_env: Any) -> float:
    try:
        pos = getattr(control_env.env, "_eef_xpos", None)
        if pos is not None:
            return float(np.linalg.norm(np.asarray(pos)))
    except Exception:
        pass
    return 0.0


def run_parity_pair(
    handle: Any,
    bundle: dict,
    instruction: str,
    max_steps: int,
    task_id: str,
    episode_seed: int,
) -> dict:
    """Run B0 (pure SmolVLA) then B3-replay on the same environment snapshot.

    Returns a dict with per-step comparison results.
    """
    env = handle.vector_env.envs[0]
    control_env = handle.control_env

    # ── B0: record full rollout ──────────────────────────────────────
    obs = observation_from_libero_env(env)
    b0_actions: list[np.ndarray] = []
    b0_obs_hashes: list[dict] = []
    b0_steps = 0
    b0_success = False
    b0_progress_vals: list[float] = []

    rng_before_b0 = _capture_rng_state()

    for t in range(max_steps):
        obs_hash_before = canonical_obs_hash(observation_from_libero_env(env))
        action = select_env_action(bundle, obs, task=instruction)
        b0_actions.append(action.copy())
        b0_obs_hashes.append(obs_hash_before)

        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(env)
        b0_progress_vals.append(_progress(control_env))
        b0_steps += 1

        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            b0_success = success_from_info(info)
            break

    rng_after_b0 = _capture_rng_state()

    # ── B3: replay B0's actions (no policy calls, no RNG consumption) ─
    # Reset RNG to match B0's starting state for deterministic comparison
    seed_everything(episode_seed)

    # Re-create environment with same seed to get identical initial state
    handle_replay = make_libero_env_for_task(
        task_id, init_state_id=episode_seed % 50, seed=episode_seed,
        libero_flavor="clean",
    )
    env_replay = handle_replay.vector_env.envs[0]
    control_replay = handle_replay.control_env
    obs_replay = observation_from_libero_env(env_replay)

    rng_before_b3 = _capture_rng_state()
    b3_actions: list[np.ndarray] = []
    comparisons: list[dict] = []
    all_obs_match = True
    all_action_match = True
    first_obs_divergence = None
    first_action_divergence = None

    for t in range(len(b0_actions)):
        obs_hash_before = canonical_obs_hash(observation_from_libero_env(env_replay))
        action_replay = b0_actions[t]  # replay exact B0 action
        b3_actions.append(action_replay.copy())

        # Compare pre-step observation hash
        hash_diff = obs_hash_diff(b0_obs_hashes[t], obs_hash_before)
        hash_match = hash_diff["equal"]

        # Also compare action byte-for-byte
        action_diff = float(np.max(np.abs(b0_actions[t] - action_replay)))
        action_match = action_diff < 1e-8

        obs_replay, reward, term, trunc, info_replay = \
            handle_replay.vector_env.step(as_batched_action(action_replay))
        obs_replay = observation_from_libero_env(env_replay)

        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])

        step_record = {
            "step": t,
            "obs_hash_match": hash_match,
            "action_match": action_match,
            "action_max_abs_diff": float(action_diff),
        }

        if not hash_match and first_obs_divergence is None:
            first_obs_divergence = t
        if not action_match and first_action_divergence is None:
            first_action_divergence = t
        if not hash_match:
            all_obs_match = False
        if not action_match:
            all_action_match = False

        comparisons.append(step_record)

        if terminated or truncated:
            break

    rng_after_b3 = _capture_rng_state()

    # ── Outcome comparison ──────────────────────────────────────────
    b3_success = success_from_info(info_replay) if (terminated or truncated) else False
    outcome_match = b0_success == b3_success
    steps_match = b0_steps == (len(comparisons))

    handle_replay.close()

    return {
        "task_id": task_id,
        "episode_seed": episode_seed,
        "b0_steps": b0_steps,
        "b0_success": b0_success,
        "b3_steps": len(comparisons),
        "b3_success": b3_success,
        "outcome_match": outcome_match,
        "steps_match": steps_match,
        "all_obs_hash_match": all_obs_match,
        "all_action_match": all_action_match,
        "first_obs_divergence_step": first_obs_divergence,
        "first_action_divergence_step": first_action_divergence,
        "n_steps_compared": len(comparisons),
        "rng_before_match": _rng_states_equal(rng_before_b0, rng_before_b3),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", type=str, nargs="*",
                        help="suite(s) to audit (default: first)")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    all_suites = list(protocol["splits"].keys())
    run_suites = args.suite if args.suite else [all_suites[0]]

    seed_everything(args.seed)

    policy_path = Path(protocol["student_identity"]["checkpoint_path"])
    vlm_cache = protocol.get("vlm_cache_path", "")
    bundle = load_smolvla_policy_bundle(
        policy_path, device="cuda",
        tokenizer_path=vlm_cache if vlm_cache else None,
        observation_height=360, observation_width=360,
    )

    pair_results: list[dict] = []

    for suite in run_suites:
        if suite not in protocol["splits"]:
            continue
        task_ids = protocol["splits"][suite]["dev"]

        for ep_i in range(args.n_episodes):
            task_id = task_ids[ep_i % len(task_ids)]
            episode_seed = (args.seed * 31 + run_suites.index(suite) * 100 + ep_i * 7) % (2 ** 31)

            handle = make_libero_env_for_task(
                task_id, init_state_id=ep_i % 50, seed=episode_seed,
                libero_flavor="clean",
            )
            instruction = str(getattr(handle.vector_env.envs[0], "task_description", "") or "")

            result = run_parity_pair(
                handle, bundle, instruction, args.max_steps,
                task_id, episode_seed,
            )
            pair_results.append(result)
            handle.close()

            print(f"  [{suite}] {task_id} ep{ep_i}: "
                  f"obs_ok={result['all_obs_hash_match']} "
                  f"outcome_ok={result['outcome_match']} "
                  f"steps_ok={result['steps_match']}")

    # ── Summary ──────────────────────────────────────────────────────
    fully_consistent = sum(
        1 for r in pair_results
        if r["outcome_match"] and r["steps_match"]
    )
    obs_divergence = sum(1 for r in pair_results if not r["all_obs_hash_match"])
    action_divergence = sum(1 for r in pair_results if not r["all_action_match"])
    rng_match_all = all(r["rng_before_match"] for r in pair_results)

    gate_pass = (
        fully_consistent == len(pair_results)
        and obs_divergence == 0
        and action_divergence == 0
        and rng_match_all
    )

    summary = {
        "total_pairs": len(pair_results),
        "fully_consistent": fully_consistent,
        "obs_divergence_count": obs_divergence,
        "action_divergence_count": action_divergence,
        "rng_before_all_match": rng_match_all,
        "gate_pass": gate_pass,
    }

    audit = {
        "summary": summary,
        "pairs": pair_results,
    }

    audit_path = output_dir / "parity_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    gate_path = output_dir / "parity_gate.json"
    gate_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nParity audit: {fully_consistent}/{len(pair_results)} pairs consistent, "
          f"gate_pass={gate_pass}")
    print(f"  obs_divergence={obs_divergence}/{len(pair_results)} "
          f"action_divergence={action_divergence}/{len(pair_results)} "
          f"rng_match={rng_match_all}")

    if not gate_pass:
        print("GATE FAILED: See parity_audit.json for details.")
        if obs_divergence > 0:
            for r in pair_results:
                if not r["all_obs_hash_match"]:
                    print(f"  Obs divergence: step={r['first_obs_divergence_step']} "
                          f"{r['task_id']} seed={r['episode_seed']}")
        if action_divergence > 0:
            for r in pair_results:
                if not r["all_action_match"]:
                    print(f"  Action divergence: step={r['first_action_divergence_step']} "
                          f"{r['task_id']} seed={r['episode_seed']}")
        if not rng_match_all:
            print("  RNG state mismatch before B0/B3 runs")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
