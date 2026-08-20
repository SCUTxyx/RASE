#!/usr/bin/env python3
"""D0-D: Trigger calibration.

Runs student rollouts on 24-32 independent states, saves snapshots at
multiple candidate trigger points, and evaluates persistent OFT recovery
from each trigger. Used to calibrate the stagnation detector timing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.collect.oracle_continuation import OracleChunkContinuation
from rase.collect.policy_step import as_batched_action, select_env_action, success_from_info
from rase.collect.pool_candidates import observation_from_libero_env
from rase.oracle.client import OracleClient
from rase.recovery.stagnation import StagnationDetector


def _progress(control_env: Any) -> float:
    try:
        pos = getattr(control_env.env, "_eef_xpos", None)
        if pos is not None:
            return float(np.linalg.norm(np.asarray(pos)))
    except Exception:
        pass
    return 0.0


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_persistent_oft_from_state(
    handle: Any, client: OracleClient, instruction: str, max_steps: int,
) -> dict:
    env = handle.vector_env.envs[0]
    obs = observation_from_libero_env(env)
    oft = OracleChunkContinuation(client, instruction=instruction,
                                   control_env=handle.control_env)
    for t in range(max_steps):
        t_action = oft.act(obs, task=instruction)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(t_action))
        obs = observation_from_libero_env(env)
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            return {"success": success_from_info(info), "steps": t + 1}
    return {"success": False, "steps": max_steps}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", type=str, nargs="*")
    parser.add_argument("--n-boundaries", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--oft-server-port", type=int, default=5555)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    plugin_conf = protocol["plugin_config"]
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

    client = OracleClient(f"tcp://127.0.0.1:{args.oft_server_port}", timeout_ms=60000)

    window = plugin_conf["stagnation_window"]
    eps = plugin_conf["stagnation_eps"]

    # Sweep over multiple trigger windows
    candidate_windows = [window, window // 2, max(window // 4, 5), window + 5]
    candidate_epss = [eps, eps * 0.5, eps * 2.0]

    results = []
    task_set: set[str] = set()
    suite_set: set[str] = set()

    for boundary_i in range(args.n_boundaries):
        suite = run_suites[boundary_i % len(run_suites)]
        if suite not in protocol["splits"]:
            continue
        task_ids = protocol["splits"][suite]["dev"]
        task_id = task_ids[boundary_i % len(task_ids)]
        seed = args.seed * 31 + boundary_i * 100

        handle = make_libero_env_for_task(
            task_id, init_state_id=boundary_i % 50, seed=seed,
            libero_flavor="clean")
        instruction = str(getattr(handle.vector_env.envs[0],
                                  "task_description", "") or "")
        obs = observation_from_libero_env(handle.vector_env.envs[0])

        # Run student until any of the candidate detectors trigger
        progress_vals = []
        trigger_info: dict[str, int] = {}
        for t in range(args.max_steps):
            action = select_env_action(bundle, obs, task=instruction)
            obs, reward, term, trunc, info = handle.vector_env.step(
                as_batched_action(action))
            obs = observation_from_libero_env(handle.vector_env.envs[0])
            progress_vals.append(_progress(handle.control_env))

            for w in candidate_windows:
                for e in candidate_epss:
                    key = f"w{w}_eps{e:.0e}"
                    if key not in trigger_info and t >= w:
                        pw = progress_vals[-w:]
                        if np.std(pw) < e and np.max(pw) > 1e-8:
                            trigger_info[key] = t

            if bool(np.asarray(term).reshape(-1)[0]) or \
               bool(np.asarray(trunc).reshape(-1)[0]):
                break

        handle.close()

        # For each trigger, re-create env and run OFT
        trigger_results = {}
        for key, trigger_t in trigger_info.items():
            h_trigger = make_libero_env_for_task(
                task_id, init_state_id=boundary_i % 50, seed=seed,
                libero_flavor="clean")
            instr_t = str(getattr(h_trigger.vector_env.envs[0],
                                  "task_description", "") or "")

            # Fast-forward to trigger_t
            obs_t = observation_from_libero_env(h_trigger.vector_env.envs[0])
            for _ in range(trigger_t):
                action = select_env_action(bundle, obs_t, task=instr_t)
                obs_t, _, _, _, _ = h_trigger.vector_env.step(as_batched_action(action))
                obs_t = observation_from_libero_env(h_trigger.vector_env.envs[0])

            oft_result = run_persistent_oft_from_state(
                h_trigger, client, instr_t, args.max_steps)
            trigger_results[key] = {
                "trigger_step": trigger_t,
                "oft_success": oft_result["success"],
                "oft_steps": oft_result["steps"],
            }
            h_trigger.close()

        task_set.add(task_id)
        suite_set.add(suite)

        results.append({
            "boundary_id": f"calib_{boundary_i:03d}",
            "task_id": task_id,
            "suite": suite,
            "seed": seed,
            "trigger_results": trigger_results,
        })

        n_rescued = sum(1 for v in trigger_results.values() if v["oft_success"])
        print(f"  [{boundary_i:03d}] {task_id}: {len(trigger_results)} triggers, "
              f"{n_rescued} OFT rescues")

    client.close()

    # Best trigger: highest rescue rate
    by_trigger: dict[str, list[bool]] = {}
    for r in results:
        for key, tr in r["trigger_results"].items():
            by_trigger.setdefault(key, []).append(tr["oft_success"])

    best_key = None
    best_rate = 0.0
    trigger_ranking = []
    for key, successes in by_trigger.items():
        rate = sum(successes) / max(len(successes), 1)
        trigger_ranking.append({"key": key, "rate": rate, "n": len(successes)})
        if rate > best_rate:
            best_rate = rate
            best_key = key

    gate_pass = best_rate >= 0.30

    summary = {
        "n_boundaries": len(results),
        "n_tasks": len(task_set),
        "n_suites": len(suite_set),
        "best_trigger": best_key,
        "best_rescue_rate": best_rate,
        "trigger_ranking": sorted(trigger_ranking, key=lambda x: -x["rate"]),
        "gate_pass": gate_pass,
    }

    (output_dir / "trigger_calibration.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "trigger_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nTrigger calibration: best='{best_key}' rate={best_rate:.1%} gate={gate_pass}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
