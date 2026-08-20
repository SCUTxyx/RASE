#!/usr/bin/env python3
"""Route C Phase 0: restore parity + recovery headroom pilot.

Uses existing RASE infrastructure: OracleChunkContinuation (ZeroMQ OFT),
make_libero_env_for_task, select_env_action, ForkableEnv for restore parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
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
from rase.envs.forkable_env import ForkableEnv
from rase.oracle.client import OracleClient


STAGNATION_WINDOW = 20
STAGNATION_EPS = 1e-4


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _progress_signal(control_env: Any) -> float:
    try:
        pos = getattr(control_env.env, "_eef_xpos", None)
        if pos is not None:
            return float(np.linalg.norm(np.asarray(pos)))
    except Exception:
        pass
    try:
        obs = observation_from_libero_env(control_env)
        if "robot0_eef_pos" in obs:
            return float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]).reshape(-1)))
    except Exception:
        pass
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", type=str, nargs="*",
                        help="suite(s) to pilot (default: first)")
    parser.add_argument("--quantile", type=int, default=8)
    parser.add_argument("--max-student-steps", type=int, default=300)
    parser.add_argument("--max-teacher-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oft-server-port", type=int, default=5555)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    plugin_conf = protocol["plugin_config"]
    all_suites = list(protocol["splits"].keys())
    run_suites = args.suite if args.suite else [all_suites[0]]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_dir = output_dir / "phase0_pilot"
    pilot_dir.mkdir(exist_ok=True)

    seed_everything(args.seed)

    policy_path = Path(protocol["student_identity"]["checkpoint_path"])
    vlm_cache = protocol.get("vlm_cache_path", "")
    bundle = load_smolvla_policy_bundle(
        policy_path, device="cuda",
        tokenizer_path=vlm_cache if vlm_cache else None,
        observation_height=360, observation_width=360,
    )

    client = OracleClient(f"tcp://127.0.0.1:{args.oft_server_port}", timeout_ms=60000)

    results = []
    recoverable_count = 0
    post_irr_count = 0
    total = 0

    for suite in run_suites:
        if suite not in protocol["splits"]:
            continue
        task_ids = protocol["splits"][suite]["train"]

        for task_id in task_ids[:args.quantile]:
            for ep_i in range(3):
                seed = args.seed * 1000 + run_suites.index(suite) * 100 + ep_i
                handle = make_libero_env_for_task(task_id, init_state_id=ep_i % 50, seed=seed,
                                                   libero_flavor="clean")
                single = handle.vector_env.envs[0]
                instruction = str(getattr(single, "task_description", "") or "")

                obs = observation_from_libero_env(single)
                start_progress = _progress_signal(handle.control_env)
                progress_vals: list[float] = []
                student_steps = 0
                success_student = False
                boundary_t = -1

                for t in range(args.max_student_steps):
                    action = select_env_action(bundle, obs, task=instruction)
                    obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
                    obs = observation_from_libero_env(single)
                    student_steps += 1
                    terminated = bool(np.asarray(term).reshape(-1)[0])
                    truncated = bool(np.asarray(trunc).reshape(-1)[0])

                    prog = _progress_signal(handle.control_env)
                    progress_vals.append(prog)

                    if terminated or truncated:
                        success_student = success_from_info(info)
                        boundary_t = t + 1
                        break

                    if t >= STAGNATION_WINDOW:
                        window = progress_vals[t - STAGNATION_WINDOW + 1:t + 1]
                        if np.std(window) < STAGNATION_EPS and max(window) > 1e-8:
                            boundary_t = t + 1
                            break

                if boundary_t < 0:
                    boundary_t = student_steps

                # OFT recovery
                success_teacher = False
                teacher_steps = 0
                if not (success_student or handle.vector_env.envs[0]._terminated if hasattr(handle.vector_env.envs[0], '_terminated') else False):
                    oft = OracleChunkContinuation(client, instruction=instruction, control_env=handle.control_env)
                    for ti in range(args.max_teacher_steps):
                        t_action = oft.act(obs, task=instruction)
                        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(t_action))
                        obs = observation_from_libero_env(single)
                        teacher_steps += 1
                        terminated = bool(np.asarray(term).reshape(-1)[0])
                        truncated = bool(np.asarray(trunc).reshape(-1)[0])
                        if terminated or truncated:
                            success_teacher = success_from_info(info)
                            break

                recoverable = success_teacher and not success_student
                post_irr = not success_student and not success_teacher
                total += 1
                if recoverable:
                    recoverable_count += 1
                if post_irr:
                    post_irr_count += 1

                rec = {"suite": suite, "task_id": task_id, "episode_seed": seed,
                       "boundary_step": boundary_t, "student_success": success_student,
                       "teacher_success": success_teacher, "recoverable": recoverable,
                       "post_irreversible": post_irr, "student_steps": student_steps,
                       "teacher_steps": teacher_steps,
                       "timestamp": datetime.now(timezone.utc).isoformat()}
                results.append(rec)
                (pilot_dir / f"{suite}_{task_id}_ep{ep_i}.json").write_text(
                    json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
                handle.close()

    client.close()

    rec_rate = recoverable_count / max(total, 1)
    pir_rate = post_irr_count / max(total, 1)

    gate = {"total_boundaries": total, "recoverable_count": recoverable_count,
            "post_irreversible_count": post_irr_count, "recovery_rate": rec_rate,
            "post_irreversible_rate": pir_rate,
            "recovery_pass": rec_rate >= 0.30, "post_irr_pass": pir_rate < 0.5}
    gate["gate_pass"] = gate["recovery_pass"] and gate["post_irr_pass"]

    (output_dir / "phase0_recoverability_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "phase0_recoverability_pilot.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Pilot: {total} boundaries, rec={recoverable_count}({rec_rate:.2%}), "
          f"pir={post_irr_count}({pir_rate:.2%}), gate={gate['gate_pass']}")
    return 0 if gate["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
