#!/usr/bin/env python3
"""R4: Paired evaluation of recovery LoRA variants.

For B2 and B3 (and optionally B1), evaluate on dev/test tasks with shared seeds.
Performs paired evaluation per task, per training seed, per episode seed.
Outputs raw results JSON for later McNemar + bootstrap analysis.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.adapt.recovery_lora import load_lora_onto_policy, set_adapter_enabled
from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.policy_step import as_batched_action, select_env_action, success_from_info
from rase.collect.pool_candidates import observation_from_libero_env


def _load_protocol(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "protocol_frozen.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _eval_one_policy(
    *,
    task_id: str,
    init_state_id: int,
    policy_bundle: dict[str, Any],
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    np.random.seed(seed)
    handle = make_libero_env_for_task(task_id, init_state_id=init_state_id, seed=seed)
    success = False
    steps = 0
    stop = "horizon"

    for _ in range(max(1, max_steps)):
        obs = observation_from_libero_env(handle.control_env.envs[0])
        action = select_env_action(policy_bundle, obs, task="")
        _o, _r, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        steps += 1
        if success_from_info(info):
            success = True
            stop = "success"
            break
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            stop = "terminated" if bool(np.asarray(term).reshape(-1)[0]) else "truncated"
            break
    handle.close()
    return {"task_id": task_id, "init_state_id": init_state_id, "seed": seed,
            "success": success, "steps": steps, "stop_reason": stop}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True, help="LoRA adapter to evaluate")
    parser.add_argument("--label", required=True, help="Label (B1, B2, B3)")
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--tasks", nargs="*", default=[], help="Task list")
    parser.add_argument("--seeds-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smolvla-checkpoint", type=Path, default=ROOT / "ckpts/smolvla_libero")
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "ckpts/SmolVLM2-500M-Instruct")
    parser.add_argument("--protocol-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load protocol for task split
    proto_dir = args.protocol_dir or output_dir.parent.parent
    protocol = _load_protocol(
        proto_dir if isinstance(proto_dir, Path) else Path(str(proto_dir)))

    # Resolve tasks
    if args.tasks:
        tasks = args.tasks
    else:
        tasks = []
        if protocol:
            for suite, splits in protocol.get("splits", {}).items():
                tasks.extend(splits.get(args.split, []))

        if not tasks:
            tasks = [
                "libero_object_000001", "libero_object_000002",
                "libero_goal_000001", "libero_goal_000002",
                "libero_spatial_000001", "libero_spatial_000002",
                "libero_10_000001", "libero_10_000002",
            ]

    print(f"Evaluating {args.label}, training_seed={args.training_seed}")
    print(f"  Tasks: {tasks}, seeds_per_task: {args.seeds_per_task}")

    # Load policy with LoRA
    bundle = load_smolvla_policy_bundle(
        Path(args.smolvla_checkpoint),
        device=args.device,
        num_steps=10, n_action_steps=10,
        tokenizer_path=Path(args.tokenizer_path),
        observation_height=360, observation_width=360,
    )
    load_lora_onto_policy(bundle["policy"], str(args.adapter_dir))
    set_adapter_enabled(
        type("H", (), {"policy": bundle["policy"], "enabled": True})(), True,
    )

    results: list[dict[str, Any]] = []
    for task_id in tasks:
        for ep_seed in range(args.seeds_per_task):
            init_state_id = (args.training_seed * 10 + ep_seed) % 50
            result = _eval_one_policy(
                task_id=task_id,
                init_state_id=init_state_id,
                policy_bundle=bundle,
                max_steps=args.max_steps,
                seed=args.training_seed * 1000 + ep_seed,
            )
            result["training_seed"] = args.training_seed
            result["variant"] = args.label
            results.append(result)

    n_success = sum(1 for r in results if r["success"])
    n_total = len(results)

    eval_path = output_dir / f"eval_{args.label}_seed{args.training_seed:02d}.json"
    summary = {
        "variant": args.label,
        "training_seed": args.training_seed,
        "n_tasks": len(tasks),
        "seeds_per_task": args.seeds_per_task,
        "n_total": n_total,
        "n_success": n_success,
        "success_rate": n_success / max(1, n_total),
        "results": results,
    }
    eval_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  Success: {n_success}/{n_total} = {summary['success_rate']:.3f}")
    print(f"  Output: {eval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
