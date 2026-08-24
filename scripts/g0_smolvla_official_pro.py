#!/usr/bin/env python3
"""Fixed-execution source audit on a provenance-verified LIBERO-PRO layout.

The program is intentionally a baseline only: no refresh, selector, replan
rule, or candidate selection is present.  Its purpose is to establish that a
frozen SmolVLA source has non-degenerate competence on the *exact* official
layout that will later be used for the C/R opportunity test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "rase-v6-official-source-audit/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ready_manifest(path: Path, *, suite: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("provenance manifest must be an object")
    if value.get("status") != "ready" or value.get("official_claim_permitted") is not True:
        raise ValueError("provenance manifest is not ready; do not run an official benchmark")
    requested = value.get("requested")
    if not isinstance(requested, dict) or requested.get("suite") != suite:
        raise ValueError("provenance manifest suite does not match --suite")
    return value


def parse_tasks(value: str) -> list[int]:
    tasks = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not tasks or min(tasks) < 1:
        raise ValueError("--tasks must be a non-empty list of positive task numbers")
    if len(set(tasks)) != len(tasks):
        raise ValueError("--tasks has duplicates")
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pro-root", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--episodes-per-task", type=int, default=8)
    parser.add_argument("--policy", type=Path, default=Path("ckpts/smolvla_libero"))
    parser.add_argument("--tokenizer", type=Path, default=Path("ckpts/SmolVLM2-500M-Instruct"))
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--native-horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes_per_task < 1 or args.native_horizon < 1:
        parser.error("--episodes-per-task and --native-horizon must be positive")
    if not np.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be non-negative and finite")

    manifest_path = args.provenance_manifest.resolve()
    provenance = read_ready_manifest(manifest_path, suite=args.suite)
    pro_root = args.pro_root.resolve()
    if str(provenance.get("runtime_root")) != str(pro_root):
        raise ValueError("--pro-root does not equal provenance runtime_root")
    tasks = parse_tasks(args.tasks)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite output: {output}")

    # Delayed imports make --help and static validation usable outside the
    # simulator environment.
    from libero.libero.utils import get_libero_path, set_libero_path
    from rase.collect.forked_rollout import InProcessSmolVLAContinuation, load_lerobot_policy_bundle
    from rase.collect.libero_env_factory import make_libero_env_for_task
    from rase.collect.policy_step import as_batched_action, success_from_info
    from rase.collect.pool_candidates import observation_from_libero_env

    policy = args.policy if args.policy.is_absolute() else ROOT / args.policy
    tokenizer = args.tokenizer if args.tokenizer.is_absolute() else ROOT / args.tokenizer
    bundle = load_lerobot_policy_bundle(
        policy, device="cuda", num_steps=args.native_horizon,
        tokenizer_path=tokenizer, observation_height=360, observation_width=360,
    )
    original_root = get_libero_path("bddl_files")
    set_libero_path(str(pro_root))
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for task_number in tasks:
            task_id = f"{args.suite}_{task_number:06d}"
            for episode in range(args.episodes_per_task):
                env_seed = args.seed + 10_000 * task_number + episode
                generation_seed = args.seed + 100_000 * task_number + episode
                handle = make_libero_env_for_task(
                    task_id, init_state_id=episode % 10, seed=env_seed,
                    observation_height=360, observation_width=360,
                    libero_clean_root="/root/autodl-tmp/src/LIBERO", libero_flavor="clean",
                )
                try:
                    single = handle.vector_env.envs[0]
                    task = str(single.task_description)
                    observation = observation_from_libero_env(single)
                    continuation = InProcessSmolVLAContinuation(
                        bundle, temperature=args.temperature, seed=generation_seed,
                    )
                    continuation.reset()
                    horizon = int(getattr(single, "_max_episode_steps", 600))
                    success = False
                    stop_reason = "horizon"
                    for step in range(horizon):
                        action = continuation.act(observation, task=task)
                        observation, _reward, terminated, truncated, info = handle.vector_env.step(
                            as_batched_action(action)
                        )
                        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                            success = bool(success_from_info(info))
                            stop_reason = "success" if success else "terminal_failure"
                            break
                    records.append({
                        "task_id": task_id,
                        "task_number": task_number,
                        "episode": episode,
                        "init_state_id": episode % 10,
                        "environment_seed": env_seed,
                        "generation_seed": generation_seed,
                        "success": success,
                        "steps": step + 1,
                        "stop_reason": stop_reason,
                    })
                finally:
                    handle.close()
                print(
                    f"official source audit {len(records)}/{len(tasks) * args.episodes_per_task} "
                    f"{task_id} ep={episode} success={records[-1]['success']}", flush=True,
                )
    finally:
        set_libero_path(original_root)

    per_task: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "successes": 0})
    for record in records:
        task_summary = per_task[record["task_id"]]
        task_summary["episodes"] += 1
        task_summary["successes"] += int(record["success"])
    successes = sum(int(record["success"]) for record in records)
    report = {
        "schema_version": SCHEMA,
        "protocol": "fixed native-horizon source execution; no selector or refresh",
        "source": {
            "policy": str(policy), "policy_sha256": sha256_file(policy / "config.json") if (policy / "config.json").is_file() else None,
            "temperature": args.temperature, "native_horizon": args.native_horizon,
        },
        "official_layout": {
            "provenance_manifest": str(manifest_path),
            "provenance_manifest_sha256": sha256_file(manifest_path),
            "runtime_root": str(pro_root),
            "variant": provenance["requested"]["variant"],
        },
        "suite": args.suite,
        "tasks": tasks,
        "episodes_per_task": args.episodes_per_task,
        "n_episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records) if records else None,
        "source_eligibility": "pass" if records and 0.15 < successes / len(records) < 0.85 else "fail",
        "per_task": dict(sorted(per_task.items())),
        "records": records,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "successes": successes, "n_episodes": len(records),
        "success_rate": report["success_rate"], "source_eligibility": report["source_eligibility"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
