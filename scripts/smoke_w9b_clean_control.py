#!/usr/bin/env python3
"""Live W9B init/success/resume smoke without formal collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rase.backends.lerobot_libero_plus import (  # noqa: E402
    _patch_lerobot_init_states,
    catalog_task_to_suite_index,
)
from rase.backends.libero_plus_paths import ensure_libero_plus_paths  # noqa: E402
from rase.collect.lerobot_libero_plus_adapter import _lerobot_env_kwargs  # noqa: E402
from rase.collect.libero_env_factory import (  # noqa: E402
    make_libero_env_for_task,
    parse_pool_task_id,
)
from rase.collect.pipeline import collect  # noqa: E402
from rase.collect.policy_step import success_from_info  # noqa: E402
from rase.collect.w9b_schedule import (  # noqa: E402
    PROTOCOL_VERSION,
    load_w9b_schedule,
)
from rase.envs.forkable_env import ForkableEnv  # noqa: E402


def _sha(array: np.ndarray) -> str:
    value = np.asarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _observation_hash(control_env: Any) -> str:
    observation = control_env.env._get_observations()
    digest = hashlib.sha256()
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        digest.update(key.encode())
        digest.update(np.asarray(observation[key]).tobytes())
    return digest.hexdigest()


def _capture(task_id: str, init_state_id: int, policy_seed: int) -> dict[str, Any]:
    handle = make_libero_env_for_task(
        task_id,
        init_state_id=init_state_id,
        seed=policy_seed,
    )
    try:
        forkable = ForkableEnv(handle.control_env)
        sim_state = np.asarray(handle.control_env.sim.get_state().flatten()).copy()
        return {
            "task_id": task_id,
            "init_state_id": init_state_id,
            "policy_seed": policy_seed,
            "sim_state": sim_state,
            "sim_state_sha256": _sha(sim_state),
            "observation_sha256": _observation_hash(handle.control_env),
            "fingerprint": forkable.task_fingerprint,
        }
    finally:
        handle.close()


def _adapter_path_capture(
    task_id: str, init_state_id: int, policy_seed: int
) -> dict[str, Any]:
    ensure_libero_plus_paths()
    _patch_lerobot_init_states()
    import gymnasium as gym
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark

    parsed = parse_pool_task_id(task_id)
    suite = benchmark.get_benchmark_dict()[parsed.suite]()
    task_index = catalog_task_to_suite_index(parsed.catalog_task_id)

    def make_single() -> LiberoEnv:
        return LiberoEnv(
            **_lerobot_env_kwargs(
                suite=suite,
                task_index=task_index,
                suite_name=parsed.suite,
                camera_name="agentview_image,robot0_eye_in_hand_image",
                init_state_id=init_state_id,
                obs_type="pixels_agent_pos",
                observation_height=360,
                observation_width=360,
                control_mode="relative",
            )
        )

    vector = gym.vector.SyncVectorEnv([make_single])
    try:
        vector.reset(seed=[policy_seed])
        single = vector.envs[0]
        sim = np.asarray(single._env.sim.get_state().flatten()).copy()
        return {
            "sim_state": sim,
            "sim_state_sha256": _sha(sim),
            "observation_sha256": _observation_hash(single._env),
            "fingerprint": ForkableEnv(single._env).task_fingerprint,
        }
    finally:
        vector.close()


def _official_eval_capture(
    task_id: str, init_state_id: int, policy_seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_libero_plus_paths()
    _patch_lerobot_init_states()
    import gymnasium as gym
    from lerobot.envs.libero import create_libero_envs

    parsed = parse_pool_task_id(task_id)
    task_index = catalog_task_to_suite_index(parsed.catalog_task_id)
    created = create_libero_envs(
        parsed.suite,
        n_envs=init_state_id + 1,
        gym_kwargs={
            "task_ids": [task_index],
            "obs_type": "pixels_agent_pos",
            "observation_height": 360,
            "observation_width": 360,
        },
        camera_name="agentview_image,robot0_eye_in_hand_image",
        init_states=True,
        env_cls=gym.vector.SyncVectorEnv,
        control_mode="relative",
    )
    vector = created[parsed.suite][task_index]
    try:
        vector.reset(seed=[policy_seed] * (init_state_id + 1))
        single = vector.envs[init_state_id]
        sim = np.asarray(single._env.sim.get_state().flatten()).copy()
        capture = {
            "sim_state": sim,
            "sim_state_sha256": _sha(sim),
            "observation_sha256": _observation_hash(single._env),
            "fingerprint": ForkableEnv(single._env).task_fingerprint,
        }

        # Force success to verify vector terminal/final_info semantics.
        for env in vector.envs:
            env._env.check_success = lambda: True
        actions = np.zeros((init_state_id + 1, 7), dtype=np.float32)
        _, reward, terminated, truncated, info = vector.step(actions)
        semantics = {
            "reward": np.asarray(reward).tolist(),
            "terminated": np.asarray(terminated).tolist(),
            "truncated": np.asarray(truncated).tolist(),
            "has_final_info": "final_info" in info,
            "success_from_info": success_from_info(info),
            "top_level_is_success": np.asarray(info["is_success"]).tolist(),
        }
        return capture, semantics
    finally:
        vector.close()


def _resume_smoke(
    output_dir: Path,
    schedule_path: Path,
    schedule_sha256: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite smoke output {output_dir}")
    config = {
        "run_name": "ngc-w9b-resume-smoke",
        "protocol": {
            "version": PROTOCOL_VERSION,
            "schedule_path": str(schedule_path),
            "schedule_sha256": schedule_sha256,
            "maximum_episodes": 140,
        },
        "adapter": None,
        "collection": {
            "output_dir": str(output_dir),
            "episodes": 60,
            "seed": 20260730,
            "schedule_batch_id": 1,
            "action_chunks_per_episode": 1,
            "snapshot_cadence_action_chunks": 2,
            "successful_snapshot_retention": 1.0,
            "dry_run": True,
            "smoke_mode": True,
        },
    }
    first = collect(config)
    manifest_path = output_dir / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    second = collect(config)
    manifest_after = manifest_path.read_bytes()
    manifest = json.loads(manifest_after)
    identities = sorted(
        (
            entry["episode_id"],
            entry["task_id"],
            entry["init_state_id"],
        )
        for entry in manifest["states"].values()
    )
    return {
        "first_states_created": first["states_created"],
        "second_states_created": second["states_created"],
        "second_episodes_skipped": second["episodes_skipped_already_in_pool"],
        "manifest_byte_stable": manifest_before == manifest_after,
        "n_entries": len(identities),
        "n_unique_identities": len(set(identities)),
        "identity_sample": identities[:3],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/ngc_w9b_clean_control_smoke.json"),
    )
    parser.add_argument(
        "--resume-output",
        type=Path,
        default=Path("runs/ngc_w9b_resume_smoke_pool"),
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path("configs/w9b_clean_control_schedule.json"),
    )
    parser.add_argument(
        "--schedule-sha256",
        default="71e61d3cd4d36469652735293b7c8e23b93fb22aa450487c58d21e085e8e1943",
    )
    parser.add_argument("--task-id", default="libero_spatial_000005")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite smoke output {args.output}")
    if not args.output.as_posix().startswith("runs/ngc_w9b_") or "smoke" not in args.output.name:
        raise ValueError("smoke output must match runs/ngc_w9b_*_smoke*")
    load_w9b_schedule(args.schedule, expected_sha256=args.schedule_sha256)

    captures = [_capture(args.task_id, init_id, 101) for init_id in (0, 1, 2)]
    pairwise = []
    for left in range(3):
        for right in range(left + 1, 3):
            delta = captures[left]["sim_state"] - captures[right]["sim_state"]
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "l2": float(np.linalg.norm(delta)),
                    "max_abs": float(np.max(np.abs(delta))),
                }
            )
    if min(item["max_abs"] for item in pairwise) <= 1e-6:
        raise AssertionError("init_state_id 0/1/2 differ only by numerical noise")

    repeat_a = _capture(args.task_id, 1, 101)
    repeat_b = _capture(args.task_id, 1, 101)
    repeated = {
        key: repeat_a[key] == repeat_b[key]
        for key in ("sim_state_sha256", "observation_sha256", "fingerprint")
    }
    if not all(repeated.values()):
        raise AssertionError(f"same init/seed is not repeatable: {repeated}")

    seed_a = _capture(args.task_id, 1, 101)
    seed_b = _capture(args.task_id, 1, 202)
    policy_seed_invariant = seed_a["sim_state_sha256"] == seed_b["sim_state_sha256"]
    if not policy_seed_invariant:
        raise AssertionError("policy_seed changed step-0 sim_state")

    adapter = _adapter_path_capture(args.task_id, 2, 303)
    official, semantics = _official_eval_capture(args.task_id, 2, 303)
    path_parity = {
        key: adapter[key] == official[key]
        for key in ("sim_state_sha256", "observation_sha256", "fingerprint")
    }
    if not all(path_parity.values()):
        raise AssertionError(f"adapter/eval path mismatch: {path_parity}")
    if (
        not semantics["success_from_info"]
        or not semantics["has_final_info"]
        or not semantics["terminated"][2]
        or semantics["truncated"][2]
    ):
        raise AssertionError(f"unexpected terminal semantics: {semantics}")

    resume = _resume_smoke(
        args.resume_output,
        args.schedule,
        args.schedule_sha256,
    )
    if (
        resume["second_states_created"] != 0
        or not resume["manifest_byte_stable"]
        or resume["n_entries"] != resume["n_unique_identities"]
    ):
        raise AssertionError(f"resume smoke failed: {resume}")

    result = {
        "schema_version": "rase-w9b-clean-control-smoke/v1",
        "status": "complete",
        "formal_collection_started": False,
        "task_id": args.task_id,
        "init_captures": [
            {key: value for key, value in capture.items() if key != "sim_state"}
            for capture in captures
        ],
        "init_pairwise_differences": pairwise,
        "same_init_repeatability": repeated,
        "policy_seed_step0_invariant": policy_seed_invariant,
        "adapter_vs_official_eval": path_parity,
        "terminal_semantics": semantics,
        "resume": resume,
        "schedule_sha256": args.schedule_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
