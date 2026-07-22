#!/usr/bin/env python3
"""Inspect the exact LIBERO/robosuite state surface used by ForkableEnv."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _describe(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return {"type": "ndarray", "dtype": value.dtype.str, "shape": list(value.shape)}
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": _name(value)}


def inspect_env(env: Any) -> dict[str, Any]:
    from rase.envs.forkable_env import ForkableEnv

    forkable = ForkableEnv(env)
    snapshot = forkable.snapshot()
    task = env.env
    robots = []
    for robot in task.robots:
        controller = robot.controller
        robots.append(
            {
                "class": _name(robot),
                "controller_class": _name(controller),
                "controller_state": {
                    key: _describe(value)
                    for key, value in snapshot.payload["robots"][len(robots)]["controller"].items()
                },
                "interpolators": {
                    name: None if value is None else _name(value)
                    for name, value in {
                        "interpolator_pos": controller.interpolator_pos,
                        "interpolator_ori": controller.interpolator_ori,
                    }.items()
                },
                "delta_buffers": list(snapshot.payload["robots"][len(robots)]["delta_buffers"]),
                "ring_buffers": list(snapshot.payload["robots"][len(robots)]["ring_buffers"]),
            }
        )
    return {
        "task_fingerprint": forkable.task_fingerprint,
        "wrapper_class": _name(env),
        "task_class": _name(task),
        "sim_state": _describe(snapshot.payload["sim_state"]),
        "env_counters": snapshot.payload["env_counters"],
        "environment_rngs": list(snapshot.payload["rng"]["environment"]),
        "observable_count": len(snapshot.payload["observables"]),
        "robots": robots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    args = parser.parse_args()

    from smoke_test import create_env

    env = create_env(
        args.bddl,
        seed=args.seed,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    try:
        print(json.dumps(inspect_env(env), indent=2, sort_keys=True))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
