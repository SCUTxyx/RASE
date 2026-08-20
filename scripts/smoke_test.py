#!/usr/bin/env python3
"""Short, opt-in deterministic fork smoke test for LIBERO-plus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rase.envs.fork_checks import fork_roundtrip


def create_env(
    bddl: str,
    *,
    seed: int = 0,
    render_gpu_device_id: int = -1,
    noise: int | None = None,
) -> Any:
    """Create a ControlEnv without importing LIBERO at module import time."""

    try:
        import numpy as np

        # LIBERO-plus fog noise historically used np.float_ (removed in NumPy 2).
        if not hasattr(np, "float_"):
            np.float_ = np.float64  # type: ignore[attr-defined]
        from libero.libero.envs.env_wrapper import ControlEnv
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO-plus is unavailable or incomplete; activate the pinned environment "
            "(the current install may also require the `wand` package)"
        ) from exc

    bddl_argument = bddl
    if noise is not None:
        if not 1 <= noise <= 50:
            raise ValueError("noise must be in [1, 50]")
        path = Path(bddl)
        # ControlEnv's LIBERO-plus extension parses this synthetic suffix.
        bddl_argument = f"{path.with_suffix('')}_view_0_0_100_0_0_initstate_0_noise_{noise}.bddl"
    env = ControlEnv(
        bddl_file_name=bddl_argument,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_gpu_device_id=render_gpu_device_id,
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(seed)
    env.reset()
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", required=True, help="Path to one LIBERO task BDDL")
    parser.add_argument("--steps", type=int, default=5, help="Short replay length (default: 5)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--noise", type=int, default=None, help="LIBERO-plus image noise level 1..50")
    parser.add_argument("--action-json", help="Optional JSON action vector; default is all zeros")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.steps <= 50:
        raise SystemExit("--steps must be in [1, 50]")
    action = json.loads(args.action_json) if args.action_json else None
    env = create_env(
        args.bddl,
        seed=args.seed,
        render_gpu_device_id=args.render_gpu_device_id,
        noise=args.noise,
    )
    try:
        fork_roundtrip(env, steps=args.steps, action=action)
    finally:
        env.close()
    print(f"PASS: two {args.steps}-step forks were deterministic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
