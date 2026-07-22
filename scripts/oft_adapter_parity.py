#!/usr/bin/env python3
"""Compare OpenVLAOFTAdapter.predict to a direct official get_vla_action path.

Run in the ``oft`` conda env with checkpoint + openvla-oft on PYTHONPATH:

  export RASE_OFT_CHECKPOINT=ckpts/oft_spatial
  export RASE_OFT_SUITE=libero_spatial
  python scripts/oft_adapter_parity.py --atol 1e-4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reference_chunk(adapter, agentview, wrist, state, instruction: str) -> np.ndarray:
    from experiments.robot.openvla_utils import get_vla_action
    from experiments.robot.robot_utils import (
        invert_gripper_action,
        normalize_gripper_action,
    )

    obs = {
        "full_image": agentview[::-1, ::-1],
        "wrist_image": wrist[::-1, ::-1],
        "state": state,
    }
    actions = get_vla_action(
        adapter._cfg,
        adapter.vla,
        adapter.processor,
        obs,
        instruction,
        action_head=adapter.action_head,
        proprio_projector=adapter.proprio_projector,
        noisy_action_projector=None,
        use_film=False,
    )
    env_actions = []
    for step in actions:
        processed = normalize_gripper_action(np.asarray(step), binarize=True)
        env_actions.append(invert_gripper_action(processed).astype(np.float32))
    return np.stack(env_actions, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from rase.oracle.openvla_oft_adapter import create_adapter

    adapter = create_adapter()
    rng = np.random.default_rng(args.seed)
    agentview = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    state = rng.standard_normal(8).astype(np.float32)
    instruction = "pick up the bowl and place it on the plate"

    predicted = adapter.predict(
        {
            "agentview": agentview[None, ...],
            "wrist": wrist[None, ...],
            "proprio": state[None, ...],
        },
        {"instructions": [instruction], "proprio_format": "policy_state"},
    )["actions"]
    reference = _reference_chunk(adapter, agentview, wrist, state, instruction)[None, ...]
    diff = float(np.max(np.abs(predicted - reference)))
    print(
        {
            "suite": adapter.suite,
            "checkpoint": str(adapter.checkpoint),
            "max_abs_diff": diff,
            "atol": args.atol,
            "pass": diff <= args.atol,
            "shape": list(predicted.shape),
        }
    )
    return 0 if diff <= args.atol else 2


if __name__ == "__main__":
    raise SystemExit(main())
