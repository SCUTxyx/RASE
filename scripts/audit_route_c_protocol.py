#!/usr/bin/env python3
"""Route C: Freeze protocol — model identity, task split, plugin config.

Output: protocol_c_frozen.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FROZEN_SEED = 202608041200


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_dict(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(canonical.encode("utf-8"))


def _partition_tasks(*, n_tasks: int, full_prefix: str, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    indices = list(range(n_tasks))
    rng.shuffle(indices)
    n_train = max(1, int(n_tasks * 0.6))
    n_dev = max(1, int(n_tasks * 0.2))
    train = [f"{full_prefix}_{i + 1:06d}" for i in sorted(indices[:n_train])]
    dev = [f"{full_prefix}_{i + 1:06d}" for i in sorted(indices[n_train : n_train + n_dev])]
    test = [f"{full_prefix}_{i + 1:06d}" for i in sorted(indices[n_train + n_dev :])]
    return {"train": train, "dev": dev, "test": test}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--smolvla-checkpoint", type=Path, default=ROOT / "ckpts" / "smolvla_libero")
    parser.add_argument("--oft-checkpoints-dir", type=Path, default=ROOT / "ckpts")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suites = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]
    all_splits: dict[str, dict[str, list[str]]] = {}
    for suite in suites:
        all_splits[suite] = _partition_tasks(n_tasks=10, full_prefix=suite, seed=args.seed)

    student_id = {
        "checkpoint_path": str(args.smolvla_checkpoint),
        "model_sha256": _sha256_file(args.smolvla_checkpoint / "model.safetensors"),
        "config_sha256": _sha256_file(args.smolvla_checkpoint / "config.json"),
    }

    teacher_suites = {"libero_object": "oft_object", "libero_goal": "oft_goal",
                      "libero_spatial": "oft_spatial", "libero_10": "oft_10"}
    teacher_ids: dict[str, Any] = {}
    for api_suite, dirname in teacher_suites.items():
        ckpt = args.oft_checkpoints_dir / dirname
        teacher_ids[api_suite] = {
            "checkpoint_path": str(ckpt),
            "config_sha256": _sha256_file(ckpt / "config.json"),
        }
        la = ckpt / "lora_adapter"
        if (la / "adapter_model.safetensors").is_file():
            teacher_ids[api_suite]["lora_adapter_sha256"] = _sha256_file(la / "adapter_model.safetensors")

    action_schema = {
        "action_dim": 7,
        "action_descriptions": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        "gripper_semantics": {"teacher_oft": "binarized_and_inverted", "student_smolvla": "lerobot_default"},
        "control_mode": "relative",
        "obs_type": "pixels_agent_pos",
        "cameras": ["agentview_image", "robot0_eye_in_hand_image"],
        "observation_height": 360,
        "observation_width": 360,
        "proprio_dim": 8,
        "chunk_size_smolvla": 50,
    }

    plugin_config = {
        "plugin_history_window": 8,
        "plugin_horizons": [4, 8, 16],
        "delta_clip_per_dim": 0.5,
        "stagnation_window": 20,
        "stagnation_eps": 1e-4,
        "handback_consecutive_progress": 3,
        "max_takeover_steps": 16,
        "action_rate_limit": 0.1,
        "teacher_max_steps": 300,
        "student_max_steps": 300,
        "min_recoverable_tasks_per_suite": 2,
        "min_unique_boundaries": 48,
        "recovery_rate_threshold": 0.30,
        "mix_g_t_ramp": [0.0, 0.3, 0.6, 1.0],
    }

    protocol = {
        "schema_version": "rase-route-c-plugin/v1",
        "frozen_seed": args.seed,
        "splits": all_splits,
        "student_identity": student_id,
        "teacher_identities": teacher_ids,
        "action_schema": action_schema,
        "plugin_config": plugin_config,
        "vlm_cache_path": str(Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"),
    }
    protocol["protocol_sha256"] = _hash_dict({k: v for k, v in protocol.items() if k != "protocol_sha256"})

    path = output_dir / "protocol_c_frozen.json"
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Route C protocol frozen: {path}")
    print(f"  split: 24 train / 8 dev / 8 test")
    print(f"  student SHA: {student_id['model_sha256'][:16]}...")
    print(f"  protocol SHA: {protocol['protocol_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
