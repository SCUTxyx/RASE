#!/usr/bin/env python3
"""R4 Phase 0A: Freeze task split, model identity, and action/normalization schema.

Output: protocol_frozen.json in the output directory.
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

LIKE = object()

FROZEN_SEED = 202608041200


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_dict(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(canonical.encode("utf-8"))


def _partition_tasks(*, suite: str, n_tasks: int, seed: int, full_prefix: str) -> dict[str, list[str]]:
    """Partition n_tasks into train/dev/test by id (full LIBERO catalog format).

    Target: 6 train / 2 dev / 2 test per suite = 10 tasks/suite in clean LIBERO.
    With fewer tasks, allocate proportionally (train first, then dev, rest test).
    """
    rng = random.Random(seed * (hash(suite) & 0x7FFFFFFF))
    indices = list(range(n_tasks))
    rng.shuffle(indices)
    n_train = max(1, int(n_tasks * 0.6))
    n_dev = max(1, int(n_tasks * 0.2))
    n_test = n_tasks - n_train - n_dev
    train = [f"{full_prefix}_{i + 1:06d}" for i in sorted(indices[:n_train])]
    dev = [f"{full_prefix}_{i + 1:06d}" for i in sorted(indices[n_train : n_train + n_dev])]
    test = [f"{full_prefix}_{i + 1:06d}" for i in sorted(indices[n_train + n_dev :])]
    return {"train": train, "dev": dev, "test": test}


def _smolvla_identity(checkpoint: str) -> dict[str, str]:
    root = Path(checkpoint).resolve()
    return {
        "checkpoint_path": str(root),
        "model_sha256": _sha256_file(root / "model.safetensors"),
        "config_sha256": _sha256_file(root / "config.json"),
    }


def _oft_identity(checkpoint: str) -> dict[str, str]:
    root = Path(checkpoint).resolve()
    info: dict[str, str] = {
        "checkpoint_path": str(root),
        "config_sha256": _sha256_file(root / "config.json"),
    }
    la = root / "lora_adapter"
    if (la / "adapter_model.safetensors").is_file():
        info["lora_adapter_sha256"] = _sha256_file(la / "adapter_model.safetensors")
        info["lora_config_sha256"] = _sha256_file(la / "adapter_config.json")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--smolvla-checkpoint", type=Path, default=ROOT / "ckpts" / "smolvla_libero")
    parser.add_argument("--oft-checkpoints-dir", type=Path, default=ROOT / "ckpts")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

    # ---- task split: 24 train / 8 dev / 8 test per suite (10 tasks each in clean LIBERO) ---
    n_clean = 10
    all_splits: dict[str, dict[str, list[str]]] = {}
    for suite in suites:
        all_splits[suite] = _partition_tasks(
            suite=suite.replace("libero_", ""),
            n_tasks=n_clean,
            seed=args.seed,
            full_prefix=suite,
        )

    # ---- model identity ---
    student_id = _smolvla_identity(str(args.smolvla_checkpoint))
    teacher_suites: dict[str, str] = {
        "libero_object": "oft_object",
        "libero_goal": "oft_goal",
        "libero_spatial": "oft_spatial",
        "libero_10": "oft_10",
    }
    teacher_ids: dict[str, Any] = {}
    for api_suite, dirname in teacher_suites.items():
        ckpt = args.oft_checkpoints_dir / dirname
        teacher_ids[api_suite] = _oft_identity(str(ckpt))

    # ---- action / normalization schema ---
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
        "n_action_steps": 1,
    }

    protocol = {
        "schema_version": "rase-recovery-distill-r4/v1",
        "timestamp_utc": "2026-08-06T08:00:00Z",
        "frozen_seed": args.seed,
        "splits": all_splits,
        "student_identity": student_id,
        "teacher_identities": teacher_ids,
        "action_schema": action_schema,
        "protocol_sha256": LIKE,  # placeholder; filled after canonical hash
    }

    # Fill deterministic hash
    filled = dict(protocol)
    filled.pop("protocol_sha256", None)
    protocol["protocol_sha256"] = _hash_dict(filled)

    path = output_dir / "protocol_frozen.json"
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== R4 Phase 0A: Protocol Frozen ===")
    sections = ["splits", "student_identity", "teacher_identities", "action_schema"]
    for sec in sections:
        print(f"  {sec}: included")
    print(f"  protocol_sha256: {protocol['protocol_sha256']}")
    print(f"  output: {path}")
    print("Phase 0A PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
