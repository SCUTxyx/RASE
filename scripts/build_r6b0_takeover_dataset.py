#!/usr/bin/env python3
"""Build the frozen initial-boundary multi-VLA takeover-risk dataset."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resize_chw(value: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)


def hashed_instruction(text: str, dim: int = 256) -> np.ndarray:
    """Deployment-safe, model-agnostic lexical instruction features."""
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    words = normalized.split()
    features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    features += [normalized[index:index + 3] for index in range(max(0, len(normalized) - 2))]
    value = np.zeros(dim, dtype=np.float32)
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        value[index] += sign
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0 else value


def observation_from_loaded(loaded: Any) -> dict[str, Any]:
    from rase.collect.pool_candidates import batch_single_gym_observation

    cache = loaded.controller_state["obs_cache"]
    controller = loaded.controller_state["robots"][0]["controller"]
    single = {
        "pixels": {
            "image": np.asarray(cache["agentview_image"]),
            "image2": np.asarray(cache["robot0_eye_in_hand_image"]),
        },
        "robot_state": {
            "eef": {
                "pos": np.asarray(cache["robot0_eef_pos"]),
                "quat": np.asarray(cache["robot0_eef_quat"]),
                "mat": np.asarray(controller["initial_ee_ori_mat"]),
            },
            "gripper": {
                "qpos": np.asarray(cache["robot0_gripper_qpos"]),
                "qvel": np.asarray(cache["robot0_gripper_qvel"]),
            },
            "joints": {
                "pos": np.asarray(cache["robot0_joint_pos"]),
                "vel": np.asarray(cache["robot0_joint_vel"]),
            },
        },
        "task": str(loaded.metadata.instruction),
    }
    return batch_single_gym_observation(single)


def policy_specs(root: Path) -> list[dict[str, Any]]:
    text = root / "ckpts/paligemma_tokenizer_35e4f46"
    return [
        {"policy_id": "smolvla_libero", "path": root / "ckpts/smolvla_libero",
         "tokenizer": None, "action_tokenizer": None},
        {"policy_id": "pi0fast_libero", "path": root / "ckpts/pi0fast_libero",
         "tokenizer": text,
         "action_tokenizer": root / "ckpts/pi0fast_action_tokenizer_79ae83e"},
        {"policy_id": "pi05_libero", "path": root / "ckpts/pi05_libero",
         "tokenizer": text, "action_tokenizer": None},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--oft-analysis", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-states", type=int, default=0)
    args = parser.parse_args()

    initial = read_json(args.initial_keys.resolve())
    atlas = read_json(args.atlas.resolve())
    oft = read_json(args.oft_analysis.resolve())
    if atlas.get("atlas_gate_status") != "ready":
        raise ValueError("R6-A atlas is not ready")
    keys = [str(value) for value in initial["state_keys"]]
    if len(keys) != 48 or len(set(keys)) != 48:
        raise ValueError("expected the frozen 48-state cohort")
    if args.max_states:
        keys = keys[: args.max_states]
    oft_rows = {str(row["state_key"]): row["oft_only_result"] for row in oft["per_task"]}

    from rase.collect.forked_rollout import InProcessLeRobotContinuation, load_lerobot_policy_bundle
    from rase.collect.state_pool import StatePool
    from rase.risk.canonical_action import summary_from_chunk
    from rase.risk.vla_action_adapters import create_vla_adapter

    pool = StatePool(Path(str(initial["pool"])).resolve())
    loaded_states = {key: pool.read_state(key, load_observations=True) for key in keys}
    rows: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    action_summary: list[np.ndarray] = []

    specs = policy_specs(ROOT)
    for spec in specs:
        policy_id = str(spec["policy_id"])
        source = []
        for seed in (0, 1):
            path = args.atlas_root / policy_id / f"seed_{seed}/summary.json"
            report = read_json(path.resolve())
            source.append({str(row["state_key"]): row for row in report["per_state"]})
        if any(set(report) != set(initial["state_keys"]) for report in source):
            raise ValueError(f"{policy_id}: source summary state mismatch")

        bundle = load_lerobot_policy_bundle(
            spec["path"], device=args.device, num_steps=10, n_action_steps=10,
            tokenizer_path=spec["tokenizer"], action_tokenizer_path=spec["action_tokenizer"],
            observation_height=360, observation_width=360,
        )
        adapter = create_vla_adapter(policy_id)
        for index, key in enumerate(keys):
            loaded = loaded_states[key]
            seed0 = source[0][key]
            continuation = InProcessLeRobotContinuation(bundle, seed=int(seed0["rollout_seed"]))
            continuation.reset()
            observation = observation_from_loaded(loaded)
            raw_actions = []
            for _ in range(10):
                raw_actions.append(np.asarray(
                    continuation.act(observation, task=loaded.metadata.instruction),
                    dtype=np.float32,
                ).reshape(-1, 7)[0])
            action = summary_from_chunk(
                adapter.to_canonical(np.stack(raw_actions))
            ).cpu().numpy().astype(np.float32)
            cache = loaded.controller_state["obs_cache"]
            image = np.stack([
                resize_chw(cache["agentview_image"], args.image_size),
                resize_chw(cache["robot0_eye_in_hand_image"], args.image_size),
            ])
            labels = [bool(source[seed][key]["source_success"]) for seed in (0, 1)]
            source_steps = [int(source[seed][key]["result"]["env_steps"]) for seed in (0, 1)]
            persistent = oft_rows[key]
            rows.append({
                "state_key": key,
                "task_id": loaded.metadata.task_id,
                "episode_id": loaded.metadata.episode_id,
                "policy_id": policy_id,
                "instruction": str(loaded.metadata.instruction),
                "source_successes": int(sum(labels)),
                "source_trials": 2,
                "source_seed_success": labels,
                "source_seed_steps": source_steps,
                "persistent_success": bool(persistent["success"]),
                "persistent_teacher_steps": int(persistent["env_steps"]),
            })
            images.append(image)
            proprio.append(np.asarray(loaded.proprio, dtype=np.float32))
            action_summary.append(action)
            print(f"R6B0_FEATURE policy={policy_id} state={index+1}/{len(keys)}", flush=True)
        del bundle
        from rase.backends import lerobot_libero_plus
        lerobot_libero_plus._POLICY_CACHE.clear()
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    policy_order = [str(spec["policy_id"]) for spec in specs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        image=np.stack(images), proprio=np.stack(proprio), action_summary=np.stack(action_summary),
        instruction=np.asarray([row["instruction"] for row in rows]),
        language_hash=np.stack([hashed_instruction(row["instruction"]) for row in rows]),
        state_key=np.asarray([row["state_key"] for row in rows]),
        task_id=np.asarray([row["task_id"] for row in rows]),
        episode_id=np.asarray([row["episode_id"] for row in rows]),
        policy_id=np.asarray([row["policy_id"] for row in rows]),
        policy_index=np.asarray([policy_order.index(row["policy_id"]) for row in rows], dtype=np.int64),
        source_successes=np.asarray([row["source_successes"] for row in rows], dtype=np.float32),
        source_trials=np.asarray([row["source_trials"] for row in rows], dtype=np.float32),
        source_seed_success=np.asarray([row["source_seed_success"] for row in rows], dtype=np.int8),
        source_seed_steps=np.asarray([row["source_seed_steps"] for row in rows], dtype=np.int32),
        persistent_success=np.asarray([row["persistent_success"] for row in rows], dtype=np.float32),
        persistent_teacher_steps=np.asarray([row["persistent_teacher_steps"] for row in rows], dtype=np.float32),
    )
    report = {
        "schema_version": "rase-r6b0-takeover-dataset/v1",
        "status": "complete" if not args.max_states else "diagnostic_smoke",
        "scientific_scope": "initial exact-state takeover feasibility; not trajectory-boundary control",
        "dataset": str(args.output.resolve()), "dataset_sha256": sha256(args.output),
        "initial_keys": str(args.initial_keys.resolve()), "initial_keys_sha256": sha256(args.initial_keys.resolve()),
        "atlas": str(args.atlas.resolve()), "atlas_sha256": sha256(args.atlas.resolve()),
        "oft_analysis": str(args.oft_analysis.resolve()), "oft_analysis_sha256": sha256(args.oft_analysis.resolve()),
        "n_rows": len(rows), "n_states": len(keys),
        "n_tasks": len({row["task_id"] for row in rows}), "policies": policy_order,
        "qualified_policies": atlas["passing_policy_pairs"],
        "feature_policy": "two RGB views + 8D proprio + instruction hash + seed0 canonical 10-step action summary + policy ID",
        "forbidden_features": ["suite", "perturb_dimension", "task_ordinal", "future_outcome"],
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
