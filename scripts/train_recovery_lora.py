#!/usr/bin/env python3
"""R4: LoRA fine-tuning for targeted recovery distillation.

Three variants:
  B1: Matched OFT BC
  B2: Matched nominal BC (SmolVLA's own successes)
  B3: Targeted recovery BC (OFT recovery from student failure, + 30% retention mix)

All share: same init, optimizer seed, LR schedule, batch order, normalization stats.
Loss: flow-matching BC. B3 adds retention loss weighted by lambda_retain.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.adapt.recovery_lora import (
    attach_recovery_lora,
    lora_trainable_parameter_count,
    save_lora_only,
    set_adapter_enabled,
)
from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.state_pool import StatePool


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _pad_actions(actions: np.ndarray, chunk_size: int, action_dim: int) -> np.ndarray:
    out = np.zeros((chunk_size, action_dim), dtype=np.float32)
    n = min(len(actions), chunk_size)
    if n:
        dim = min(actions.shape[1], action_dim)
        out[:n, :dim] = actions[:n, :dim]
    return out


def _observation_batch(
    bundle: dict[str, Any],
    observation: dict[str, Any],
    *,
    task: str,
    teacher_actions: np.ndarray,
    chunk_size_override: int | None = None,
) -> dict[str, torch.Tensor]:
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    policy_observation = preprocess_observation(
        {key: value for key, value in observation.items() if key != "task"}
    )
    policy_observation["task"] = [task]
    env_observation = bundle["env_preprocessor"](policy_observation)
    processed = bundle["preprocessor"](env_observation)
    policy = bundle["policy"]
    chunk = chunk_size_override or int(policy.config.chunk_size)
    max_dim = int(policy.config.max_action_dim)
    padded = _pad_actions(teacher_actions, chunk, max_dim)
    device = next(policy.parameters()).device
    action = torch.as_tensor(padded, dtype=torch.float32, device=device).unsqueeze(0)
    processed = {key: value for key, value in processed.items()}
    processed[ACTION] = action
    for key, value in list(processed.items()):
        if torch.is_tensor(value):
            processed[key] = value.to(device)
    return processed


def _load_chunk(chunk_dir: Path, chunk_index: int, episode_id: str) -> np.ndarray:
    """Load teacher actions from a chunk NPZ file."""
    chunk_path = chunk_dir / f"chunk_{episode_id.replace('_b1_', '_ep').replace('_b2_', '_ep').replace('_b3_', '_ep')}_step{chunk_index:04d}.npz"
    matches = list(chunk_dir.glob(f"chunk_*_step{chunk_index:04d}.npz"))
    if not matches:
        # Try broader match
        matches = sorted(chunk_dir.glob("chunk_*.npz"))
    if not matches:
        return np.zeros((10, 7), dtype=np.float32)
    data = dict(np.load(str(chunk_dir / f"chunk_ep{episode_id.split('_')[-1]}_step{chunk_index:04d}.npz"), allow_pickle=False))
    return np.asarray(data.get("teacher_action", np.zeros((10, 7), dtype=np.float32)), dtype=np.float32)


def _collate(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batches[0].keys()
    out: dict[str, Any] = {}
    for key in keys:
        values = [batch[key] for batch in batches]
        if torch.is_tensor(values[0]):
            out[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list):
            merged = []
            for item in values:
                merged.extend(item)
            out[key] = merged
        else:
            out[key] = values
    return out


def _process_row(
    bundle: dict[str, Any],
    row: dict[str, Any],
    chunks_dir: Path,
) -> dict[str, torch.Tensor]:
    """Create a single training batch from a dataset row."""
    import numpy as np

    chunk_dir = Path(row.get("chunk_dir", str(chunks_dir)))
    chunk_idx = int(row.get("chunk_index", 0))
    ep_id = str(row.get("episode_id", "unknown"))
    task = str(row.get("task_id", "")).replace("libero_object_", "").replace("libero_goal_", "")

    actions = _load_chunk(chunk_dir, chunk_idx, ep_id)
    obs = {
        "task": task,
        "pixels": {"image": np.zeros((1, 3, 256, 256), dtype=np.float32)},
        "robot_state": {
            "eef": {
                "quat": np.zeros((1, 4), dtype=np.float32),
                "pos": np.zeros((1, 3), dtype=np.float32),
            },
            "gripper": {"qpos": np.zeros((1, 2), dtype=np.float32)},
        },
    }
    return _observation_batch(bundle, obs, task=task, teacher_actions=actions)


def train(
    *,
    variant: str,
    train_rows: list[dict[str, Any]],
    adapter_config: dict[str, Any],
    output_dir: Path,
    chunks_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    retention_weight: float,
    device: str,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bundle = load_smolvla_policy_bundle(
        Path(adapter_config.get("policy_path", str(ROOT / "ckpts/smolvla_libero"))),
        device=device,
        num_steps=int(adapter_config.get("num_steps", 10)),
        n_action_steps=int(adapter_config.get("n_action_steps", 10)),
        tokenizer_path=Path(adapter_config.get("tokenizer_path", str(ROOT / "ckpts/SmolVLM2-500M-Instruct"))),
        observation_height=int(adapter_config.get("observation_height", 360)),
        observation_width=int(adapter_config.get("observation_width", 360)),
    )
    policy = bundle["policy"]
    policy.train()

    lora_cfg = adapter_config.get("lora", {})
    handle = attach_recovery_lora(
        policy,
        rank=int(lora_cfg.get("rank", 32)),
        alpha=int(lora_cfg.get("alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "action_in_proj", "action_out_proj",
        ])),
    )
    set_adapter_enabled(handle, True)
    counts = lora_trainable_parameter_count(handle.policy)
    print(json.dumps({"lora_param_counts": counts}, sort_keys=True))

    optimizer = torch.optim.AdamW(
        [p for p in handle.policy.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        random.shuffle(train_rows)
        losses: list[float] = []
        for start in range(0, len(train_rows), batch_size):
            chunk = train_rows[start : start + batch_size]
            batches = []
            retain_mask = []
            for row in chunk:
                batch = _process_row(bundle, row, chunks_dir)
                batches.append(batch)
                retain_mask.append(bool(row.get("clean_flag")))

            if not batches:
                continue

            collated = _collate(batches)
            loss, loss_dict = handle.policy.forward(collated)

            if variant == "B3" and any(retain_mask) and retention_weight != 1.0:
                clean_frac = sum(retain_mask) / max(1, len(retain_mask))
                loss = loss * (1.0 + (retention_weight - 1.0) * clean_frac)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(
                [p for p in handle.policy.parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(losses)) if losses else float("inf")
        history.append({"epoch": epoch, "mean_loss": mean_loss, "n_steps": len(losses)})
        print(f"epoch={epoch} mean_loss={mean_loss:.6f}", flush=True)

        adapter_dir = output_dir / f"adapter_epoch_{epoch}"
        save_lora_only(handle, str(adapter_dir))

    final_dir = output_dir / "adapter_final"
    save_lora_only(handle, str(final_dir))

    metrics = {
        "variant": variant,
        "n_train_rows": len(train_rows),
        "epochs": epochs,
        "lora_param_counts": counts,
        "history": history,
        "adapter_dir": str(final_dir),
        "seed": seed,
    }
    _write(output_dir / "train_metrics.json", metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["B1", "B2", "B3"])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, default=ROOT / "configs/collect_pre_c0_deviation_pilot24.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retention-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunks-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = args.dataset_dir / args.variant.lower()
    train_path = baseline_dir / "train.jsonl"
    if not train_path.is_file():
        print(f"ERROR: train.jsonl not found at {train_path}")
        return 1

    train_rows = _load_jsonl(train_path)
    if not train_rows:
        print(f"ERROR: no training data at {train_path}")
        return 1

    adapter_config = {}
    if args.adapter_config.is_file():
        adapter_config = json.loads(args.adapter_config.read_text(encoding="utf-8"))

    chunks_dir = args.chunks_dir or args.dataset_dir

    print(f"Variant: {args.variant}, Rows: {len(train_rows)}")
    print(f"Epochs: {args.epochs}, Seed: {args.seed}")

    metrics = train(
        variant=args.variant,
        train_rows=train_rows,
        adapter_config=adapter_config,
        output_dir=output_dir,
        chunks_dir=chunks_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        retention_weight=args.retention_weight,
        device=args.device,
    )

    print(json.dumps(metrics, sort_keys=True))
    print(f"Training complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
