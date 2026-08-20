#!/usr/bin/env python3
"""Train SmolVLA recovery LoRA with OFT action distillation + clean retention."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from rase.adapt.pre_c1 import load_protocol_lock
from rase.adapt.recovery_lora import (
    attach_recovery_lora,
    lora_trainable_parameter_count,
    save_lora_only,
    set_adapter_enabled,
)
from rase.collect.forked_rollout import load_smolvla_policy_bundle, restore_pool_state
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
    chunk = int(policy.config.chunk_size)
    max_dim = int(policy.config.max_action_dim)
    padded = _pad_actions(teacher_actions, chunk, max_dim)
    device = next(policy.parameters()).device
    action = torch.as_tensor(padded, dtype=torch.float32, device=device).unsqueeze(0)
    processed = {key: value for key, value in processed.items()}
    processed[ACTION] = action
    # Ensure tensors on device
    for key, value in list(processed.items()):
        if torch.is_tensor(value):
            processed[key] = value.to(device)
    return processed


def _unpack_chunk_observation(packed: dict[str, Any]) -> dict[str, Any]:
    pixels: dict[str, Any] = {}
    if "pixels_image" in packed:
        pixels["image"] = np.asarray(packed["pixels_image"])[None, ...]
    if "pixels_image2" in packed:
        pixels["image2"] = np.asarray(packed["pixels_image2"])[None, ...]
    observation: dict[str, Any] = {"pixels": pixels}
    robot_state: dict[str, Any] = {}
    for key, value in packed.items():
        if not str(key).startswith("rs_"):
            continue
        parts = str(key)[3:].split(".")
        cursor: dict[str, Any] = robot_state
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = np.asarray(value)[None, ...]
    if robot_state:
        observation["robot_state"] = robot_state
    if "agent_pos" in packed:
        observation["agent_pos"] = np.asarray(packed["agent_pos"], dtype=np.float32)[None, ...]
    task = packed.get("task")
    if task is not None:
        observation["task"] = str(np.asarray(task).item() if hasattr(task, "item") else task)
    return observation


def _row_cache_key(row: dict[str, Any]) -> str:
    sample_id = row.get("sample_id")
    if sample_id:
        return str(sample_id).replace("/", "_").replace(":", "_")
    chunk_index = row.get("chunk_index")
    if chunk_index is not None:
        return f"{row['state_key']}__chunk_{int(chunk_index):04d}"
    return str(row["state_key"])


def _cache_or_build_batch(
    *,
    cache_dir: Path,
    pool: StatePool,
    bundle: dict[str, Any],
    row: dict[str, Any],
    libero_plus_root: str | None,
    observation_height: int,
    observation_width: int,
) -> dict[str, torch.Tensor]:
    cache_path = cache_dir / f"{_row_cache_key(row)}.pt"
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        device = next(bundle["policy"].parameters()).device
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in payload.items()}

    chunk_path = row.get("chunk_path")
    use_chunk = bool(chunk_path) and Path(str(chunk_path)).is_file()
    if use_chunk:
        packed = dict(np.load(str(chunk_path), allow_pickle=False))
        observation = _unpack_chunk_observation(packed)
        # Legacy chunks without robot_state cannot train; fall back to restore@fork
        # only for chunk_index==0. Mid-traj chunks must be rewritten first.
        if "robot_state" not in observation:
            if int(row.get("chunk_index") or 0) != 0:
                raise RuntimeError(
                    f"chunk missing robot_state and not fork chunk: {chunk_path}"
                )
            use_chunk = False
        else:
            task = str(observation.pop("task", "") or "")
            actions = np.asarray(packed["oft_action_chunk"], dtype=np.float32)
            batch = _observation_batch(
                bundle, observation, task=task, teacher_actions=actions
            )

    if not use_chunk:
        restored = restore_pool_state(
            pool,
            str(row["state_key"]),
            libero_plus_root=libero_plus_root,
            observation_height=observation_height,
            observation_width=observation_width,
        )
        try:
            observation = observation_from_libero_env(restored.handle.vector_env.envs[0])
            task = str(
                getattr(restored.handle.vector_env.envs[0], "task_description", "") or ""
            )
            if chunk_path and Path(str(chunk_path)).is_file():
                actions = np.asarray(
                    np.load(str(chunk_path), allow_pickle=False)["oft_action_chunk"],
                    dtype=np.float32,
                )
            else:
                actions = np.asarray(row["teacher_actions"], dtype=np.float32)
            batch = _observation_batch(
                bundle, observation, task=task, teacher_actions=actions
            )
        finally:
            restored.close()
    cpu_batch = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_batch, cache_path)
    return batch


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    train_cfg = dict(lock["train"])
    lora_cfg = dict(lock["lora"])
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    adapter = dict(cfg.get("adapter_config") or {})
    pool = StatePool(Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve())

    rows = _load_jsonl(args.dataset_jsonl.resolve())
    splits = json.loads(args.splits_json.read_text(encoding="utf-8"))
    train_eps = set(splits["train_episodes"])
    train_rows = [row for row in rows if str(row["episode_id"]) in train_eps]
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.smoke:
        train_rows = train_rows[: max(2, min(4, len(train_rows)))]
        train_cfg["epochs"] = 1
        train_cfg["batch_size"] = 1
    if not train_rows:
        raise SystemExit("no train rows")

    # Variable-length language tokens: keep batch_size=1 for stable collation.
    train_cfg["batch_size"] = 1

    seed = int(train_cfg.get("seed", 2_026_080_405))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bundle = load_smolvla_policy_bundle(
        Path(adapter.get("policy_path") or "ckpts/smolvla_libero"),
        device=str(train_cfg.get("device", "cuda")),
        num_steps=int(adapter.get("num_steps", 10)),
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=Path(adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct"),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    policy = bundle["policy"]
    policy.train()
    handle = attach_recovery_lora(
        policy,
        rank=int(lora_cfg["rank"]),
        alpha=int(lora_cfg["alpha"]),
        dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
    )
    set_adapter_enabled(handle, True)
    counts = lora_trainable_parameter_count(handle.policy)
    print(json.dumps({"lora_param_counts": counts}, sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in handle.policy.parameters() if p.requires_grad],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    retain_w = float(train_cfg["retain_loss_weight"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    history = []

    for epoch in range(epochs):
        random.shuffle(train_rows)
        losses = []
        for start in range(0, len(train_rows), batch_size):
            chunk = train_rows[start : start + batch_size]
            batches = []
            retain_mask = []
            for row in chunk:
                batch = _cache_or_build_batch(
                    cache_dir=cache_dir,
                    pool=pool,
                    bundle=bundle,
                    row=row,
                    libero_plus_root=adapter.get("libero_plus_root"),
                    observation_height=int(adapter.get("observation_height", 360)),
                    observation_width=int(adapter.get("observation_width", 360)),
                )
                batches.append(batch)
                retain_mask.append(bool(row.get("clean_flag")))
            collated = _collate(batches)
            loss, loss_dict = handle.policy.forward(collated)
            # Up-weight clean retention samples.
            if any(retain_mask) and retain_w != 1.0:
                # Scalar loss already averaged; apply retain multiplier softly.
                clean_frac = sum(retain_mask) / len(retain_mask)
                loss = loss * (1.0 + (retain_w - 1.0) * clean_frac)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(
                [p for p in handle.policy.parameters() if p.requires_grad],
                float(train_cfg.get("grad_clip", 1.0)),
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            print(
                f"epoch={epoch} step={start} loss={losses[-1]:.6f} "
                f"dict_loss={loss_dict.get('loss')}",
                flush=True,
            )
        mean_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch, "mean_loss": mean_loss, "n_steps": len(losses)})
        adapter_dir = output_dir / f"adapter_epoch_{epoch}"
        save_lora_only(handle, str(adapter_dir))
        print(f"PRE_C1_LORA_EPOCH_DONE epoch={epoch} mean_loss={mean_loss}", flush=True)

    final_dir = output_dir / "adapter_final"
    save_lora_only(handle, str(final_dir))
    metrics = {
        "schema_version": "rase-pre-c1-lora-train/v1",
        "naming": "recovery LoRA / OFT action distillation",
        "not_runtime_oft": True,
        "not_smolvla_flow_api_guidance": True,
        "n_train_rows": len(train_rows),
        "epochs": epochs,
        "lora_param_counts": counts,
        "history": history,
        "adapter_dir": str(final_dir),
    }
    _write(output_dir / "train_metrics.json", metrics)
    print(json.dumps(metrics, sort_keys=True))
    print(f"PRE_C1_LORA_TRAIN_DONE output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
