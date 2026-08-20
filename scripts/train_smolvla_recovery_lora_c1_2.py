#!/usr/bin/env python3
"""PRE-C1.2 train: 9+1 batch schedule + native flow horizon weighting."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rase.adapt.pre_c1_2 import (
    choose_batch_kind,
    load_protocol_lock,
    native_flow_forward_weighted,
    sample_clean_row,
    sample_recovery_row,
)
from rase.adapt.recovery_lora import (
    attach_recovery_lora,
    lora_trainable_parameter_count,
    save_lora_only,
    set_adapter_enabled,
)
from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.collect.state_pool import StatePool

# Reuse batch builders from C1 trainer.
from train_smolvla_recovery_lora import (  # type: ignore
    _cache_or_build_batch,
    _collate,
    _load_jsonl,
    _write,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--enable-horizon-weighting",
        action="store_true",
        help="E4: piecewise native flow weighting. E3 leaves this off.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    train_cfg = dict(lock["train"])
    lora_cfg = dict(lock["lora"])
    sched = dict(lock["batch_schedule"])
    src_w = dict(lock["recovery_source_weights"])
    offset_w = {
        int(k): float(v) for k, v in dict(lock["dagger_sources"]["weight_by_offset"]).items()
    }
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
        train_rows = train_rows[: max(4, min(8, len(train_rows)))]
        train_cfg["epochs"] = 1

    student_rows = [
        r
        for r in train_rows
        if str(r.get("source")) in {"student_query_state", "teacher_suffix_after_student_query"}
        or str(r.get("dataset_role")) == "student_state_recovery"
    ]
    original_rows = [
        r
        for r in train_rows
        if str(r.get("source")) == "original_recovery"
        or (
            str(r.get("dataset_role")) == "original_recovery"
            and not bool(r.get("clean_flag"))
        )
    ]
    # Fallback: any non-clean as original if roles missing.
    if not original_rows:
        original_rows = [r for r in train_rows if not bool(r.get("clean_flag"))]
    clean_rows = [r for r in train_rows if bool(r.get("clean_flag"))]
    if not clean_rows:
        raise SystemExit("no clean retention rows in train split")
    if not student_rows and not original_rows:
        raise SystemExit("no recovery rows in train split")

    seed = int(train_cfg.get("seed", 2_026_080_405))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    history = []
    steps_per_epoch = max(len(train_rows), 10)
    enable_w = bool(args.enable_horizon_weighting)

    for epoch in range(epochs):
        losses = []
        by_source: dict[str, list[float]] = defaultdict(list)
        by_anchor: dict[str, list[float]] = defaultdict(list)
        for step in range(steps_per_epoch):
            kind = choose_batch_kind(
                step,
                cycle_length=int(sched["cycle_length"]),
                clean_batches=int(sched["clean_batches"]),
            )
            if kind == "clean":
                row = dict(sample_clean_row(clean_rows, rng=rng))
                source_name = "clean"
            else:
                row = dict(
                    sample_recovery_row(
                        student_rows=student_rows or original_rows,
                        original_rows=original_rows or student_rows,
                        student_weight=float(src_w["student_state_recovery"]),
                        original_weight=float(src_w["original_recovery"]),
                        offset_weights=offset_w,
                        rng=rng,
                    )
                )
                source_name = str(row.get("source") or row.get("dataset_role") or "recovery")
            batch = _cache_or_build_batch(
                cache_dir=cache_dir,
                pool=pool,
                bundle=bundle,
                row=row,
                libero_plus_root=adapter.get("libero_plus_root"),
                observation_height=int(adapter.get("observation_height", 360)),
                observation_width=int(adapter.get("observation_width", 360)),
            )
            collated = _collate([batch])
            loss, metrics = native_flow_forward_weighted(
                handle.policy,
                collated,
                enable_weighting=enable_w,
            )
            if kind == "clean" and float(train_cfg.get("retain_loss_weight", 1.0)) != 1.0:
                loss = loss * float(train_cfg["retain_loss_weight"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(
                [p for p in handle.policy.parameters() if p.requires_grad],
                float(train_cfg.get("grad_clip", 1.0)),
            )
            optimizer.step()
            loss_f = float(loss.detach().cpu())
            losses.append(loss_f)
            by_source[source_name].append(loss_f)
            anchor = str(row.get("anchor_id") or row.get("state_key") or "na")
            by_anchor[anchor].append(loss_f)
            if step % 20 == 0:
                print(
                    f"epoch={epoch} step={step} kind={kind} source={source_name} "
                    f"loss={loss_f:.6f} prefix4={metrics.get('loss_prefix_4')} "
                    f"weighting={enable_w}",
                    flush=True,
                )
        mean_loss = float(np.mean(losses)) if losses else 0.0
        history.append(
            {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "n_steps": len(losses),
                "loss_by_source": {k: float(np.mean(v)) for k, v in by_source.items()},
                "loss_by_anchor": {
                    k: float(np.mean(v)) for k, v in sorted(by_anchor.items())[:32]
                },
                "horizon_weighting": enable_w,
            }
        )
        save_lora_only(handle, str(output_dir / f"adapter_epoch_{epoch}"))
        print(f"PRE_C1_2_LORA_EPOCH_DONE epoch={epoch} mean_loss={mean_loss}", flush=True)

    final_dir = output_dir / "adapter_final"
    save_lora_only(handle, str(final_dir))
    metrics_out = {
        "schema_version": "rase-pre-c1-2-lora-train/v1",
        "naming": "recovery LoRA / OFT action distillation",
        "not_runtime_oft": True,
        "horizon_weighting": enable_w,
        "batch_schedule": sched,
        "recovery_source_weights": src_w,
        "n_train_rows": len(train_rows),
        "n_student_rows": len(student_rows),
        "n_original_rows": len(original_rows),
        "n_clean_rows": len(clean_rows),
        "epochs": epochs,
        "lora_param_counts": counts,
        "history": history,
        "adapter_dir": str(final_dir),
        "auxiliary_sampled_action_mse": False,
    }
    _write(output_dir / "train_metrics.json", metrics_out)
    print(json.dumps({k: metrics_out[k] for k in metrics_out if k != "history"}, sort_keys=True))
    print(f"PRE_C1_2_LORA_TRAIN_DONE output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
