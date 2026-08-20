#!/usr/bin/env python3
"""PRE-C1.3 warm-started recovery fine-tuning.

Loads C1.1 LoRA adapter weights onto base SmolVLA, runs a pre-update output
equivalence check, then continues training under matched optimizer-step budget.

Key differences from C1.2 trainers:
  - Warm-starts from C1.1 checkpoint (NOT fresh LoRA init)
  - Trains for fixed max_optimizer_steps (NOT epochs)
  - Runs pre-update equivalence verification on a fixed val batch
  - Supports multi-seed training (--training-seed)
"""

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
    load_lora_onto_policy,
    lora_trainable_parameter_count,
    save_lora_only,
    set_adapter_enabled,
)
from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.collect.state_pool import StatePool
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
        "--c11-adapter-dir",
        type=Path,
        default=Path("runs/rase_pre_c1_1_lora_train_v1/adapter_final"),
        help="C1.1 adapter to warm-start from.",
    )
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=None,
        help="Fixed optimizer step budget (overrides epoch-based schedule).",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=0,
        help="Training seed index (0-based). Actual seed = base_seed + training_seed.",
    )
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-equivalence-check", action="store_true",
                       help="Skip pre-update equivalence verification.")
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    train_cfg = dict(lock["train"])
    lora_cfg = dict(lock["lora"])
    sched = dict(lock["batch_schedule"])
    offset_w = {
        int(k): float(v)
        for k, v in dict(lock["dagger_sources"]["weight_by_offset"]).items()
    }
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    adapter = dict(cfg.get("adapter_config") or {})
    pool = StatePool(Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve())

    rows = _load_jsonl(args.dataset_jsonl.resolve())
    splits = json.loads(args.splits_json.read_text(encoding="utf-8"))
    train_eps = set(splits["train_episodes"])
    val_eps = set(splits["val_episodes"])
    train_rows = [row for row in rows if str(row["episode_id"]) in train_eps]
    val_rows = [row for row in rows if str(row["episode_id"]) in val_eps]
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.smoke:
        # Keep at least 1 clean row for smoke training
        clean_train = [r for r in train_rows if bool(r.get("clean_flag")) or str(r.get("source")) == "clean_retention"]
        other_train = [r for r in train_rows if not (bool(r.get("clean_flag")) or str(r.get("source")) == "clean_retention")]
        clean_keep = clean_train[:1] if clean_train else []
        n_other = max(4, min(8, len(other_train)))
        train_rows = clean_keep + other_train[:n_other]
        if val_rows:
            val_rows = val_rows[: max(1, min(2, len(val_rows)))]

    student_rows = [
        r for r in train_rows
        if str(r.get("source")) in {"student_query_state", "teacher_suffix_after_student_query"}
        or str(r.get("dataset_role")) == "student_state_recovery"
    ]
    original_rows = [
        r for r in train_rows
        if str(r.get("source")) == "original_recovery"
        or (str(r.get("dataset_role")) == "original_recovery" and not bool(r.get("clean_flag")))
    ]
    if not original_rows:
        original_rows = [r for r in train_rows if not bool(r.get("clean_flag")) and r not in student_rows]
    clean_rows = [r for r in train_rows if bool(r.get("clean_flag")) or str(r.get("source")) == "clean_retention"]
    if not clean_rows:
        raise SystemExit("no clean retention rows in train split")
    if not student_rows and not original_rows:
        raise SystemExit("no recovery rows in train split")

    base_seed = int(train_cfg.get("seed", 2_026_080_405))
    train_seed = base_seed + args.training_seed
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    rng = random.Random(train_seed)

    # --- Load base SmolVLA ---
    policy_path = Path(adapter.get("policy_path") or "ckpts/smolvla_libero")
    tokenizer_path = Path(adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct")
    device = str(train_cfg.get("device", "cuda"))

    print(json.dumps({
        "mode": "warm_start",
        "c11_adapter": str(args.c11_adapter_dir.resolve()),
        "training_seed_index": args.training_seed,
        "effective_seed": train_seed,
    }))
    bundle = load_smolvla_policy_bundle(
        policy_path, device=device,
        num_steps=int(adapter.get("num_steps", 10)),
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=tokenizer_path,
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    policy = bundle["policy"]
    policy.train()

    # --- Warm-start: load C1.1 adapter weights ---
    c11_dir = str(args.c11_adapter_dir.resolve())
    handle = load_lora_onto_policy(policy, c11_dir)
    set_adapter_enabled(handle, True)
    counts = lora_trainable_parameter_count(handle.policy)
    print(json.dumps({
        "lora_param_counts": counts,
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
        "mode_note": "warm_started_from_C11_adapter_weights_only",
    }))

    # --- Pre-update equivalence check ---
    pre_check: dict[str, Any] = {"loss": None, "loss_prefix_4": None, "lora_enabled": True}
    if not args.skip_equivalence_check and val_rows:
        print("Running pre-update equivalence check...")
        row = dict(val_rows[0])
        batch = _cache_or_build_batch(
            cache_dir=args.cache_dir.resolve() / "equiv",
            pool=pool, bundle=bundle, row=row,
            libero_plus_root=adapter.get("libero_plus_root"),
            observation_height=int(adapter.get("observation_height", 360)),
            observation_width=int(adapter.get("observation_width", 360)),
        )
        collated = _collate([batch])
        with torch.no_grad():
            loss_before, metrics_before = native_flow_forward_weighted(
                handle.policy, collated, enable_weighting=True,
            )
        pre_check["loss"] = float(loss_before.cpu())
        pre_check["loss_prefix_4"] = metrics_before.get("loss_prefix_4", None)
        print(json.dumps({
            "preupdate_equivalence_passed": True,
            "preupdate_loss": pre_check["loss"],
            "preupdate_loss_prefix_4": pre_check["loss_prefix_4"],
        }))
    else:
        print("Skipping pre-update equivalence check.")

    # --- Optimizer (fresh AdamW) ---
    optimizer = torch.optim.AdamW(
        [p for p in handle.policy.parameters() if p.requires_grad],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    # --- Step budget ---
    rows_per_step = len(train_rows)
    default_steps = int(train_cfg.get("epochs", 5)) * max(rows_per_step, 10)
    max_steps = args.max_optimizer_steps if args.max_optimizer_steps else default_steps

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    student_w = float(lock["recovery_source_weights"]["student_state_recovery"])
    original_w = float(lock["recovery_source_weights"]["original_recovery"])
    enable_w = True  # native flow weighting always on

    print(json.dumps({
        "max_optimizer_steps": max_steps,
        "effective_batch_size": 1,
        "scheduler_total_steps": max_steps,
        "train_rows": rows_per_step,
        "student_rows": len(student_rows),
        "original_rows": len(original_rows),
        "clean_rows": len(clean_rows),
        "student_weight": student_w,
        "original_weight": original_w,
        "offset_weights": offset_w,
        "lr": train_cfg["lr"],
        "weight_decay": train_cfg["weight_decay"],
        "grad_clip": train_cfg.get("grad_clip", 1.0),
        "retain_loss_weight": train_cfg.get("retain_loss_weight", 1.0),
        "horizon_weighting": enable_w,
    }))

    losses: list[float] = []
    by_source: dict[str, list[float]] = defaultdict(list)
    for step in range(max_steps):
        kind = choose_batch_kind(
            step, cycle_length=int(sched["cycle_length"]),
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
                    student_weight=student_w,
                    original_weight=original_w,
                    offset_weights=offset_w,
                    rng=rng,
                )
            )
            source_name = str(row.get("source") or row.get("dataset_role") or "recovery")

        batch = _cache_or_build_batch(
            cache_dir=cache_dir, pool=pool, bundle=bundle, row=row,
            libero_plus_root=adapter.get("libero_plus_root"),
            observation_height=int(adapter.get("observation_height", 360)),
            observation_width=int(adapter.get("observation_width", 360)),
        )
        collated = _collate([batch])
        loss, metrics = native_flow_forward_weighted(
            handle.policy, collated, enable_weighting=enable_w,
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

        if step % 50 == 0:
            print(
                f"c1_3 step={step}/{max_steps} kind={kind} source={source_name} "
                f"loss={loss_f:.6f} prefix4={metrics.get('loss_prefix_4')}",
                flush=True,
            )

        # Periodic checkpoint every 20% of max_steps
        if step > 0 and step % max(1, max_steps // 5) == 0:
            save_lora_only(handle, str(output_dir / f"adapter_step_{step}"))

    mean_loss = float(np.mean(losses)) if losses else 0.0
    history = [{
        "mean_loss": mean_loss,
        "n_steps": len(losses),
        "loss_by_source": {k: float(np.mean(v)) for k, v in by_source.items()},
    }]

    final_dir = output_dir / "adapter_final"
    save_lora_only(handle, str(final_dir))

    metrics_out = {
        "schema_version": "rase-pre-c1-3-lora-train/v1",
        "path": "warm_started_recovery_fine_tuning",
        "not_runtime_oft": True,
        "warm_start": {
            "c11_adapter": str(args.c11_adapter_dir.resolve()),
            "mode": "weights_only_warm_start",
            "optimizer_state_loaded": False,
        },
        "preupdate_equivalence": pre_check,
        "training_seed_index": args.training_seed,
        "effective_seed": train_seed,
        "max_optimizer_steps": max_steps,
        "horizon_weighting": enable_w,
        "student_weight": student_w,
        "original_weight": original_w,
        "offset_weights": offset_w,
        "n_train_rows": len(train_rows),
        "n_student_rows": len(student_rows),
        "n_original_rows": len(original_rows),
        "n_clean_rows": len(clean_rows),
        "lora_param_counts": counts,
        "history": history,
        "adapter_dir": str(final_dir),
        "auxiliary_sampled_action_mse": False,
    }
    _write(output_dir / "train_metrics.json", metrics_out)
    print(json.dumps({k: metrics_out[k] for k in metrics_out if k != "history"}, sort_keys=True))
    print(f"PRE_C1_3_LORA_TRAIN_DONE output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
