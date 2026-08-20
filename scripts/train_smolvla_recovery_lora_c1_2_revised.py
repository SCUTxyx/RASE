#!/usr/bin/env python3
"""Revised PRE-C1.2 train: early recoverable student states + short-horizon native flow.

Paused until R0 decision allows a revised-training branch. Legacy E3/E4 remains separate.
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
    attach_recovery_lora,
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

ALLOWED_BRANCHES = {
    "recoverability_aware_dagger_early_query",
    "first_action_correction_residual",
    "short_horizon_corrective_plus_aware_dagger",
    "repeated_correction_handback",
    "revised_short_horizon_training",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--r0-decision-json",
        type=Path,
        default=Path("runs/rase_pre_c1_2_r0_decision_v1.json"),
    )
    parser.add_argument(
        "--allow-without-r0",
        action="store_true",
        help="Escape hatch for engineering smoke only.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--student-weight",
        type=float,
        default=0.85,
        help="Recovery-batch weight for student_query_state (original gets remainder).",
    )
    args = parser.parse_args()

    if not args.allow_without_r0:
        if not args.r0_decision_json.is_file():
            raise SystemExit(
                f"missing R0 decision ({args.r0_decision_json}); refuse revised training"
            )
        decision = json.loads(args.r0_decision_json.read_text(encoding="utf-8"))
        if decision.get("blocked"):
            raise SystemExit(f"R0 decision blocked: {decision.get('reason')}")
        branch = str(decision.get("branch") or "")
        if branch not in ALLOWED_BRANCHES:
            raise SystemExit(
                f"R0 branch={branch} does not authorize revised training; "
                f"allowed={sorted(ALLOWED_BRANCHES)}"
            )
        if branch == "stop_dagger_inspect_optimization_capacity_target":
            raise SystemExit("R0 says stop and inspect optimization/capacity; no revised train")
    else:
        decision = {"branch": "allow_without_r0", "provisional": True}

    lock = load_protocol_lock(args.protocol_lock)
    train_cfg = dict(lock["train"])
    lora_cfg = dict(lock["lora"])
    sched = dict(lock["batch_schedule"])
    # Force query-state dominance; ignore long suffix offsets.
    offset_w = {0: 1.0}
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
        if str(r.get("source")) == "student_query_state"
        or (
            str(r.get("dataset_role")) == "student_state_recovery"
            and int(r.get("offset_from_student_state", 0) or 0) == 0
        )
    ]
    original_rows = [
        r
        for r in train_rows
        if str(r.get("source")) == "original_recovery"
        or str(r.get("dataset_role")) == "original_recovery"
    ]
    clean_rows = [r for r in train_rows if bool(r.get("clean_flag")) or str(r.get("source")) == "clean_retention"]
    if not clean_rows:
        raise SystemExit("no clean retention rows in revised train split")
    if not student_rows and not original_rows:
        raise SystemExit("no recovery rows in revised train split")

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
    print(json.dumps({"lora_param_counts": counts, "r0_branch": decision.get("branch")}, sort_keys=True), flush=True)

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
    student_w = float(args.student_weight)
    original_w = max(0.0, 1.0 - student_w)
    # Always use short-horizon native weighting for revised path.
    enable_w = True

    for epoch in range(epochs):
        losses = []
        by_source: dict[str, list[float]] = defaultdict(list)
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
                        student_weight=student_w,
                        original_weight=original_w,
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
            if step % 20 == 0:
                print(
                    f"revised epoch={epoch} step={step} kind={kind} source={source_name} "
                    f"loss={loss_f:.6f} prefix4={metrics.get('loss_prefix_4')}",
                    flush=True,
                )
        mean_loss = float(np.mean(losses)) if losses else 0.0
        history.append(
            {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "n_steps": len(losses),
                "loss_by_source": {k: float(np.mean(v)) for k, v in by_source.items()},
            }
        )
        save_lora_only(handle, str(output_dir / f"adapter_epoch_{epoch}"))
        print(f"PRE_C1_2_REVISED_LORA_EPOCH_DONE epoch={epoch} mean_loss={mean_loss}", flush=True)

    final_dir = output_dir / "adapter_final"
    save_lora_only(handle, str(final_dir))
    metrics_out = {
        "schema_version": "rase-pre-c1-2-revised-lora-train/v1",
        "path": "revised_short_horizon_after_r0",
        "not_runtime_oft": True,
        "r0_decision": {
            "branch": decision.get("branch"),
            "provisional": decision.get("provisional"),
            "path": str(args.r0_decision_json),
        },
        "horizon_weighting": True,
        "student_weight": student_w,
        "original_weight": original_w,
        "offset_weights": offset_w,
        "primary_targets": ["R_adapted_1", "R_adapted_4", "return_to_stable"],
        "terminal_8pp_is_final_gate_only": True,
        "n_train_rows": len(train_rows),
        "n_student_rows": len(student_rows),
        "n_original_rows": len(original_rows),
        "n_clean_rows": len(clean_rows),
        "epochs": epochs,
        "lora_param_counts": counts,
        "history": history,
        "adapter_dir": str(final_dir),
        "auxiliary_sampled_action_mse": False,
        "residual_note": "First revised version uses short-horizon native flow on early recoverable states; explicit residual head is a follow-on if R0 branch requests it.",
    }
    _write(output_dir / "train_metrics.json", metrics_out)
    print(json.dumps({k: metrics_out[k] for k in metrics_out if k != "history"}, sort_keys=True))
    print(f"PRE_C1_2_REVISED_LORA_TRAIN_DONE output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
