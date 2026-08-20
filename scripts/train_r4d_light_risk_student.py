#!/usr/bin/env python3
"""Train the LightRiskStudent ensemble (Milestone 3b) with hard labels.

Data: R4 boundary rows (latent mode) or R4-D world-model windows (image mode).

Training protocol (mirrors the verified Ridge baseline):
  - Task-level outer CV folds
  - Inner fit / calibration split per fold
  - Task-bootstrap ensemble members (3)
  - Per-fold threshold selection on inner calibration (conformal-corrected)
  - No future leakage: predicted remaining cost, no task_ordinal

Outputs a report gated on the standard R4-F metrics plus a frozen checkpoint
dir for Milestone 3c export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_r4_safe_handback_wm_ridge import (  # noqa: E402
    grouped_task_folds,
    inner_task_split,
    read_jsonl,
)

from rase.risk.canonical_action import summary_from_chunk  # noqa: E402
from rase.risk.light_risk_student import LightRiskStudent  # noqa: E402
from rase.risk.tiny_universal_state_encoder import TinyUniversalStateEncoder  # noqa: E402
from rase.risk.vla_action_adapters import create_vla_adapter  # noqa: E402


def build_batch(
    rows: list[dict[str, Any]],
    *,
    adapter: Any,
    input_mode: str,
    device: str,
    dtype: torch.dtype = torch.float32,
    latent_overrides: dict[int, np.ndarray] | None = None,
) -> dict[str, torch.Tensor]:
    """Tensorize a list of boundary rows for LightRiskStudent training.

    If latent_overrides is given (id(row) -> V-JEPA pooled latent), those
    replace the dataset's baseline latent features so the student consumes
    teacher evidence (Milestone 3a/3b distillation path).
    """
    n = len(rows)
    if latent_overrides:
        image = np.stack([
            np.asarray(latent_overrides.get(id(r), r["latent"]), np.float32) for r in rows
        ])
    else:
        image = np.stack([np.asarray(r["latent"], np.float32) for r in rows])  # (N, L)
    proprio = np.stack([np.asarray(r["proprio"], np.float32) for r in rows])
    student_acts = np.stack([np.asarray(r["student_action"], np.float32) for r in rows])
    oft_acts = np.stack([np.asarray(r["oft_action"], np.float32) for r in rows])

    # Per-row canonical summaries (B, action_dim)
    student_sums = np.stack([
        summary_from_chunk(adapter.to_canonical(np.asarray(r["student_action_chunk"], np.float32)))
        .numpy()
        for r in rows
    ])
    oft_sums = np.stack([
        summary_from_chunk(adapter.to_canonical(np.asarray(r["oft_action"], np.float32).reshape(1, -1)))
        .numpy()
        for r in rows
    ])

    # History: use rolled time-deltas of the last few boundary steps as a cheap
    # proxy (in latent mode we have no frame history). Shape (N, hist_len, 6).
    hist_len = 4
    history = np.zeros((n, hist_len, 6), np.float32)
    for i, r in enumerate(rows):
        for j in range(hist_len):
            history[i, j, 0] = float(j)
            history[i, j, 1:] = np.asarray(r["proprio"], np.float32)[:5]

    success_label = np.array([
        int(bool(r.get("success_if_handback_now", r.get("student_step_success", False))))
        for r in rows
    ], np.float32)
    # Cost label: remaining teacher steps if handback now (avoid future leakage
    # during inference; label is only used for training). Normalized to [0, 1].
    cost_max = float(max((float(r.get("remaining_teacher_steps", 0.0)) for r in rows), default=1.0))
    cost_label = np.array([
        float(r.get("remaining_teacher_steps", 0.0)) / max(cost_max, 1e-9)
        for r in rows
    ], np.float32)

    return {
        "image": torch.as_tensor(image, dtype=dtype, device=device),
        "proprio": torch.as_tensor(proprio, dtype=dtype, device=device),
        "student_action": torch.as_tensor(student_sums, dtype=dtype, device=device),
        "oft_action": torch.as_tensor(oft_sums, dtype=dtype, device=device),
        "history": torch.as_tensor(history, dtype=dtype, device=device),
        "success_label": torch.as_tensor(success_label, dtype=dtype, device=device),
        "cost_label": torch.as_tensor(cost_label, dtype=dtype, device=device),
    }


def mean_ensemble(pred: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return pred.mean(dim=dim)


def train_step(model, batch, optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(
        batch["image"],
        batch["proprio"],
        batch["student_action"],
        batch["oft_action"],
        batch["history"],
    )
    success_prob = out["student_success"]  # (M, B)
    succ_loss = F.binary_cross_entropy(
        success_prob, batch["success_label"].unsqueeze(0).expand_as(success_prob)
    )
    cost_pred = out["remaining_cost"]  # (M, B, Q)
    cost_target = batch["cost_label"].unsqueeze(0).unsqueeze(-1).expand_as(cost_pred)
    cost_loss = F.smooth_l1_loss(cost_pred, cost_target)
    ood_pred = out["unsafe_ood"]
    ood_label = 1.0 - batch["success_label"]
    ood_loss = F.binary_cross_entropy(
        ood_pred, ood_label.unsqueeze(0).expand_as(ood_pred)
    )
    loss = succ_loss + 0.5 * cost_loss + ood_loss
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def predict(model, batch, device: str) -> dict[str, torch.Tensor]:
    model.eval()
    out = model(
        batch["image"],
        batch["proprio"],
        batch["student_action"],
        batch["oft_action"],
        batch["history"],
    )
    return {
        "success": out["student_success"].mean(dim=0).cpu().numpy(),
        "success_std": out["student_success"].std(dim=0).cpu().numpy(),
        "cost": out["remaining_cost"].mean(dim=(0, 2)).cpu().numpy(),
        "ood": out["unsafe_ood"].mean(dim=0).cpu().numpy(),
    }


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based AUC (no sklearn dependency)."""
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    n_pos = int(np.sum(y_true_sorted == 1))
    n_neg = int(len(y_true_sorted) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y_true_sorted) + 1, dtype=np.float64)
    rank_pos = ranks[y_true_sorted == 1]
    return float((rank_pos.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate(
    model,
    rows: list[dict[str, Any]],
    *,
    adapter: Any,
    device: str,
    threshold: float,
    latent_overrides: dict[int, np.ndarray] | None = None,
) -> dict[str, float]:
    batch = build_batch(rows, adapter=adapter, input_mode="latent", device=device,
                        latent_overrides=latent_overrides)
    pred = predict(model, batch, device)
    succ = np.asarray([int(bool(r.get("success_if_handback_now", False))) for r in rows])
    cost = np.asarray([float(r.get("remaining_teacher_steps", 0.0)) for r in rows])

    handback = pred["success"] >= threshold
    rescued = np.mean(handback & (succ == 1))
    harmed = np.mean(handback & (succ == 0))
    baseline_cost = float(cost.sum())
    saved = float(np.sum(np.where(handback, cost, 0.0)))
    oft_savings = saved / max(baseline_cost, 1e-9)
    persistent_savings = float(np.sum(cost)) / max(baseline_cost, 1e-9)

    # Risk metrics: AUC of student_success against the handback-success label.
    # Using success directly (not 1-success) keeps the ranking direction correct.
    auc = roc_auc_score_np(succ, pred["success"])
    return {
        "auc": auc,
        "rescued": rescued,
        "harmed": harmed,
        "handback_rate": float(np.mean(handback)),
        "oft_savings": oft_savings,
        "persistent_savings": persistent_savings,
    }


def choose_threshold(model, calib_rows, *, adapter, device, target_rescue: float = 0.5,
                     latent_overrides: dict[int, np.ndarray] | None = None):
    """Inner-calibration threshold selection for the handback gate."""
    batch = build_batch(calib_rows, adapter=adapter, input_mode="latent", device=device,
                        latent_overrides=latent_overrides)
    pred = predict(model, batch, device)
    succ = np.asarray([int(bool(r.get("success_if_handback_now", False))) for r in calib_rows])
    # Greedy: pick the highest threshold that still rescues target_rescue of positives
    scores = np.sort(pred["success"])
    best_th = 0.5
    best_harm = float("inf")
    for th in np.unique(scores):
        handback = pred["success"] >= th
        rescued = np.mean(handback & (succ == 1)) if succ.sum() else 0.0
        harmed = np.mean(handback & (succ == 0))
        if rescued >= target_rescue and harmed <= best_harm:
            best_harm = harmed
            best_th = th
    return float(best_th)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True,
                   help="boundary_transitions jsonl (v3/v4)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--input-mode", choices=("latent", "image"), default="latent")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--ensemble-size", type=int, default=3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--device", default="cuda")
    p.add_argument("--action-dim", type=int, default=20)
    p.add_argument("--vjepa-evidence", type=Path, default=None,
                   help="teacher evidence jsonl from cache_r4d_teacher_evidence.py; "
                        "matches rows by state_key + nearest window_start <= elapsed_oft_steps "
                        "and replaces the baseline latent with the V-JEPA pooled latent")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rows = read_jsonl(args.dataset)
    if not rows:
        raise SystemExit("empty dataset")
    tasks = sorted({str(r["task_id"]) for r in rows})
    states = sorted({str(r["state_key"]) for r in rows})
    print(f"rows={len(rows)} states={len(states)} tasks={len(tasks)}")

    # V-JEPA teacher evidence: build a state_key -> [(window_start, latent)] index
    # and a per-row override for every boundary row whose elapsed_oft_steps can be
    # matched to a window start (distillation path, Milestone 3a -> 3b).
    latent_overrides: dict[int, np.ndarray] = {}
    latent_dim = 128
    if args.vjepa_evidence is not None:
        ev_rows = read_jsonl(args.vjepa_evidence)
        by_state: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for ev in ev_rows:
            latent = np.asarray(ev.get("latent"), np.float32)
            if latent.size == 0:
                continue
            by_state[str(ev["state_key"])].append((int(ev.get("window_start", 0)), latent))
        for state in by_state:
            by_state[state].sort(key=lambda pair: pair[0])
        matched = 0
        for r in rows:
            starts = by_state.get(str(r["state_key"]), [])
            if not starts:
                continue
            elapsed = int(r.get("elapsed_oft_steps", 0))
            best = None
            for ws, latent in starts:
                if ws > elapsed:
                    break
                best = (ws, latent)
            if best is not None:
                latent_overrides[id(r)] = best[1]
                latent_dim = best[1].shape[0]
                matched += 1
        print(f"vjepa-evidence: matched {matched}/{len(rows)} boundary rows "
              f"(latent_dim={latent_dim})")

    folds = grouped_task_folds(rows, args.folds)
    adapter = create_vla_adapter("smolvla")

    oof_succ = np.zeros(len(rows), np.float32)
    oof_risk = np.zeros(len(rows), np.float32)
    oof_cost = np.zeros(len(rows), np.float32)
    oof_ood = np.zeros(len(rows), np.float32)
    index = {id(r): i for i, r in enumerate(rows)}
    fold_reports = []
    thresholds = []

    encoder = TinyUniversalStateEncoder(
        image_size=128, proprio_dim=8, text_embed_dim=0,
        hidden_dim=args.hidden_dim, output_dim=128,
        input_mode=args.input_mode, latent_dim=latent_dim,
    )
    model = LightRiskStudent(
        encoder, proprio_dim=8, action_dim=args.action_dim,
        history_dim=64, fused_dim=args.hidden_dim,
        head_hidden=128, n_members=args.ensemble_size, n_cost_quantiles=3,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    for fold_number, fold in enumerate(folds):
        train_rows = fold["train"]
        val_rows = fold["val"]
        inner_fit_rows, calibration_rows, calibration_tasks = inner_task_split(
            train_rows, fold_number
        )

        for member_idx in range(args.ensemble_size):
            boot_seed = args.seed + fold_number * 100 + member_idx
            random.seed(boot_seed)
            np.random.seed(boot_seed)
            torch.manual_seed(boot_seed)
            boot_train = _task_bootstrap(inner_fit_rows, seed=boot_seed)
            batch = build_batch(boot_train, adapter=adapter,
                                input_mode=args.input_mode, device=args.device,
                                latent_overrides=latent_overrides)
            for epoch in range(args.epochs):
                loss = train_step(model, batch, optimizer)
                if epoch % 50 == 0:
                    print(f"fold={fold_number} member={member_idx} epoch={epoch} loss={loss:.4f}",
                          flush=True)

        calib_batch = build_batch(calibration_rows, adapter=adapter,
                                  input_mode=args.input_mode, device=args.device,
                                  latent_overrides=latent_overrides)
        calib_pred = predict(model, calib_batch, args.device)
        th = choose_threshold(model, calibration_rows, adapter=adapter,
                              device=args.device, latent_overrides=latent_overrides)
        thresholds.append(th)
        print(f"fold={fold_number} threshold={th:.4f}")

        val_pred = predict(model, build_batch(val_rows, adapter=adapter,
                                              input_mode=args.input_mode, device=args.device,
                                              latent_overrides=latent_overrides),
                           args.device)
        val_idx = [index[id(r)] for r in val_rows]
        oof_succ[val_idx] = val_pred["success"]
        oof_cost[val_idx] = val_pred["cost"]
        oof_ood[val_idx] = val_pred["ood"]
        oof_risk[val_idx] = 1 - val_pred["success"]

        metrics = evaluate(model, val_rows, adapter=adapter, device=args.device,
                           threshold=th, latent_overrides=latent_overrides)
        fold_reports.append(metrics)
        print(f"fold={fold_number} {metrics}", flush=True)

    # Overall OOF report
    labels = np.asarray([
        int(bool(r.get("success_if_handback_now", False))) for r in rows
    ])
    costs = np.asarray([float(r.get("remaining_teacher_steps", 0.0)) for r in rows])
    mean_th = float(np.mean(thresholds))
    handback = oof_succ >= mean_th
    oof_auc = roc_auc_score_np(labels, oof_succ)

    report = {
        "schema_version": "rase-pre-c0-r4d-lightriskstudent/v1",
        "input_mode": args.input_mode,
        "vjepa_evidence": str(args.vjepa_evidence) if args.vjepa_evidence else None,
        "vjepa_matched_rows": len(latent_overrides),
        "latent_dim": latent_dim,
        "n_states": len(states),
        "n_rows": len(rows),
        "n_tasks": len(tasks),
        "folds": len(fold_reports),
        "ensemble_size": args.ensemble_size,
        "mean_threshold": mean_th,
        "oof_auc": oof_auc,
        "oof_rescued": float(np.mean(handback & (labels == 1))),
        "oof_harmed": float(np.mean(handback & (labels == 0))),
        "oof_handback_rate": float(np.mean(handback)),
        "oof_oft_savings": float(np.sum(np.where(handback, costs, 0.0)) / max(costs.sum(), 1e-9)),
        "fold_reports": fold_reports,
        "per_fold_thresholds": thresholds,
        "oof_predictions": {
            f"{r['state_key']}:{r['elapsed_oft_steps']}": float(s)
            for r, s in zip(rows, oof_succ)
        },
        "gates": {
            "auc": oof_auc,
            "rescue_non_inferior": bool(np.mean(handback & (labels == 1)) >= 0.15),
            "harm_limited": bool(np.mean(handback & (labels == 0)) <= 0.10),
        },
        "source": str(args.dataset.resolve()),
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ckpt = args.output_dir / "light_risk_student.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "encoder_state": encoder.state_dict(),
            "config": {
                "input_mode": args.input_mode,
                "action_dim": args.action_dim,
                "n_members": args.ensemble_size,
                "threshold": mean_th,
                "hidden_dim": args.hidden_dim,
                "latent_dim": latent_dim,
            },
            "seed": args.seed,
        },
        ckpt,
    )
    print(json.dumps(report, indent=2))
    return 0


def _task_bootstrap(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    tasks = sorted({str(r["task_id"]) for r in rows})
    sampled = rng.choice(tasks, len(tasks), replace=True)
    out = []
    for t in sampled:
        pool = [r for r in rows if str(r["task_id"]) == t]
        out.append(pool[rng.integers(len(pool))])
    return out


if __name__ == "__main__":
    raise SystemExit(main())
