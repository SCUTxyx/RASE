#!/usr/bin/env python3
"""R4 Safe-Handback Ablation Matrix.

Evaluates ablations of the R4 safe-handback system on the boundary dataset:
- Dynamics backend (ridge/linear/persistence)
- History features (with/without suite encoding)
- Ensemble sizes
- Hidden dimensions
- Cost credit
- Label type (live exact vs historical finite-arm)

Each ablation is evaluated using grouped-task-held-out folds with the same
fold assignments, making results directly comparable.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_r4_safe_handback_wm_ridge import (
    SUITE_MAP,
    DecisionMLP,
    Standardizer,
    _compute_history_features,
    _mean_std,
    _predict_decision,
    _stack,
    _tensorize,
    _vec,
    choose_threshold,
    grouped_task_folds,
    inner_task_split,
    objective,
    read_jsonl,
    _state_policy,
    validate_rows,
)
from rase.dynamics_backend import (
    DynamicsBackend,
    LinearDynamicsBackend,
    PersistenceBackend,
    RidgeDynamicsBackend,
)


# ---------------------------------------------------------------------------
# Simplified array building (without history features)
# ---------------------------------------------------------------------------

def build_arrays_no_history(rows: list[dict[str, Any]],
                            stats: dict[str, Standardizer] | None = None):
    """Build arrays WITHOUT task-history features for ablation."""
    latent = _stack(rows, "latent")
    proprio = _stack(rows, "proprio")
    student = _stack(rows, "student_action")
    oft = _stack(rows, "oft_action")
    chunk = _stack(rows, "student_action_chunk")
    next_student = _stack(rows, "next_latent_student")
    next_oft = _stack(rows, "next_latent_oft")
    time_features = np.asarray(
        [[float(r["elapsed_oft_steps"]) / max(1.0, float(r["horizon"])),
          float(r["simulator_timestep"]) / max(1.0, float(r["horizon"]))]
         for r in rows],
        dtype=np.float32,
    )
    if stats is None:
        stats = {
            "latent": Standardizer(latent),
            "proprio": Standardizer(proprio),
            "action": Standardizer(np.concatenate([student, oft], axis=0)),
            "chunk": Standardizer(chunk),
        }
    state = np.concatenate(
        [stats["latent"](latent), stats["proprio"](proprio), time_features], axis=1
    )
    decision = np.concatenate([stats["chunk"](chunk), stats["action"](oft)], axis=1)
    actions = np.stack([stats["action"](student), stats["action"](oft)], axis=1)
    action_type = np.zeros((len(rows), 2, 2), dtype=np.float32)
    action_type[:, 0, 0] = 1.0
    action_type[:, 1, 1] = 1.0
    transitions = np.concatenate([actions, action_type], axis=2)
    next_latents = np.stack([stats["latent"](next_student), stats["latent"](next_oft)], axis=1)
    current = stats["latent"](latent)[:, None, :]
    delta = next_latents - current
    terminal = np.asarray(
        [[r["student_step_terminal"], r["oft_step_terminal"]] for r in rows],
        np.float32,
    )
    handback = np.asarray([r["success_if_handback_now"] for r in rows], np.float32)
    persistent = np.asarray([r["success_if_continue_oft"] for r in rows], np.float32)
    remaining = np.asarray(
        [float(r["remaining_teacher_steps"]) / max(1.0, float(r["persistent_executed_oft_steps"]))
         for r in rows], np.float32,
    )
    return {
        "state": state, "decision": decision, "transition": transitions,
        "delta": delta, "terminal": terminal, "handback": handback,
        "risk": 1.0 - handback, "persistent": persistent, "remaining": remaining,
        "latent": stats["latent"](latent),
    }, stats


# ---------------------------------------------------------------------------
# Label comparison: live exact vs historical finite-arm
# ---------------------------------------------------------------------------

def compute_label_misalignment(rows: list[dict]) -> dict:
    """Compute misalignment between historical finite-arm labels and exact live labels."""
    total = 0
    mismatches = 0
    false_positives = 0  # historical says handback-safe but live disagrees
    false_negatives = 0  # historical says not-safe but live says safe
    for row in rows:
        if "historical_success_if_handback_now" not in row or row["historical_success_if_handback_now"] is None:
            continue
        total += 1
        hist = bool(row["historical_success_if_handback_now"])
        live = bool(row["success_if_handback_now"])
        if hist != live:
            mismatches += 1
            if hist and not live:
                false_positives += 1
            else:
                false_negatives += 1
    return {
        "total_comparable_rows": total,
        "mismatches": mismatches,
        "mismatch_rate": mismatches / max(1, total),
        "false_positives": false_positives,  # historical says safe but live says not
        "false_negatives": false_negatives,  # historical says not-safe but live says safe
    }


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    rows: list[dict],
    folds: list[dict],
    *,
    backend: DynamicsBackend,
    hidden_dim: int,
    ensemble_size: int,
    epochs: int,
    patience: int,
    lr: float,
    lcb_z: float,
    cost_credit: float,
    seed: int,
    device: str,
    use_history: bool = True,
) -> dict[str, Any]:
    """Run a single ablation configuration and return results."""
    index_map = {id(row): i for i, row in enumerate(rows)}

    oof = {key: np.zeros(len(rows), np.float32) for key in ("risk", "handback", "persistent", "remaining")}
    oof_std = {key: np.zeros(len(rows), np.float32) for key in oof}
    dyn_mse_list = []
    persistence_mse_list = []

    for fold_number, fold in enumerate(folds):
        train_rows = fold["train"]
        val_rows = fold["val"]
        inner_fit_rows, calibration_rows, _ = inner_task_split(train_rows, fold_number)

        build_fn = build_arrays_no_history
        if use_history:
            from train_r4_safe_handback_wm_ridge import build_arrays as build_arrays_history
            build_fn = build_arrays_history

        train_np, stats = build_fn(inner_fit_rows)
        full_train_np, _ = build_fn(train_rows, stats)
        val_np, _ = build_fn(val_rows, stats)

        # Fit dynamics backend
        backend_fold = copy.deepcopy(backend)
        backend_fold.fit(train_np)
        # Refit on full train
        backend_full = copy.deepcopy(backend)
        backend_full.fit(full_train_np)

        ridge_delta_train = backend_fold.predict(train_np)
        ridge_delta_val = backend_full.predict(val_np)

        # Dynamics MSE
        mask = (1.0 - val_np["terminal"])[..., None]
        dyn_error = ((ridge_delta_val - val_np["delta"]) ** 2) * mask
        base_error = (val_np["delta"] ** 2) * mask
        denom = max(1.0, float(mask.sum()) * val_np["delta"].shape[-1])
        dyn_mse_list.append(float(dyn_error.sum() / denom))
        persistence_mse_list.append(float(base_error.sum() / denom))

        # Decision MLP ensemble
        member_val = []
        for member_idx in range(ensemble_size):
            mlp = DecisionMLP(
                train_np["state"].shape[1],
                train_np["decision"].shape[1],
                ridge_delta_train.shape[2],
                hidden_dim,
            ).to(device)

            train_t = _tensorize(train_np, device)
            ridge_t = torch.as_tensor(ridge_delta_train, device=device)
            calib_np, _ = build_fn(calibration_rows, stats)
            ridge_calib = backend_fold.predict(calib_np)
            calib_t = _tensorize(calib_np, device)
            ridge_calib_t = torch.as_tensor(ridge_calib, device=device)

            np.random.seed(seed + fold_number * 100 + member_idx)
            torch.manual_seed(seed + fold_number * 100 + member_idx)

            opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-3)
            best_loss, best_state, stale, best_epoch = float("inf"), None, 0, 0

            for epoch in range(epochs):
                mlp.train()
                opt.zero_grad()
                pred = mlp(train_t["state"], train_t["decision"], ridge_t)
                loss = objective(pred, train_t)
                loss.backward()
                nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                opt.step()
                mlp.eval()
                with torch.no_grad():
                    val_loss = float(objective(mlp(calib_t["state"], calib_t["decision"], ridge_calib_t), calib_t).item())
                if val_loss < best_loss - 1e-5:
                    best_loss = val_loss
                    best_state = copy.deepcopy(mlp.state_dict())
                    best_epoch = epoch + 1
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    break

            if best_state is None:
                raise RuntimeError("no checkpoint")
            mlp.load_state_dict(best_state)

            # Refit on full train
            train_full_np, _ = build_fn(train_rows, stats)
            ridge_train_full = backend_full.predict(train_full_np)
            mlp2 = DecisionMLP(
                train_full_np["state"].shape[1], train_full_np["decision"].shape[1],
                ridge_train_full.shape[2], hidden_dim,
            ).to(device)

            train_full_t = _tensorize(train_full_np, device)
            ridge_full_t_t = torch.as_tensor(ridge_train_full, device=device)
            val_t = _tensorize(val_np, device)
            ridge_val_t = torch.as_tensor(ridge_delta_val, device=device)

            torch.manual_seed(seed + 50000 + fold_number * 100 + member_idx)
            opt2 = torch.optim.AdamW(mlp2.parameters(), lr=lr, weight_decay=1e-3)
            for _ in range(best_epoch):
                mlp2.train()
                opt2.zero_grad()
                loss_f = objective(mlp2(train_full_t["state"], train_full_t["decision"], ridge_full_t_t), train_full_t)
                loss_f.backward()
                nn.utils.clip_grad_norm_(mlp2.parameters(), 1.0)
                opt2.step()

            member_val.append(_predict_decision(mlp2, val_t["state"], val_t["decision"], ridge_val_t))

        val_mean, val_std = _mean_std(member_val)
        for local, row in enumerate(val_rows):
            gi = index_map[id(row)]
            for key in oof:
                oof[key][gi] = float(val_mean[key][local])
                oof_std[key][gi] = float(val_std[key][local])

    # Global threshold selection (diagnostic only; per-fold inner-calibration is the proper protocol)
    final_threshold, final_policy = choose_threshold(
        rows, oof, oof_std, z=lcb_z, cost_credit=cost_credit)
    selector = _state_policy(
        rows, oof["handback"], oof_std["handback"],
        predicted_remaining=oof.get("remaining"),
        threshold=final_threshold, z=lcb_z, cost_credit=cost_credit)

    return {
        "dynamics_mse": float(np.mean(dyn_mse_list)),
        "persistence_mse": float(np.mean(persistence_mse_list)),
        "dynamics_improvement": 1.0 - float(np.mean(dyn_mse_list)) / max(float(np.mean(persistence_mse_list)), 1e-12),
        "selected_threshold": final_threshold,
        "selector": {k: v for k, v in selector.items() if k != "decisions"},
        "deployment": final_policy,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    validate_rows(rows)
    folds = grouped_task_folds(rows, 6)

    # Label misalignment
    label_mis = compute_label_misalignment(rows)
    print(f"\nLabel misalignment: {label_mis['mismatch_rate']:.4f} "
          f"({label_mis['mismatches']}/{label_mis['total_comparable_rows']})")

    # Ablation configurations
    configs = [
        {"name": "ridge_h96_ens3_hist_cost010", "backend": RidgeDynamicsBackend(1000),
         "hidden_dim": 96, "ensemble_size": 3, "use_history": True, "cost_credit": 0.10},
        {"name": "ridge_h96_ens5_hist_cost005", "backend": RidgeDynamicsBackend(1000),
         "hidden_dim": 96, "ensemble_size": 5, "use_history": True, "cost_credit": 0.05},
        {"name": "ridge_h128_ens3_hist_cost010", "backend": RidgeDynamicsBackend(1000),
         "hidden_dim": 128, "ensemble_size": 3, "use_history": True, "cost_credit": 0.10},
        {"name": "linear_h96_ens3_hist_cost010", "backend": LinearDynamicsBackend(1000),
         "hidden_dim": 96, "ensemble_size": 3, "use_history": True, "cost_credit": 0.10},
        {"name": "persistence_h96_ens3_hist_cost010", "backend": PersistenceBackend(),
         "hidden_dim": 96, "ensemble_size": 3, "use_history": True, "cost_credit": 0.10},
        {"name": "ridge_h96_ens3_nohist_cost010", "backend": RidgeDynamicsBackend(1000),
         "hidden_dim": 96, "ensemble_size": 3, "use_history": False, "cost_credit": 0.10},
        {"name": "ridge_h96_ens3_hist_cost020", "backend": RidgeDynamicsBackend(1000),
         "hidden_dim": 96, "ensemble_size": 3, "use_history": True, "cost_credit": 0.20},
    ]

    results = {"label_misalignment": label_mis, "ablations": {}}
    print(f"\n{'Ablation':<42} {'DynImp':>8} {'Delta':>7} {'Savings':>8} {'FBs':>4} {'FB%':>7} {'T':>5}")
    print("-" * 95)

    for cfg in configs:
        name = cfg.pop("name")
        print(f"  Running {name}...")
        r = run_ablation(rows, folds, seed=20260808, epochs=200, patience=30,
                         lr=3e-4, lcb_z=1.64, device=args.device, **cfg)
        results["ablations"][name] = r
        results["ablations"][name]["config"] = {k: str(v) if not isinstance(v, (int, float, bool)) else v
                                                  for k, v in cfg.items()}
        cfg["name"] = name
        sel = r["selector"]
        print(f"  {name:<42} {r['dynamics_improvement']:8.4f} {sel['success_minus_persistent']:+7.4f} "
              f"{sel['oft_step_savings_fraction']:8.4f} {sel['false_handbacks']:4d} "
              f"{sel['false_handback_rate_persistent_rescuable']:7.4f} {r['selected_threshold']:5.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": "rase-pre-c0-r4-ablation-matrix/v1",
        "dataset": str(args.dataset.resolve()),
        "n_rows": len(rows),
        "n_states": len({str(r["state_key"]) for r in rows}),
        **results,
    }, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nResults: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
