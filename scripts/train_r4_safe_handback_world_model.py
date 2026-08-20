#!/usr/bin/env python3
"""Train and gate the PRE-C0-R4 risk-aware safe-handback world model.

The input contains paired one-step Student/OFT latent transitions at several
boundaries per exact state.  All reported predictions are out-of-fold with
logical tasks held out as groups.  The controller uses ensemble lower bounds
and an explicit teacher-step cost; it never trains the rejected success-only
operator selector.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _vec(row: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(row[key], dtype=np.float32).reshape(-1)
    if value.size == 0:
        raise ValueError(f"{row.get('state_key')}: empty {key}")
    return value


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        raise ValueError("boundary dataset is empty")
    seen = set()
    dims = {}
    for row in rows:
        key = (str(row["state_key"]), int(row["elapsed_oft_steps"]))
        if key in seen:
            raise ValueError(f"duplicate state/boundary: {key}")
        seen.add(key)
        local = {
            "latent": _vec(row, "latent").size,
            "proprio": _vec(row, "proprio").size,
            "student_action": _vec(row, "student_action").size,
            "oft_action": _vec(row, "oft_action").size,
            "student_chunk": _vec(row, "student_action_chunk").size,
        }
        if dims and local != dims:
            raise ValueError(f"feature dimensions differ at {key}: {local} vs {dims}")
        dims = local
        for next_key in ("next_latent_student", "next_latent_oft"):
            if _vec(row, next_key).size != dims["latent"]:
                raise ValueError(f"latent transition dimension mismatch at {key}")
        if dims["student_action"] != 7 or dims["oft_action"] != 7:
            raise ValueError(f"actions must flatten to 7 dimensions: {key}")
    return dims


class Standardizer:
    def __init__(self, values: np.ndarray) -> None:
        self.mean = values.mean(0).astype(np.float32)
        self.std = values.std(0).clip(1e-5).astype(np.float32)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def _stack(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.stack([_vec(row, key) for row in rows]).astype(np.float32)


def build_arrays(rows: list[dict[str, Any]], stats: dict[str, Standardizer] | None = None):
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
    decision = np.concatenate(
        [stats["chunk"](chunk), stats["action"](oft)], axis=1
    )
    actions = np.stack([stats["action"](student), stats["action"](oft)], axis=1)
    action_type = np.zeros((len(rows), 2, 2), dtype=np.float32)
    action_type[:, 0, 0] = 1.0
    action_type[:, 1, 1] = 1.0
    transitions = np.concatenate([actions, action_type], axis=2)
    next_latents = np.stack(
        [stats["latent"](next_student), stats["latent"](next_oft)], axis=1
    )
    current = stats["latent"](latent)[:, None, :]
    delta = next_latents - current
    terminal = np.asarray(
        [[r["student_step_terminal"], r["oft_step_terminal"]] for r in rows],
        dtype=np.float32,
    )
    handback = np.asarray([r["success_if_handback_now"] for r in rows], np.float32)
    persistent = np.asarray([r["success_if_continue_oft"] for r in rows], np.float32)
    remaining = np.asarray(
        [float(r["remaining_teacher_steps"]) / max(1.0, float(r["persistent_executed_oft_steps"]))
         for r in rows],
        np.float32,
    )
    return {
        "state": state,
        "decision": decision,
        "transition": transitions,
        "delta": delta,
        "terminal": terminal,
        "handback": handback,
        "risk": 1.0 - handback,
        "persistent": persistent,
        "remaining": remaining,
    }, stats


def _tensorize(data: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in data.items()
    }


class SafeHandbackWorldModel(nn.Module):
    def __init__(self, state_dim: int, decision_dim: int, latent_dim: int,
                 transition_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + transition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, latent_dim)
        self.terminal_head = nn.Linear(hidden_dim, 1)
        self.decision = nn.Sequential(
            nn.Linear(hidden_dim + decision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.handback_head = nn.Linear(hidden_dim, 1)
        self.persistent_head = nn.Linear(hidden_dim, 1)
        self.remaining_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, decision: torch.Tensor,
                transition: torch.Tensor):
        encoded = self.state_encoder(state)
        batch, arms, _ = transition.shape
        repeated = encoded[:, None, :].expand(-1, arms, -1)
        dyn = self.dynamics(torch.cat([repeated, transition], dim=-1))
        delta = self.delta_head(dyn)
        terminal = self.terminal_head(dyn).squeeze(-1)
        dec = self.decision(torch.cat([encoded, decision], dim=-1))
        return {
            "delta": delta,
            "terminal": terminal,
            "risk": self.risk_head(dec).squeeze(-1),
            "handback": self.handback_head(dec).squeeze(-1),
            "persistent": self.persistent_head(dec).squeeze(-1),
            "remaining": self.remaining_head(dec).squeeze(-1),
        }


def _positive_weight(target: torch.Tensor) -> torch.Tensor:
    positive = target.sum().clamp_min(1.0)
    negative = (target.numel() - target.sum()).clamp_min(1.0)
    return (negative / positive).clamp(0.25, 6.0)


def objective(pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor],
              dynamics_weight: float) -> torch.Tensor:
    nonterminal = (1.0 - target["terminal"])[..., None]
    dyn_error = F.smooth_l1_loss(pred["delta"], target["delta"], reduction="none")
    dyn_denominator = nonterminal.sum().clamp_min(1.0) * pred["delta"].shape[-1]
    dyn_loss = (dyn_error * nonterminal).sum() / dyn_denominator
    term_loss = F.binary_cross_entropy_with_logits(pred["terminal"], target["terminal"])
    risk_loss = F.binary_cross_entropy_with_logits(
        pred["risk"], target["risk"], pos_weight=_positive_weight(target["risk"])
    )
    hand_loss = F.binary_cross_entropy_with_logits(
        pred["handback"], target["handback"],
        pos_weight=_positive_weight(target["handback"]),
    )
    persistent_loss = F.binary_cross_entropy_with_logits(
        pred["persistent"], target["persistent"],
        pos_weight=_positive_weight(target["persistent"]),
    )
    remaining_loss = F.smooth_l1_loss(torch.sigmoid(pred["remaining"]), target["remaining"])
    return (
        dynamics_weight * dyn_loss
        + 0.25 * term_loss
        + risk_loss
        + hand_loss
        + 0.5 * persistent_loss
        + 0.25 * remaining_loss
    )


def _predict(model: nn.Module, data: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred = model(data["state"], data["decision"], data["transition"])
    return {
        "delta": pred["delta"].cpu().numpy(),
        "terminal": torch.sigmoid(pred["terminal"]).cpu().numpy(),
        "risk": torch.sigmoid(pred["risk"]).cpu().numpy(),
        "handback": torch.sigmoid(pred["handback"]).cpu().numpy(),
        "persistent": torch.sigmoid(pred["persistent"]).cpu().numpy(),
        "remaining": torch.sigmoid(pred["remaining"]).cpu().numpy(),
    }


def fit_member(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], *,
               device: str, seed: int, epochs: int, lr: float, hidden_dim: int,
               patience: int, dynamics_weight: float):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_np, stats = build_arrays(train_rows)
    val_np, _ = build_arrays(val_rows, stats)
    train = _tensorize(train_np, device)
    val = _tensorize(val_np, device)
    model = SafeHandbackWorldModel(
        train_np["state"].shape[1],
        train_np["decision"].shape[1],
        train_np["delta"].shape[2],
        train_np["transition"].shape[2],
        hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_loss, best_state, stale, best_epoch = float("inf"), None, 0, 0
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        pred = model(train["state"], train["decision"], train["transition"])
        loss = objective(pred, train, dynamics_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(val["state"], val["decision"], val["transition"])
            val_loss = float(objective(val_pred, val, dynamics_weight).item())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, stats, _predict(model, train), _predict(model, val), {
        "best_val_loss": best_loss, "best_epoch": best_epoch
    }


def fit_full_member(rows: list[dict[str, Any]], *, device: str, seed: int,
                    epochs: int, lr: float, hidden_dim: int,
                    dynamics_weight: float):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_np, stats = build_arrays(rows)
    data = _tensorize(data_np, device)
    config = {
        "state_dim": data_np["state"].shape[1],
        "decision_dim": data_np["decision"].shape[1],
        "latent_dim": data_np["delta"].shape[2],
        "transition_dim": data_np["transition"].shape[2],
        "hidden_dim": hidden_dim,
    }
    model = SafeHandbackWorldModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        pred = model(data["state"], data["decision"], data["transition"])
        loss = objective(pred, data, dynamics_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    normalizers = {key: value.state_dict() for key, value in stats.items()}
    return model, config, normalizers


def fit_fixed_member(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], *,
                     device: str, seed: int, epochs: int, lr: float,
                     hidden_dim: int, dynamics_weight: float):
    """Fit without inspecting the outer held-out fold and predict that fold."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_np, stats = build_arrays(train_rows)
    eval_np, _ = build_arrays(eval_rows, stats)
    train = _tensorize(train_np, device)
    model = SafeHandbackWorldModel(
        train_np["state"].shape[1],
        train_np["decision"].shape[1],
        train_np["delta"].shape[2],
        train_np["transition"].shape[2],
        hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(max(1, int(epochs))):
        model.train()
        optimizer.zero_grad()
        pred = model(train["state"], train["decision"], train["transition"])
        loss = objective(pred, train, dynamics_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    eval_data = _tensorize(eval_np, device)
    return model, stats, _predict(model, eval_data)


def average_precision(y: Iterable[float], p: Iterable[float]) -> float:
    y_arr = np.asarray(list(y), np.float32)
    p_arr = np.asarray(list(p), np.float32)
    if y_arr.sum() == 0:
        return float("nan")
    order = np.argsort(-p_arr)
    ranked = y_arr[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / y_arr.sum())


def roc_auc(y: Iterable[float], p: Iterable[float]) -> float:
    y_arr = np.asarray(list(y), np.float32)
    p_arr = np.asarray(list(p), np.float32)
    positive, negative = p_arr[y_arr == 1], p_arr[y_arr == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = [(a > b) + 0.5 * (a == b) for a in positive for b in negative]
    return float(np.mean(comparisons))


def grouped_task_folds(rows: list[dict[str, Any]], n_folds: int) -> list[dict[str, Any]]:
    by_suite: dict[str, list[str]] = defaultdict(list)
    for task in sorted({str(row["task_id"]) for row in rows}):
        suite = next(str(row["suite"]) for row in rows if str(row["task_id"]) == task)
        by_suite[suite].append(task)
    fold_tasks = [set() for _ in range(min(n_folds, len({r["task_id"] for r in rows})))]
    offset = 0
    for suite, tasks in sorted(by_suite.items()):
        for index, task in enumerate(sorted(tasks)):
            fold_tasks[(offset + index) % len(fold_tasks)].add(task)
        offset = (offset + len(tasks)) % len(fold_tasks)
    folds = []
    for index, held_out in enumerate(fold_tasks):
        if not held_out:
            continue
        train = [row for row in rows if str(row["task_id"]) not in held_out]
        val = [row for row in rows if str(row["task_id"]) in held_out]
        if train and val:
            folds.append({"fold": index, "held_out_tasks": sorted(held_out),
                          "train": train, "val": val})
    return folds


def inner_task_split(rows: list[dict[str, Any]], rotation: int) -> tuple[list[dict], list[dict], list[str]]:
    """Reserve task groups for early stopping/calibration inside an outer fold."""
    by_suite: dict[str, list[str]] = defaultdict(list)
    for task in sorted({str(row["task_id"]) for row in rows}):
        suite = next(str(row["suite"]) for row in rows if str(row["task_id"]) == task)
        by_suite[suite].append(task)
    calibration_tasks = set()
    for offset, (_, tasks) in enumerate(sorted(by_suite.items())):
        if len(tasks) > 1:
            calibration_tasks.add(tasks[(rotation + offset) % len(tasks)])
    if not calibration_tasks:
        all_tasks = sorted({str(row["task_id"]) for row in rows})
        calibration_tasks.add(all_tasks[rotation % len(all_tasks)])
    fit = [row for row in rows if str(row["task_id"]) not in calibration_tasks]
    calibration = [row for row in rows if str(row["task_id"]) in calibration_tasks]
    if not fit or not calibration:
        raise ValueError("inner task split produced an empty partition")
    return fit, calibration, sorted(calibration_tasks)


def _state_policy(
    rows: list[dict[str, Any]],
    handback: np.ndarray,
    persistent: np.ndarray,
    handback_std: np.ndarray,
    persistent_std: np.ndarray,
    *,
    threshold: float,
    z: float,
    cost_credit: float,
) -> dict[str, Any]:
    by_state: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_state[str(row["state_key"])].append(index)
    decisions = []
    for state, indices in sorted(by_state.items()):
        indices.sort(key=lambda i: int(rows[i]["elapsed_oft_steps"]))
        selected = None
        for index in indices:
            row = rows[index]
            hand_lcb = float(handback[index] - z * handback_std[index])
            persistent_lcb = float(persistent[index] - z * persistent_std[index])
            cost = cost_credit * (
                float(row["remaining_teacher_steps"])
                / max(1.0, float(row["persistent_executed_oft_steps"]))
            )
            if hand_lcb >= threshold and hand_lcb >= persistent_lcb - cost:
                selected = index
                break
        reference = rows[indices[0]]
        persistent_success = bool(reference["success_if_continue_oft"])
        persistent_steps = int(reference["persistent_executed_oft_steps"])
        if selected is None:
            success = persistent_success
            executed = persistent_steps
            action = "CONTINUE_OFT"
            boundary = None
        else:
            success = bool(rows[selected]["success_if_handback_now"])
            executed = int(rows[selected]["elapsed_oft_steps"])
            action = "HAND_BACK_TO_STUDENT"
            boundary = executed
        decisions.append(
            {
                "state_key": state,
                "task_id": str(reference["task_id"]),
                "action": action,
                "boundary": boundary,
                "success": success,
                "persistent_success": persistent_success,
                "executed_oft_steps": executed,
                "persistent_executed_oft_steps": persistent_steps,
                "false_handback": bool(
                    selected is not None and persistent_success and not success
                ),
            }
        )
    return summarize_decisions(decisions)


def summarize_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(decisions)
    persistent_successes = sum(d["persistent_success"] for d in decisions)
    selected_successes = sum(d["success"] for d in decisions)
    persistent_steps = sum(d["persistent_executed_oft_steps"] for d in decisions)
    selected_steps = sum(d["executed_oft_steps"] for d in decisions)
    false_handbacks = sum(d["false_handback"] for d in decisions)
    handbacks = sum(d["action"] == "HAND_BACK_TO_STUDENT" for d in decisions)
    return {
        "n_states": n,
        "success_rate": selected_successes / max(1, n),
        "persistent_success_rate": persistent_successes / max(1, n),
        "success_minus_persistent": (selected_successes - persistent_successes) / max(1, n),
        "executed_oft_steps": selected_steps,
        "persistent_executed_oft_steps": persistent_steps,
        "oft_step_savings_fraction": 1.0 - selected_steps / max(1, persistent_steps),
        "handback_rate": handbacks / max(1, n),
        "false_handbacks": false_handbacks,
        "false_handback_rate_persistent_rescuable": false_handbacks / max(1, persistent_successes),
        "false_handback_rate_conditional": false_handbacks / max(1, handbacks),
        "decisions": decisions,
    }


def choose_threshold(rows: list[dict[str, Any]], mean: dict[str, np.ndarray],
                     std: dict[str, np.ndarray], *, z: float,
                     cost_credit: float) -> tuple[float, dict[str, Any]]:
    candidates = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
    feasible = []
    all_results = []
    for threshold in candidates:
        result = _state_policy(
            rows, mean["handback"], mean["persistent"],
            std["handback"], std["persistent"], threshold=threshold,
            z=z, cost_credit=cost_credit,
        )
        all_results.append((threshold, result))
        if (
            result["success_minus_persistent"] >= -0.05
            and result["false_handback_rate_persistent_rescuable"] <= 0.05
        ):
            feasible.append((threshold, result))
    pool = feasible or all_results
    threshold, result = max(
        pool,
        key=lambda item: (
            item[1]["oft_step_savings_fraction"],
            item[1]["success_rate"],
            item[0],
        ),
    )
    result = {key: value for key, value in result.items() if key != "decisions"}
    result["feasible_on_training"] = bool(feasible)
    return threshold, result


def _mean_std(predictions: list[dict[str, np.ndarray]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    keys = predictions[0]
    mean = {key: np.mean([pred[key] for pred in predictions], axis=0) for key in keys}
    std = {key: np.std([pred[key] for pred in predictions], axis=0) for key in keys}
    return mean, std


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    gate_group = parser.add_mutually_exclusive_group(required=True)
    gate_group.add_argument(
        "--collection-report",
        type=Path,
        help="Merged v3 collection report with a passed live safe-handback gate",
    )
    gate_group.add_argument(
        "--allow-ungated-smoke",
        action="store_true",
        help="Permit code-path smoke tests only; never writes deployment checkpoints",
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dynamics-weight", type=float, default=1.0)
    parser.add_argument("--lcb-z", type=float, default=1.64)
    parser.add_argument("--cost-credit", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    collection_report = None
    if args.collection_report:
        collection_report = json.loads(args.collection_report.read_text())
        if collection_report.get("schema_version") != "rase-pre-c0-r4-boundary-merged/v3":
            raise SystemExit("formal training requires a merged v3 boundary collection")
        if collection_report.get("safe_handback_status") != "ready":
            raise SystemExit(
                "live safe-handback gate is closed: "
                + repr(collection_report.get("safe_handback_reasons"))
            )
        expected_dataset = Path(str(collection_report.get("output", ""))).resolve()
        if expected_dataset != args.dataset.resolve():
            raise SystemExit(
                f"dataset/report mismatch: {args.dataset.resolve()} != {expected_dataset}"
            )
    rows = read_jsonl(args.dataset)
    if collection_report and int(collection_report.get("n_rows", -1)) != len(rows):
        raise SystemExit("dataset row count differs from the passed collection report")
    dims = validate_rows(rows)
    tasks = sorted({str(row["task_id"]) for row in rows})
    states = sorted({str(row["state_key"]) for row in rows})
    if len(tasks) < 3:
        raise SystemExit("need at least three logical tasks for held-out evaluation")
    folds = grouped_task_folds(rows, args.folds)
    if len(folds) < 3:
        raise SystemExit("need at least three nonempty task-held-out folds")

    index = {id(row): i for i, row in enumerate(rows)}
    oof = {key: np.zeros(len(rows), np.float32) for key in (
        "risk", "handback", "persistent", "remaining"
    )}
    oof_std = {key: np.zeros(len(rows), np.float32) for key in oof}
    fold_reports = []
    all_decisions = []
    dynamics_squared_error = []
    persistence_squared_error = []

    for fold_number, fold in enumerate(folds):
        train_rows = fold["train"]
        val_rows = fold["val"]
        inner_fit_rows, calibration_rows, calibration_tasks = inner_task_split(
            train_rows, fold_number
        )
        member_calibration, member_val = [], []
        member_meta = []
        for member in range(args.ensemble_size):
            _, _, _, calibration_pred, meta = fit_member(
                inner_fit_rows,
                calibration_rows,
                device=args.device,
                seed=args.seed + fold_number * 100 + member,
                epochs=args.epochs,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                patience=args.patience,
                dynamics_weight=args.dynamics_weight,
            )
            _, _, val_pred = fit_fixed_member(
                train_rows,
                val_rows,
                device=args.device,
                seed=args.seed + 50_000 + fold_number * 100 + member,
                epochs=int(meta["best_epoch"]),
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                dynamics_weight=args.dynamics_weight,
            )
            member_calibration.append(calibration_pred)
            member_val.append(val_pred)
            meta["outer_refit_epochs"] = int(meta["best_epoch"])
            member_meta.append(meta)
        calibration_mean, calibration_std = _mean_std(member_calibration)
        val_mean, val_std = _mean_std(member_val)
        threshold, calibration_policy = choose_threshold(
            calibration_rows, calibration_mean, calibration_std, z=args.lcb_z,
            cost_credit=args.cost_credit,
        )
        val_policy = _state_policy(
            val_rows,
            val_mean["handback"],
            val_mean["persistent"],
            val_std["handback"],
            val_std["persistent"],
            threshold=threshold,
            z=args.lcb_z,
            cost_credit=args.cost_credit,
        )
        all_decisions.extend(val_policy.pop("decisions"))
        # Dynamics are already in each member's train-fitted normalized space;
        # compare ensemble delta against that member-normalized target.
        _, member_stats = build_arrays(train_rows)
        normalized_val, _ = build_arrays(val_rows, member_stats)
        mask = (1.0 - normalized_val["terminal"])[..., None]
        dyn_error = ((val_mean["delta"] - normalized_val["delta"]) ** 2) * mask
        base_error = (normalized_val["delta"] ** 2) * mask
        denominator = max(1.0, float(mask.sum()) * normalized_val["delta"].shape[-1])
        dynamics_squared_error.append(float(dyn_error.sum() / denominator))
        persistence_squared_error.append(float(base_error.sum() / denominator))
        for local, row in enumerate(val_rows):
            global_index = index[id(row)]
            for key in oof:
                oof[key][global_index] = float(val_mean[key][local])
                oof_std[key][global_index] = float(val_std[key][local])
        fold_reports.append(
            {
                "fold": fold["fold"],
                "held_out_tasks": fold["held_out_tasks"],
                "inner_calibration_tasks": calibration_tasks,
                "n_inner_fit_rows": len(inner_fit_rows),
                "n_calibration_rows": len(calibration_rows),
                "n_train_rows": len(train_rows),
                "n_val_rows": len(val_rows),
                "threshold": threshold,
                "calibration_policy": calibration_policy,
                "val_policy": {key: value for key, value in val_policy.items() if key != "decisions"},
                "members": member_meta,
            }
        )

    selector = summarize_decisions(all_decisions)
    selector_decisions = selector.pop("decisions")
    y_hand = np.asarray([row["success_if_handback_now"] for row in rows], np.float32)
    y_risk = 1.0 - y_hand
    y_persistent = np.asarray([row["success_if_continue_oft"] for row in rows], np.float32)
    dyn_mse = float(np.mean(dynamics_squared_error))
    persistence_mse = float(np.mean(persistence_squared_error))
    hand_ap = average_precision(y_hand, oof["handback"])
    hand_auc = roc_auc(y_hand, oof["handback"])
    risk_ap = average_precision(y_risk, oof["risk"])
    risk_auc = roc_auc(y_risk, oof["risk"])
    persistent_auc = roc_auc(y_persistent, oof["persistent"])
    prevalence = float(y_hand.mean())
    final_threshold, final_policy = choose_threshold(
        rows, oof, oof_std, z=args.lcb_z, cost_credit=args.cost_credit
    )
    threshold_sweep = []
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]:
        value = _state_policy(
            rows,
            oof["handback"],
            oof["persistent"],
            oof_std["handback"],
            oof_std["persistent"],
            threshold=threshold,
            z=args.lcb_z,
            cost_credit=args.cost_credit,
        )
        value.pop("decisions")
        threshold_sweep.append({"threshold": threshold, **value})
    gates = {
        "latent_prediction_improves_10pct": bool(
            dyn_mse <= 0.90 * max(persistence_mse, 1e-12)
        ),
        "handback_auc_ge_070": bool(hand_auc >= 0.70),
        "handback_ap_above_prevalence": bool(hand_ap > prevalence),
        "false_handback_le_005": bool(
            selector["false_handback_rate_persistent_rescuable"] <= 0.05
        ),
        "success_within_005_of_persistent": bool(
            selector["success_minus_persistent"] >= -0.05
        ),
        "oft_step_reduction_ge_020": bool(
            selector["oft_step_savings_fraction"] >= 0.20
        ),
    }
    gate_status = "ready_for_closed_loop_pilot" if all(gates.values()) else "not_ready"
    report = {
        "schema_version": "rase-pre-c0-r4-safe-handback-wm/v1",
        "status": (
            "smoke_only_not_for_pilot" if args.allow_ungated_smoke else gate_status
        ),
        "gate_evaluation_status": gate_status,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "collection_report": (
            str(args.collection_report.resolve()) if args.collection_report else None
        ),
        "collection_gate": (
            collection_report.get("safe_handback_status")
            if collection_report else "bypassed_for_smoke"
        ),
        "n_rows": len(rows),
        "n_states": len(states),
        "n_tasks": len(tasks),
        "tasks": tasks,
        "feature_dims": dims,
        "n_folds": len(folds),
        "ensemble_size": args.ensemble_size,
        "dynamics_mse": dyn_mse,
        "persistence_mse": persistence_mse,
        "dynamics_improvement": 1.0 - dyn_mse / max(persistence_mse, 1e-12),
        "handback_prevalence": prevalence,
        "handback_ap": hand_ap,
        "handback_auc": hand_auc,
        "risk_ap": risk_ap,
        "risk_auc": risk_auc,
        "persistent_auc": persistent_auc,
        "selector_oof": selector,
        "deployment_policy": {
            "handback_lcb_threshold": final_threshold,
            "lcb_z": args.lcb_z,
            "cost_credit": args.cost_credit,
            "selection_source": "task-held-out OOF predictions on train tasks",
            "oof_policy_at_selected_threshold": final_policy,
        },
        "diagnostic_oof_threshold_sweep": threshold_sweep,
        "gates": gates,
        "folds": fold_reports,
        "note": (
            "Outer OOF folds hold out logical tasks and are never inspected for early "
            "stopping or threshold selection. Each threshold and epoch count comes from "
            "an inner task-group calibration split; models are then refit on the complete "
            "outer-training partition. Simulator forks provide labels only."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "oof_predictions.jsonl"
    prediction_records = []
    for index, row in enumerate(rows):
        prediction_records.append({
            "state_key": str(row["state_key"]),
            "task_id": str(row["task_id"]),
            "suite": str(row["suite"]),
            "elapsed_oft_steps": int(row["elapsed_oft_steps"]),
            "persistent_executed_oft_steps": int(row["persistent_executed_oft_steps"]),
            "remaining_teacher_steps": int(row["remaining_teacher_steps"]),
            "success_if_handback_now": bool(row["success_if_handback_now"]),
            "success_if_continue_oft": bool(row["success_if_continue_oft"]),
            **{f"{key}_mean": float(oof[key][index]) for key in oof},
            **{f"{key}_std": float(oof_std[key][index]) for key in oof_std},
        })
    prediction_path.write_text("".join(
        json.dumps(record, sort_keys=True) + "\n" for record in prediction_records
    ))
    report["diagnostic_oof_predictions"] = str(prediction_path.resolve())
    report["diagnostic_oof_predictions_sha256"] = hashlib.sha256(
        prediction_path.read_bytes()
    ).hexdigest()
    report["deployment_artifacts_written"] = False
    if report["status"] == "ready_for_closed_loop_pilot" and not args.allow_ungated_smoke:
        best_epochs = [
            int(member["best_epoch"])
            for fold in fold_reports
            for member in fold["members"]
        ]
        final_epochs = max(1, int(round(float(np.median(best_epochs)))))
        dataset_sha = report["dataset_sha256"]
        for member in range(args.ensemble_size):
            model, config, normalizers = fit_full_member(
                rows,
                device=args.device,
                seed=args.seed + 10_000 + member,
                epochs=final_epochs,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                dynamics_weight=args.dynamics_weight,
            )
            torch.save(
                {
                    "schema_version": "rase-pre-c0-r4-safe-handback-checkpoint/v1",
                    "state_dict": model.state_dict(),
                    "config": config,
                    "normalizers": normalizers,
                    "member": member,
                    "seed": args.seed + 10_000 + member,
                    "training_rows": len(rows),
                    "training_states": states,
                    "training_tasks": tasks,
                    "dataset_sha256": dataset_sha,
                    "handback_lcb_threshold": final_threshold,
                    "lcb_z": args.lcb_z,
                    "cost_credit": args.cost_credit,
                    "epochs": final_epochs,
                },
                args.output_dir / f"member_{member:02d}.pt",
            )
        report["deployment_artifacts_written"] = True
        report["deployment_final_epochs"] = final_epochs
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "oof_state_decisions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selector_decisions)
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ready_for_closed_loop_pilot" else 2


if __name__ == "__main__":
    raise SystemExit(main())
