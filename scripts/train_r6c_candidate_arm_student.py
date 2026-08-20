#!/usr/bin/env python3
"""Train the no-world-model candidate-arm baseline (CandidateArmStudent).

Uses the candidate-arm dataset built by ``build_candidate_arm_dataset.py``
(rows are boundaries of trajectory groups; every row carries per-arm outcome
labels ``arm_success`` / ``arm_teacher_steps`` for CONTINUE_SOURCE and
ENTER_PERSISTENT_OFT).

Evaluation is identical to the R6-C protocol: 5 task-held-out outer folds,
three task-bootstrap ensemble members, 1.64-sigma LCB of source success,
two-boundary dwell, and the R6-C per-policy gate (success gap >= -5pp,
false continue <= 5%, savings >= 20%).  Output schema is compatible with
``audit_r6c_dynamic_stability.py`` so the same 4/5-seed stage gate applies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.risk.light_risk_student import CandidateArmStudent  # noqa: E402
from rase.risk.tiny_universal_state_encoder import TinyUniversalStateEncoder  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def folds(tasks: list[str], count: int, seed: int) -> list[set[str]]:
    values = sorted(set(tasks))
    random.Random(seed).shuffle(values)
    return [set(values[index::count]) for index in range(count)]


def calibration_tasks(train_tasks: set[str], fold: int, count: int = 6) -> set[str]:
    values = sorted(train_tasks)
    offset = (fold * count) % len(values)
    rotated = values[offset:] + values[:offset]
    return set(rotated[: min(count, max(1, len(values) // 3))])


def normalize(values: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[fit].mean(0)
    std = values[fit].std(0)
    std[std < 1e-5] = 1.0
    return (values - mean) / std, mean, std


def group_boundaries(data: dict[str, np.ndarray], idx: np.ndarray) -> list[str]:
    order = np.argsort(idx, kind="stable")
    group_ids: list[str] = []
    for position in order:
        group = str(data["group_id"][idx[position]])
        group_ids.append(group)
    return group_ids


def controller_metrics(data: dict[str, np.ndarray], idx: np.ndarray,
                       lcb: np.ndarray, threshold: float, dwell: int) -> dict:
    """Two-boundary dwell controller; baseline is ENTER_PERSISTENT_OFT at t=0."""
    group_ids = group_boundaries(data, idx)
    by_group: dict[str, list[tuple[int, float]]] = {}
    for local, group in zip(range(len(idx)), group_ids):
        by_group.setdefault(group, []).append((int(idx[local]), float(lcb[local])))
    entered = 0
    false_continue = 0
    controller_success = 0
    controller_teacher = 0.0
    baseline_success = 0
    baseline_teacher = 0.0
    n_trajectories = 0
    trajectory_records = []
    for group, members in by_group.items():
        members.sort(key=lambda item: item[0])
        rows = [item[0] for item in members]
        lcbs = np.asarray([item[1] for item in members])
        arm_success = data["arm_success"][rows]  # (n, n_arms)
        arm_steps = data["arm_teacher_steps"][rows]  # (n, n_arms)
        persistent = arm_success[:, 1].astype(bool)
        psteps = arm_steps[:, 1].astype(float)
        source_success = bool(arm_success[0, 0])
        baseline_success += int(persistent[0])
        baseline_teacher += float(psteps[0])
        n_trajectories += 1
        streak = 0
        enter_at = None
        for position, row in enumerate(rows):
            if lcbs[position] < threshold:
                streak += 1
                if streak >= dwell:
                    enter_at = position
                    break
            else:
                streak = 0
        if enter_at is not None:
            entered += 1
            controller_success += int(persistent[enter_at])
            controller_teacher += float(psteps[enter_at])
        else:
            controller_success += int(source_success)
            if (not source_success) and persistent[0]:
                false_continue += 1
        trajectory_records.append({
            "group_id": group, "state_key": str(data["state_key"][rows[0]]),
            "task_id": str(data["task_id"][rows[0]]),
            "n_boundaries": len(rows),
            "entered_persistent": enter_at is not None,
            "source_success": bool(source_success),
            "persistent_success_at_0": bool(persistent[0]),
            "controller_success": bool(controller_success),
            "controller_teacher_steps": float(controller_teacher),
            "baseline_teacher_steps": float(psteps[0]),
        })
    return {
        "episodes": float(n_trajectories),
        "entered": float(entered),
        "successes": float(controller_success),
        "baseline_successes": float(baseline_success),
        "success_gap": float((controller_success - baseline_success) / n_trajectories),
        "false_continue": float(false_continue),
        "false_continue_rate": float(false_continue / max(1, baseline_success)),
        "teacher_steps": float(controller_teacher),
        "baseline_teacher_steps": float(baseline_teacher),
        "savings": float(1.0 - controller_teacher / max(1.0, baseline_teacher)),
        "trajectories": trajectory_records,
    }


def select_threshold(data: dict[str, np.ndarray], idx: np.ndarray, lcb: np.ndarray,
                     dwell: int) -> tuple[float, dict]:
    best = None
    values = sorted(set([-0.01, 1.01, *np.linspace(0, 1, 101).tolist(), *lcb.tolist()]))
    for threshold in values:
        metrics = controller_metrics(data, idx, lcb, threshold, dwell)
        if metrics["success_gap"] < -0.05 or metrics["false_continue_rate"] > 0.05:
            continue
        rank = (metrics["savings"], metrics["success_gap"], -threshold)
        if best is None or rank > best[0]:
            best = (rank, float(threshold), metrics)
    if best is None:
        return 1.01, controller_metrics(data, idx, lcb, 1.01, dwell)
    return best[1], best[2]


def eligible_masks(data: dict[str, np.ndarray], mode: str, target: str | None,
                   source: str | None) -> tuple[np.ndarray, np.ndarray, bool]:
    policy = data["policy_id"]
    if mode == "per_vla":
        if not target:
            raise ValueError("per_vla requires --target-policy")
        mask = policy == target
        return mask, mask, False
    if mode == "zero_shot":
        if not target or not source:
            raise ValueError("zero_shot requires target and source")
        return policy == source, policy == target, False
    if mode == "loo":
        if not target:
            raise ValueError("loo requires --target-policy")
        return policy != target, policy == target, False
    raise ValueError(mode)


def train_member(data: dict[str, np.ndarray], fit_idx: np.ndarray, *,
                 use_policy_id: bool, seed: int, epochs: int, device: str) -> CandidateArmStudent:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    task_values = sorted(set(data["task_id"][fit_idx].tolist()))
    rng = random.Random(seed ^ 0xB0075A9)
    sampled = [rng.choice(task_values) for _ in task_values]
    boot = np.concatenate([fit_idx[data["task_id"][fit_idx] == task] for task in sampled])
    prop, _, _ = normalize(data["proprio"], boot)
    action, _, _ = normalize(data["action_summary"], boot)
    history, _, _ = normalize(data["history"], boot)
    progress, _, _ = normalize(data["elapsed_progress"].reshape(-1, 1), boot)

    n_policies = int(data["policy_index"].max()) + 1
    encoder = TinyUniversalStateEncoder(
        image_size=96, proprio_dim=8, text_embed_dim=256, hidden_dim=128,
        output_dim=128, input_mode="image",
    )
    n_arms = int(data["arm_ids"].shape[0])
    wm_dim = 0
    wm_tensor = None
    if "_wm_aligned" in data:
        first = data["_wm_aligned"][boot[0]]
        wm_dim = int(first.shape[0])
        wm_tensor = torch.stack([torch.as_tensor(data["_wm_aligned"][i], device=device)
                                 for i in boot])
    model = CandidateArmStudent(
        encoder, n_arms=n_arms, proprio_dim=8, action_dim=data["action_summary"].shape[1],
        history_dim=data["history"].shape[1], fused_dim=128, head_hidden=128,
        n_members=1, n_cost_quantiles=3, use_unsafe_head=False, wm_dim=wm_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    q = torch.tensor([0.1, 0.5, 0.9], device=device)

    image = torch.as_tensor(data["image"][boot].astype(np.float32) / 255.0, device=device)
    p = torch.as_tensor(prop[boot], device=device)
    a = torch.as_tensor(action[boot], device=device)
    h = torch.as_tensor(history[boot], device=device)
    lang = torch.as_tensor(data["language_hash"][boot], device=device)
    arm_succ = torch.as_tensor(data["arm_success"][boot], device=device)  # (n, n_arms)
    arm_steps = torch.as_tensor(data["arm_teacher_steps"][boot], device=device)  # (n, n_arms)
    target_cost = torch.log1p(arm_steps)
    withins = torch.stack([
        torch.as_tensor(data["source_within_8"][boot], device=device),
        torch.as_tensor(data["source_within_16"][boot], device=device),
        torch.as_tensor(data["source_within_32"][boot], device=device),
    ], dim=-1)

    for _ in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        out = model(image, p, a, h, lang, wm_tensor)
        source_loss = F.binary_cross_entropy(
            out["source_success"][0], arm_succ[:, 0])
        within_loss = F.binary_cross_entropy(
            out["source_within"][0], withins)
        arm_loss = F.binary_cross_entropy(
            out["arm_success"][0].reshape(-1), arm_succ.reshape(-1))
        error = target_cost.unsqueeze(-1) - out["arm_cost"][0]  # (B, n_arms, Q)
        cost_loss = torch.maximum(q * error, (q - 1) * error).mean()
        loss = source_loss + 0.5 * within_loss + 0.5 * arm_loss + 0.12 * cost_loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    model.eval()
    model._r6_prop_mean = torch.as_tensor(data["proprio"][boot].mean(0), device=device)
    prop_std = data["proprio"][boot].std(0); prop_std[prop_std < 1e-5] = 1.0
    model._r6_prop_std = torch.as_tensor(prop_std, device=device)
    model._r6_action_mean = torch.as_tensor(data["action_summary"][boot].mean(0), device=device)
    action_std = data["action_summary"][boot].std(0); action_std[action_std < 1e-5] = 1.0
    model._r6_action_std = torch.as_tensor(action_std, device=device)
    model._r6_history_mean = torch.as_tensor(data["history"][boot].mean(0), device=device)
    history_std = data["history"][boot].std(0); history_std[history_std < 1e-5] = 1.0
    model._r6_history_std = torch.as_tensor(history_std, device=device)
    return model


@torch.no_grad()
def predict(model: CandidateArmStudent, data: dict[str, np.ndarray], idx: np.ndarray,
            device: str) -> dict[str, np.ndarray]:
    image = torch.as_tensor(data["image"][idx].astype(np.float32) / 255.0, device=device)
    prop = (torch.as_tensor(data["proprio"][idx], device=device) - model._r6_prop_mean) / model._r6_prop_std
    action = (torch.as_tensor(data["action_summary"][idx], device=device) - model._r6_action_mean) / model._r6_action_std
    history = (torch.as_tensor(data["history"][idx], device=device) - model._r6_history_mean) / model._r6_history_std
    lang = torch.as_tensor(data["language_hash"][idx], device=device)
    wm = None
    if "_wm_aligned" in data:
        wm = torch.stack([torch.as_tensor(data["_wm_aligned"][i], device=device) for i in idx])
    out = model(image, prop, action, history, lang, wm)
    return {
        "source": out["source_success"].mean(0).cpu().numpy(),
        "within": out["source_within"].mean(0).cpu().numpy(),
        "arm_success": out["arm_success"].mean(0).cpu().numpy(),
        "arm_cost": torch.expm1(out["arm_cost"]).mean(0).cpu().numpy(),
    }


def load_wm_features(path: Path) -> dict[str, np.ndarray]:
    """Load the R6-D world-model feature cache and align by group + elapsed.

    Returns {group_id: {elapsed_source_steps: np.ndarray (wm_dim,)}}.
    """
    from rase.risk.wm_features import feature_vector

    aligned: dict[str, dict[int, np.ndarray]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        vector = feature_vector(row)
        if vector is None:
            continue
        group = str(row["group_id"])
        elapsed = int(row["elapsed_source_steps"])
        aligned.setdefault(group, {})[elapsed] = vector
    return aligned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["per_vla", "zero_shot", "loo"], required=True)
    parser.add_argument("--target-policy")
    parser.add_argument("--source-policy")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260810)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--dwell", type=int, default=2)
    parser.add_argument("--lcb-z", type=float, default=1.6448536269514722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wm-features", type=Path, default=None,
                        help="R6-D world-model feature cache jsonl (pre-registered ablation)")
    args = parser.parse_args()

    report = json.loads(args.dataset_report.read_text())
    if report.get("status") != "complete" or report.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("dataset/report lock mismatch")
    raw = np.load(args.dataset)
    data = {key: raw[key] for key in raw.files}
    if "arm_success" not in data or "arm_teacher_steps" not in data:
        raise ValueError("candidate-arm dataset missing arm outcome labels")

    wm_aligned: dict[str, dict[int, np.ndarray]] = {}
    wm_dim = 0
    if args.wm_features is not None:
        wm_aligned = load_wm_features(args.wm_features)
        if not wm_aligned:
            raise ValueError("--wm-features produced no aligned feature rows")
        wm_dim = next(iter(next(iter(wm_aligned.values())).values())).shape[0]
        data["_wm_aligned"] = np.empty(len(data["group_id"]), dtype=object)
        missing = 0
        for position, (group, elapsed) in enumerate(zip(data["group_id"], data["elapsed_source_steps"])):
            vector = wm_aligned.get(str(group), {}).get(int(elapsed))
            if vector is None:
                missing += 1
                vector = np.zeros(wm_dim, dtype=np.float32)
            data["_wm_aligned"][position] = vector
        if missing:
            print(f"WARNING: {missing}/{len(data['group_id'])} rows have no WM features", file=sys.stderr)
    train_mask, eval_mask, use_policy_id = eligible_masks(
        data, args.mode, args.target_policy, args.source_policy)
    task_folds = folds(data["task_id"].tolist(), args.folds, args.fold_seed)
    predictions: list[dict] = []
    fold_reports = []
    for fold, validation_tasks in enumerate(task_folds):
        train_tasks = set(data["task_id"].tolist()) - validation_tasks
        cal_tasks = calibration_tasks(train_tasks, fold)
        fit_tasks = train_tasks - cal_tasks
        fit_idx = np.where(train_mask & np.isin(data["task_id"], list(fit_tasks)))[0]
        cal_idx = np.where(train_mask & np.isin(data["task_id"], list(cal_tasks)))[0]
        val_idx = np.where(eval_mask & np.isin(data["task_id"], list(validation_tasks)))[0]
        if min(len(fit_idx), len(cal_idx), len(val_idx)) == 0:
            raise ValueError(f"fold {fold} has an empty partition")
        models = [train_member(data, fit_idx, use_policy_id=use_policy_id,
                               seed=args.seed + fold * 1009 + member * 7919,
                               epochs=args.epochs, device=args.device)
                  for member in range(args.members)]
        cal_pred = [predict(model, data, cal_idx, args.device) for model in models]
        cal_source = np.stack([value["source"] for value in cal_pred])
        cal_lcb = np.clip(cal_source.mean(0) - args.lcb_z * cal_source.std(0), 0, 1)
        val_pred = [predict(model, data, val_idx, args.device) for model in models]
        source = np.stack([value["source"] for value in val_pred])
        lcb = np.clip(source.mean(0) - args.lcb_z * source.std(0), 0, 1)
        threshold, _ = select_threshold(data, cal_idx, cal_lcb, args.dwell)
        metrics = controller_metrics(data, val_idx, lcb, threshold, args.dwell)
        fold_reports.append({
            "fold": fold, "threshold": threshold,
            "fit_rows": len(fit_idx), "calibration_rows": len(cal_idx),
            "validation_rows": len(val_idx),
            "validation_metrics": {key: value for key, value in metrics.items() if key != "trajectories"},
        })
        for local, index in enumerate(val_idx):
            predictions.append({
                "index": int(index), "fold": fold,
                "group_id": str(data["group_id"][index]),
                "state_key": str(data["state_key"][index]),
                "task_id": str(data["task_id"][index]),
                "policy_id": str(data["policy_id"][index]),
                "elapsed_source_steps": int(data["elapsed_source_steps"][index]),
                "source_success": bool(data["source_success"][index]),
                "persistent_success": bool(data["arm_success"][index, 1]),
                "source_mean": float(source[:, local].mean()),
                "source_std": float(source[:, local].std()),
                "source_lcb": float(lcb[local]),
                "threshold": float(threshold),
                "risky": bool(lcb[local] < threshold),
                "arm_success_mean": float(np.stack([value["arm_success"][local] for value in val_pred]).mean()),
                "arm_cost_q50": float(np.stack([value["arm_cost"][local, :, 1] for value in val_pred]).mean()),
            })
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions.sort(key=lambda row: (row["group_id"], row["elapsed_source_steps"]))
    all_idx = np.asarray([row["index"] for row in predictions], dtype=int)
    all_lcb = np.asarray([row["source_lcb"] for row in predictions], dtype=float)
    threshold = float(np.asarray([row["threshold"] for row in predictions]).mean())
    metrics = controller_metrics(data, all_idx, all_lcb, threshold, args.dwell)
    by_policy = {}
    for policy in sorted(set(data["policy_id"][all_idx].tolist())):
        mask = data["policy_id"][all_idx] == policy
        by_policy[policy] = {key: value for key, value in
                             controller_metrics(data, all_idx[mask], all_lcb[mask], threshold, args.dwell).items()
                             if key != "trajectories"}
    gate = (metrics["success_gap"] >= -0.05 and metrics["false_continue_rate"] <= 0.05
            and metrics["savings"] >= 0.20 and all(
                value["success_gap"] >= -0.05 and value["false_continue_rate"] <= 0.05
                and value["savings"] >= 0.20 for value in by_policy.values()))
    result = {
        "schema_version": ("rase-r6c-candidate-arm-oof-wm/v1" if args.wm_features is not None
                           else "rase-r6c-candidate-arm-oof/v1"),
        "status": "complete",
        "scientific_scope": "no-world-model candidate-arm baseline; two-boundary dwell",
        "wm_features": None if args.wm_features is None else str(args.wm_features.resolve()),
        "wm_feature_dim": wm_dim if args.wm_features is not None else None,
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
        "mode": args.mode, "target_policy": args.target_policy, "source_policy": args.source_policy,
        "seed": args.seed, "fold_seed": args.fold_seed, "folds": args.folds,
        "members": args.members, "epochs": args.epochs, "dwell": args.dwell, "lcb_z": args.lcb_z,
        "metrics": {key: value for key, value in metrics.items() if key != "trajectories"},
        "metrics_by_policy": by_policy,
        "seed_gate_passed": gate,
        "fold_reports": fold_reports,
        "trajectory_records": metrics["trajectories"],
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in
                      ["mode", "target_policy", "seed", "metrics", "metrics_by_policy",
                       "seed_gate_passed"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
