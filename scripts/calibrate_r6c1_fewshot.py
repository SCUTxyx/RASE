#!/usr/bin/env python3
"""R6-C.2: few-shot calibration curve for the shared-core generalization claim.

For every training seed and each held-out "new" VLA, train a shared risk core
on the other VLA(s), then measure how target-VLA performance improves as the
calibration set grows from 0 to 8/16/32 target trajectory groups:

  N = 0  pure zero-shot (challenge metric only): source-derived descriptor
         prior + source-calibrated thresholds.
  N > 0  target descriptor estimated from N groups; the per-VLA FiLM
         descriptor-conditioned FiLM adapter is fine-tuned on those N groups; controller
         thresholds are fit on the same N groups; evaluation happens on the
         held-out target tasks (never on calibration groups).

Methodological red lines:
  - calibration groups are task-disjoint from evaluation groups;
  - the shared core is trained only on source tasks (no target leak);
  - every VLA task, seed and replica stays in one outer fold;
  - the curve is a generalization claim, not the R6-C.1 gate (zero-shot is
    reported as a challenge metric only).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.train_r6c1_early_selector as trainer  # noqa: E402
from rase.risk.light_risk_student import CandidateArmStudent  # noqa: E402
from rase.risk.tiny_universal_state_encoder import TinyUniversalStateEncoder  # noqa: E402


def sample_cal_groups(data: dict[str, np.ndarray], pool_idx: np.ndarray,
                      target_policy: str, n_groups: int, rng: random.Random,
                      forbid_tasks: set[str]) -> tuple[np.ndarray, list[str]]:
    """Sample ``n_groups`` trajectory groups of the target VLA from ``pool_idx``.

    Groups are task-disjoint from ``forbid_tasks`` (evaluation tasks).  Returns
    the sampled row indices and the sampled group ids.
    """
    pool = pool_idx[data["policy_id"][pool_idx] == target_policy]
    pool = pool[~np.isin(data["task_id"][pool], list(forbid_tasks))]
    groups = sorted({str(data["group_id"][i]) for i in pool})
    if n_groups > len(groups):
        raise ValueError(f"requested {n_groups} calibration groups but only "
                         f"{len(groups)} task-disjoint target groups available")
    chosen_groups = set(rng.sample(groups, n_groups))
    mask = np.isin(data["group_id"][pool], list(chosen_groups))
    idx = pool[mask]
    return idx, sorted(chosen_groups)


def fine_tune_film(model: CandidateArmStudent, data: dict[str, np.ndarray],
                   fit_idx: np.ndarray, target_policy: str,
                   descriptor: np.ndarray, *, device: str,
                   epochs: int = 40, lr: float = 1e-3) -> None:
    """Fine-tune only the descriptor-conditioned FiLM adapter on target groups.

    All other weights (shared core, heads, encoder) stay frozen; only
    ``policy_film`` (and the descriptor MLP that feeds it) are updated.
    """
    params = []
    if model.policy_film is not None:
        params += list(model.policy_film.parameters())
    if model.descriptor_mlp is not None:
        params += list(model.descriptor_mlp.parameters())
    if not params:
        raise ValueError("model has no calibration adapter to fine-tune")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in params:
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)

    image = torch.as_tensor(data["image"][fit_idx].astype(np.float32) / 255.0, device=device)
    prop = ((torch.as_tensor(data["proprio"][fit_idx], device=device)
             - model._r6_prop_mean) / model._r6_prop_std)
    action = ((torch.as_tensor(data["action_summary"][fit_idx], device=device)
               - model._r6_action_mean) / model._r6_action_std)
    history = ((torch.as_tensor(data["history"][fit_idx], device=device)
                - model._r6_history_mean) / model._r6_history_std)
    lang = torch.as_tensor(data["language_hash"][fit_idx], device=device)
    desc = torch.as_tensor(np.tile(descriptor, (len(fit_idx), 1)), device=device)
    arm_succ = torch.as_tensor(data["arm_success"][fit_idx], device=device)
    arm_steps = torch.as_tensor(data["arm_teacher_steps"][fit_idx], device=device)
    target_cost = torch.log1p(arm_steps)
    withins = torch.stack([
        torch.as_tensor(data["source_within_8"][fit_idx], device=device),
        torch.as_tensor(data["source_within_16"][fit_idx], device=device),
        torch.as_tensor(data["source_within_32"][fit_idx], device=device),
    ], dim=-1)
    adv_target = arm_succ[:, 1] - arm_succ[:, 0]
    q = torch.tensor([0.1, 0.5, 0.9], device=device)

    for _ in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        out = model(image, prop, action, history, lang, policy_descriptor=desc)
        source_loss = F.binary_cross_entropy(out["source_success"][0], arm_succ[:, 0])
        within_loss = F.binary_cross_entropy(out["source_within"][0], withins)
        arm_loss = F.binary_cross_entropy(out["arm_success"][0].reshape(-1), arm_succ.reshape(-1))
        error = target_cost.unsqueeze(-1) - out["arm_cost"][0]
        cost_loss = torch.maximum(q * error, (q - 1) * error).mean()
        adv_loss = F.mse_loss(out["advantage"][0], adv_target)
        loss = (source_loss + 0.5 * within_loss + 0.5 * arm_loss
                + 0.12 * cost_loss + 0.5 * adv_loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); optimizer.step()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="10 11 12 13 14")
    parser.add_argument("--shot-sizes", default="0,8,16,32")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260810)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lcb-z", type=float, default=1.6448536269514722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--oof-root", type=Path, default=None,
                        help="optional shared_calib OOF root for comparison reference")
    args = parser.parse_args()

    report = json.loads(args.dataset_report.read_text())
    if report.get("status") != "complete" or report.get("dataset_sha256") != trainer.sha256(args.dataset):
        raise ValueError("dataset/report lock mismatch")
    if report.get("protocol_sha256") != trainer.sha256(args.protocol):
        raise ValueError("dataset/protocol lock mismatch")
    data = {key: np.load(args.dataset)[key] for key in np.load(args.dataset).files}
    if "arm_success" not in data or "arm_teacher_steps" not in data:
        raise ValueError("candidate-arm dataset missing arm outcome labels")

    shot_sizes = [int(value) for value in args.shot_sizes.split(",")]
    policies = sorted(set(data["policy_id"].tolist()))
    cohort_role = (data["cohort_role"] if "cohort_role" in data
                   else np.asarray(["natural"] * len(data["group_id"])))
    natural_mask = cohort_role == "natural"
    task_folds = trainer.folds(data["task_id"][natural_mask].tolist(),
                               args.folds, args.fold_seed)

    # Fold assignment cache: which fold each task belongs to (all VLA states,
    # seeds and replicas of a task stay in the same fold).
    task_fold_of: dict[str, int] = {}
    for fold, tasks in enumerate(task_folds):
        for task in tasks:
            task_fold_of[str(task)] = fold

    points: list[dict] = []
    for seed_value in (int(value) for value in args.seeds.split()):
        for target in policies:
            source_policies = [policy for policy in policies if policy != target]
            # Each (seed, target) uses fresh calibration sampling.
            target_salt = int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], "big")
            rng = random.Random(seed_value ^ target_salt)
            source_mask = data["policy_id"] == source_policies[0]
            if len(source_policies) > 1:
                source_mask = np.logical_or(source_mask, data["policy_id"] == source_policies[1])
            target_mask = data["policy_id"] == target

            for fold in range(args.folds):
                validation_tasks = task_folds[fold]
                train_tasks = set(data["task_id"].tolist()) - validation_tasks
                cal_tasks = trainer.calibration_tasks(train_tasks, fold)
                fit_tasks = train_tasks - cal_tasks
                fit_idx = np.where(source_mask & np.isin(data["task_id"], list(fit_tasks)))[0]
                cal_idx = np.where(source_mask & natural_mask
                                   & np.isin(data["task_id"], list(cal_tasks)))[0]
                val_idx = np.where(target_mask & natural_mask
                                   & np.isin(data["task_id"], list(validation_tasks)))[0]
                if min(len(fit_idx), len(cal_idx), len(val_idx)) == 0:
                    raise ValueError(f"fold {fold} has an empty partition")
                descriptor_idx = fit_idx[natural_mask[fit_idx]]
                descriptors = trainer.policy_descriptors(data, descriptor_idx)
                models = [
                    trainer.train_member(data, fit_idx,
                                         seed=seed_value + fold * 1009 + member * 7919,
                                         epochs=args.epochs, device=args.device,
                                         n_policies=0, policy_descriptor_dim=8,
                                         use_calibration_adapter=True,
                                         use_advantage_head=True,
                                         descriptors=descriptors)
                    for member in range(args.members)
                ]
                cal_pred = [trainer.predict(model, data, cal_idx, args.device, descriptors)
                            for model in models]
                cal_lcb = np.clip(np.stack([value["source"] for value in cal_pred]).mean(0)
                                  - args.lcb_z * np.stack([value["source"] for value in cal_pred]).std(0), 0, 1)
                cal_adv_members = np.stack([value["advantage"] for value in cal_pred])
                cal_adv = (cal_adv_members.mean(0)
                           - args.lcb_z * cal_adv_members.std(0))
                source_params, _ = trainer.select_controller(data, cal_idx, cal_lcb, cal_adv)

                # Calibration pool: target rows on train tasks (never validation).
                pool_idx = np.where(target_mask & natural_mask
                                    & np.isin(data["task_id"], list(train_tasks)))[0]
                for n_shots in shot_sizes:
                    # Each shot size starts from the same base shared core: clone
                    # the trained members so fine-tuning never leaks across sizes.
                    shot_models = [copy.deepcopy(model) for model in models]
                    if n_shots == 0:
                        # Pure zero-shot: source descriptor prior + source thresholds.
                        target_descriptor = descriptors[source_policies[0]]
                        params = source_params
                        calibration_group_count = 0
                        desc_for_target = dict(descriptors)
                        desc_for_target[target] = target_descriptor
                    else:
                        cal_rows, cal_groups = sample_cal_groups(
                            data, pool_idx, target, n_shots,
                            rng, forbid_tasks=set(validation_tasks))
                        target_descriptor = trainer._descriptor_for(data, cal_rows, target)
                        desc_for_target = dict(descriptors)
                        desc_for_target[target] = target_descriptor
                        for model in shot_models:
                            fine_tune_film(model, data, cal_rows, target,
                                           target_descriptor, device=args.device)
                        cal_pred = [trainer.predict(model, data, cal_rows, args.device, desc_for_target)
                                    for model in shot_models]
                        cal_lcb = np.clip(np.stack([value["source"] for value in cal_pred]).mean(0)
                                          - args.lcb_z * np.stack([value["source"] for value in cal_pred]).std(0), 0, 1)
                        cal_adv_members = np.stack([value["advantage"] for value in cal_pred])
                        cal_adv = (cal_adv_members.mean(0)
                                   - args.lcb_z * cal_adv_members.std(0))
                        params, _ = trainer.select_controller(data, cal_rows, cal_lcb, cal_adv)
                        calibration_group_count = len(cal_groups)

                    val_pred = [trainer.predict(model, data, val_idx, args.device, desc_for_target)
                                for model in shot_models]
                    source = np.stack([value["source"] for value in val_pred])
                    lcb = np.clip(source.mean(0) - args.lcb_z * source.std(0), 0, 1)
                    adv_members = np.stack([value["advantage"] for value in val_pred])
                    adv = adv_members.mean(0) - args.lcb_z * adv_members.std(0)
                    metrics = trainer.controller_early_window(
                        data, val_idx, lcb, adv, params["risk_thr"], params["adv_thr"])
                    points.append({
                        "seed": seed_value, "target_policy": target,
                        "fold": fold, "n_shots": n_shots,
                        "calibration_groups": int(calibration_group_count),
                        "episodes": metrics["episodes"],
                        "entered": metrics["entered"],
                        "successes": metrics["successes"],
                        "baseline_successes": metrics["baseline_successes"],
                        "false_continue": metrics["false_continue"],
                        "missed_rescue": metrics["missed_rescue"],
                        "rescue_opportunities": metrics["rescue_opportunities"],
                        "paired_harm": metrics["paired_harm"],
                        "teacher_steps": metrics["teacher_steps"],
                        "baseline_teacher_steps": metrics["baseline_teacher_steps"],
                        "success_gap": metrics["success_gap"],
                        "false_continue_rate": metrics["false_continue_rate"],
                        "absolute_paired_harm": metrics["absolute_paired_harm"],
                        "conditional_missed_rescue_rate": metrics["conditional_missed_rescue_rate"],
                        "savings": metrics["savings"],
                    })
                    del shot_models
                del models
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Aggregate the curve: fold-correct counts per (seed, target, n_shots),
    # then task-cluster intervals across the pooled points.
    def _summary(rows: list[dict]) -> dict:
        m = trainer.fold_correct_aggregate([{k: float(v) for k, v in row.items()
                                             if k in ("episodes", "entered", "successes",
                                                      "baseline_successes", "false_continue",
                                                      "missed_rescue", "rescue_opportunities", "paired_harm",
                                                      "teacher_steps", "baseline_teacher_steps")}
                                            for row in rows])
        return m

    curve: list[dict] = []
    for target in policies:
        for n_shots in shot_sizes:
            group_rows = [p for p in points if p["target_policy"] == target and p["n_shots"] == n_shots]
            m = _summary(group_rows)
            curve.append({
                "target_policy": target, "n_shots": n_shots,
                "n_fold_points": len(group_rows),
                "success_gap": m["success_gap"],
                "false_continue_rate": m["false_continue_rate"],
                "absolute_paired_harm": m["absolute_paired_harm"],
                "conditional_missed_rescue_rate": m["conditional_missed_rescue_rate"],
                "savings": m["savings"],
            })

    result = {
        "schema_version": "rase-r6c1-fewshot-curve/v1",
        "status": "complete",
        "scientific_scope": ("shared risk core + descriptor-conditioned calibration adaptation "
                             "curve; zero-shot is a challenge metric only"),
        "dataset": str(args.dataset.resolve()), "dataset_sha256": trainer.sha256(args.dataset),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": trainer.sha256(args.protocol),
        "seeds": [int(v) for v in args.seeds.split()], "shot_sizes": shot_sizes,
        "folds": args.folds, "fold_seed": args.fold_seed, "members": args.members,
        "epochs": args.epochs, "lcb_z": args.lcb_z,
        "curve": curve,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"curve": curve}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
