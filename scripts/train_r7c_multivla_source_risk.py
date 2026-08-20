#!/usr/bin/env python3
"""Task-held-out shared multi-VLA source-risk OOF with fixed conditioning modes."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.risk.light_risk_student import SourceRiskStudent  # noqa: E402
from rase.risk.multi_vla_descriptor import (  # noqa: E402
    DESCRIPTOR_VERSION,
    descriptors_by_policy,
)
from rase.risk.r7_source_protocol import (  # noqa: E402
    FOLD_SEED,
    N_FOLDS,
    calibration_tasks,
    task_folds,
)
from rase.risk.tiny_universal_state_encoder import TinyUniversalStateEncoder  # noqa: E402
from scripts.train_r7a_source_risk_probe import (  # noqa: E402
    fit_platt,
    metrics,
    normalize,
    sha256,
    task_bootstrap,
)

MODES = ("pooled", "shared_id", "shared_desc", "shared_calib")


def _condition_arrays(
    data: dict[str, np.ndarray], indices: np.ndarray, *, mode: str,
    descriptor_map: dict[str, np.ndarray] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    policy_index = None
    descriptor = None
    if mode == "shared_id":
        policy_index = torch.as_tensor(data["policy_index"][indices], dtype=torch.long)
    if mode in {"shared_desc", "shared_calib"}:
        if descriptor_map is None:
            raise ValueError("descriptor mode requires outer-fit descriptors")
        descriptor = torch.as_tensor(np.stack([
            descriptor_map[str(policy)] for policy in data["policy_id"][indices]
        ]), dtype=torch.float32)
    return policy_index, descriptor


def train_member(
    data: dict[str, np.ndarray], fit_idx: np.ndarray, *, mode: str,
    policy_count: int, seed: int, epochs: int, device: str,
) -> tuple[SourceRiskStudent, dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tasks = sorted(set(data["task_id"][fit_idx].tolist()))
    rng = random.Random(seed ^ 0xA17C0DE)
    sampled_tasks = [rng.choice(tasks) for _ in tasks]
    boot = np.concatenate([
        fit_idx[data["task_id"][fit_idx] == task] for task in sampled_tasks
    ])
    proprio, prop_mean, prop_std = normalize(data["proprio"], boot)
    action, action_mean, action_std = normalize(data["action_summary"], boot)
    descriptor_map = (
        descriptors_by_policy(data, fit_idx)
        if mode in {"shared_desc", "shared_calib"} else None
    )
    encoder = TinyUniversalStateEncoder(
        image_size=96, proprio_dim=8,
        text_embed_dim=data["language_hash"].shape[1],
        hidden_dim=128, output_dim=128, dropout=0.1, input_mode="image",
    )
    model = SourceRiskStudent(
        encoder, action_dim=data["action_summary"].shape[1], fused_dim=128,
        head_hidden=128, n_members=1, dropout=0.1,
        n_policies=policy_count if mode == "shared_id" else 0,
        policy_descriptor_dim=80 if descriptor_map is not None else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    image = torch.as_tensor(data["image"][boot].astype(np.float32) / 255.0, device=device)
    prop = torch.as_tensor(proprio[boot], device=device)
    act = torch.as_tensor(action[boot], device=device)
    text = torch.as_tensor(data["language_hash"][boot], device=device)
    policy_index, descriptor = _condition_arrays(
        data, boot, mode=mode, descriptor_map=descriptor_map,
    )
    if policy_index is not None:
        policy_index = policy_index.to(device)
    if descriptor is not None:
        descriptor = descriptor.to(device)
    target = torch.as_tensor(data["source_failure"][boot], device=device)
    positives = float(target.sum().item())
    pos_weight = torch.tensor(
        min(8.0, max(0.125, (len(target) - positives) / max(1.0, positives))),
        device=device,
    )
    for _ in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logit = model(
            image, prop, act, text, policy_index=policy_index,
            policy_descriptor=descriptor,
        )["source_failure_logit"][0]
        loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    return model, {
        "prop_mean": prop_mean, "prop_std": prop_std,
        "action_mean": action_mean, "action_std": action_std,
    }, descriptor_map


@torch.no_grad()
def predict(
    model: SourceRiskStudent, stats: dict[str, np.ndarray],
    descriptor_map: dict[str, np.ndarray] | None,
    data: dict[str, np.ndarray], idx: np.ndarray, *, mode: str, device: str,
) -> np.ndarray:
    image = torch.as_tensor(data["image"][idx].astype(np.float32) / 255.0, device=device)
    prop = torch.as_tensor(
        (data["proprio"][idx] - stats["prop_mean"]) / stats["prop_std"], device=device,
    )
    action = torch.as_tensor(
        (data["action_summary"][idx] - stats["action_mean"]) / stats["action_std"],
        device=device,
    )
    text = torch.as_tensor(data["language_hash"][idx], device=device)
    policy_index, descriptor = _condition_arrays(
        data, idx, mode=mode, descriptor_map=descriptor_map,
    )
    if policy_index is not None:
        policy_index = policy_index.to(device)
    if descriptor is not None:
        descriptor = descriptor.to(device)
    return model(
        image, prop, action, text, policy_index=policy_index,
        policy_descriptor=descriptor,
    )["source_failure_logit"][0].cpu().numpy()


def gate_for_subset(
    labels: np.ndarray, probability: np.ndarray, task_id: np.ndarray,
    suite: np.ndarray, *, seed: int, bootstrap_samples: int,
) -> tuple[dict, dict, dict, dict]:
    overall = metrics(labels, probability)
    by_suite = {
        name: metrics(labels[suite == name], probability[suite == name])
        for name in sorted(set(suite.tolist()))
    }
    bootstrap = task_bootstrap(
        labels, probability, task_id, seed=seed, samples=bootstrap_samples,
    )
    gate = {
        "auroc_at_least_0p75": overall["auroc"] >= 0.75,
        "bootstrap_auroc_lower_at_least_0p65": bootstrap["auroc"]["lower_95"] >= 0.65,
        "ap_above_prevalence_at_least_0p10": overall["ap_above_prevalence"] >= 0.10,
        "ece_at_most_0p10": overall["ece_10_equal_width"] <= 0.10,
        "all_suite_auroc_above_0p60": all(row["auroc"] > 0.60 for row in by_suite.values()),
    }
    return overall, by_suite, bootstrap, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold-seed", type=int, default=FOLD_SEED)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    report = json.loads(args.dataset_report.read_text())
    if report.get("status") != "frozen" or report.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("multi-VLA dataset/report is not frozen or hash bound")
    with np.load(args.dataset, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    policies = sorted(set(data["policy_id"].tolist()))
    if len(policies) < 2 or int(report.get("states_per_policy", -1)) != 192:
        raise ValueError("multi-VLA OOF requires >=2 aligned qualified policy cohorts")
    policy_count = len(policies)
    if sorted(set(data["policy_index"].tolist())) != list(range(policy_count)):
        raise ValueError("policy_index is not contiguous")

    folds = task_folds(data["task_id"], data["suite"], count=args.folds, seed=args.fold_seed)
    logits = np.full(len(data["source_failure"]), np.nan, dtype=np.float64)
    probabilities = np.full_like(logits, np.nan)
    fold_reports = []
    all_tasks = set(data["task_id"].tolist())
    for fold, validation_tasks in enumerate(folds):
        train_tasks = all_tasks - validation_tasks
        cal_tasks = calibration_tasks(
            train_tasks, data["task_id"], data["suite"], fold=fold, seed=args.fold_seed,
        )
        fit_tasks = train_tasks - cal_tasks
        fit_idx = np.flatnonzero(np.isin(data["task_id"], list(fit_tasks)))
        cal_idx = np.flatnonzero(np.isin(data["task_id"], list(cal_tasks)))
        val_idx = np.flatnonzero(np.isin(data["task_id"], list(validation_tasks)))
        for policy in policies:
            for partition, indices in (("fit", fit_idx), ("calibration", cal_idx)):
                selected = indices[data["policy_id"][indices] == policy]
                if len(np.unique(data["source_failure"][selected])) != 2:
                    raise ValueError(f"fold {fold} {policy} {partition} lacks both labels")
        member_cal, member_val = [], []
        for member in range(args.members):
            member_seed = args.seed + fold * 1009 + member * 7919
            model, stats, descriptor_map = train_member(
                data, fit_idx, mode=args.mode, policy_count=policy_count,
                seed=member_seed, epochs=args.epochs, device=args.device,
            )
            member_cal.append(predict(
                model, stats, descriptor_map, data, cal_idx,
                mode=args.mode, device=args.device,
            ))
            member_val.append(predict(
                model, stats, descriptor_map, data, val_idx,
                mode=args.mode, device=args.device,
            ))
        cal_logit = np.mean(member_cal, axis=0)
        val_logit = np.mean(member_val, axis=0)
        logits[val_idx] = val_logit
        calibration = {}
        if args.mode == "shared_calib":
            for policy in policies:
                cal_mask = data["policy_id"][cal_idx] == policy
                val_mask = data["policy_id"][val_idx] == policy
                temperature, bias = fit_platt(
                    cal_logit[cal_mask], data["source_failure"][cal_idx][cal_mask],
                )
                probabilities[val_idx[val_mask]] = 1.0 / (
                    1.0 + np.exp(-(val_logit[val_mask] / temperature + bias))
                )
                calibration[policy] = {"temperature": temperature, "bias": bias}
        else:
            temperature, bias = fit_platt(cal_logit, data["source_failure"][cal_idx])
            probabilities[val_idx] = 1.0 / (1.0 + np.exp(-(val_logit / temperature + bias)))
            calibration["pooled"] = {"temperature": temperature, "bias": bias}
        fold_reports.append({
            "fold": fold, "fit_tasks": sorted(fit_tasks),
            "calibration_tasks": sorted(cal_tasks),
            "validation_tasks": sorted(validation_tasks),
            "calibration": calibration,
        })
    if not np.isfinite(probabilities).all():
        raise AssertionError("multi-VLA OOF predictions are incomplete")

    labels = data["source_failure"].astype(np.float64)
    by_policy, by_policy_suite, policy_bootstrap, policy_gate = {}, {}, {}, {}
    for index, policy in enumerate(policies):
        selected = np.flatnonzero(data["policy_id"] == policy)
        overall, suites, bootstrap, gate = gate_for_subset(
            labels[selected], probabilities[selected], data["task_id"][selected],
            data["suite"][selected], seed=args.seed + index * 101,
            bootstrap_samples=args.bootstrap_samples,
        )
        by_policy[policy] = overall
        by_policy_suite[policy] = suites
        policy_bootstrap[policy] = bootstrap
        policy_gate[policy] = gate
    overall = metrics(labels, probabilities)
    status = "PASS" if all(all(gate.values()) for gate in policy_gate.values()) else "FAIL"
    result = {
        "schema_version": "rase-r7c-multivla-source-risk-oof/v1",
        "status": status, "mode": args.mode, "seed": args.seed,
        "fold_seed": args.fold_seed, "folds": args.folds,
        "members": args.members, "epochs": args.epochs,
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
        "policies": policies, "metrics": overall,
        "metrics_by_policy": by_policy,
        "metrics_by_policy_suite": by_policy_suite,
        "task_bootstrap_by_policy": policy_bootstrap,
        "gate_by_policy": policy_gate,
        "descriptor_version": DESCRIPTOR_VERSION if args.mode in {"shared_desc", "shared_calib"} else None,
        "descriptor_contract": "outer-fit tasks only; outcome-free" if args.mode in {"shared_desc", "shared_calib"} else None,
        "fold_reports": fold_reports,
        "forbidden_and_absent": [
            "OFT labels/actions/cost", "policy outcome rate as feature",
            "future frames", "validation labels in descriptor",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".predictions.npz"),
        state_key=data["state_key"], task_id=data["task_id"], suite=data["suite"],
        policy_id=data["policy_id"], source_failure=labels.astype(np.float32),
        raw_oof_logit=logits.astype(np.float32),
        calibrated_oof_probability=probabilities.astype(np.float32),
    )
    print(json.dumps({
        "status": status, "mode": args.mode,
        "metrics_by_policy": by_policy, "gate_by_policy": policy_gate,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
