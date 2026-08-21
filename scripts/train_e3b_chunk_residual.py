#!/usr/bin/env python3
"""Train a root-balanced, task-group-isolated E3-B residual chunk ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rase.recovery.e3b_chunk_residual import (  # noqa: E402
    HORIZON,
    make_network,
    state_features,
    vision_features,
)


SUITES = ("spatial", "object", "goal", "long")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def teacher_outcomes(root: Path) -> dict[str, bool]:
    result = {}
    for suite in SUITES:
        payload = load(root / "a" / suite / "summary.json")
        for row in payload["per_state"]:
            result[str(row["state_key"])] = bool(row["direct_oft_success"])
    return result


def task_split(task: str, suite: str) -> str:
    digest = hashlib.sha256(f"e3b-train-v1|{suite}|{task}".encode()).hexdigest()
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, action="append", required=True)
    parser.add_argument("--teacher-outcomes", type=Path, action="append", required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    args = parser.parse_args()
    if len(args.collection_dir) != len(args.teacher_outcomes):
        raise ValueError("collection and teacher outcome directories must align")

    partition = load(args.partition.resolve())
    metadata = {str(row["state_key"]): row for row in partition["records"]}
    teacher = {}
    for path in args.teacher_outcomes:
        teacher.update(teacher_outcomes(path.resolve()))
    source_success = {}
    sample_paths: dict[str, list[Path]] = {}
    for collection in args.collection_dir:
        root = collection.resolve()
        for suite in SUITES:
            summary = load(root / suite / "summary.json")
            for row in summary["per_state_arm"]:
                key = str(row["state_key"])
                if row["arm"] == "source_h8":
                    if key in source_success and source_success[key] != bool(row["success"]):
                        raise ValueError(f"source outcome drift across DAgger rounds: {key}")
                    source_success[key] = bool(row["success"])
                elif row["arm"] == "persistent_h8":
                    sample_paths.setdefault(key, []).append(Path(row["sample_artifact"]))
    keys = sorted(set(metadata) & set(source_success) & set(teacher) & set(sample_paths))
    if not keys:
        raise ValueError("no aligned training roots")

    tasks_by_suite: dict[str, set[str]] = {}
    for key in keys:
        row = metadata[key]
        tasks_by_suite.setdefault(str(row["suite"]), set()).add(str(row["logical_task_id"]))
    dev_tasks = set()
    for suite, tasks in tasks_by_suite.items():
        ordered = sorted(tasks, key=lambda task: task_split(task, suite))
        if len(ordered) >= 3:
            dev_tasks.add(ordered[0])

    states = []
    visions = []
    targets = []
    gates = []
    roots = []
    splits = []
    strata = {}
    excluded = Counter()
    for key in keys:
        if source_success[key]:
            stratum = "identity"
        elif teacher[key]:
            stratum = "correction"
        else:
            excluded["capability_gap"] += 1
            continue
        strata[key] = stratum
        for sample_path in sample_paths[key]:
            with np.load(sample_path, allow_pickle=False) as archive:
                n = len(archive["source_chunk"])
                for index in range(n):
                    states.append(
                        state_features(
                            archive["proprio"][index], archive["source_chunk"][index],
                            archive["history"][index], archive["language_hash"][index],
                        )
                    )
                    visions.append(
                        vision_features(archive["agentview"][index], archive["wrist"][index])
                    )
                    target = (
                        np.zeros((HORIZON, 7), dtype=np.float32)
                        if stratum == "identity"
                        else np.asarray(archive["teacher_chunk"][index] - archive["source_chunk"][index], dtype=np.float32)
                    )
                    targets.append(np.clip(target, -2.0, 2.0).reshape(-1))
                    gates.append(float(stratum == "correction"))
                    roots.append(key)
                    splits.append("dev" if metadata[key]["logical_task_id"] in dev_tasks else "train")

    x_state = np.asarray(states, dtype=np.float32)
    x_vision = np.asarray(visions, dtype=np.float32)
    y_delta = np.asarray(targets, dtype=np.float32)
    y_gate = np.asarray(gates, dtype=np.float32)
    splits_array = np.asarray(splits)
    train_mask = splits_array == "train"
    dev_mask = splits_array == "dev"
    if not train_mask.any() or not dev_mask.any():
        raise ValueError("task-group split produced an empty fold")
    state_mean = x_state[train_mask].mean(axis=0)
    state_std = x_state[train_mask].std(axis=0)
    state_std[state_std < 1e-5] = 1.0
    x_state = (x_state - state_mean) / state_std

    roots_array = np.asarray(roots)
    weights = np.zeros(len(roots), dtype=np.float32)
    for split in ("train", "dev"):
        mask = splits_array == split
        for label in (0.0, 1.0):
            class_mask = mask & (y_gate == label)
            class_roots = sorted(set(roots_array[class_mask]))
            if not class_roots:
                continue
            for key in class_roots:
                root_mask = class_mask & (roots_array == key)
                weights[root_mask] = 0.5 / len(class_roots) / root_mask.sum()
        total = weights[mask].sum()
        if total > 0:
            weights[mask] /= total

    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensors = {
        "state": torch.from_numpy(x_state).to(device),
        "vision": torch.from_numpy(x_vision).to(device),
        "delta": torch.from_numpy(y_delta).to(device),
        "gate": torch.from_numpy(y_gate).to(device),
        "weight": torch.from_numpy(weights).to(device),
        "train": torch.from_numpy(train_mask).to(device),
        "dev": torch.from_numpy(dev_mask).to(device),
    }
    state_dicts = []
    seed_metrics = []
    dev_gate_predictions = []
    dev_delta_predictions = []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = make_network().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best = None
        best_loss = float("inf")
        patience = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            delta, gate = model(tensors["state"][tensors["train"]], tensors["vision"][tensors["train"]])
            target_delta = tensors["delta"][tensors["train"]]
            target_gate = tensors["gate"][tensors["train"]]
            weight = tensors["weight"][tensors["train"]]
            delta_loss = F.smooth_l1_loss(delta, target_delta, reduction="none").mean(dim=1)
            gate_loss = F.binary_cross_entropy_with_logits(gate, target_gate, reduction="none")
            loss = ((delta_loss + 0.25 * gate_loss) * weight).sum()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                dev_delta, dev_gate = model(tensors["state"][tensors["dev"]], tensors["vision"][tensors["dev"]])
                dev_weight = tensors["weight"][tensors["dev"]]
                dev_loss = (
                    F.smooth_l1_loss(dev_delta, tensors["delta"][tensors["dev"]], reduction="none").mean(dim=1)
                    + 0.25 * F.binary_cross_entropy_with_logits(dev_gate, tensors["gate"][tensors["dev"]], reduction="none")
                )
                score = float((dev_loss * dev_weight).sum().cpu())
            if score < best_loss - 1e-6:
                best_loss = score
                best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 20:
                break
        assert best is not None
        model.load_state_dict(best)
        model.eval()
        with torch.no_grad():
            predicted_delta, predicted_gate = model(tensors["state"][tensors["dev"]], tensors["vision"][tensors["dev"]])
        dev_delta_predictions.append(predicted_delta.cpu().numpy())
        dev_gate_predictions.append(torch.sigmoid(predicted_gate).cpu().numpy())
        state_dicts.append(best)
        seed_metrics.append({"seed": seed, "best_dev_loss": best_loss, "epochs": epoch + 1})

    delta_prediction = np.mean(dev_delta_predictions, axis=0)
    gate_prediction = np.mean(dev_gate_predictions, axis=0)
    dev_target = y_delta[dev_mask]
    dev_gate_target = y_gate[dev_mask]
    delta_mse = float(np.mean((delta_prediction - dev_target) ** 2))
    identity_mse = float(np.mean(dev_target**2))
    candidates = np.linspace(0.25, 0.75, 21)
    threshold_rows = []
    for threshold in candidates:
        prediction = gate_prediction >= threshold
        positive = dev_gate_target == 1
        negative = ~positive
        tpr = float(prediction[positive].mean()) if positive.any() else 0.0
        fpr = float(prediction[negative].mean()) if negative.any() else 0.0
        threshold_rows.append((threshold, tpr, fpr, 0.5 * (tpr + 1.0 - fpr)))
    feasible = [row for row in threshold_rows if row[2] <= 0.20]
    selected_threshold = max(feasible or threshold_rows, key=lambda row: (row[3], row[1], -row[0]))

    payload = {
        "schema_version": "rase-e3b-chunk-residual/v1",
        "state_dicts": state_dicts,
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "gate_threshold": float(selected_threshold[0]),
        "horizon": HORIZON,
        "seeds": args.seeds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    report = {
        "schema_version": "rase-e3b-chunk-residual-training/v1",
        "status": "complete",
        "n_roots_aligned": len(keys),
        "n_roots_used": len(strata),
        "root_strata": dict(Counter(strata.values())),
        "excluded": dict(excluded),
        "n_samples": len(roots),
        "n_train_samples": int(train_mask.sum()),
        "n_dev_samples": int(dev_mask.sum()),
        "dev_tasks": sorted(dev_tasks),
        "dev_delta_mse": delta_mse,
        "dev_identity_baseline_mse": identity_mse,
        "dev_mse_improvement": 0.0 if identity_mse == 0 else 1.0 - delta_mse / identity_mse,
        "gate_threshold": float(selected_threshold[0]),
        "dev_gate_tpr": selected_threshold[1],
        "dev_gate_fpr": selected_threshold[2],
        "dev_gate_balanced_accuracy": selected_threshold[3],
        "seed_metrics": seed_metrics,
        "checkpoint": str(args.output.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("n_roots_used", "root_strata", "dev_mse_improvement", "dev_gate_tpr", "dev_gate_fpr")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
