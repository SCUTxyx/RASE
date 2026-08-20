#!/usr/bin/env python3
"""Train a root-balanced stepwise residual model from exact-root E3 demos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_e3_residual_dataset import canonical_instruction, language_hash  # noqa: E402
from scripts.train_e3_residual_ridge import group_folds, image_grid_features  # noqa: E402


def build_features(data: Mapping[str, np.ndarray], variant: str) -> np.ndarray:
    state = np.concatenate(
        [
            np.asarray(data["proprio"], dtype=np.float32),
            np.asarray(data["source_action"], dtype=np.float32),
            np.asarray(data["language_hash"], dtype=np.float32),
        ],
        axis=1,
    )
    if variant == "state":
        return state
    if variant == "state_vision":
        return np.concatenate(
            [state, image_grid_features(data["agentview"]), image_grid_features(data["wrist"])],
            axis=1,
        )
    raise ValueError(f"unknown feature variant: {variant}")


def fit_weighted_ridge(
    x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, alpha: float
) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weight = np.asarray(sample_weight, dtype=np.float64)
    weight = weight / weight.sum()
    x_mean = np.sum(x * weight[:, None], axis=0)
    x_var = np.sum((x - x_mean) ** 2 * weight[:, None], axis=0)
    x_scale = np.sqrt(x_var)
    x_scale[x_scale < 1e-6] = 1.0
    xs = (x - x_mean) / x_scale
    y_mean = np.sum(y * weight[:, None], axis=0)
    yc = y - y_mean
    weighted_x = xs * np.sqrt(weight[:, None])
    weighted_y = yc * np.sqrt(weight[:, None])
    gram = weighted_x.T @ weighted_x + float(alpha) * np.eye(xs.shape[1])
    coef = np.linalg.solve(gram, weighted_x.T @ weighted_y)
    return {
        "x_mean": x_mean.astype(np.float32),
        "x_scale": x_scale.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "weight": coef.astype(np.float32),
    }


def predict(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xs = (np.asarray(x, dtype=np.float32) - model["x_mean"]) / model["x_scale"]
    return xs @ model["weight"] + model["y_mean"]


def weighted_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    correction: np.ndarray,
) -> dict[str, float]:
    error = prediction - target
    output: dict[str, float] = {}
    for label, mask in (("overall", np.ones(len(target), dtype=bool)), ("correction", correction), ("identity", ~correction)):
        if not mask.any():
            continue
        local_weight = weights[mask] / weights[mask].sum()
        output[f"{label}_mse"] = float(np.sum(np.mean(error[mask] ** 2, axis=1) * local_weight))
        output[f"{label}_mae"] = float(np.sum(np.mean(np.abs(error[mask]), axis=1) * local_weight))
        output[f"{label}_predicted_delta_abs_mean"] = float(
            np.sum(np.mean(np.abs(prediction[mask]), axis=1) * local_weight)
        )
    return output


def atomic_save(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument(
        "--root-dataset",
        type=Path,
        help="optional Phase0G root dataset supplying active-suffix identity retention",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-steps-per-root", type=int, default=128)
    args = parser.parse_args()
    summary = json.loads((args.demo_dir.resolve() / "summary.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError("step-demo collection is incomplete")
    demo_checks = dict(summary.get("checks") or {})
    if not (
        demo_checks.get("correction_roots_at_least_20")
        and demo_checks.get("correction_steps_at_least_1000")
    ):
        raise ValueError("step-demo correction coverage gate must PASS")

    pieces: dict[str, list[np.ndarray]] = {
        key: [] for key in ("proprio", "source_action", "delta_target", "agentview", "wrist")
    }
    root_ids = []
    group_ids = []
    modes = []
    for row in summary.get("records") or []:
        if row.get("status") != "complete":
            continue
        with np.load(row["artifact"], allow_pickle=False) as archive:
            length = len(archive["source_action"])
            if length > args.max_steps_per_root:
                indices = np.linspace(0, length - 1, args.max_steps_per_root).round().astype(int)
            else:
                indices = np.arange(length)
            for key in pieces:
                pieces[key].append(archive[key][indices])
        instruction = str(row["instruction"])
        root_ids.extend([str(row["state_key"])] * len(indices))
        group = f"{row['suite']}|{canonical_instruction(instruction)}"
        group_ids.extend([group] * len(indices))
        modes.extend([str(row["mode"])] * len(indices))

    root_identity_examples = 0
    root_identity_roots = 0
    if args.root_dataset:
        with np.load(args.root_dataset.resolve(), allow_pickle=False) as archive:
            identity = np.asarray(archive["source_success"], dtype=bool)
            root_identity_roots = int(identity.sum())
            for index in np.flatnonzero(identity):
                actions = np.asarray(archive["source_action"][index], dtype=np.float32)
                count = len(actions)
                pieces["proprio"].append(
                    np.repeat(archive["proprio"][index][None, ...], count, axis=0)
                )
                pieces["source_action"].append(actions)
                pieces["delta_target"].append(np.zeros_like(actions))
                pieces["agentview"].append(
                    np.repeat(archive["agentview"][index][None, ...], count, axis=0)
                )
                pieces["wrist"].append(
                    np.repeat(archive["wrist"][index][None, ...], count, axis=0)
                )
                root = str(archive["state_key"][index])
                group = str(archive["group_id"][index])
                root_ids.extend([root] * count)
                group_ids.extend([group] * count)
                modes.extend(["identity_active_suffix_root"] * count)
                root_identity_examples += count

    arrays = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
    arrays["language_hash"] = np.stack(
        [language_hash(group.split("|", 1)[1]) for group in group_ids]
    ).astype(np.float32)
    roots = np.asarray(root_ids)
    groups = np.asarray(group_ids)
    correction = np.asarray(modes) == "successful_recovery_replay"
    y = arrays["delta_target"].astype(np.float32)

    # Equal total weight for correction/identity, then equal roots within class,
    # then equal steps within root. This prevents long recovery traces from
    # dominating either the correction or retention objective.
    sample_weight = np.zeros(len(y), dtype=np.float64)
    for class_mask in (correction, ~correction):
        class_roots = sorted(set(roots[class_mask].tolist()))
        for root in class_roots:
            mask = class_mask & (roots == root)
            sample_weight[mask] = 0.5 / len(class_roots) / int(mask.sum())
    sample_weight /= sample_weight.sum()

    folds = group_folds(groups.tolist(), args.folds)
    candidates = []
    for variant in ("state", "state_vision"):
        x = build_features(arrays, variant)
        for alpha in (0.001, 0.01, 0.1, 1.0, 10.0):
            oof = np.zeros_like(y)
            for validation in folds:
                training = ~validation
                model = fit_weighted_ridge(x[training], y[training], sample_weight[training], alpha)
                oof[validation] = predict(model, x[validation])
            candidates.append(
                {
                    "variant": variant,
                    "alpha": alpha,
                    "input_dim": int(x.shape[1]),
                    "metrics": weighted_metrics(y, oof, sample_weight, correction),
                    "oof": oof,
                }
            )
    selected = min(candidates, key=lambda row: (row["metrics"]["overall_mse"], row["input_dim"], row["alpha"]))
    x = build_features(arrays, selected["variant"])
    model = fit_weighted_ridge(x, y, sample_weight, selected["alpha"])
    zero = weighted_metrics(y, np.zeros_like(y), sample_weight, correction)
    checks = {
        "group_cv_correction_mse_improves_zero_by_10pct": selected["metrics"]["correction_mse"] <= 0.90 * zero["correction_mse"],
        "group_cv_identity_mean_abs_delta_at_most_0_10": selected["metrics"]["identity_predicted_delta_abs_mean"] <= 0.10,
        "model_finite": all(np.isfinite(value).all() for value in model.values()),
    }
    demo_sha = hashlib.sha256((args.demo_dir.resolve() / "summary.json").read_bytes()).hexdigest()
    atomic_save(
        args.model_output.resolve(), **model,
        feature_variant=np.asarray(selected["variant"]),
        alpha=np.asarray(selected["alpha"], dtype=np.float32),
        image_size=np.asarray(arrays["agentview"].shape[1], dtype=np.int64),
        language_dim=np.asarray(arrays["language_hash"].shape[1], dtype=np.int64),
        demo_summary_sha256=np.asarray(demo_sha),
    )
    report = {
        "schema_version": "rase-e3-step-residual-training/v1",
        "status": "complete",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "scientific_scope": "development_group_cv_stepwise_model_selection",
        "demo_dir": str(args.demo_dir.resolve()),
        "demo_summary_sha256": demo_sha,
        "n_samples": len(y),
        "n_roots": len(set(roots.tolist())),
        "n_groups": len(set(groups.tolist())),
        "n_correction_samples": int(correction.sum()),
        "n_identity_samples": int((~correction).sum()),
        "root_identity_dataset": str(args.root_dataset.resolve()) if args.root_dataset else None,
        "root_identity_roots": root_identity_roots,
        "root_identity_examples": root_identity_examples,
        "selected": {
            "variant": selected["variant"], "alpha": selected["alpha"],
            "input_dim": selected["input_dim"],
            "parameter_count": int(selected["input_dim"] * y.shape[1] + y.shape[1]),
            "group_cv_metrics": selected["metrics"],
            "final_fit_metrics": weighted_metrics(y, predict(model, x), sample_weight, correction),
        },
        "zero_residual_baseline": zero,
        "checks": checks,
        "candidates": [
            {"variant": row["variant"], "alpha": row["alpha"], "input_dim": row["input_dim"], "metrics": row["metrics"]}
            for row in candidates
        ],
        "claim_boundary": "PASS authorizes outcome-independent closed-loop rollout only; it is not task-success evidence.",
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": report["decision"], **report["selected"]}, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
