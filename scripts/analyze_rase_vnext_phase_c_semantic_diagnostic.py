#!/usr/bin/env python3
"""Low-cost task/state/action alignment diagnostic on the Phase-C pilot.

The analysis is exploratory: it reuses development labels, excludes practical
ties, and cannot override Phase-B or unlock pooled/multi-VLA claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.libero import LIBERO_ACTION_SEMANTICS, LIBERO_MOTION_SEMANTIC_MAP
from rase.vnext.phase_c_pilot import (
    bootstrap_task_difference,
    raw_action_feature_vector,
    stable_seed,
    task_folds,
    trace_feature_vector,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def signed_text_hash(text: str, *, dimension: int = 64) -> np.ndarray:
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    result = np.zeros(dimension, dtype=np.float64)
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        result[index] += sign
    norm = np.linalg.norm(result)
    return result / norm if norm else result


def image_statistics(image: np.ndarray, *, blocks: int = 4) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64) / 255.0
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {value.shape}")
    pooled = []
    for y_indices in np.array_split(np.arange(value.shape[0]), blocks):
        for x_indices in np.array_split(np.arange(value.shape[1]), blocks):
            pooled.extend(value[np.ix_(y_indices, x_indices)].mean(axis=(0, 1)))
    dx = np.abs(np.diff(value, axis=1)).mean(axis=(0, 1))
    dy = np.abs(np.diff(value, axis=0)).mean(axis=(0, 1))
    return np.concatenate((pooled, value.mean(axis=(0, 1)), value.std(axis=(0, 1)), dx, dy))


def load_instruction(manifest: dict[str, Any], root_id: str) -> str:
    root = next(row for row in manifest["roots"] if str(row["root_id"]) == root_id)
    reference = str(root["restore_state_ref"])
    if not reference.startswith("state_pool:") or "#" not in reference:
        raise ValueError(f"unsupported restore reference {reference}")
    pool_text, state_key = reference[len("state_pool:"):].split("#", 1)
    pool = Path(pool_text)
    pool_manifest = json.loads((pool / "manifest.json").read_text())
    relative = pool_manifest["states"][state_key]["path"]
    metadata = json.loads((pool / relative / "meta.json").read_text())
    instruction = str(metadata.get("instruction", ""))
    if not instruction:
        raise ValueError(f"missing instruction for {root_id}")
    return instruction


def ridge_oof_pair(
    train_features: np.ndarray, swapped_features: np.ndarray, labels: np.ndarray,
    tasks: list[str], folds_by_task: dict[str, int], *, alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    normal = np.full(len(labels), np.nan, dtype=np.float64)
    swapped = np.full(len(labels), np.nan, dtype=np.float64)
    for fold in sorted(set(folds_by_task.values())):
        test = np.array([folds_by_task[task] == fold for task in tasks])
        train = ~test
        mean = train_features[train].mean(axis=0)
        scale = train_features[train].std(axis=0)
        scale[scale < 1e-8] = 1.0
        x_train = (train_features[train] - mean) / scale
        x_test = (train_features[test] - mean) / scale
        x_swap = (swapped_features[test] - mean) / scale
        design = np.column_stack((np.ones(train.sum()), x_train))
        penalty = np.eye(design.shape[1]) * alpha
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ labels[train])
        normal[test] = np.column_stack((np.ones(test.sum()), x_test)) @ beta
        swapped[test] = np.column_stack((np.ones(test.sum()), x_swap)) @ beta
    if not np.isfinite(normal).all() or not np.isfinite(swapped).all():
        raise RuntimeError("incomplete OOF predictions")
    return normal, swapped


def build_dataset(feature_dir: Path, *, tie_margin: float) -> dict[str, Any]:
    contract = json.loads((feature_dir / "EXPORT_CONTRACT.json").read_text())
    manifest = json.loads(Path(contract["source_manifest"]).read_text())
    point_values: set[str] = set()
    raw_rows: list[dict[str, Any]] = []
    instruction_cache: dict[str, str] = {}
    for meta_path in sorted((feature_dir / "groups").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("status") != "COMPLETE":
            continue
        operators = list(meta["operator_order"])
        if operators != ["continue.source", "requery.source"]:
            raise ValueError(f"unexpected operator order {operators}")
        utility = [float(meta["outcomes"][operator]["utility"]) for operator in operators]
        difference = utility[0] - utility[1]
        if abs(difference) <= tie_margin:
            continue
        root_id = str(meta["root_id"])
        if root_id not in instruction_cache:
            instruction_cache[root_id] = load_instruction(manifest, root_id)
        with np.load(meta["features_path"], allow_pickle=False) as arrays:
            actions = arrays["actions"].copy()
            masks = arrays["action_step_mask"].copy()
            proprio = arrays["proprio"].astype(np.float64) * arrays["proprio_mask"]
            visual = np.concatenate((
                image_statistics(arrays["image_agentview"]),
                image_statistics(arrays["image_wrist"]),
            ))
        raw = [raw_action_feature_vector(actions[i], masks[i]) for i in range(2)]
        trace = [trace_feature_vector(
            actions[i], masks[i], semantics=LIBERO_ACTION_SEMANTICS,
            policy_id=str(meta["policy_id"]), semantic_map=LIBERO_MOTION_SEMANTIC_MAP,
        ) for i in range(2)]
        point = str(meta["decision_point_id"])
        point_values.add(point)
        raw_rows.append({
            "task": str(meta["task_id"]), "suite": str(meta["suite"]),
            "point": point, "label": 1.0 if difference > 0 else -1.0,
            "text": signed_text_hash(instruction_cache[root_id]),
            "proprio": proprio, "visual": visual,
            "raw_delta": raw[0] - raw[1], "trace_delta": trace[0] - trace[1],
        })
    point_vocab = tuple(sorted(point_values))
    contexts, text_only, state_only, raw_delta, trace_delta = [], [], [], [], []
    tasks, labels, suites = [], [], {}
    for row in raw_rows:
        point = np.zeros(len(point_vocab), dtype=np.float64)
        point[point_vocab.index(row["point"])] = 1.0
        state = np.concatenate((point, row["proprio"], row["visual"]))
        context = np.concatenate((row["text"], state))
        text_only.append(row["text"])
        state_only.append(state)
        contexts.append(context)
        raw_delta.append(row["raw_delta"])
        trace_delta.append(row["trace_delta"])
        tasks.append(row["task"]); labels.append(row["label"])
        suites[row["task"]] = row["suite"]
    context = np.stack(contexts); text = np.stack(text_only); state = np.stack(state_only)
    raw = np.stack(raw_delta); trace = np.stack(trace_delta)
    labels_array = np.asarray(labels, dtype=np.float64)
    order = sorted(range(len(tasks)), key=lambda i: (stable_seed("semantic-shuffle", tasks[i], i), i))
    source = order[-1:] + order[:-1]
    shuffled_trace = np.empty_like(trace)
    for target, origin in zip(order, source): shuffled_trace[target] = trace[origin]
    rng = np.random.default_rng(20270817)
    cproj = rng.normal(size=(context.shape[1], 32)) / np.sqrt(context.shape[1])
    tproj = rng.normal(size=(trace.shape[1], 32)) / np.sqrt(trace.shape[1])

    def semantic_features(delta: np.ndarray) -> np.ndarray:
        interaction = (context @ cproj) * (delta @ tproj)
        return np.column_stack((context, delta, interaction))

    features = {
        "S0_constant": np.ones((len(tasks), 1)),
        "S1_task_text": text,
        "S2_state_visual": state,
        "S3_task_state_visual": context,
        "A1_raw_delta": raw,
        "A2_trace_delta": trace,
        "A3_task_state_trace": np.column_stack((context, trace)),
        "A4_task_x_state_x_trace": semantic_features(trace),
        "A4_trace_shuffled": semantic_features(shuffled_trace),
    }
    swapped = {
        "S0_constant": features["S0_constant"],
        "S1_task_text": features["S1_task_text"],
        "S2_state_visual": features["S2_state_visual"],
        "S3_task_state_visual": features["S3_task_state_visual"],
        "A1_raw_delta": -raw,
        "A2_trace_delta": -trace,
        "A3_task_state_trace": np.column_stack((context, -trace)),
        "A4_task_x_state_x_trace": semantic_features(-trace),
        "A4_trace_shuffled": semantic_features(-shuffled_trace),
    }
    return {"features": features, "swapped": swapped, "labels": labels_array,
            "tasks": tasks, "suites": suites}


def analyze(data: dict[str, Any], *, replicates: int) -> dict[str, Any]:
    labels = data["labels"]; tasks = data["tasks"]; suites = data["suites"]
    results: dict[str, list[dict[str, float]]] = {name: [] for name in data["features"]}
    per_task: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in data["features"]
    }
    for seed in range(5):
        folds = task_folds(tasks, suites, seed=seed, folds=5)
        for name, matrix in data["features"].items():
            prediction, swap_prediction = ridge_oof_pair(
                matrix, data["swapped"][name], labels, tasks, folds,
            )
            correct = (prediction * labels > 0).astype(np.float64)
            swapped_correct = (swap_prediction * (-labels) > 0).astype(np.float64)
            flip = (np.sign(prediction) == -np.sign(swap_prediction)).astype(np.float64)
            results[name].append({
                "seed": seed, "pairwise_accuracy": float(correct.mean()),
                "swapped_pairwise_accuracy": float(swapped_correct.mean()),
                "prediction_flip_rate": float(flip.mean()),
            })
            for task, value in zip(tasks, correct): per_task[name][task].append(float(value))
    summary = {}
    for name, rows in results.items():
        summary[name] = {
            "pairwise_accuracy_mean": float(np.mean([r["pairwise_accuracy"] for r in rows])),
            "pairwise_accuracy_std": float(np.std([r["pairwise_accuracy"] for r in rows])),
            "swapped_pairwise_accuracy_mean": float(np.mean([r["swapped_pairwise_accuracy"] for r in rows])),
            "prediction_flip_rate_mean": float(np.mean([r["prediction_flip_rate"] for r in rows])),
            "seeds": rows,
        }
    primary = "A4_task_x_state_x_trace"
    controls = ("S0_constant", "S1_task_text", "S2_state_visual", "S3_task_state_visual")
    best_control = max(controls, key=lambda name: summary[name]["pairwise_accuracy_mean"])
    gain, interval = bootstrap_task_difference(
        per_task[primary], per_task[best_control], replicates=replicates, seed=202710,
    )
    directional = sum(
        left["pairwise_accuracy"] > right["pairwise_accuracy"]
        for left, right in zip(results[primary], results[best_control])
    )
    return {
        "schema_version": "rase-vnext-phase-c-semantic-diagnostic/v1",
        "status": "EXPLORATORY_NOT_A_GATE",
        "scientific_scope": "A_PARTIAL_PI0FAST_DEVELOPMENT_ONLY",
        "informative_groups": len(labels), "informative_tasks": len(set(tasks)),
        "class_counts": {"continue.source": int((labels > 0).sum()),
                         "requery.source": int((labels < 0).sum())},
        "primary": {
            "model": primary, "best_no_action_control": best_control,
            "task_bootstrap_gain": gain, "task_bootstrap_95_ci": interval,
            "directional_seeds": directional,
            "trace_minus_shuffled_gain": (
                summary[primary]["pairwise_accuracy_mean"]
                - summary["A4_trace_shuffled"]["pairwise_accuracy_mean"]
            ),
            "no_action_gap": (
                summary[primary]["pairwise_accuracy_mean"]
                - summary[best_control]["pairwise_accuracy_mean"]
            ),
        },
        "models": summary,
        "limitations": [
            "development labels reused; not sealed", "only 13 informative tasks expected",
            "visual baseline uses deterministic low-cost image statistics, not a trained VLM",
            "cannot override B_FAIL or unlock pooled multi-VLA claims",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie-margin", type=float, default=0.01)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    data = build_dataset(args.feature_dir.resolve(), tie_margin=args.tie_margin)
    result = analyze(data, replicates=args.bootstrap_replicates)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
