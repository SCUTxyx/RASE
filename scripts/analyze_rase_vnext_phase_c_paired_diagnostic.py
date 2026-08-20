#!/usr/bin/env python3
"""Exploratory direct same-root ranking diagnostic for the Phase C pilot.

This does not replace the frozen primary analysis.  It asks whether fitting the
paired practical winner directly repairs the signal before any new rollout or
representation scaling is authorized.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.phase_c_pilot import (
    bootstrap_task_difference,
    ridge_oof_predictions,
    stable_seed,
    task_folds,
)
from analyze_rase_vnext_phase_c_pilot import atomic_json, load_dataset


def _shuffled_trace(records: list[dict[str, object]]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    by_operator: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_operator[str(row["operator"])].append(index)
    for operator, indices in sorted(by_operator.items()):
        ordered = sorted(
            indices,
            key=lambda index: (
                stable_seed("paired-shuffle", operator, records[index]["group_id"]),
                index,
            ),
        )
        source = ordered[-1:] + ordered[:-1]
        for target_index, source_index in zip(ordered, source):
            result[target_index] = np.asarray(records[source_index]["trace"], dtype=np.float64)
    return result


def build_pairs(
    records: list[dict[str, object]], *, tie_margin: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], dict[str, str]]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_group[str(row["group_id"])].append(index)
    shuffled = _shuffled_trace(records)
    point_values = tuple(sorted({str(row["decision_point"]) for row in records}))

    contexts: list[np.ndarray] = []
    raw_deltas: list[np.ndarray] = []
    trace_deltas: list[np.ndarray] = []
    shuffled_deltas: list[np.ndarray] = []
    labels: list[float] = []
    tasks: list[str] = []
    suites: dict[str, str] = {}
    for group in sorted(by_group):
        indices = by_group[group]
        candidates = {str(records[index]["operator"]): index for index in indices}
        if set(candidates) != {"continue.source", "requery.source"}:
            continue
        left = candidates["continue.source"]
        right = candidates["requery.source"]
        difference = float(records[left]["utility"]) - float(records[right]["utility"])
        if abs(difference) <= tie_margin:
            continue
        row = records[left]
        point = np.zeros(len(point_values), dtype=np.float64)
        point[point_values.index(str(row["decision_point"]))] = 1.0
        proprio = np.asarray(row["proprio"], dtype=np.float64)
        contexts.append(np.concatenate((point, proprio)))
        raw_deltas.append(
            np.asarray(records[left]["raw"], dtype=np.float64)
            - np.asarray(records[right]["raw"], dtype=np.float64)
        )
        trace_deltas.append(
            np.asarray(records[left]["trace"], dtype=np.float64)
            - np.asarray(records[right]["trace"], dtype=np.float64)
        )
        shuffled_deltas.append(shuffled[left] - shuffled[right])
        labels.append(1.0 if difference > 0 else -1.0)
        task = str(row["task"])
        tasks.append(task)
        suites[task] = str(row["suite"])

    context = np.stack(contexts)
    raw = np.stack(raw_deltas)
    trace = np.stack(trace_deltas)
    trace_shuffled = np.stack(shuffled_deltas)
    rng = np.random.default_rng(20270815)
    projection = rng.normal(size=(trace.shape[1], 16)) / np.sqrt(trace.shape[1])

    def interacted(delta: np.ndarray) -> np.ndarray:
        projected = delta @ projection
        interaction = np.einsum("ni,nj->nij", context, projected).reshape(len(context), -1)
        return np.column_stack((context, delta, interaction))

    features = {
        "P0_constant": np.ones((len(context), 1), dtype=np.float64),
        "P1_state_context": context,
        "P2_raw_delta": raw,
        "P3_trace_delta": trace,
        "P4_state_trace": np.column_stack((context, trace)),
        "P5_state_x_trace": interacted(trace),
        "P5_state_x_trace_shuffled": interacted(trace_shuffled),
    }
    return features, np.asarray(labels), tasks, suites


def analyze(dataset: dict[str, object], *, tie_margin: float, replicates: int) -> dict[str, object]:
    features, labels, tasks, suites = build_pairs(dataset["records"], tie_margin=tie_margin)
    seed_rows: dict[str, list[dict[str, float]]] = {name: [] for name in features}
    per_task: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in features
    }
    for seed in range(5):
        folds = task_folds(tasks, suites, seed=seed, folds=5)
        for name, matrix in features.items():
            prediction = ridge_oof_predictions(matrix, labels, tasks, folds, alpha=1.0)
            correct = (prediction * labels > 0).astype(np.float64)
            seed_rows[name].append({
                "seed": seed,
                "pairwise_accuracy": float(correct.mean()),
            })
            for task, value in zip(tasks, correct):
                per_task[name][task].append(float(value))

    summary: dict[str, object] = {}
    for name, rows in seed_rows.items():
        values = [row["pairwise_accuracy"] for row in rows]
        summary[name] = {
            "pairwise_accuracy_mean": float(np.mean(values)),
            "pairwise_accuracy_std": float(np.std(values)),
            "seeds": rows,
        }
    primary = "P5_state_x_trace"
    controls = ("P0_constant", "P1_state_context", "P2_raw_delta", "P4_state_trace")
    best_control = max(
        controls, key=lambda name: float(summary[name]["pairwise_accuracy_mean"]),
    )
    gain, interval = bootstrap_task_difference(
        per_task[primary], per_task[best_control], replicates=replicates, seed=202709,
    )
    shuffled_gain = (
        float(summary[primary]["pairwise_accuracy_mean"])
        - float(summary["P5_state_x_trace_shuffled"]["pairwise_accuracy_mean"])
    )
    directional = sum(
        left["pairwise_accuracy"] > right["pairwise_accuracy"]
        for left, right in zip(seed_rows[primary], seed_rows[best_control])
    )
    return {
        "schema_version": "rase-vnext-phase-c-paired-diagnostic/v1",
        "status": "EXPLORATORY_NOT_A_GATE",
        "tie_margin": tie_margin,
        "informative_groups": len(labels),
        "informative_tasks": len(set(tasks)),
        "class_counts": {
            "continue.source": int((labels > 0).sum()),
            "requery.source": int((labels < 0).sum()),
        },
        "primary": {
            "model": primary,
            "best_control": best_control,
            "task_bootstrap_gain": gain,
            "task_bootstrap_95_ci": interval,
            "directional_seeds": directional,
            "trace_minus_shuffled_gain": shuffled_gain,
        },
        "models": summary,
        "interpretation_rule": (
            "This diagnostic can identify target-formulation failure, but cannot unlock "
            "Phase D or override B_FAIL_REPRODUCIBILITY."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie-margin", type=float, default=0.01)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    dataset = load_dataset(args.feature_dir.resolve())
    result = analyze(
        dataset, tie_margin=args.tie_margin, replicates=args.bootstrap_replicates,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
