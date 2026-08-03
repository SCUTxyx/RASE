#!/usr/bin/env python3
"""Analyze a complete same-state intervention matrix by task, suite, and step."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    disagreements = left_only + right_only
    if disagreements == 0:
        return 1.0
    tail = sum(
        math.comb(disagreements, index)
        for index in range(min(left_only, right_only) + 1)
    ) / (2**disagreements)
    return min(1.0, 2.0 * tail)


def _summary(rows: list[dict[str, Any]], operator_ids: list[str]) -> dict[str, Any]:
    if not rows:
        return {"n_states": 0}
    fixed = {
        operator_id: float(
            np.mean([row["success"][operator_id] for row in rows])
        )
        for operator_id in operator_ids
    }
    best_fixed_id = max(operator_ids, key=lambda name: fixed[name])
    oracle = float(
        np.mean([max(row["success"].values()) for row in rows])
    )
    unique_winners: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    for row in rows:
        best = max(row["success"].values())
        winners = [
            name
            for name in operator_ids
            if np.isclose(row["success"][name], best)
        ]
        if len(winners) == 1:
            unique_winners[winners[0]] += 1
        pattern_counts[
            "".join("1" if row["success"][name] > 0.5 else "0" for name in operator_ids)
        ] += 1
    return {
        "n_states": len(rows),
        "n_tasks": len({row["task_id"] for row in rows}),
        "n_episodes": len({row["episode_id"] for row in rows}),
        "per_operator_success_rate": fixed,
        "best_fixed_operator": best_fixed_id,
        "best_fixed_success_rate": fixed[best_fixed_id],
        "same_state_oracle_success_rate": oracle,
        "oracle_minus_best_fixed": oracle - fixed[best_fixed_id],
        "n_no_operator_support": int(sum(
            np.isclose(max(row["success"].values()), 0.0) for row in rows
        )),
        "n_all_operator_success": int(sum(
            all(np.isclose(row["success"][name], 1.0) for name in operator_ids)
            for row in rows
        )),
        "unique_winner_counts": dict(sorted(unique_winners.items())),
        "success_pattern_counts": dict(sorted(pattern_counts.items())),
    }


def _bootstrap_episode_gap(
    rows: list[dict[str, Any]],
    operator_ids: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    episode_ids = sorted(by_episode)
    if not episode_ids or replicates < 1:
        return {"replicates": 0, "n_episode_clusters": len(episode_ids)}
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(replicates):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        bootstrap_rows = [row for episode_id in sampled for row in by_episode[episode_id]]
        gaps.append(_summary(bootstrap_rows, operator_ids)["oracle_minus_best_fixed"])
    return {
        "replicates": replicates,
        "seed": seed,
        "cluster_unit": "source_episode",
        "n_episode_clusters": len(episode_ids),
        "oracle_gap_percentile_95_ci": [
            float(np.quantile(gaps, 0.025)),
            float(np.quantile(gaps, 0.975)),
        ],
        "fraction_gap_positive": float(np.mean(np.asarray(gaps) > 0.0)),
    }


def analyze_matrix(
    snapshots: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    operator_ids: list[str],
    *,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 20260801,
) -> dict[str, Any]:
    snapshot_by_id = {row["snapshot_id"]: row for row in snapshots}
    if len(snapshot_by_id) != len(snapshots):
        raise ValueError("duplicate snapshots")
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for outcome in outcomes:
        if outcome["snapshot_id"] not in snapshot_by_id:
            raise ValueError("outcome references unknown snapshot")
        if outcome["operator_id"] not in operator_ids:
            raise ValueError("outcome references unknown operator")
        if outcome.get("observed") and not outcome.get("proxy", False):
            values[(outcome["snapshot_id"], outcome["operator_id"])].append(
                float(bool(outcome.get("success")))
            )
    rows = []
    for snapshot_id, snapshot in snapshot_by_id.items():
        success = {}
        for operator_id in operator_ids:
            arm = values[(snapshot_id, operator_id)]
            if arm:
                success[operator_id] = float(np.mean(arm))
        if len(success) != len(operator_ids):
            continue
        perturbation = dict(snapshot.get("perturbation") or {})
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "task_id": str(snapshot["task_id"]),
                "episode_id": str(snapshot["episode_id"]),
                "suite": str(snapshot.get("suite") or "unknown"),
                "step": int(snapshot["step"]),
                "perturbation_dimension": str(
                    perturbation.get("dimension") or "unknown"
                ),
                "perturbation_level": int(perturbation.get("level") or 0),
                "dimension_level": (
                    f"{perturbation.get('dimension') or 'unknown'}:"
                    f"L{int(perturbation.get('level') or 0)}"
                ),
                "success": success,
            }
        )
    if not rows:
        raise ValueError("matrix contains no complete states")

    reference_id = next(
        (name for name in operator_ids if name.startswith("continue_")), None
    )
    pairwise = {}
    if reference_id is not None:
        for operator_id in operator_ids:
            if operator_id == reference_id:
                continue
            reference_only = int(sum(
                row["success"][reference_id] > row["success"][operator_id]
                and not np.isclose(
                    row["success"][reference_id], row["success"][operator_id]
                )
                for row in rows
            ))
            operator_only = int(sum(
                row["success"][operator_id] > row["success"][reference_id]
                and not np.isclose(
                    row["success"][operator_id], row["success"][reference_id]
                )
                for row in rows
            ))
            pairwise[operator_id] = {
                "continue_only_states": reference_only,
                "operator_only_states": operator_only,
                "tied_states": len(rows) - reference_only - operator_only,
                "mcnemar_exact_p": _mcnemar_exact(reference_only, operator_only),
            }

    grouped = {}
    for field in (
        "suite",
        "perturbation_dimension",
        "perturbation_level",
        "dimension_level",
        "step",
    ):
        grouped[field] = {
            str(value): _summary(
                [row for row in rows if row[field] == value], operator_ids
            )
            for value in sorted({row[field] for row in rows}, key=str)
        }
    return {
        "schema_version": "rase-intervention-matrix-analysis/v1",
        "operator_order": operator_ids,
        "overall": _summary(rows, operator_ids),
        "episode_cluster_bootstrap": _bootstrap_episode_gap(
            rows,
            operator_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "pairwise_vs_continue": pairwise,
        "by_group": grouped,
        "per_state": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.bootstrap_replicates < 0:
        raise SystemExit("--bootstrap-replicates must be non-negative")
    matrix_dir = args.matrix_dir.resolve()
    registry = _read_json(matrix_dir / "operators.json")
    operator_ids = [str(row["operator_id"]) for row in registry["operators"]]
    result = analyze_matrix(
        _read_jsonl(matrix_dir / "snapshots.jsonl"),
        _read_jsonl(matrix_dir / "outcomes.jsonl"),
        operator_ids,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = (args.output or matrix_dir / "analysis.json").resolve()
    _write_json(output, result)
    print(
        json.dumps(
            {
                "overall": result["overall"],
                "pairwise_vs_continue": result["pairwise_vs_continue"],
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
