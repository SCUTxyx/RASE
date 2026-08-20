#!/usr/bin/env python3
"""Export PRE-C0 arms and fit a minimal candidate-conditioned risk scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rase.risk import (
    evaluate_selector_baselines,
    export_candidate_rows,
    fit_logistic_scorer,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_states(rollout_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(rollout_dir.glob("*.json")):
        if path.name == "run_manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "rase-pre-c0-corrective-rollout/v1":
            continue
        rows.append(payload)
    if not rows:
        raise SystemExit(f"no PRE-C0 rollout JSON under {rollout_dir}")
    return rows


def _episode_split(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float = 0.25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = sorted({str(row["episode_id"]) for row in rows})
    rng = np.random.default_rng(seed)
    order = list(episodes)
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_fraction)))
    val_eps = set(order[:n_val])
    train = [row for row in rows if str(row["episode_id"]) not in val_eps]
    val = [row for row in rows if str(row["episode_id"]) in val_eps]
    if not train or not val:
        raise SystemExit("episode split produced empty train or val")
    return train, val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--export-json", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2_026_080_405)
    args = parser.parse_args()

    states = _load_states(args.rollout_dir.resolve())
    candidates = export_candidate_rows(states)
    _write(
        args.export_json,
        {
            "schema_version": "rase-pre-c0-candidate-export/v1",
            "n_states": len(states),
            "n_candidates": len(candidates),
            "candidates": candidates,
        },
    )

    train_rows, val_rows = _episode_split(candidates, seed=args.seed)
    x_cand_train = np.asarray([row["x_candidate"] for row in train_rows], dtype=np.float64)
    x_hist_train = np.asarray([row["x_history"] for row in train_rows], dtype=np.float64)
    y_train = np.asarray([float(row["success"]) for row in train_rows], dtype=np.float64)
    candidate_scorer = fit_logistic_scorer(
        x_cand_train, y_train, kind="candidate_conditioned", seed=args.seed
    )
    history_scorer = fit_logistic_scorer(
        x_hist_train, y_train, kind="history_only", seed=args.seed + 1
    )
    _write(
        args.model_json,
        {
            "schema_version": "rase-pre-c0-candidate-risk-model/v1",
            "seed": args.seed,
            "train_episodes": sorted({str(r["episode_id"]) for r in train_rows}),
            "val_episodes": sorted({str(r["episode_id"]) for r in val_rows}),
            "candidate_scorer": {
                "kind": candidate_scorer.kind,
                "bias": candidate_scorer.bias,
                "feature_dim": candidate_scorer.feature_dim,
                "weights": candidate_scorer.weights.tolist(),
            },
            "history_scorer": {
                "kind": history_scorer.kind,
                "bias": history_scorer.bias,
                "feature_dim": history_scorer.feature_dim,
                "weights": history_scorer.weights.tolist(),
            },
        },
    )

    eval_payload = evaluate_selector_baselines(
        val_rows,
        candidate_scorer=candidate_scorer,
        history_scorer=history_scorer,
    )
    eval_payload["schema_version"] = "rase-pre-c0-candidate-selector-eval/v1"
    eval_payload["split"] = "episode_held_out_val"
    eval_payload["n_train_candidates"] = len(train_rows)
    eval_payload["n_val_candidates"] = len(val_rows)
    _write(args.eval_json, eval_payload)
    print(json.dumps({k: eval_payload[k] for k in ("n_states", "success_rates", "headroom_pp")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
