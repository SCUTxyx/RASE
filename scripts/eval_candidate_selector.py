#!/usr/bin/env python3
"""Evaluate a saved candidate risk scorer against selector baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rase.risk import CandidateRiskScorer, evaluate_selector_baselines, export_candidate_rows


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_scorer(payload: dict[str, Any], key: str) -> CandidateRiskScorer:
    raw = payload[key]
    return CandidateRiskScorer(
        weights=np.asarray(raw["weights"], dtype=np.float64),
        bias=float(raw["bias"]),
        feature_dim=int(raw["feature_dim"]),
        kind=str(raw["kind"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--val-episodes-only",
        action="store_true",
        help="If set, evaluate only episodes listed as val in the model JSON.",
    )
    args = parser.parse_args()

    model = json.loads(args.model_json.read_text(encoding="utf-8"))
    states = []
    for path in sorted(args.rollout_dir.glob("*.json")):
        if path.name == "run_manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "rase-pre-c0-corrective-rollout/v1":
            continue
        states.append(payload)
    candidates = export_candidate_rows(states)
    if args.val_episodes_only:
        val_eps = set(model.get("val_episodes") or [])
        candidates = [row for row in candidates if str(row["episode_id"]) in val_eps]
    candidate_scorer = _load_scorer(model, "candidate_scorer")
    history_scorer = _load_scorer(model, "history_scorer")
    eval_payload = evaluate_selector_baselines(
        candidates,
        candidate_scorer=candidate_scorer,
        history_scorer=history_scorer,
    )
    eval_payload["schema_version"] = "rase-pre-c0-candidate-selector-eval/v1"
    eval_payload["model"] = str(args.model_json)
    _write(args.output, eval_payload)
    print(json.dumps(eval_payload["success_rates"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
