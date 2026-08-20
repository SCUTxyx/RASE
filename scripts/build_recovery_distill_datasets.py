#!/usr/bin/env python3
"""R4: 3-way dataset construction from collected demonstrations.

Inputs:
  - B1 (matched OFT) chunks from collection
  - B2 (matched nominal) chunks from collection
  - B3 (targeted recovery) chunks from collection

Outputs:
  - train.jsonl / dev.jsonl with rows per chunk (observation ref + teacher action)
  - splits.json with train/dev split
  - data_gate.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FROZEN_SPLIT_SEED = 202608041200


def _load_index(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _build_retention_rows(b2_index: list[dict[str, Any]], b2_chunks_dir: Path, retention_frac: float) -> list[dict[str, Any]]:
    """Select retention episodes from B2 successes."""
    clean = [e for e in b2_index if e.get("success")]
    random.shuffle(clean)
    n = max(1, int(len(clean) * retention_frac))
    selected = clean[:n]
    rows: list[dict[str, Any]] = []
    for ep in selected:
        for chunk_idx in range(ep.get("chunk_count", 0)):
            rows.append({
                "episode_id": ep["episode_id"],
                "task_id": ep["task_id"],
                "data_stream": "clean_retention",
                "chunk_index": chunk_idx,
                "chunk_dir": str(b2_chunks_dir),
                "clean_flag": True,
            })
    return rows


def _build_stream_rows(index: list[dict[str, Any]], chunks_dir: Path, stream: str, clean_flag: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ep in index:
        for chunk_idx in range(ep.get("chunk_count", 0)):
            rows.append({
                "episode_id": ep["episode_id"],
                "task_id": ep["task_id"],
                "data_stream": stream,
                "chunk_index": chunk_idx,
                "chunk_dir": str(chunks_dir),
                "clean_flag": clean_flag,
            })
    return rows


def _split_rows(rows: list[dict[str, Any]], train_frac: float, seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_ep: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_ep.setdefault(row["episode_id"], []).append(row)
    ep_ids = list(by_ep)
    rng.shuffle(ep_ids)
    n_train = int(len(ep_ids) * train_frac)
    train_eps = set(ep_ids[:n_train])
    dev_eps = set(ep_ids[n_train:])
    train_rows = [row for ep in ep_ids if ep in train_eps for row in by_ep[ep]]
    dev_rows = [row for ep in ep_ids if ep in dev_eps for row in by_ep[ep]]
    return {
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "train_episodes": sorted(train_eps),
        "dev_episodes": sorted(dev_eps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--b1-index", type=Path, required=True, help="B1 collection index")
    parser.add_argument("--b2-index", type=Path, required=True, help="B2 collection index")
    parser.add_argument("--b3-index", type=Path, required=True, help="B3 collection index")
    parser.add_argument("--b1-chunks-dir", type=Path, required=True)
    parser.add_argument("--b2-chunks-dir", type=Path, required=True)
    parser.add_argument("--b3-chunks-dir", type=Path, required=True)
    parser.add_argument("--retention-frac", type=float, default=0.30, help="Retention fraction for B3")
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=FROZEN_SPLIT_SEED)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    b1_index = _load_index(args.b1_index)
    b2_index = _load_index(args.b2_index)
    b3_index = _load_index(args.b3_index)

    # B1: matched OFT data
    b1_streams = _build_stream_rows(b1_index, args.b1_chunks_dir, "matched_oft")

    # B2: matched nominal
    b2_streams = _build_stream_rows(b2_index, args.b2_chunks_dir, "matched_nominal")

    # B3: targeted recovery (70%) + retention (30%)
    b3_recovery = _build_stream_rows(b3_index, args.b3_chunks_dir, "targeted_recovery")
    b3_retention = _build_retention_rows(b2_index, args.b2_chunks_dir, args.retention_frac)

    n_recovery = len(b3_recovery)
    n_retention = len(b3_retention)
    n_b3_retention = max(1, int(n_recovery * args.retention_frac / (1 - args.retention_frac)))
    if n_b3_retention < len(b3_retention):
        b3_retention = b3_retention[:n_b3_retention]

    b3_combined = b3_recovery + b3_retention
    random.shuffle(b3_combined)

    # Split
    b1_splits = _split_rows(b1_streams, args.train_frac, args.seed)
    b2_splits = _split_rows(b2_streams, args.train_frac, args.seed + 1)
    b3_splits = _split_rows(b3_combined, args.train_frac, args.seed + 2)

    # Write per-baseline files
    for baseline, splits in [("B1", b1_splits), ("B2", b2_splits), ("B3", b3_splits)]:
        baseline_dir = output_dir / baseline.lower()
        baseline_dir.mkdir(parents=True, exist_ok=True)

        train_path = baseline_dir / "train.jsonl"
        with train_path.open("w", encoding="utf-8") as f:
            for row in splits["train_rows"]:
                f.write(json.dumps(row, sort_keys=True) + "\n")

        dev_path = baseline_dir / "dev.jsonl"
        with dev_path.open("w", encoding="utf-8") as f:
            for row in splits["dev_rows"]:
                f.write(json.dumps(row, sort_keys=True) + "\n")

        split_path = baseline_dir / "splits.json"
        split_path.write_text(json.dumps({
            "train_episodes": splits["train_episodes"],
            "dev_episodes": splits["dev_episodes"],
            "n_train": len(splits["train_rows"]),
            "n_dev": len(splits["dev_rows"]),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Data gate
    n_b3_recovery = len([r for r in b3_splits["train_rows"] if r["data_stream"] == "targeted_recovery"])
    n_b3_retention_total = len([r for r in b3_splits["train_rows"] if r["data_stream"] == "clean_retention"])

    threshold = 20  # minimum recovery chunks
    gate_passed = n_b3_recovery >= threshold

    gate = {
        "schema_version": "rase-recovery-distill-r4/data-gate/v1",
        "b1_matched_oft": {"n_train": len(b1_splits["train_rows"]), "n_dev": len(b1_splits["dev_rows"])},
        "b2_matched_nominal": {"n_train": len(b2_splits["train_rows"]), "n_dev": len(b2_splits["dev_rows"])},
        "b3_targeted_recovery": {"n_train": n_b3_recovery, "n_retention_train": n_b3_retention_total, "n_dev": len(b3_splits["dev_rows"])},
        "gate_passed": gate_passed,
        "threshold": threshold,
    }

    gate_path = output_dir / "data_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"\nData gate: {'PASSED' if gate_passed else 'FAILED'} (need >= {threshold} recovery chunks, have {n_b3_recovery})")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
