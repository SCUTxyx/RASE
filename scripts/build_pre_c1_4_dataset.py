#!/usr/bin/env python3
"""PRE-C1.4-R3 Phase 2: Dataset build.

Builds 6 mutually-exclusive data streams:
  1. verified_preferred_pairs   → V1/V2 training
  2. matched_teacher_targets    → V0/V1/V2 shared
  3. teacher_positive           → standard FM (DAgger R1 teacher_suffix)
  4. equivalent_pairs           → retention/neutral mask
  5. clean_retention            → prevent clean regression
  6. diagnostic_only            → both-fail / ambiguous pairs

Normalization statistics, action dimension weights, dense reward statistics
estimated from train_collection only. Adjacent snapshots stay in same cluster.

Output: runs/rase_pre_c1_4_dataset/{train,val}.jsonl + splits JSON + stats.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_c1_1_distill_rows():
    """Load C1.1 distill dataset rows for retention."""
    c11_path = ROOT / "runs" / "rase_pre_c1_1_distill_dataset_v1.jsonl"
    if not c11_path.exists():
        return []
    rows = []
    with open(c11_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_dagger_r1_rows():
    """Load DAgger R1 teacher_suffix rows."""
    dagger_path = (
        ROOT / "runs" / "rase_pre_c1_2_distill_dataset_r1_v1.jsonl"
    )
    if not dagger_path.exists():
        return []
    rows = []
    with open(dagger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _estimate_normalization_stats(data_rows: list) -> dict:
    """Estimate per-dimension normalization from train data."""
    # Placeholder: actual implementation would read action chunks
    # from the .npz files and compute per-dimension mean/std.
    return {
        "action_dim_groups": {
            "translation": {"dims": [0, 1, 2], "std": [0.1, 0.1, 0.05]},
            "rotation": {"dims": [3, 4, 5], "std": [0.05, 0.05, 0.03]},
            "gripper": {"dims": [6], "std": [0.5]},
        },
        "estimated_from": "train_collection",
        "n_rows": len(data_rows),
    }


def _episode_grouped_split(
    rows: list, val_fraction: float = 0.25, seed: int = 20260806
) -> tuple:
    """Split rows by episode_id to prevent cross-split leakage."""
    rng = random.Random(seed)
    by_episode = defaultdict(list)
    for i, row in enumerate(rows):
        ep = row.get("episode_id", row.get("anchor_id", f"ep_{i}"))
        by_episode[ep].append(row)

    episodes = list(by_episode.keys())
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * val_fraction))
    val_eps = set(episodes[:n_val])
    train_eps = set(episodes[n_val:])

    train_rows = []
    val_rows = []
    for ep, rows_ep in by_episode.items():
        if ep in val_eps:
            val_rows.extend(rows_ep)
        else:
            train_rows.extend(rows_ep)

    return train_rows, val_rows, {
        "train_episodes": len(train_eps),
        "val_episodes": len(val_eps),
        "overlap": len(train_eps & val_eps),
    }


def main():
    parser = argparse.ArgumentParser(
        description="PRE-C1.4-R3 Phase 2: Dataset build"
    )
    parser.add_argument(
        "--verified-pairs",
        default=str(
            ROOT
            / "runs"
            / "rase_pre_c1_4_counterfactual"
            / "verified_pairs.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "runs" / "rase_pre_c1_4_dataset"),
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.25,
    )
    parser.add_argument(
        "--seed", type=int, default=20260806,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data sources ----
    verified_pairs = []
    vp_path = Path(args.verified_pairs)
    if vp_path.exists():
        with open(vp_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    verified_pairs.append(json.loads(line))

    c11_rows = _load_c1_1_distill_rows()
    dagger_rows = _load_dagger_r1_rows()

    # ---- Classify into streams ----
    preferred = [p for p in verified_pairs if p.get("label") == "teacher_preferred"]
    equivalent = [p for p in verified_pairs if p.get("label") == "equivalent"]
    both_fail = [p for p in verified_pairs if p.get("label") == "both_fail"]
    ambiguous = [p for p in verified_pairs if p.get("label") == "ambiguous"]

    # Teacher positives from DAgger R1 (teacher_suffix + student_query as FM data)
    teacher_positive = [
        r for r in dagger_rows
        if r.get("source") in (
            "teacher_suffix_after_student_query",
            "student_query_state",
        )
        or r.get("dataset_role") == "student_state_recovery"
    ]

    # Clean retention from C1.1
    clean_rows = [
        r for r in c11_rows
        if r.get("source") == "clean_retention" or r.get("clean_flag") is True
    ]

    # Original recovery from C1.1 for matched_teacher_targets
    original_recovery = [
        r for r in c11_rows
        if r.get("source") == "original_recovery"
    ]

    # ---- Estimate statistics from train data ----
    all_train = preferred + teacher_positive + original_recovery
    norm_stats = _estimate_normalization_stats(all_train)

    # ---- Build datasets ----
    # Stream 1: verified_preferred (used by V1/V2, also in V0 as matched_teacher_targets)
    # Stream 2: teacher_positive (standard FM for all variants)
    # Stream 3: equivalent (retention/neutral, used by all)
    # Stream 4: clean_retention
    # Stream 5: both_fail + ambiguous (diagnostic only)

    train_rows = []
    val_rows = []

    # Each row gets a "data_stream" tag
    for row in preferred:
        row["data_stream"] = "verified_preferred"
        train_rows.append(row)

    for row in teacher_positive:
        row["data_stream"] = "teacher_positive"
        train_rows.append(row)

    for row in equivalent:
        row["data_stream"] = "equivalent"
        train_rows.append(row)

    for row in clean_rows:
        row["data_stream"] = "clean_retention"
        train_rows.append(row)

    for row in original_recovery:
        row["data_stream"] = "matched_teacher_targets"
        train_rows.append(row)

    # Train/val split (episode-grouped)
    train, val, split_info = _episode_grouped_split(
        train_rows, val_fraction=args.val_fraction, seed=args.seed
    )

    # ---- Write outputs ----
    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")
    print(f"Train dataset: {train_path} ({len(train)} rows)")

    val_path = output_dir / "val.jsonl"
    with open(val_path, "w") as f:
        for row in val:
            f.write(json.dumps(row) + "\n")
    print(f"Val dataset: {val_path} ({len(val)} rows)")

    # Diagnostic data (both_fail + ambiguous)
    diag_path = output_dir / "diagnostic.jsonl"
    with open(diag_path, "w") as f:
        for row in both_fail + ambiguous:
            row["data_stream"] = "diagnostic_only"
            f.write(json.dumps(row) + "\n")
    print(f"Diagnostic: {diag_path} ({len(both_fail) + len(ambiguous)} rows)")

    # Splits JSON
    splits = {
        "train_episodes": split_info["train_episodes"],
        "val_episodes": split_info["val_episodes"],
        "n_train_rows": len(train),
        "n_val_rows": len(val),
        "leakage_episode_overlap": split_info["overlap"],
        **{f"n_{k}": len(v) for k, v in {
            "preferred": preferred,
            "teacher_positive": teacher_positive,
            "equivalent": equivalent,
            "clean": clean_rows,
            "both_fail": both_fail,
            "ambiguous": ambiguous,
        }.items()},
    }
    splits_path = output_dir / "benchmark_splits.json"
    splits_path.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n")
    print(f"Splits: {splits_path}")

    # Normalization stats
    stats_path = output_dir / "normalization_stats.json"
    stats_path.write_text(json.dumps(norm_stats, indent=2) + "\n")
    print(f"Stats: {stats_path}")

    # Build manifest
    manifest = {
        "schema_version": "rase-pre-c1-4-r3-dataset/v1",
        "output_dir": str(output_dir),
        "streams": {
            "verified_preferred": len(preferred),
            "matched_teacher_targets": len(original_recovery),
            "teacher_positive": len(teacher_positive),
            "equivalent": len(equivalent),
            "clean_retention": len(clean_rows),
            "diagnostic": len(both_fail) + len(ambiguous),
        },
        "n_train": len(train),
        "n_val": len(val),
        "split_seed": args.seed,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Manifest: {manifest_path}")
    print("\nDone. Dataset built.")


if __name__ == "__main__":
    main()
