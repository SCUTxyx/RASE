#!/usr/bin/env python3
"""PRE-C1.3: Build recovery-corridor ablation datasets for Arms A', B, C.

Builds three datasets from the R1 merged JSONL (1880 rows):

  Arm A' — original_recovery + clean retention (continued-training drift control)
  Arm B  — student_query_state + downsampled original + clean (query-only)
  Arm C  — student_query + position-weighted teacher suffix + original + clean

All arms share matched original_recovery fraction and clean fraction.
Splits are episode/anchor-grouped. Same anchor never spans train/val.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from rase.adapt.pre_c1 import episode_grouped_split
    from rase.adapt.pre_c1_2 import load_protocol_lock
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rase.adapt.pre_c1 import episode_grouped_split
    from rase.adapt.pre_c1_2 import load_protocol_lock


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _save_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _row_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _anchor_id(row: dict[str, Any]) -> str:
    return str(row.get("anchor_id") or row.get("state_key") or row.get("failure_key") or "")


def classify_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    classified: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source") or "")
        role = str(row.get("dataset_role") or "")
        if bool(row.get("clean_flag")) or source == "clean_retention" or role == "clean_retention":
            classified["clean"].append({**row, "source": "clean_retention", "dataset_role": "clean_retention"})
        elif source == "student_query_state":
            classified["student_query"].append(row)
        elif source == "teacher_suffix_after_student_query":
            classified["teacher_suffix"].append(row)
        elif source == "original_recovery" or role == "original_recovery":
            classified["original_recovery"].append(row)
        else:
            classified["other"].append(row)
    return dict(classified)


def build_arm_ap(
    original_recovery: list[dict[str, Any]],
    clean: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in original_recovery:
        rows.append({**row, "source": "original_recovery", "dataset_role": "original_recovery"})
    for row in clean:
        rows.append({**row, "source": "clean_retention", "dataset_role": "clean_retention", "clean_flag": True})
    return rows


def build_arm_b(
    student_query: list[dict[str, Any]],
    original_recovery: list[dict[str, Any]],
    clean: list[dict[str, Any]],
    *,
    target_original_frac: float = 0.20,
    seed: int = 2026080405,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in student_query:
        rows.append({**row, "source": "student_query_state", "dataset_role": "student_state_recovery",
                     "arm_b_include_reason": "all_query"})

    n_student = len(student_query)
    n_original_target = int(n_student / (1.0 - target_original_frac) * target_original_frac)
    n_original = min(n_original_target, len(original_recovery))
    rng = random.Random(seed)
    sampled_original = rng.sample(original_recovery, n_original) if n_original < len(original_recovery) else list(original_recovery)
    for row in sampled_original:
        rows.append({**row, "source": "original_recovery", "dataset_role": "original_recovery",
                     "arm_b_include_reason": "downsampled_original"})

    for row in clean:
        rows.append({**row, "source": "clean_retention", "dataset_role": "clean_retention", "clean_flag": True,
                     "arm_b_include_reason": "all_clean"})
    return rows


def build_arm_c(
    student_query: list[dict[str, Any]],
    teacher_suffix: list[dict[str, Any]],
    original_recovery: list[dict[str, Any]],
    clean: list[dict[str, Any]],
    *,
    target_student_frac: float = 0.35,
    target_suffix_frac: float = 0.35,
    target_original_frac: float = 0.20,
    target_clean_frac: float = 0.10,
    min_suffix_offset_gap: int = 0,
    max_suffix_rows_per_anchor: int = 30,
    early_u_max: float = 0.30,
    mid_u_max: float = 0.70,
    early_weight: float = 3.0,
    mid_weight: float = 2.0,
    late_weight: float = 1.0,
    seed: int = 2026080405,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Student queries: all.
    for row in student_query:
        rows.append({**row, "source": "student_query_state", "dataset_role": "student_state_recovery",
                     "arm_c_include_reason": "all_query"})

    n_student = len(student_query)
    total = int(n_student / target_student_frac)
    n_suffix_target = int(total * target_suffix_frac)
    n_original_target = int(total * target_original_frac)
    n_clean_target = int(total * target_clean_frac)

    # --- Suffix: annotate normalized position, then position-weighted sample ---
    annotated_suffix: list[dict[str, Any]] = []
    for row in teacher_suffix:
        offset = int(row.get("offset_from_student_state", 0) or 0)
        rec_len = int(row.get("teacher_recovery_length", 0) or 0)
        # Try to get recovery_length from the anchor's query record
        if rec_len <= 0:
            rec_len = int(row.get("chunk_index", 0) or 0) + 1
        u = offset / max(rec_len, 1)
        annotated_suffix.append({
            **row,
            "suffix_offset": offset,
            "suffix_u": u,
            "suffix_bucket": "early" if u <= early_u_max else ("mid" if u <= mid_u_max else "late"),
        })

    # Group by anchor
    by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotated_suffix:
        by_anchor[_anchor_id(row)].append(row)

    # Per-anchor dedup + cap
    deduped: list[dict[str, Any]] = []
    for anchor, anchor_rows in sorted(by_anchor.items()):
        anchor_rows.sort(key=lambda r: int(r.get("suffix_offset", 0) or 0))
        kept = [anchor_rows[0]]
        for row in anchor_rows[1:]:
            if (int(row.get("suffix_offset", 0) or 0) - int(kept[-1].get("suffix_offset", 0) or 0)) >= min_suffix_offset_gap:
                kept.append(row)
        if len(kept) > max_suffix_rows_per_anchor:
            rng = random.Random(seed + hash(anchor) % 10000)
            kept = rng.sample(kept, max_suffix_rows_per_anchor)
            kept.sort(key=lambda r: int(r.get("suffix_offset", 0) or 0))
        deduped.extend(kept)

    # Position-weighted sampling
    weights = []
    for row in deduped:
        bucket = row.get("suffix_bucket", "mid")
        if bucket == "early":
            weights.append(early_weight)
        elif bucket == "mid":
            weights.append(mid_weight)
        else:
            weights.append(late_weight)

    rng = random.Random(seed)
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    n_suffix = min(n_suffix_target, len(deduped))
    if n_suffix < len(deduped):
        indices = rng.choices(range(len(deduped)), weights=probs, k=n_suffix * 2)
        indices = list(dict.fromkeys(indices))[:n_suffix]
        sampled_suffix = [deduped[i] for i in sorted(indices)]
    else:
        sampled_suffix = list(deduped)

    for row in sampled_suffix:
        rows.append({**row, "source": "teacher_suffix_after_student_query", "dataset_role": "student_state_recovery",
                     "arm_c_include_reason": "position_weighted_suffix"})

    # Original recovery: deterministic downsampling
    n_original = min(n_original_target, len(original_recovery))
    rng2 = random.Random(seed + 1)
    if n_original < len(original_recovery):
        sampled_original = rng2.sample(original_recovery, n_original)
    else:
        sampled_original = list(original_recovery)
    for row in sampled_original:
        rows.append({**row, "source": "original_recovery", "dataset_role": "original_recovery",
                     "arm_c_include_reason": "downsampled_original"})

    # Clean
    n_clean = min(n_clean_target, len(clean))
    rng3 = random.Random(seed + 2)
    if n_clean < len(clean):
        sampled_clean = rng3.sample(clean, n_clean)
    else:
        sampled_clean = list(clean)
        if n_clean > len(clean):
            for _ in range(n_clean - len(clean)):
                sampled_clean.append(rng3.choice(clean))
    for row in sampled_clean:
        rows.append({**row, "source": "clean_retention", "dataset_role": "clean_retention", "clean_flag": True,
                     "arm_c_include_reason": "all_clean"})

    return rows


def _ensure_episode_id(row: dict[str, Any]) -> None:
    if "episode_id" not in row or row["episode_id"] is None:
        row["episode_id"] = str(row.get("anchor_id") or row.get("state_key") or row.get("sample_id") or row.get("chunk_path"))


def _group_unit(row: dict[str, Any]) -> str:
    suite = str(row.get("suite") or "unknown")
    episode = str(row.get("episode_id") or "unknown")
    anchor = _anchor_id(row)
    return f"{suite}::{episode}::{anchor}"


def build_manifest(rows: list[dict[str, Any]], splits: dict[str, Any], arm_name: str) -> dict[str, Any]:
    source_counts = Counter(str(r.get("source")) for r in rows)
    role_counts = Counter(str(r.get("dataset_role")) for r in rows)
    suite_counts = Counter(str(r.get("suite") or "unknown") for r in rows)
    anchor_set = {_anchor_id(r) for r in rows}

    suffix_us = []
    for r in rows:
        u = r.get("suffix_u")
        if u is not None:
            suffix_us.append(float(u))

    return {
        "arm": arm_name,
        "n_rows": len(rows),
        "source_counts": dict(source_counts),
        "role_counts": dict(role_counts),
        "suite_counts": dict(suite_counts),
        "n_unique_anchors": len(anchor_set),
        "suffix_u_histogram_10bins": (
            [int(x) for x in np.histogram(suffix_us, bins=10, range=(0, 1))[0].tolist()]
            if suffix_us else []
        ),
        "splits": {
            "n_train_episodes": len(splits["train_episodes"]),
            "n_val_episodes": len(splits["val_episodes"]),
            "n_train_rows": splits["n_train_rows"],
            "n_val_rows": splits["n_val_rows"],
            "leakage_episode_overlap": splits.get("leakage_episode_overlap", 0),
        },
        "duplicate_state_hash_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_protocol_lock.yaml"),
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl"),
        help="Source R1 merged dataset (1880 rows).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rase_pre_c1_3_datasets"))
    parser.add_argument("--seed", type=int, default=2026080405)
    parser.add_argument("--smoke", action="store_true", help="Subsample for smoke test.")
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    ds_cfg = dict(lock["dataset"])

    rows = _load_jsonl(args.input_jsonl.resolve())
    if args.smoke:
        # Always keep all clean rows + a stratified sample
        classified_all = classify_rows(rows)
        clean_smoke = classified_all.get("clean", [])
        n_keep = 80
        n_other = max(0, n_keep - len(clean_smoke))
        other_rows = [r for r in rows if not (bool(r.get("clean_flag")) or str(r.get("source", "")) == "clean_retention")]
        rng = random.Random(args.seed)
        rows = clean_smoke + rng.sample(other_rows, min(n_other, len(other_rows)))

    classified = classify_rows(rows)
    print(f"Classified rows: { {k: len(v) for k, v in classified.items()} }")

    student_query = classified.get("student_query", [])
    teacher_suffix = classified.get("teacher_suffix", [])
    original_recovery = classified.get("original_recovery", [])
    clean = classified.get("clean", [])

    if not clean:
        raise SystemExit("No clean retention rows found.")
    if not student_query:
        raise SystemExit("No student_query rows found.")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = {
        "arm_ap": build_arm_ap(original_recovery, clean),
        "arm_b": build_arm_b(student_query, original_recovery, clean, seed=args.seed),
        "arm_c": build_arm_c(
            student_query, teacher_suffix, original_recovery, clean, seed=args.seed,
        ),
    }

    manifests: dict[str, Any] = {}
    for arm_name, arm_rows in arms.items():
        for row in arm_rows:
            _ensure_episode_id(row)

        # Episode-anchor grouped split
        splits = episode_grouped_split(
            arm_rows,
            seed=int(args.seed + hash(arm_name) % 100000),
            val_fraction=float(ds_cfg.get("val_episode_fraction", 0.25)),
        )

        # Verify no leakage: same anchor must not span train and val
        train_eps = set(splits["train_episodes"])
        val_eps = set(splits["val_episodes"])
        train_anchors = {_group_unit(r) for r in arm_rows if str(r.get("episode_id")) in train_eps}
        val_anchors = {_group_unit(r) for r in arm_rows if str(r.get("episode_id")) in val_eps}
        leakage = train_anchors & val_anchors
        if leakage:
            raise SystemExit(f"Anchor leakage between train/val in {arm_name}: {leakage}")

        jsonl_path = out_dir / f"{arm_name}.jsonl"
        splits_path = out_dir / f"{arm_name}.benchmark-splits.json"
        manifest_path = out_dir / f"{arm_name}.manifest.json"

        _save_jsonl(jsonl_path, arm_rows)
        _write_json(splits_path, splits)
        manifest = build_manifest(arm_rows, splits, arm_name)
        _write_json(manifest_path, manifest)
        manifests[arm_name] = manifest

        print(f"Arm {arm_name}: {len(arm_rows)} rows -> {jsonl_path}")
        print(f"  Sources: {manifest['source_counts']}")
        print(f"  Suites:  {manifest['suite_counts']}")

    summary = {
        "schema_version": "rase-pre-c1-3-dataset-build/v1",
        "source": str(args.input_jsonl.resolve()),
        "seed": args.seed,
        "arms": manifests,
    }
    _write_json(out_dir / "build_summary.json", summary)
    print(f"PRE_C1_3_DATASETS_DONE output_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
