#!/usr/bin/env python3
"""Build revised recovery dataset: early student_query_state ∩ successful OFT (+ light clean)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rase.adapt.pre_c1 import episode_grouped_split
from rase.adapt.pre_c1_2 import load_protocol_lock


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _query_index(row: dict[str, Any]) -> int:
    qid = str(row.get("query_id") or "")
    if "__q" not in qid:
        return 10**9
    try:
        return int(qid.rsplit("__q", 1)[-1])
    except ValueError:
        return 10**9


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
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("runs/rase_pre_c1_2_revised_dataset_r1_v1.jsonl"),
    )
    parser.add_argument(
        "--splits-output",
        type=Path,
        default=Path("runs/rase_pre_c1_2_revised_dataset_r1_v1.benchmark-splits.json"),
    )
    parser.add_argument(
        "--qc-json",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_revised_dataset_qc_r1.json"),
    )
    parser.add_argument("--max-query-index", type=int, default=8, help="Keep early queries only.")
    parser.add_argument(
        "--keep-triggers",
        nargs="+",
        default=["anchor_start", "progress_stall", "first_divergence"],
    )
    parser.add_argument("--drop-suffix", action="store_true", default=True)
    parser.add_argument("--keep-suffix", action="store_true", help="Override drop-suffix.")
    parser.add_argument("--keep-original-recovery-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026080405)
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    rows = _load_jsonl(args.input_jsonl.resolve())
    drop_suffix = bool(args.drop_suffix) and not bool(args.keep_suffix)
    keep_triggers = set(args.keep_triggers)

    selected: list[dict[str, Any]] = []
    rejected = Counter()
    for row in rows:
        source = str(row.get("source") or "")
        role = str(row.get("dataset_role") or "")
        if bool(row.get("clean_flag")) or source == "clean_retention" or role == "clean_retention":
            selected.append({**row, "source": "clean_retention", "dataset_role": "clean_retention"})
            continue
        if not bool(row.get("teacher_rollout_success", True)):
            rejected["failed_teacher"] += 1
            continue
        if source == "teacher_suffix_after_student_query" or (
            int(row.get("offset_from_student_state", 0) or 0) > 0 and source != "original_recovery"
        ):
            if drop_suffix:
                rejected["suffix_dropped"] += 1
                continue
        if source == "student_query_state" or (
            role == "student_state_recovery" and int(row.get("offset_from_student_state", 0) or 0) == 0
        ):
            trigger = str(row.get("query_trigger") or "")
            q_index = _query_index(row)
            early = q_index <= int(args.max_query_index)
            trigger_ok = (not keep_triggers) or (trigger in keep_triggers) or early
            if not trigger_ok and not early:
                rejected["late_query"] += 1
                continue
            if not early and trigger == "periodic":
                rejected["late_periodic"] += 1
                continue
            selected.append(
                {
                    **row,
                    "source": "student_query_state",
                    "dataset_role": "student_state_recovery",
                    "revised_keep_reason": "early_or_priority_trigger",
                    "query_index": q_index,
                }
            )
            continue
        if source == "original_recovery" or role == "original_recovery":
            # Downsample original recovery so student states dominate.
            # Deterministic keep by hash of sample_id.
            sample_id = str(row.get("sample_id") or row.get("chunk_path") or row.get("state_key"))
            digest = sum(ord(c) for c in sample_id) % 1000
            keep = digest < int(1000 * float(args.keep_original_recovery_frac))
            if not keep:
                rejected["original_downsampled"] += 1
                continue
            selected.append(
                {
                    **row,
                    "source": "original_recovery",
                    "dataset_role": "original_recovery",
                    "revised_keep_reason": "initial_deviation_frac",
                }
            )
            continue
        rejected["other"] += 1

    for row in selected:
        if "episode_id" not in row or row["episode_id"] is None:
            row["episode_id"] = str(row.get("anchor_id") or row.get("state_key") or row.get("sample_id"))

    splits = episode_grouped_split(
        selected,
        seed=int(args.seed),
        val_fraction=float(lock["dataset"].get("val_episode_fraction", 0.25)),
    )
    out = args.output_jsonl.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write(args.splits_output.resolve(), splits)
    qc = {
        "schema_version": "rase-pre-c1-2-revised-dataset-qc/v1",
        "n_input": len(rows),
        "n_selected": len(selected),
        "rejected": dict(rejected),
        "source_counts": dict(Counter(str(r.get("source")) for r in selected)),
        "role_counts": dict(Counter(str(r.get("dataset_role")) for r in selected)),
        "drop_suffix": drop_suffix,
        "max_query_index": int(args.max_query_index),
        "keep_triggers": sorted(keep_triggers),
        "splits": {
            "n_train_episodes": len(splits["train_episodes"]),
            "n_val_episodes": len(splits["val_episodes"]),
            "n_train_rows": splits["n_train_rows"],
            "n_val_rows": splits["n_val_rows"],
            "leakage_episode_overlap": splits["leakage_episode_overlap"],
        },
        "training_objective_note": "short-horizon corrective / early recoverable occupancy; terminal 8pp is final gate only",
    }
    _write(args.qc_json.resolve(), qc)
    print(json.dumps(qc, sort_keys=True))
    print(f"PRE_C1_2_REVISED_DATASET_DONE output={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
