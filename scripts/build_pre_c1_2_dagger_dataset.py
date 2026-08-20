#!/usr/bin/env python3
"""Build PRE-C1.2 distill JSONL from DAgger rows + original PRE-C1.1 recovery + clean."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rase.adapt.pre_c1 import episode_grouped_split
from rase.adapt.pre_c1_2 import dagger_qc_report, load_protocol_lock


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_dagger_dir(dagger_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(dagger_dir.glob("*.json")):
        if path.name in {"dagger_qc.json", "run_manifest.json"}:
            continue
        if path.name.endswith("_qc.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "rase-pre-c1-2-dagger-run/v1":
            continue
        rows.extend(payload.get("accepted_rows") or [])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_protocol_lock.yaml"),
    )
    parser.add_argument("--dagger-dir", type=Path, required=True)
    parser.add_argument(
        "--original-dataset-jsonl",
        type=Path,
        default=Path("runs/rase_pre_c1_1_distill_dataset_v1.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("runs/rase_pre_c1_2_distill_dataset_v1.jsonl"),
    )
    parser.add_argument(
        "--splits-output",
        type=Path,
        default=Path("runs/rase_pre_c1_2_distill_dataset_v1.benchmark-splits.json"),
    )
    parser.add_argument(
        "--qc-json",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_dataset_qc.json"),
    )
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    ds_cfg = dict(lock["dataset"])
    dagger_rows = _load_dagger_dir(args.dagger_dir.resolve())
    original = _load_jsonl(args.original_dataset_jsonl.resolve()) if args.original_dataset_jsonl.exists() else []

    samples: list[dict[str, Any]] = []
    for row in dagger_rows:
        samples.append(
            {
                **row,
                "dataset_role": "student_state_recovery",
                "clean_flag": False,
            }
        )
    for row in original:
        if bool(row.get("clean_flag")):
            samples.append({**row, "source": "clean_retention", "dataset_role": "clean_retention"})
        else:
            samples.append(
                {
                    **row,
                    "source": "original_recovery",
                    "dataset_role": "original_recovery",
                    "offset_from_student_state": row.get("offset_from_student_state", 0),
                }
            )

    # Ensure episode_id present for splits.
    for row in samples:
        if "episode_id" not in row or row["episode_id"] is None:
            row["episode_id"] = str(row.get("anchor_id") or row.get("state_key") or row.get("sample_id"))

    splits = episode_grouped_split(
        samples,
        seed=int(ds_cfg.get("split_seed", 2026080405)),
        val_fraction=float(ds_cfg.get("val_episode_fraction", 0.25)),
    )
    out = args.output_jsonl.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write(args.splits_output.resolve(), splits)

    qc_dagger = dagger_qc_report(dagger_rows)
    role_counts = Counter(str(r.get("dataset_role")) for r in samples)
    source_counts = Counter(str(r.get("source")) for r in samples)
    qc = {
        "schema_version": "rase-pre-c1-2-dataset-qc/v1",
        "n_samples": len(samples),
        "role_counts": dict(role_counts),
        "source_counts": dict(source_counts),
        "dagger_qc": qc_dagger,
        "splits": {
            "n_train_episodes": len(splits["train_episodes"]),
            "n_val_episodes": len(splits["val_episodes"]),
            "n_train_rows": splits["n_train_rows"],
            "n_val_rows": splits["n_val_rows"],
            "leakage_episode_overlap": splits["leakage_episode_overlap"],
        },
        "batch_schedule": dict(lock["batch_schedule"]),
        "recovery_source_weights": dict(lock["recovery_source_weights"]),
    }
    _write(args.qc_json.resolve(), qc)
    print(json.dumps(qc, sort_keys=True))
    print(f"PRE_C1_2_DATASET_DONE output={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
