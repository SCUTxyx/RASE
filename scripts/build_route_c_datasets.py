#!/usr/bin/env python3
"""D2-B: Build four training datasets from verified boundaries and existing data.

Datasets:
  clean          - original C1.1 successful trajectories (retention)
  nominal-OFT    - arbitrary OFT success trajectories + clean
  recovery-OFT   - verified recoverable boundary OFT trajectories + clean
  residual       - verified boundary teacher_delta targets

Outputs JSONL files consumable by train_route_c_plugin.py (residual) and
train_route_c_direct_recovery.py (LoRA variants).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_clean_dataset(output_dir: Path):
    """Placeholder: links to existing C1.1 clean trajectories."""
    clean_dir = output_dir / "clean"
    clean_dir.mkdir(exist_ok=True)
    manifest_file = clean_dir / "manifest.json"
    manifest_file.write_text(json.dumps({
        "dataset": "clean",
        "source": "C1.1 successful rollouts",
        "n_samples": 0,
        "note": "Populate with actual C1.1 trajectories from pool/",
    }, indent=2) + "\n", encoding="utf-8")


def build_nominal_oft_dataset(output_dir: Path):
    """Placeholder: links to arbitrary OFT success data."""
    nom_dir = output_dir / "nominal_oft"
    nom_dir.mkdir(exist_ok=True)
    (nom_dir / "manifest.json").write_text(json.dumps({
        "dataset": "nominal_oft",
        "source": "any OFT success trajectory + clean",
        "n_samples": 0,
        "note": "Populate from existing OFT collection runs",
    }, indent=2) + "\n", encoding="utf-8")


def build_recovery_oft_dataset(verified_boundaries_file: Path, output_dir: Path):
    """Extract OFT recovery trajectories from verified boundaries."""
    rev_dir = output_dir / "recovery_oft"
    rev_dir.mkdir(exist_ok=True)

    boundaries = json.loads(verified_boundaries_file.read_text(encoding="utf-8"))
    samples_written = 0
    for b in boundaries:
        trajectory = b.get("persistent_teacher_trajectory", [])
        if not trajectory:
            continue
        task_id = b["task_id"]
        for t, step_data in enumerate(trajectory):
            if t > 0 and t % 4 != 0:
                continue  # subsample: take every 4th step for balanced sampling
            sample = {
                "task_id": task_id,
                "suite": b["suite"],
                "boundary_id": b["boundary_id"],
                "step": t,
                "action": step_data.get("action", []),
                "split": "train",
            }
            (rev_dir / f"{b['boundary_id']}_step_{t:04d}.json").write_text(
                json.dumps(sample, ensure_ascii=False), encoding="utf-8")
            samples_written += 1

    (rev_dir / "manifest.json").write_text(json.dumps({
        "dataset": "recovery_oft",
        "source": str(verified_boundaries_file),
        "n_boundaries": len(boundaries),
        "n_samples": samples_written,
        "sampling": "every 4th step",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  recovery_oft: {samples_written} samples from {len(boundaries)} boundaries")


def build_residual_dataset(verified_boundaries_file: Path, output_dir: Path):
    """Extract residual (teacher_delta) targets from verified boundaries."""
    res_dir = output_dir / "residual"
    res_dir.mkdir(exist_ok=True)

    boundaries = json.loads(verified_boundaries_file.read_text(encoding="utf-8"))
    samples_written = 0
    for b in boundaries:
        trajectory = b.get("persistent_teacher_trajectory", [])
        if len(trajectory) < 2:
            continue
        # For residual, use the first teacher action as delta target
        # (student action would need to be collected at the same state)
        first_action = trajectory[0].get("action", [])
        sample = {
            "task_id": b["task_id"],
            "suite": b["suite"],
            "boundary_id": b["boundary_id"],
            "delta_target": first_action,
            "student_action": [0.0] * len(first_action),
            "obs_features": None,
            "split": "train",
            "note": "student_action must be filled from student rollout at boundary",
        }
        (res_dir / f"{b['boundary_id']}_residual.json").write_text(
            json.dumps(sample, ensure_ascii=False), encoding="utf-8")
        samples_written += 1

    (res_dir / "manifest.json").write_text(json.dumps({
        "dataset": "residual",
        "source": str(verified_boundaries_file),
        "n_boundaries": len(boundaries),
        "n_samples": samples_written,
        "note": "student_action at boundary must be populated for training",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  residual: {samples_written} samples from {len(boundaries)} boundaries")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verified-boundaries", type=Path, default=None,
                        help="D2-A verified boundaries JSON file")
    parser.add_argument("--datasets", type=str, nargs="*",
                        default=["clean", "nominal_oft", "recovery_oft", "residual"],
                        choices=["clean", "nominal_oft", "recovery_oft", "residual"])
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(exist_ok=True)

    for ds in args.datasets:
        print(f"Building {ds}...")
        if ds == "clean":
            build_clean_dataset(dataset_dir)
        elif ds == "nominal_oft":
            build_nominal_oft_dataset(dataset_dir)
        elif ds == "recovery_oft":
            if args.verified_boundaries and args.verified_boundaries.is_file():
                build_recovery_oft_dataset(args.verified_boundaries, dataset_dir)
            else:
                print("  Skipped: --verified-boundaries not provided")
        elif ds == "residual":
            if args.verified_boundaries and args.verified_boundaries.is_file():
                build_residual_dataset(args.verified_boundaries, dataset_dir)
            else:
                print("  Skipped: --verified-boundaries not provided")

    summary = {
        "output_dir": str(dataset_dir),
        "datasets_built": args.datasets,
    }
    (dataset_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nDatasets built: {dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
