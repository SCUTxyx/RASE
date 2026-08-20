#!/usr/bin/env python3
"""Create an audited canonical copy of legacy E3 step-demo artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical_steps(value: np.ndarray, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (length, 1, 7):
        array = array[:, 0, :]
    if array.shape != (length, 7):
        raise ValueError(f"{name}: expected ({length}, 7), got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    output.mkdir(parents=True)
    records = []
    repaired_broadcast_deltas = 0
    canonical_identity_deltas = 0
    for original in summary.get("records") or []:
        row = dict(original)
        source_json = source / f"{row['state_key']}.json"
        if row.get("status") != "complete":
            write_json(output / source_json.name, row)
            records.append(row)
            continue
        artifact = Path(row["artifact"])
        with np.load(artifact, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        length = int(row["n_steps"])
        source_action = canonical_steps(arrays["source_action"], name="source_action", length=length)
        target_action = canonical_steps(arrays["target_action"], name="target_action", length=length)
        old_delta_shape = tuple(np.asarray(arrays["delta_target"]).shape)
        delta_target = np.ascontiguousarray(target_action - source_action, dtype=np.float32)
        if old_delta_shape != delta_target.shape:
            repaired_broadcast_deltas += 1
        if row.get("mode") == "identity_source_success":
            if not np.array_equal(delta_target, np.zeros_like(delta_target)):
                raise ValueError(f"identity target is not zero for {row['state_key']}")
            canonical_identity_deltas += 1
        arrays["source_action"] = source_action
        arrays["target_action"] = target_action
        arrays["delta_target"] = delta_target
        for key in ("proprio", "agentview", "wrist"):
            if len(arrays[key]) != length:
                raise ValueError(f"{row['state_key']} {key} length mismatch")
        target_npz = output / artifact.name
        save_npz(target_npz, arrays)
        row["artifact"] = str(target_npz)
        row["canonical_action_shape"] = [length, 7]
        write_json(output / source_json.name, row)
        records.append(row)
    migrated = dict(summary)
    migrated.update(
        {
            "schema_version": "rase-e3-step-demos/v2-canonical",
            "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "source_dir": str(source),
            "records": records,
            "canonicalization_audit": {
                "n_complete_artifacts": sum(row.get("status") == "complete" for row in records),
                "n_broadcast_deltas_repaired": repaired_broadcast_deltas,
                "n_identity_zero_delta_verified": canonical_identity_deltas,
                "all_action_arrays_T_by_7": True,
                "delta_recomputed_as_target_minus_source": True,
            },
        }
    )
    write_json(output / "summary.json", migrated)
    print(json.dumps(migrated["canonicalization_audit"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
