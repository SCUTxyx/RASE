#!/usr/bin/env python3
"""Export dual-oracle state/candidate labels to JSONL for QC and warm-start."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-oracle", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits-output", type=Path, default=None)
    parser.add_argument("--benchmark-splits-output", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=20260727)
    parser.add_argument("--traces-dir", type=Path, default=None)
    args = parser.parse_args()

    from rase.collect.candidates import load_artifact
    from rase.collect.dataset_export import (
        EXPORT_SCHEMA_VERSION,
        build_grouped_benchmark_splits,
        build_recovery_rows,
        split_state_keys,
    )
    from rase.collect.state_pool import StatePool

    summary_path = args.dual_oracle.resolve()
    pool = StatePool(args.pool.resolve())
    candidates_dir = args.candidates_dir.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def metadata_for(state_key):
        return pool.read_state(state_key, load_observations=False).metadata

    def artifact_for(state_key):
        path = candidates_dir / f"{state_key}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, load_artifact(path)

    traces_dir = args.traces_dir.resolve() if args.traces_dir else None
    rows = build_recovery_rows(
        summary,
        metadata_for=metadata_for,
        artifact_for=artifact_for,
        traces_dir=traces_dir,
    )
    _atomic_write(
        args.output.resolve(),
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    splits = split_state_keys(rows)
    splits_path = (
        args.splits_output.resolve()
        if args.splits_output
        else args.output.resolve().with_suffix(".splits.json")
    )
    _atomic_write(
        splits_path,
        json.dumps(
            {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "source": str(summary_path),
                "n_rows": len(rows),
                "n_states": len({row["state_key"] for row in rows}),
                "splits": splits,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    benchmark_splits = build_grouped_benchmark_splits(rows, seed=args.split_seed)
    benchmark_splits_path = (
        args.benchmark_splits_output.resolve()
        if args.benchmark_splits_output
        else args.output.resolve().with_suffix(".benchmark-splits.json")
    )
    _atomic_write(
        benchmark_splits_path,
        json.dumps(
            {
                **benchmark_splits,
                "source": str(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "states": len({row["state_key"] for row in rows}),
                "output": str(args.output.resolve()),
                "splits": str(splits_path),
                "benchmark_splits": str(benchmark_splits_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
