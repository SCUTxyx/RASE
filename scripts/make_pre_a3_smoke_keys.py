#!/usr/bin/env python3
"""Build a plumbing-only smoke key artifact from the frozen PRE-A0 12-state set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "runs/rase_pre_a0_strict_resample12_keys_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/rase_pre_a3_smoke12_keys_v1.json",
    )
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    records = list(source["records"])[: args.n]
    # Assign synthetic splits only for plumbing; never confirmatory.
    for index, row in enumerate(records):
        row = dict(row)
        row["split"] = ("train", "val", "test", "train")[index % 4]
        row["cell"] = f"{row['perturbation_dimension']}:L{row['perturbation_level']}"
        records[index] = row
    payload = {
        "artifact_version": "rase-pre-a3-state-keys/v1",
        "selection_uses_outcomes": False,
        "exclude_from_flagship_hidden_test": True,
        "plumbing_only": True,
        "n_states": len(records),
        "n_tasks": len({row["task_id"] for row in records}),
        "pool": source.get("source") or "pre_a0_dev",
        "state_keys": [row["state_key"] for row in records],
        "records": records,
        "durations": [0, 8, 16, 32, 64, 96, 128],
        "execution_mode": "live_closed_loop_oft_prefix",
        "note": "Plumbing smoke only; not usable for confirmatory PRE-A3 claims.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "n_states": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
