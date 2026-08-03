#!/usr/bin/env python3
"""Freeze an outcome-independent 12-state PRE-A0 suite×cell pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

SUITES = ("Spatial", "Object", "Goal", "Long")
CELLS = (("clean", 0), ("camera", 1), ("robot", 1))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _keys_hash(keys: list[str]) -> str:
    raw = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def freeze(source: dict[str, Any], source_path: Path) -> dict[str, Any]:
    if source.get("artifact_version") != "rase-replacement-initial-keys/v1":
        raise ValueError("unexpected Phase 1A initial-key artifact")
    if source.get("selection_uses_outcomes") is not False:
        raise ValueError("source key artifact is not outcome-independent")
    reset = dict(source.get("reset_semantics") or {})
    if (
        int(reset.get("snapshot_policy_step", -1)) != 0
        or int(reset.get("source_actions_before_snapshot", -1)) != 0
    ):
        raise ValueError("source states are not zero-source-action reset states")
    records = [dict(row) for row in source.get("records") or []]
    selected: list[dict[str, Any]] = []
    for suite in SUITES:
        for dimension, level in CELLS:
            matches = [
                row
                for row in records
                if str(row["suite"]) == suite
                and str(row["perturbation_dimension"]) == dimension
                and int(row["perturbation_level"]) == level
            ]
            if not matches:
                raise ValueError(f"empty stratum {suite}/{dimension}:L{level}")
            # Frozen metadata-only selection; outcomes never enter ordering.
            matches.sort(
                key=lambda row: (
                    int(row["design_request_index"]),
                    str(row["task_id"]),
                    str(row["state_key"]),
                )
            )
            selected.append(matches[0])
    keys = [str(row["state_key"]) for row in selected]
    tasks = [str(row["task_id"]) for row in selected]
    episodes = [str(row["episode_id"]) for row in selected]
    if len(keys) != 12 or len(set(keys)) != 12:
        raise ValueError("PRE-A0 requires exactly 12 unique states")
    if len(set(tasks)) != 12 or len(set(episodes)) != 12:
        raise ValueError("PRE-A0 requires unique tasks and episodes")
    return {
        "artifact_version": "rase-pre-a0-state-keys/v1",
        "status": "frozen",
        "use_for": "development-only candidate opportunity pilot",
        "exclude_from_flagship_hidden_test": True,
        "selection_uses_outcomes": False,
        "selection_rule": (
            "minimum design_request_index per suite×(clean:L0,camera:L1,robot:L1)"
        ),
        "selection_fields": [
            "suite",
            "perturbation_dimension",
            "perturbation_level",
            "design_request_index",
            "task_id",
            "state_key",
        ],
        "source": str(source_path.resolve()),
        "source_artifact_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "n_states": len(keys),
        "n_tasks": len(set(tasks)),
        "n_episodes": len(set(episodes)),
        "state_keys": keys,
        "state_keys_sha256": _keys_hash(keys),
        "post_selection_source_outcomes": dict(
            sorted(
                Counter(
                    "success" if bool(row["source_only_success"]) else "failure"
                    for row in selected
                ).items()
            )
        ),
        "records": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(_read(args.source), args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_states": result["n_states"],
                "state_keys_sha256": result["state_keys_sha256"],
                "post_selection_source_outcomes": result[
                    "post_selection_source_outcomes"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

