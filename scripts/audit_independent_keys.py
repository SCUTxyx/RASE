#!/usr/bin/env python3
"""Audit frozen decision keys against an independent collection design."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _checksum(keys: list[str]) -> str:
    raw = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def audit(
    keys: dict[str, Any], design: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    records = list(keys.get("records") or [])
    state_keys = [str(value) for value in keys.get("state_keys") or []]
    protocol = config["protocol"]
    expected_n = int(protocol["expected_states"])
    expected_step = int(protocol["decision_step"])
    expected_suffix = int(protocol["expected_suffix_steps"])
    planned = {
        (
            str(row["episode_id"]),
            str(row["task_id"]),
            str(row["suite"]),
            str(row["dimension"]),
            int(row["level"]),
        )
        for row in design.get("records") or []
    }
    observed = {
        (
            str(row["episode_id"]),
            str(row["task_id"]),
            str(row["suite"]),
            str(row["perturbation_dimension"]),
            int(row["perturbation_level"]),
        )
        for row in records
    }
    tasks = [str(row["task_id"]) for row in records]
    episodes = [str(row["episode_id"]) for row in records]
    reasons = []
    if design.get("status") != "ready":
        reasons.append("frozen design is not ready")
    if len(records) != expected_n or len(state_keys) != expected_n:
        reasons.append(f"expected {expected_n} states")
    invalid_keys = len(state_keys) != len(set(state_keys))
    invalid_checksum = keys.get("state_keys_sha256") != _checksum(state_keys)
    if invalid_keys or invalid_checksum:
        reasons.append("state keys are duplicate or checksum-invalid")
    if len(tasks) != len(set(tasks)):
        reasons.append("task ids are not unique")
    if len(episodes) != len(set(episodes)):
        reasons.append("episode ids are not unique")
    if observed != planned:
        reasons.append("observed task/episode/cell identities differ from frozen design")
    if any(int(row["step"]) != expected_step for row in records):
        reasons.append("decision step differs from preregistered step")
    if any(int(row["suffix_steps"]) != expected_suffix for row in records):
        reasons.append("active suffix length differs from preregistered length")
    return {
        "schema_version": "rase-independent-keys-audit/v1",
        "status": "ready" if not reasons else "not_ready",
        "reasons": reasons,
        "n_states": len(records),
        "n_unique_tasks": len(set(tasks)),
        "n_unique_episodes": len(set(episodes)),
        "step_counts": dict(sorted(Counter(int(row["step"]) for row in records).items())),
        "suffix_length_counts": dict(
            sorted(Counter(int(row["suffix_steps"]) for row in records).items())
        ),
        "design_identity_match": observed == planned,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(_read(args.state_keys_json), _read(args.design), _read(args.config))
    _write(args.output.resolve(), result)
    print(json.dumps({"output": str(args.output.resolve()), **result}, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
