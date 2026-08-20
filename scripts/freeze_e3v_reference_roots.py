#!/usr/bin/env python3
"""Freeze an outcome-selected development cohort for E3-V reference viability.

E3-V is a training-data diagnostic, so selecting roots where the frozen source
failed is intentional.  The artifact records that outcomes were used and must
never be relabelled as a held-out eligibility/test cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


SCHEMA = "rase-e3v-reference-roots/v1"


def canonical_sha256(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def balanced_records(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Round-robin tasks, prioritizing roots missed by the single reference."""
    by_task: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(
        rows,
        key=lambda item: (
            bool(item["single_reference_success"]),
            int(item.get("step", 0)),
            str(item["state_key"]),
        ),
    ):
        by_task[str(row["task_id"])].append(row)
    selected: list[dict[str, Any]] = []
    task_ids = sorted(by_task)
    while task_ids and (limit <= 0 or len(selected) < limit):
        remaining: list[str] = []
        for task_id in task_ids:
            queue = by_task[task_id]
            if queue and (limit <= 0 or len(selected) < limit):
                selected.append(queue.popleft())
            if queue:
                remaining.append(task_id)
        task_ids = remaining
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opportunity", type=Path, required=True)
    parser.add_argument(
        "--source-summary",
        type=Path,
        help="optional Smol summary used when the opportunity rows omit source outcomes",
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--rollouts-per-state", type=int, default=4)
    parser.add_argument("--policy-id", default="pi0fast_libero")
    args = parser.parse_args()
    if args.max_states < 0 or args.rollouts_per_state < 1:
        raise ValueError("invalid cohort size or rollout count")

    source = read_json(args.opportunity.resolve())
    source_outcomes: dict[str, bool] = {}
    if args.source_summary:
        source_summary = read_json(args.source_summary.resolve())
        for row in source_summary.get("per_pair") or []:
            source_outcomes[str(row["state_key"])] = bool(
                row.get("continue_smol_active_chunk")
            )
    source_rows = source.get("per_state") or []
    # Phase0g stores the fully joined three-arm rows inside its analysis block.
    # Prefer the top-level layout when it already carries source outcomes.
    if not source_rows or not any(
        "continue_success" in row or "continue_smol_active_chunk_success" in row
        for row in source_rows
    ):
        source_rows = (
            ((source.get("three_operator") or {}).get("overall") or {}).get("per_state")
            or source_rows
        )
    pool_manifest = read_json(args.pool.resolve() / "manifest.json")
    pool_states = dict(pool_manifest.get("states") or {})
    rows: list[dict[str, Any]] = []
    for item in source_rows:
        state_key = str(item["state_key"])
        explicit_source = item.get(
            "continue_success", item.get("continue_smol_active_chunk_success")
        )
        if explicit_source is None and state_key not in source_outcomes:
            raise ValueError(
                f"source outcome missing for {state_key}; provide --source-summary"
            )
        source_success = bool(
            source_outcomes[state_key] if explicit_source is None else explicit_source
        )
        if source_success:
            continue
        reference_success = bool(
            item.get("fallback_success", item.get("direct_oft_success", False))
        )
        pool_item = dict(pool_states.get(state_key) or {})
        task_id = str(item.get("task_id") or pool_item.get("task_id") or "")
        episode_id = str(item.get("episode_id") or pool_item.get("episode_id") or "")
        if not task_id or not episode_id:
            raise ValueError(f"missing pool metadata for {state_key}")
        rows.append(
            {
                "state_key": state_key,
                "task_id": task_id,
                "episode_id": episode_id,
                "suite": str(item["suite"]),
                "step": int(item.get("step", pool_item.get("step", 0))),
                "source_success": False,
                "single_reference_success": reference_success,
                "single_reference_stop_reason": str(
                    item.get("fallback_stop_reason")
                    or next(
                        (
                            arm.get("stop_reason", "")
                            for arm in item.get("arms") or []
                            if arm.get("arm_label") == "direct_oft"
                        ),
                        "",
                    )
                ),
            }
        )
    records = balanced_records(rows, args.max_states)
    if not records:
        raise ValueError("no source-failure roots were selected")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "frozen",
        "scientific_scope": "development_only_reference_viability",
        "selection_uses_outcomes": True,
        "selection_rule": "source_failure; task_round_robin; single_reference_failure_first",
        "source_opportunity": str(args.opportunity.resolve()),
        "pool": str(args.pool.resolve()),
        "policy_id": args.policy_id,
        "rollouts_per_state": args.rollouts_per_state,
        "n_states": len(records),
        "n_tasks": len({row["task_id"] for row in records}),
        # Compatibility with the existing OFT trajectory generator.  Records
        # remain canonical; this is only their ordered key projection.
        "state_keys": [row["state_key"] for row in records],
        "records": records,
        "records_sha256": canonical_sha256(records),
    }
    payload["protocol_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("n_states", "n_tasks", "protocol_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
