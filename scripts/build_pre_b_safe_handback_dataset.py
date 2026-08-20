#!/usr/bin/env python3
"""Build PRE-B safe-handback labels from a passed PRE-A3 duration audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_dataset(
    audit: dict[str, Any],
    *,
    require_gate_pass: bool = True,
) -> dict[str, Any]:
    if require_gate_pass and not audit.get("gate_pass"):
        raise ValueError(
            "refusing to build PRE-B dataset while termination gate is closed: "
            f"status={audit.get('status')}"
        )
    rows = []
    for state in audit["per_state"]:
        outcomes = state["outcomes"]
        base = bool(state["base_success"])
        for duration, success in outcomes.items():
            h = int(duration)
            if h == 0:
                continue
            rows.append(
                {
                    "state_key": state["state_key"],
                    "task_id": state["task_id"],
                    "suite": state["suite"],
                    "cell": state["cell"],
                    "split": state.get("split", "train"),
                    "elapsed_recovery_steps": h,
                    "handback_success": bool(success),
                    "base_success_at_h0": base,
                    "false_handback_harm": bool(base and not success),
                    "rescue": bool((not base) and success),
                    "direct_oft_success": bool(state["direct_oft_success"]),
                    "label": "handback_ok" if success else "continue_recovery",
                }
            )
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    return {
        "schema_version": "rase-pre-b-safe-handback/v1",
        "n_rows": len(rows),
        "n_states": len(audit["per_state"]),
        "source_audit_status": audit.get("status"),
        "split_counts": split_counts,
        "label_definition": (
            "handback_ok iff frozen base succeeds after live OFT prefix of "
            "elapsed_recovery_steps"
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-unpassed-gate",
        action="store_true",
        help="Dev-only escape hatch; never use for paper claims.",
    )
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    dataset = build_dataset(audit, require_gate_pass=not args.allow_unpassed_gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "n_rows": dataset["n_rows"],
                "n_states": dataset["n_states"],
                "split_counts": dataset["split_counts"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
