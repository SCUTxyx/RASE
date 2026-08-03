#!/usr/bin/env python3
"""Run the preregistered same-state operator opportunity gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--continue-operator-id", default=None)
    parser.add_argument("--min-complete-snapshots", type=int, default=20)
    parser.add_argument("--min-oracle-gap", type=float, default=0.05)
    parser.add_argument("--min-winning-operators", type=int, default=3)
    parser.add_argument("--min-tasks-per-winning-operator", type=int, default=2)
    parser.add_argument("--min-repeats-per-arm", type=int, default=1)
    parser.add_argument("--allow-missing-continue", action="store_true")
    parser.add_argument("--allow-zero-harm", action="store_true")
    parser.add_argument("--allow-zero-futility", action="store_true")
    args = parser.parse_args()

    from rase.interventions.dataset import OpportunityGate, opportunity_audit, parse_registry
    from rase.interventions.schema import InterventionOutcome, InterventionSnapshot

    specs = parse_registry(_read_json(args.registry.resolve()))
    snapshots = [InterventionSnapshot.from_dict(row) for row in _read_jsonl(args.snapshots)]
    outcomes = [InterventionOutcome.from_dict(row) for row in _read_jsonl(args.outcomes)]
    gate = OpportunityGate(
        min_complete_snapshots=args.min_complete_snapshots,
        min_oracle_gap=args.min_oracle_gap,
        min_winning_operators=args.min_winning_operators,
        min_tasks_per_winning_operator=args.min_tasks_per_winning_operator,
        min_repeats_per_arm=args.min_repeats_per_arm,
        require_continue=not args.allow_missing_continue,
        require_harm=not args.allow_zero_harm,
        require_futility=not args.allow_zero_futility,
    )
    result = opportunity_audit(
        snapshots,
        outcomes,
        specs,
        gate=gate,
        success_reward=args.success_reward,
        continue_operator_id=args.continue_operator_id,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_complete_snapshots": result["n_complete_snapshots"],
                "oracle_minus_best_fixed": result["same_state"][
                    "oracle_minus_best_fixed"
                ],
                "reasons": result["reasons"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "ready_for_method" else 2


if __name__ == "__main__":
    raise SystemExit(main())
