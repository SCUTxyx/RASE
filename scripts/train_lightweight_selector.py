#!/usr/bin/env python3
"""Audit, train, and evaluate the lightweight three-arm escalation selector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{number}: {error}") from error
    return rows


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--min-train-states", type=int, default=30)
    args = parser.parse_args()

    from rase.selector.lightweight import (
        audit_selector_dataset,
        evaluate_selector,
        fit_lightweight_selector,
    )

    rows = _read_jsonl(args.dataset)
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    audit = audit_selector_dataset(
        rows,
        splits,
        success_reward=args.success_reward,
        min_train_states=args.min_train_states,
    )
    _write(args.output_dir / "readiness_audit.json", audit.to_dict())
    if not audit.ready:
        print(json.dumps(audit.to_dict(), indent=2), flush=True)
        print("NOT_READY: no model was trained; see readiness_audit.json", flush=True)
        return 2

    split_map = splits.get("splits", splits)
    row_by_key = {str(row["state_key"]): row for row in rows}
    train_rows = [row_by_key[str(key)] for key in split_map["train"]]
    model = fit_lightweight_selector(
        train_rows,
        ridge=args.ridge,
        success_reward=args.success_reward,
    )
    _write(args.output_dir / "model.json", model.to_dict())
    metrics = {
        split: evaluate_selector(
            model, [row_by_key[str(key)] for key in split_map.get(split, [])]
        )
        for split in ("train", "val", "test")
    }
    _write(args.output_dir / "metrics.json", metrics)
    print(
        json.dumps(
            {
                "status": "complete",
                "n_parameters": model.n_parameters,
                "test": metrics.get("test"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
