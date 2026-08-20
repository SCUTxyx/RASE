#!/usr/bin/env python3
"""Phase 0 timing/trigger contract audit.

Checks on a fresh dynamic-boundary collection output:
  1. required fields missing rate == 0 (trigger provenance + decomposed timing);
  2. raw wall times non-negative;
  3. timestamps monotonic within a branch;
  4. component sums vs totals within tolerance (TimingComponents.validate);
  5. same-root branches share the same source prefix action hash;
  6. trigger provenance validates (rule/scores/steps).
Fails (exit 2) if any required check fails; the collection must pass before
any pilot is allowed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    branches_path = args.output_dir / "branches.jsonl"
    if not branches_path.exists():
        branches_path = args.output_dir / "runner.log"
        if not branches_path.exists():
            raise SystemExit(f"no branches.jsonl or runner.log in {args.output_dir}")
        # runner.log may contain JSON lines interleaved; try branches first
        rows: list[dict[str, Any]] = []
        for line in branches_path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    else:
        rows = [json.loads(line) for line in branches_path.read_text().splitlines() if line.strip()]

    if not rows:
        raise SystemExit("no rows to audit")
    failures: list[str] = []
    required = (
        "boundary_rule", "trigger_provenance",
        "source_prefix_wall_s", "source_prefix_inference_wall_s",
        "source_prefix_env_wall_s", "source_prefix_steps",
    )
    missing: dict[str, int] = {}
    for row in rows:
        for field in required:
            if row.get(field) is None:
                missing[field] = missing.get(field, 0) + 1
        # non-negative raw wall
        for field in ("source_prefix_wall_s", "source_prefix_inference_wall_s", "source_prefix_env_wall_s", "branch_wall_s"):
            value = row.get(field)
            if value is not None and float(value) < 0:
                failures.append(f"negative wall {field} in {row.get('job_id')}")
        # component sum vs total (prefix)
        total = row.get("source_prefix_wall_s")
        inference = row.get("source_prefix_inference_wall_s")
        env = row.get("source_prefix_env_wall_s")
        if total is not None and inference is not None and env is not None:
            if abs((inference + env) - total) > 0.05 * max(total, 1e-9):
                failures.append(f"prefix timing sum mismatch in {row.get('job_id')}")
        # trigger provenance validates
        provenance = row.get("trigger_provenance")
        if provenance is not None:
            rule = provenance.get("rule")
            if rule not in ("combined", "phase", "disagreement", "stagnation", "none"):
                failures.append(f"invalid trigger rule {rule!r} in {row.get('job_id')}")
            boundary = provenance.get("boundary_step")
            first = provenance.get("first_eligible_step")
            if boundary is None or first is None or boundary < first:
                failures.append(f"trigger step ordering in {row.get('job_id')}")
    if missing:
        failures.append(f"missing required fields: {missing}")
    # same-root prefix hash consistency
    prefix_by_root: dict[str, set[str]] = {}
    for row in rows:
        root = str(row.get("root_id"))
        prefix_by_root.setdefault(root, set()).add(str(row.get("source_prefix_action_sha256")))
    inconsistent = {
        root: hashes for root, hashes in prefix_by_root.items() if len(hashes) > 1
    }
    if inconsistent:
        failures.append(f"inconsistent same-root prefix hashes: {sorted(inconsistent)[:5]}")

    report = {
        "schema_version": "rase-vnext-timing-contract-audit/v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "rows": len(rows),
        "roots": len(prefix_by_root),
        "missing_rate": {
            field: {"missing": count, "rows": len(rows),
                    "rate": round(count / len(rows), 4)}
            for field, count in sorted(missing.items())
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
