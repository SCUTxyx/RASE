#!/usr/bin/env python3
"""A0: rebuild the real per-candidate cost ledger from branch/capture records.

Every (root, replica, operator) gets one row with the raw cost components and
their provenance:

  - sunk prefix cost (shared by all candidates, never re-penalized);
  - incremental decision cost from the decision point onward:
      requery extra inference, fallback pre-query count, fallback inference
      steps, branch wall time, end-to-end env steps;
  - normalized protocol costs (query/fallback/latency) preserved as-is;
  - wall-time and normalized cost both kept, with units and timing boundaries.

Missing fields are reported with a missing-rate table, never invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    branches_path = args.output_dir / "branches.jsonl"
    if not branches_path.exists():
        raise SystemExit(f"missing {branches_path}")
    rows = [json.loads(line) for line in branches_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("empty branches.jsonl")

    # Group by (root, replica).
    by_unit: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("capability_status") != "executable":
            continue
        if row.get("execution_status") == "not_selected":
            continue
        key = (str(row["root_id"]), int(row["seed_ledger"]["exact_repeat_replica"]))
        by_unit[key][str(row["operator_id"])] = row

    ledger: list[dict[str, Any]] = []
    missing: dict[str, int] = defaultdict(int)
    field_total: dict[str, int] = defaultdict(int)
    for unit, operators in sorted(by_unit.items()):
        cont = operators.get("continue.source")
        for operator, row in sorted(operators.items()):
            entry: dict[str, Any] = {
                "root_id": unit[0],
                "replica": unit[1],
                "task_id": str(row["task_id"]),
                "suite": str(row["suite"]),
                "operator": operator,
                "success": bool(row.get("success")),
            }
            # --- sunk prefix (shared; report only, never re-penalized) ---
            entry["source_prefix_steps"] = int(row.get("source_prefix_steps") or 0)
            prefix_wall = row.get("source_prefix_wall_s")
            field_total["source_prefix_wall_s"] += 1
            if prefix_wall is None:
                missing["source_prefix_wall_s"] += 1
            entry["source_prefix_wall_s"] = prefix_wall

            # --- incremental decision cost ---
            entry["intervention_query_count"] = int(row.get("intervention_query_count") or 0)
            entry["fallback_steps"] = int(row.get("fallback_steps") or 0)
            entry["post_decision_env_steps"] = int(row.get("post_decision_env_steps") or 0)
            branch_wall = row.get("branch_wall_s")
            field_total["branch_wall_s"] += 1
            if branch_wall is None:
                missing["branch_wall_s"] += 1
            entry["branch_wall_s"] = branch_wall
            # incremental wall time relative to continue (same unit) if present
            if cont is not None and branch_wall is not None and cont.get("branch_wall_s") is not None:
                entry["incremental_wall_s_vs_continue"] = round(
                    float(branch_wall) - float(cont["branch_wall_s"]), 6,
                )
            else:
                entry["incremental_wall_s_vs_continue"] = None
            # --- normalized protocol costs (kept as-is) ---
            for name in ("query_cost", "fallback_cost", "latency_cost"):
                value = row.get(name)
                field_total[name] += 1
                if value is None:
                    missing[name] += 1
                entry[name] = value
            entry["utility"] = row.get("utility")
            entry["incremental_cost"] = round(
                float(row.get("query_cost") or 0.0)
                + float(row.get("fallback_cost") or 0.0)
                + float(row.get("latency_cost") or 0.0), 6,
            )
            ledger.append(entry)

    report = {
        "schema_version": "rase-vnext-cost-ledger/v1",
        "status": "frozen",
        "source_branches": str(branches_path.resolve()),
        "source_branches_sha256": sha256(branches_path),
        "rows_total": len(rows),
        "ledger_rows": len(ledger),
        "units": len(by_unit),
        "timing_boundaries": {
            "branch_wall_s": "per-branch wall time inside collector (rollout phase); "
                             "prefix wall not persisted in rows (missing)",
            "incremental_wall_s_vs_continue": "branch_wall_s(operator) - branch_wall_s(continue) "
                                              "on the same (root, replica)",
            "units": "seconds for wall fields; steps for env/fallback steps; "
                     "dimensionless normalized costs from the frozen protocol",
        },
        "missing_rate": {
            name: {"missing": count, "total": field_total[name],
                   "rate": round(count / field_total[name], 4) if field_total[name] else 0.0}
            for name, count in sorted(missing.items())
        },
        "sunk_vs_incremental": (
            "source prefix cost is shared by all candidates and reported only; "
            "incremental_cost = query + fallback + latency normalized costs "
            "incurred from the decision point onward"
        ),
        "ledger": ledger,
    }
    atomic_json(args.output, report)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "ledger_rows": len(ledger),
        "units": len(by_unit),
        "missing_rate": report["missing_rate"],
        "samples": ledger[:2],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
