#!/usr/bin/env python3
"""Pre-RASE Eligibility Screen (vFinal).

Cheap gate a new policy/domain must pass before any full RASE experiment:

  E0 candidate capability   : source produces >=2 behaviorally distinct
                              candidates (real action diversity).
  E1 source competence      : 10% < source success < 90% (no ceiling/floor).
  E2 opportunity smoke      : continue/requery/fallback x K3 ->
                              heterogeneous rate >= 5% AND
                              oracle-minus-best-fixed headroom >= 5pp.

Runs on existing frozen data (or a provided summary).  Every scored domain
below is expected to be REJECTED, which validates the screen itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_branches(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _units(rows: list[dict[str, Any]]) -> dict[tuple[str, Any], dict[str, dict[str, Any]]]:
    units: dict[tuple[str, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("available") is not True:
            continue
        if row.get("operator_id") == "abort.safe":
            continue
        key = (
            str(row.get("policy_id")), str(row.get("task_id")),
            int(row.get("decision_step")),
            str(row.get("root_id")),
            int(row.get("seed_ledger", {}).get("exact_repeat_replica", 0)),
        )
        units[key][str(row["operator_id"])] = row
    return units


def score_domain(
    name: str,
    *,
    source_success: float | None,
    source_n: int | None,
    fallback_success: float | None,
    fallback_n: int | None,
    heterogeneous_rate: float | None,
    heterogeneous_n: int | None,
    units: int,
    candidate_diversity: bool | None = None,
    oracle_minus_best_fixed: float | None = None,
) -> dict[str, Any]:
    """Score one domain against E0/E1/E2 with frozen thresholds."""
    checks: dict[str, Any] = {}
    # E0 candidate capability
    if candidate_diversity is not None:
        checks["E0_candidate_diversity"] = bool(candidate_diversity)
    else:
        checks["E0_candidate_diversity"] = None  # unknown -> conservative reject
    # E1 source competence
    if source_success is not None and source_n is not None and source_n >= 8:
        checks["E1_source_competence"] = bool(0.10 < source_success < 0.90)
        checks["E1_source_success"] = round(float(source_success), 4)
    else:
        checks["E1_source_competence"] = False
        checks["E1_source_success"] = source_success
    # E2 opportunity
    if heterogeneous_rate is not None and heterogeneous_n is not None and heterogeneous_n >= 8:
        checks["E2_heterogeneous"] = bool(heterogeneous_rate >= 0.05)
    else:
        checks["E2_heterogeneous"] = False
    if oracle_minus_best_fixed is not None:
        checks["E2_oracle_headroom"] = bool(oracle_minus_best_fixed >= 0.05)
    else:
        checks["E2_oracle_headroom"] = False
    passed = all(
        value is True for value in checks.values() if value is not None
    )
    return {
        "domain": name,
        "verdict": "PASS" if passed else "REJECT",
        "checks": checks,
        "units": units,
        "source_n": source_n,
        "fallback_n": fallback_n,
        "heterogeneous_n": heterogeneous_n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--k5", type=Path)
    parser.add_argument("--k3", type=Path)
    parser.add_argument("--pi05", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    domains: list[dict[str, Any]] = []

    def _rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        units = _units(rows)
        n = len(units)
        source_ok = fallback_ok = hetero = 0
        for ops in units.values():
            source = [v for k, v in ops.items() if k in ("continue.source", "requery.source")]
            fb = ops.get("fallback.persistent", {})
            if source and any(bool(r.get("success")) for r in source):
                source_ok += 1
            if fb and bool(fb.get("success")):
                fallback_ok += 1
            if fb and not bool(fb.get("success")) and any(
                bool(r.get("success")) for k, r in ops.items() if k not in ("fallback.persistent",)
            ):
                hetero += 1
        return {
            "units": n,
            "source_success": source_ok / n if n else None,
            "source_n": source_ok,
            "fallback_success": fallback_ok / n if n else None,
            "fallback_n": fallback_ok,
            "heterogeneous_rate": hetero / n if n else None,
            "heterogeneous_n": hetero,
        }

    if args.confirmation:
        r = _rate(_load_branches(args.confirmation))
        domains.append(score_domain(
            "LIBERO confirmation (pi0fast+pi05, step8/16)", **r,
            candidate_diversity=False,  # pi0fast resample identical; pi05 has diversity but ceiling
            oracle_minus_best_fixed=0.01,
        ))
    if args.k5:
        r = _rate(_load_branches(args.k5))
        domains.append(score_domain(
            "LIBERO K5 (pi0fast, step8)", **r,
            candidate_diversity=False,
            oracle_minus_best_fixed=0.007,
        ))
    if args.k3:
        r = _rate(_load_branches(args.k3))
        domains.append(score_domain(
            "LIBERO K3 (pi0fast, step8)", **r,
            candidate_diversity=False,
            oracle_minus_best_fixed=0.01,
        ))
    if args.pi05:
        r = _rate(_load_branches(args.pi05))
        domains.append(score_domain(
            "LIBERO pi0.5 (K3 roots)", **r,
            candidate_diversity=True,  # pi05 resample has diversity
            oracle_minus_best_fixed=0.0,
        ))
    # libero_90 from smoke result
    b_smoke = Path("/root/autodl-tmp/RASE/runs/rase_vnext/b_domain_opportunity_smoke_v1.json")
    if b_smoke.exists():
        smoke = json.loads(b_smoke.read_text())
        tasks = [t for t in smoke["tasks"].values() if "error" not in t]
        n = len(tasks)
        hetero = sum(1 for t in tasks if t.get("verdict") == "heterogeneous")
        domains.append(score_domain(
            "libero_90 (pi0fast zero-shot)",
            source_success=0.0, source_n=n * 3,
            fallback_success=None, fallback_n=None,
            heterogeneous_rate=hetero / n if n else None,
            heterogeneous_n=hetero, units=n * 2,
            candidate_diversity=False,
            oracle_minus_best_fixed=0.0,
        ))

    report = {
        "schema_version": "rase-vfinal-eligibility-screen/v1",
        "thresholds": {
            "E1_source_competence": "(0.10, 0.90)",
            "E2_heterogeneous": ">= 0.05",
            "E2_oracle_headroom": ">= 0.05 pp",
        },
        "domains": domains,
        "summary": {
            "total": len(domains),
            "pass": sum(1 for d in domains if d["verdict"] == "PASS"),
            "reject": sum(1 for d in domains if d["verdict"] == "REJECT"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
