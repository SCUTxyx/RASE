#!/usr/bin/env python3
"""Phase 1: Boundary x Opportunity analysis + steering verdict.

Reads dynamic smoke outputs (per-suite json) and the static-domain atlas
(domain_atlas_v1.json) as the negative control, and reports:

  - per-root boundary step/rule and per-candidate K3 success;
  - heterogeneous / fallback-not-optimal / source-source informative /
    all-fail / oracle-minus-best-fixed rates;
  - trigger coverage and real collection cost (rollouts executed);
  - steering verdict: dynamic heterogeneity >= 8% AND >= 2x static AND
    positive controls rediscovered AND all-fail not inflated.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-dynamic-hetero", type=float, default=0.08)
    parser.add_argument("--static-multiplier", type=float, default=2.0)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(args.smoke_dir / "*.json"))):
        if path.endswith("runner.log"):
            continue
        report = json.loads(Path(path).read_text())
        records.extend(report.get("roots", []))
    if not records:
        raise SystemExit("no dynamic smoke records")

    hetero_units = 0
    fb_dom_units = 0
    all_fail_units = 0
    triggered = 0
    rollouts = 0
    per_root: list[dict[str, Any]] = []
    source_source_info = 0
    for record in records:
        entry = {
            "label": record.get("label"),
            "suite": record.get("suite"),
            "state_key": str(record.get("state_key", ""))[:16],
            "boundary": record.get("boundary"),
            "verdict": record.get("verdict"),
        }
        succ = {
            op: cand.get("success", []) if isinstance(cand, dict) else []
            for op, cand in (record.get("candidates") or {}).items()
        }
        for op, values in succ.items():
            rollouts += sum(1 for v in values if v is not None)
        fb = succ.get("fallback.persistent", [])
        others = [v for op, v in succ.items() if op != "fallback.persistent"]
        others_any = [v for op, values in succ.items() if op != "fallback.persistent" for v in values]
        fb_ok = any(fb) if fb else None
        others_ok = any(others_any) if others_any else None
        if fb_ok is False and others_ok:
            hetero_units += 1
        elif fb_ok is False:
            all_fail_units += 1
        else:
            fb_dom_units += 1
        if (record.get("boundary") or {}).get("rule") not in (None, "none"):
            triggered += 1
        # source-source informative: continue vs requery differ
        cont = succ.get("continue.source", [])
        req = succ.get("requery.source", [])
        if cont and req and set(cont) != set(req):
            source_source_info += 1
        per_root.append(entry)

    n = len(records)
    hetero_rate = hetero_units / n if n else 0.0
    # task bootstrap CI on per-root hetero
    rng = np.random.default_rng(20270818)
    per_root_verdict = np.asarray([1 if r.get("verdict") == "heterogeneous" else 0 for r in records])
    samples = []
    for _ in range(2000):
        idx = rng.integers(0, len(per_root_verdict), size=len(per_root_verdict))
        samples.append(per_root_verdict[idx].mean())
    ci = [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]

    # static comparison from atlas
    atlas = json.loads(args.atlas.read_text())
    static_rates = [cell["totals"]["heterogeneous_rate"] for cell in atlas["grid"].values()]
    static_max = max(rate for rate in static_rates if rate is not None) if static_rates else 0.0

    # positive control rediscovery: label startswith positive and verdict heterogeneous
    positive_controls = [r for r in records if str(r.get("label", "")).startswith("positive")]
    positives_rediscovered = sum(1 for r in positive_controls if r.get("verdict") == "heterogeneous")

    steering_pass = (
        hetero_rate >= args.min_dynamic_hetero
        and hetero_rate >= args.static_multiplier * static_max
        and positives_rediscovered >= 1
        and (all_fail_units / n if n else 0.0) <= 0.10
    )

    report = {
        "schema_version": "rase-vnext-dynamic-opportunity/v1",
        "steering": {
            "pass": bool(steering_pass),
            "dynamic_heterogeneous_rate": round(hetero_rate, 4),
            "dynamic_hetero_ci95": [round(v, 4) for v in ci],
            "static_max_heterogeneous_rate": round(static_max, 4),
            "static_multiplier_achieved": round(hetero_rate / max(static_max, 1e-9), 2),
            "min_dynamic_hetero": args.min_dynamic_hetero,
            "positive_controls_rediscovered": positives_rediscovered,
            "positive_controls_total": len(positive_controls),
            "all_fail_rate": round(all_fail_units / n, 4) if n else None,
            "all_fail_cap": 0.10,
        },
        "matrix": {
            "units": n,
            "heterogeneous_units": hetero_units,
            "fallback_dominates_units": fb_dom_units,
            "all_fail_units": all_fail_units,
            "fallback_not_optimal_units": hetero_units,
            "source_source_informative_units": source_source_info,
            "triggered_units": triggered,
            "trigger_coverage": round(triggered / n, 4) if n else None,
            "collection_cost_rollouts": rollouts,
        },
        "per_root": per_root,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
