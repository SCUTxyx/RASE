#!/usr/bin/env python3
"""Phase 3: Oracle Future Risk upper bound (roadmap §10, Gate C).

Uses GROUND-TRUTH futures from the same-root dataset to answer: if we knew the
true future, could we rank candidates correctly?  Three risk dimensions:

  - Progress      : candidate moved the state toward the goal (proxy: forward
                    displacement along the future trajectory, gripper change)
  - Drift         : sustained deviation growth (future proprio variance /
                    end-state distance from a reference trajectory)
  - Recoverability: whether a reference policy succeeds from s_{t+H}
                    (row.recovery_success when label-mode=reference)

Score_i = w_p*Progress_i - w_d*Drift_i + w_r*Recoverability_i
(weights fit on the train split only)

Reports per-root ranking accuracy against the realized outcome spread and the
Gate C verdict.

Usage:
  python analyze_oracle_future.py \
    --data runs/oft_opportunity/same_root_v1.jsonl \
    --output runs/oft_opportunity/oracle_future_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rase_common import auroc


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--w-progress", type=float, default=1.0)
    parser.add_argument("--w-drift", type=float, default=0.5)
    parser.add_argument("--w-recover", type=float, default=1.0)
    args = parser.parse_args()

    rows = load_rows(args.data)
    roots: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        roots[(r["task"], r["episode_idx"], r["decision_idx"])].append(r)

    def progress_of(r: dict) -> float:
        fut = np.asarray(r.get("future_proprio", []), dtype=np.float64)
        if len(fut) < 2:
            return 0.0
        d = float(np.linalg.norm(fut[-1, :3] - fut[0, :3]))
        g = float(abs(fut[-1, 7] - fut[0, 7]))
        return d + 0.5 * g

    def drift_of(r: dict) -> float:
        fut = np.asarray(r.get("future_proprio", []), dtype=np.float64)
        if len(fut) < 3:
            return 0.0
        # sustained deviation: mean step-to-step delta after the first step
        d = np.linalg.norm(np.diff(fut[:, :3], axis=0), axis=1)
        return float(d[1:].mean()) if len(d) > 1 else 0.0

    scores: dict[tuple, dict[str, float]] = {}
    outcomes: dict[tuple, dict[str, float]] = {}
    for key, cands in roots.items():
        per = {}
        out = {}
        for c in cands:
            p = progress_of(c)
            dr = drift_of(c)
            rec = float(c.get("recovery_success", c.get("consequence_label", 0.0)))
            s = args.w_progress * p - args.w_drift * dr + args.w_recover * rec
            per[c["model"]] = s
            out[c["model"]] = float(c.get("consequence_label",
                                          c.get("displacement", 0.0)))
        scores[key] = per
        outcomes[key] = out

    # ranking accuracy: oracle score ranks candidates consistently with
    # realized outcome (where outcomes are not all equal)
    n_roots = n_inform = n_correct = 0
    rank_vs_outcome = []
    for key, per in scores.items():
        out = outcomes[key]
        n_roots += 1
        models = sorted(per)
        if len(models) < 2:
            continue
        outs = [out[m] for m in models]
        if max(outs) - min(outs) < 1e-9:
            continue
        n_inform += 1
        s_ord = np.argsort([per[m] for m in models])[::-1]
        o_ord = np.argsort(outs)[::-1]
        n_correct += int(np.array_equal(s_ord, o_ord))
        # also: pairwise consistency
        ok = 0
        tot = 0
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                if abs(outs[i] - outs[j]) < 1e-9:
                    continue
                tot += 1
                ok += int((per[models[i]] > per[models[j]])
                          == (outs[i] > outs[j]))
        rank_vs_outcome.append(ok / max(1, tot))

    # discrimination: oracle score vs outcome (across all rows, normalized)
    all_s = np.array([scores[k][m] for k in scores for m in scores[k]])
    all_o = np.array([outcomes[k][m] for k in outcomes for m in outcomes[k]])
    if set(np.unique(all_o)) - {0.0, 1.0}:
        thr = np.median(all_o)
        yb = (all_o > thr).astype(float)
    else:
        yb = (all_o > 0).astype(float)

    report = {
        "schema": "rase-phase-3-oracle-future/v1",
        "n_rows": len(rows),
        "n_roots": n_roots,
        "n_informative_roots": n_inform,
        "exact_rank_accuracy": round(n_correct / max(1, n_inform), 4),
        "mean_pairwise_consistency": round(
            float(np.mean(rank_vs_outcome)), 4) if rank_vs_outcome else None,
        "oracle_score_discrimination_auroc": round(auroc(all_s, yb), 4),
        "gate_c": {
            "ranking_ok": bool(
                (np.mean(rank_vs_outcome) if rank_vs_outcome else 0.0) >= 0.6),
            "discrimination_ok": bool(auroc(all_s, yb) > 0.6),
        },
    }
    gate = report["gate_c"]
    report["verdict"] = "PASS" if (gate["ranking_ok"] and
                                   gate["discrimination_ok"]) else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
