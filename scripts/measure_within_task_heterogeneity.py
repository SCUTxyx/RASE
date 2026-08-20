#!/usr/bin/env python3
"""P2: within-task opportunity mining on same-root data.

Measures:
  H_within   = P(argmax_i q_i changes between two roots of the same task)
               (per-task fraction of root pairs with different argmax,
               averaged over tasks; bootstrap CI over tasks)
  argmax model distributions per task
  oracle gain decomposition (state-level vs task-level vs suite-level oracle)
  state-level oracle gain over suite-favorite default

Gate (per plan §11): H_within >= 5% AND OracleGain_within >= 5pp -> PASS.
The current dataset has no binary success/recovery outcome, so it can measure
H_within and q-space headroom but not the second criterion in percentage
points.  If H_within fails, the joint gate is still definitively FAIL;
otherwise it is INDETERMINATE until success-labelled rollouts are available.
Expectation: Gate B already measured h_within = 0 on this data.

Usage (server, oft env):
  python measure_within_task_heterogeneity.py \
    --data runs/oft_opportunity/same_root_w1.jsonl \
    --output runs/oft_opportunity/crr_p2_heterogeneity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crr_common import load_rows, group_roots, q_scores


def argmax_per_root(roots: dict, q: np.ndarray, by_key: dict) -> dict:
    out = {}
    for rk, rs in roots.items():
        idx = [by_key[(root_key(r), r["model"])] for r in rs]
        qs = q[idx]
        best = rs[int(np.argmax(qs))]["model"]
        out[rk] = best
    return out


def root_key(row: dict) -> tuple:
    return (row["task"], row["episode_idx"], row["decision_idx"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--w2", type=float, default=0.5)
    ap.add_argument("--bootstrap-n", type=int, default=2000)
    args = ap.parse_args()

    rows = load_rows(args.data)
    roots = group_roots(rows)
    q = q_scores(rows, w2=args.w2)
    by_key = {(root_key(r), r["model"]): i for i, r in enumerate(rows)}
    best = argmax_per_root(roots, q, by_key)

    # per-task argmax stability
    tasks = {}
    for rk, b in best.items():
        tasks.setdefault(rk[0], []).append((rk, b))

    per_task = {}
    h_within_global = []
    for task, items in sorted(tasks.items()):
        models = [b for _, b in items]
        pairs = [(i, j) for i in range(len(models)) for j in range(i + 1, len(models))]
        diff = sum(1 for i, j in pairs if models[i] != models[j]) if pairs else 0
        frac = diff / len(pairs) if pairs else float("nan")
        h_within_global.append((task, frac, len(items)))
        from collections import Counter
        per_task[task] = {
            "n_roots": len(items),
            "argmax_counts": dict(Counter(models)),
            "pairwise_flip_fraction": frac,
        }

    # bootstrap over tasks (roots within task are the unit of sampling is
    # complex; use task-level bootstrap of the per-task flip fraction)
    rng = np.random.default_rng(0)
    vals = np.array([f for _, f, _ in h_within_global if not np.isnan(f)])
    stats = []
    for _ in range(args.bootstrap_n):
        v = rng.choice(vals, size=len(vals), replace=True)
        stats.append(float(v.mean()))
    h_mean = float(vals.mean()) if len(vals) else float("nan")
    h_ci = [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

    # oracle gain decomposition on q (z-units)
    # suite favorite: spatial -> oft_spatial, object -> oft_object
    fav_gain = []      # state-level oracle over suite-favorite
    task_oracle_gain = []  # task-level oracle over suite-favorite
    for rk, rs in roots.items():
        suite = rs[0]["suite"]
        fav = "oft_spatial" if suite == "libero_spatial" else "oft_object"
        qs = {r["model"]: q[by_key[(root_key(r), r["model"])]] for r in rs}
        q_fav = qs[fav]
        q_best_state = max(qs.values())
        fav_gain.append(q_best_state - q_fav)
    # task-level oracle: best fixed model per task
    for task, items in tasks.items():
        task_models = {}
        for rk, b in items:
            suite = None
            for r in roots[rk]:
                suite = r["suite"]
            task_models.setdefault(suite, []).append(b)
        # only for tasks with >=2 roots
        for suite, ms in task_models.items():
            if len(ms) < 2:
                continue
            fav = "oft_spatial" if suite == "libero_spatial" else "oft_object"
            # best per task by mean q
            qsum = {}
            for rk, b in items:
                suite_r = next(r for r in roots[rk])["suite"]
                if suite_r != suite:
                    continue
                qs = {r["model"]: q[by_key[(root_key(r), r["model"])]]
                      for r in roots[rk]}
                for m, v in qs.items():
                    qsum.setdefault(m, []).append(v)
            best_m = max(qsum, key=lambda m: float(np.mean(qsum[m])))
            for rk, b in items:
                suite_r = roots[rk][0]["suite"]
                if suite_r != suite:
                    continue
                qs = {r["model"]: q[by_key[(root_key(r), r["model"])]]
                      for r in roots[rk]}
                task_oracle_gain.append(qs[best_m] - qs[fav])

    rep = {
        "schema": "rase-p2-within-task-heterogeneity/v1",
        "n_rows": len(rows), "n_roots": len(roots),
        "q": {"w2": args.w2},
        "h_within": {
            "definition": "P(argmax_i q_i differs between two roots of the "
                          "same task); bootstrap over tasks",
            "per_task": per_task,
            "mean": h_mean,
            "ci95": h_ci,
        },
        "oracle_gain_vs_suite_favorite": {
            "state_level_mean": float(np.mean(fav_gain)),
            "task_level_mean": float(np.mean(task_oracle_gain))
                              if task_oracle_gain else float("nan"),
            "n_roots_state": len(fav_gain),
            "n_roots_task": len(task_oracle_gain),
        },
        "gate_p2": {
            "verdict": "FAIL" if h_mean < 0.05 else "INDETERMINATE",
            "threshold_h_within": 0.05,
            "threshold_oracle_gain_success_pp": 5.0,
            "oracle_gain_success_pp": None,
            "note": ("h_within<5% -> current domain is task-level routing; "
                     "per plan: change domain or shrink claim (no P3). The "
                     "dataset has no success/recovery label, so q-space gain "
                     "must not be reported as success percentage points."),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "h_within_mean": h_mean, "h_within_ci95": h_ci,
        "gate_p2": rep["gate_p2"]["verdict"],
        "state_oracle_gain": rep["oracle_gain_vs_suite_favorite"]["state_level_mean"],
        "task_oracle_gain": rep["oracle_gain_vs_suite_favorite"]["task_level_mean"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
