#!/usr/bin/env python3
"""Audit corrected same-root recoverability labels before CRR training.

The primary target is a pre-registered multi-horizon survival value:

    q_recovery = mean_K 1[task recovered within K reference steps]

This avoids selecting one favorable cutoff after seeing results and keeps the
label task-relevant.  Candidate execution is required to be one native chunk
from the common root, followed by a closed-loop reference continuation from
the branch endpoint s_{t+H}.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def root_key(row):
    return (row["task"], row["episode_idx"], row["decision_idx"])


def recovered_by(row, budget):
    return int(bool(row.get("candidate_success")) or
               (bool(row.get("recovery_success")) and
                int(row.get("reference_steps_used", budget + 1)) <= budget))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--budgets", default="64,128,256")
    args = ap.parse_args()

    rows = [json.loads(x) for x in args.data.read_text().splitlines()]
    budgets = [int(x) for x in args.budgets.split(",")]
    roots = {}
    for r in rows:
        roots.setdefault(root_key(r), []).append(r)

    bad_start = sum(r.get("reference_start_state") != "s_t_plus_h" for r in rows)
    bad_mode = sum(r.get("candidate_rollout_mode") != "single_chunk" for r in rows)
    bad_len = sum(int(r.get("future_steps", 0)) >
                  int(r.get("candidate_native_chunk_len", 0)) for r in rows)

    by_budget = {}
    for budget in budgets:
        y = np.array([recovered_by(r, budget) for r in rows])
        informative = 0
        for rs in roots.values():
            vals = [recovered_by(r, budget) for r in rs]
            informative += int(len(set(vals)) > 1)
        by_budget[str(budget)] = {
            "positive_rate": float(y.mean()),
            "n_positive": int(y.sum()),
            "n_rows": len(rows),
            "informative_roots": informative,
            "informative_root_fraction": float(informative / len(roots)),
        }

    q = np.array([np.mean([recovered_by(r, k) for k in budgets]) for r in rows])
    q_by_key = {(root_key(r), r["model"]): float(v) for r, v in zip(rows, q)}

    # Does the legacy displacement proxy preserve the task-relevant ranking?
    agree, comparable = 0, 0
    root_q_spans = []
    task_best_sets = {}
    task_root_q = {}
    oracle_gain = []
    nonfavorite_wins = 0
    for rk, rs in roots.items():
        qv = {r["model"]: q_by_key[(rk, r["model"])] for r in rs}
        root_q_spans.append(max(qv.values()) - min(qv.values()))
        top = tuple(sorted(m for m, v in qv.items()
                           if abs(v - max(qv.values())) < 1e-12))
        task_best_sets.setdefault(rk[0], []).append(top)
        task_root_q.setdefault(rk[0], []).append(qv)
        suite = rs[0]["suite"]
        fav = {"libero_spatial": "oft_spatial",
               "libero_object": "oft_object"}.get(suite)
        if fav in qv:
            oracle_gain.append(max(qv.values()) - qv[fav])
            nonfavorite_wins += int(any(m != fav and v > qv[fav] + 1e-12
                                        for m, v in qv.items()))
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                ri, rj = rs[i], rs[j]
                dq = qv[ri["model"]] - qv[rj["model"]]
                if abs(dq) < 1e-12:
                    continue
                di = float(np.linalg.norm(np.asarray(ri["s_th_proprio"])[:3] -
                                          np.asarray(ri["s_t_proprio"])[:3]))
                dj = float(np.linalg.norm(np.asarray(rj["s_th_proprio"])[:3] -
                                          np.asarray(rj["s_t_proprio"])[:3]))
                comparable += 1
                agree += int((di > dj) == (dq > 0))

    per_task = {}
    flips = []
    task_best_fixed_gains = []
    for task, sets in task_best_sets.items():
        # Strict H_within excludes ties: tie-vs-winner is not evidence that a
        # different candidate is actually better at runtime.
        unique = [s[0] for s in sets if len(s) == 1]
        pairs = [(i, j) for i in range(len(unique))
                 for j in range(i + 1, len(unique))]
        flip = (sum(unique[i] != unique[j] for i, j in pairs) / len(pairs)) \
            if pairs else float("nan")
        if not np.isnan(flip):
            flips.append(flip)
        qrows = task_root_q[task]
        models = sorted(set().union(*(set(x) for x in qrows)))
        mean_by_model = {m: float(np.mean([x[m] for x in qrows if m in x]))
                         for m in models}
        best_fixed = max(mean_by_model, key=mean_by_model.get)
        fixed_gain = float(np.mean([max(x.values()) - x[best_fixed]
                                    for x in qrows]))
        task_best_fixed_gains.append(fixed_gain)
        per_task[task] = {"n_roots": len(sets),
                          "best_set_counts": {str(k): v for k, v in Counter(sets).items()},
                          "n_unique_winner_roots": len(unique),
                          "strict_pairwise_flip_fraction": flip,
                          "best_fixed_model": best_fixed,
                          "state_oracle_gain_over_best_fixed": fixed_gain}

    nondegenerate = any(0.10 <= v["positive_rate"] <= 0.90 and
                        v["informative_root_fraction"] >= 0.10
                        for v in by_budget.values())
    provenance_ok = bad_start == 0 and bad_mode == 0 and bad_len == 0
    rep = {
        "schema": "rase-corrected-recovery-label-audit/v1",
        "n_rows": len(rows), "n_roots": len(roots), "budgets": budgets,
        "provenance": {"bad_reference_start": bad_start,
                       "bad_candidate_rollout_mode": bad_mode,
                       "candidate_steps_exceed_native_chunk": bad_len,
                       "pass": provenance_ok},
        "by_budget": by_budget,
        "q_recovery": {
            "definition": "mean_K I(recovered within K), K pre-registered",
            "unique_values": sorted(float(x) for x in np.unique(q)),
            "informative_root_fraction": float(np.mean(np.asarray(root_q_spans) > 0)),
            "oracle_gain_over_suite_favorite_mean": float(np.mean(oracle_gain))
            if oracle_gain else None,
            "nonfavorite_strict_win_fraction": float(nonfavorite_wins / len(roots)),
            "state_oracle_gain_over_task_best_fixed_mean":
                float(np.mean(task_best_fixed_gains)) if task_best_fixed_gains else None,
            "legacy_displacement_pairwise_agreement": float(agree / comparable)
            if comparable else None,
            "n_comparable_pairs": comparable,
        },
        "within_task": {"mean_strict_pairwise_flip_fraction": float(np.mean(flips))
                        if flips else None, "per_task": per_task},
        "label_pilot_gate": {
            "verdict": "PASS" if provenance_ok and nondegenerate else "FAIL",
            "requirements": "provenance PASS and at least one pre-registered "
                            "budget with 10%-90% positives and >=10% informative roots",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"label_pilot_gate": rep["label_pilot_gate"],
                      "by_budget": by_budget,
                      "q_recovery": rep["q_recovery"],
                      "within_task_mean": rep["within_task"]["mean_strict_pairwise_flip_fraction"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
