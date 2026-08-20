#!/usr/bin/env python3
"""Phase 2.5: counterfactual opportunity — do same-root candidates actually
diverge?  (roadmap §9, Gate B)

From the same-root dataset (collect_same_root.py output) computes per-root:
  - candidate action diversity (canonical chunk distance)
  - future divergence: pairwise distance between candidate future trajectories
  - outcome spread: |consequence_label_i - consequence_label_j|
and aggregates:
  - fraction of roots with non-degenerate divergence (Gate B pass)
  - within-state comparative advantage (root has a strictly better candidate)
  - H_within-style signal: does the best candidate flip across decision points
    within the same task? (roadmap §18)

Verdict:
  PASS  -> candidate futures genuinely diverge; proceed to Oracle Future Risk
  FAIL  -> change candidate provider / decision boundary
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rase_common import canonical_chunk_features


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def traj_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-step euclidean distance between two proprio trajectories
    (shape (H,8)), comparing aligned steps."""
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1)))


def object_shift(a: list, b: list) -> float:
    """Mean |Δxyz| over common object bodies between two pose snapshots."""
    da = {n: np.asarray(p, dtype=np.float64) for n, p in (a or [])}
    db = {n: np.asarray(p, dtype=np.float64) for n, p in (b or [])}
    common = set(da) & set(db)
    if not common:
        return float("nan")
    return float(np.mean([np.linalg.norm(da[n] - db[n]) for n in common]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--divergence-threshold", type=float, default=0.05,
                        help="root counts as divergent if mean pairwise "
                             "future distance >= this")
    args = parser.parse_args()

    rows = load_rows(args.data)
    roots: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        roots[(r["task"], r["episode_idx"], r["decision_idx"])].append(r)

    n_roots = len(roots)
    n_multi = 0          # roots with >= 2 candidates
    n_divergent = 0      # roots with non-degenerate future divergence
    n_advantage = 0      # roots with a strictly better candidate (outcome)
    action_div = []
    future_div = []
    object_div = []
    outcome_spread = []
    per_task_flips: dict[str, int] = defaultdict(int)
    per_task_decisions: dict[str, int] = defaultdict(int)

    for key, cands in roots.items():
        task = key[0]
        if len(cands) < 2:
            continue
        n_multi += 1
        per_task_decisions[task] += 1
        # candidate action diversity
        chunks = [np.asarray(c["chunk_raw"], dtype=np.float64).reshape(-1, 7)
                  for c in cands]
        ad = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                ci = canonical_chunk_features(chunks[i])
                cj = canonical_chunk_features(chunks[j])
                ad.append(float(np.linalg.norm(ci - cj)))
        action_div.append(float(np.mean(ad)) if ad else float("nan"))
        # future divergence
        futs = [np.asarray(c["future_proprio"], dtype=np.float64)
                for c in cands if c.get("future_proprio")]
        fd = []
        for i in range(len(futs)):
            for j in range(i + 1, len(futs)):
                fd.append(traj_distance(futs[i], futs[j]))
        fd_mean = float(np.mean(fd)) if fd else float("nan")
        future_div.append(fd_mean)
        # object-level divergence (privileged state; more task-sensitive)
        objs = [c.get("s_th_objects") for c in cands]
        od = []
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                s = object_shift(objs[i], objs[j])
                if s == s:  # not nan
                    od.append(s)
        obj_div_mean = float(np.mean(od)) if od else float("nan")
        object_div.append(obj_div_mean)
        # outcome spread
        cons = [c.get("consequence_label", c.get("displacement", 0.0))
                for c in cands]
        spread = float(np.max(cons)) - float(np.min(cons))
        outcome_spread.append(spread)
        if fd_mean >= args.divergence_threshold:
            n_divergent += 1
        if spread > 1e-6:
            n_advantage += 1
        # best candidate flips within task (across decision points)
        best_models = [c["model"] for c in cands
                       if c.get("consequence_label", c.get("displacement", 0.0))
                       == max(cons)]
        if len(set(best_models)) > 1:
            per_task_flips[task] += 1

    n = max(1, n_multi)
    flip_tasks = sum(1 for t, d in per_task_decisions.items()
                     if per_task_flips[t] > 0 and d >= 2)
    h_within = flip_tasks / max(1, len(per_task_decisions))

    report = {
        "schema": "rase-phase-2.5-counterfactual-opportunity/v1",
        "n_rows": len(rows),
        "n_roots": n_roots,
        "n_multi_candidate_roots": n_multi,
        "fraction_roots_divergent": round(n_divergent / n, 4),
        "fraction_roots_with_advantage": round(n_advantage / n, 4),
        "action_diversity": {
            "mean": round(float(np.nanmean(action_div)), 4) if action_div else None,
            "median": round(float(np.nanmedian(action_div)), 4) if action_div else None,
            "q05": round(float(np.nanpercentile(action_div, 5)), 4) if action_div else None,
        },
        "future_divergence": {
            "mean": round(float(np.nanmean(future_div)), 4) if future_div else None,
            "median": round(float(np.nanmedian(future_div)), 4) if future_div else None,
            "fraction_ge_threshold": round(
                sum(1 for d in future_div if d >= args.divergence_threshold) / n, 4),
        },
        "object_divergence": {
            "mean": round(float(np.nanmean(object_div)), 4) if object_div else None,
            "median": round(float(np.nanmedian(object_div)), 4) if object_div else None,
            "fraction_ge_threshold": round(
                sum(1 for d in object_div if d >= args.divergence_threshold) / n, 4)
            if object_div else None,
        },
        "outcome_spread_mean": round(float(np.mean(outcome_spread)), 4)
        if outcome_spread else None,
        "h_within_tasks": round(h_within, 4),
        "tasks_with_flips": flip_tasks,
        "n_tasks": len(per_task_decisions),
        "gate_b": {
            "threshold": args.divergence_threshold,
            "candidate_diversity_ok":
                bool(np.nanmedian(action_div) > 1e-3) if action_div else False,
            "future_divergence_ok":
                bool(n_divergent / n >= 0.3) if n else False,
            "object_divergence_ok":
                bool((sum(1 for d in object_div
                          if d >= args.divergence_threshold) / n) >= 0.3)
                if object_div else False,
            "within_state_advantage_ok":
                bool(n_advantage / n >= 0.1) if n else False,
        },
    }
    gate = report["gate_b"]
    report["verdict"] = (
        "PASS" if gate["candidate_diversity_ok"] and
        gate["future_divergence_ok"] and gate["within_state_advantage_ok"]
        else "FAIL")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
