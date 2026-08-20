#!/usr/bin/env python3
"""R6-C.1A: early-window model-free opportunity audit.

WARNING (methodological red line #1): this audit only measures *opportunity*.
No learned model is involved, so nothing here establishes feature
separability.  Separability must be answered by the task-held-out probe in
R6-C.1C.  The report is deliberately named "opportunity" everywhere.

For every trajectory group (existing B1.2 collection, 143 groups after the
frozen exclusion manifest, zero new collection) we construct five label-based
strategies:

1. CONTINUE_SOURCE                  - never switch; success = source final
2. ENTER_OFT@t0                     - switch at the first boundary (t0)
3. CONTINUE_TO_t16_THEN_OFT         - run to the t16 boundary, then switch if
                                      OFT persistent success holds there
4. privileged success oracle        - pick any successful option
                                      (source / enter@t0 / enter@t16)
5. privileged cost-aware oracle     - among the successful options pick the
                                      cheapest in OFT teacher steps

t={0,16} are the only boundaries available in the existing collection (t=8 is
collected in R6-C.1B).  Per-VLA and per-suite reports cover:

- success rate and success gap relative to ENTER_OFT@t0;
- OFT teacher-step savings relative to ENTER_OFT@t0;
- paired harm / rescue;
- number of t0-rescuable groups and groups rescuable only at t0 (already
  unrescuable at t16);
- state counts that are source-safe / OFT-safe / both fail;
- task-cluster bootstrap intervals.

R6-C.1A gate (per VLA, all must hold):

- cost-aware oracle success gap >= -5pp;
- OFT savings >= 30%;
- >= 20 decision-divergence groups;
- >= 10 source-failed AND early-OFT-rescuable groups;
- opportunity spans all four suites and >= 12 distinct real tasks.

If the opportunity is insufficient we may only proceed to R6-C.1B to collect
more data -- never to declare the method failed.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_exclusions(path: Path) -> set[tuple[str, int, str]]:
    if path is None:
        return set()
    data = json.loads(path.read_text())
    return {(str(e[0]), int(e[1]), str(e[2])) for e in data["excluded"]}


def boundary_map(members: list[dict]) -> dict[int, dict]:
    return {int(b["elapsed_source_steps"]): b for b in members}


def policy_outcomes(group: dict, boundaries: dict[int, dict]) -> dict:
    """Per-group label-derived outcomes for the five strategies.

    Returns a dict keyed by strategy name with fields:
    success, oft_teacher_steps, entered (bool), policy_id, task_id, suite,
    state_key, group_id.
    """
    source_success = bool(group["rows"][0]["source_final_success"])
    b0 = boundaries.get(0)
    b16 = boundaries.get(16)
    p0 = bool(b0["persistent_success_if_enter_now"]) if b0 is not None else False
    steps0 = float(b0["persistent_teacher_steps_if_enter_now"] or 0.0) if b0 is not None else float("inf")
    p16 = bool(b16["persistent_success_if_enter_now"]) if b16 is not None else False
    steps16 = float(b16["persistent_teacher_steps_if_enter_now"] or 0.0) if b16 is not None else float("inf")

    def base(success, teacher, entered):
        return {"success": bool(success), "oft_teacher_steps": float(teacher),
                "entered": bool(entered)}

    strategies = {
        "CONTINUE_SOURCE": base(source_success, 0.0, False),
        "ENTER_OFT@t0": base(p0, steps0, True),
        # run to t16 then switch if OFT still succeeds; otherwise continue source
        "CONTINUE_TO_t16_THEN_OFT": base(
            (p16 or (not p16 and source_success)),
            steps16 if p16 else 0.0,
            p16),
    }
    # privileged success oracle: any successful option, pick source first
    if source_success:
        oracle_success, oracle_steps, oracle_entered = True, 0.0, False
    elif p0:
        oracle_success, oracle_steps, oracle_entered = True, steps0, True
    elif p16:
        oracle_success, oracle_steps, oracle_entered = True, steps16, True
    else:
        oracle_success, oracle_steps, oracle_entered = False, 0.0, False
    strategies["privileged_success_oracle"] = base(
        oracle_success, oracle_steps, oracle_entered)

    # privileged cost-aware early oracle: cheapest successful option
    options = [
        (source_success, 0.0, False),
        (p0, steps0, True),
        (p16, steps16, True),
    ]
    feasible = [(c, e) for s, c, e in options if s]
    if feasible:
        cost, entered = min(feasible, key=lambda item: item[0])
        strategies["privileged_cost_aware_early_oracle"] = base(True, cost, entered)
    else:
        strategies["privileged_cost_aware_early_oracle"] = base(False, 0.0, False)

    for strategy in strategies.values():
        strategy.update({
            "policy_id": str(group["policy_id"]),
            "task_id": str(group["task_id"]),
            "suite": str(group["suite"]),
            "state_key": str(group["state_key"]),
            "group_id": str(group["group_id"]),
            "source_success": source_success,
            "rescuable_t0": p0,
            "rescuable_t16": p16,
        })
    return strategies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="B1.2 collection output root (contains suite_*)")
    parser.add_argument("--exclusions", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()

    metadata_paths = sorted(glob.glob(str(args.input_root / "suite_*" / "*" / "seed_*" / "*__seed*.json")))
    metadata_paths = [p for p in metadata_paths if Path(p).name != "report.json"]
    if not metadata_paths:
        raise SystemExit(f"no trajectory metadata under {args.input_root}")
    excluded = load_exclusions(args.exclusions)

    groups: dict[str, dict] = {}
    for path_string in metadata_paths:
        data = json.loads(Path(path_string).read_text())
        if not data["rows"]:
            continue
        policy_id = str(data["rows"][0]["policy_id"])
        seed_index = int(data["rows"][0]["seed_index"])
        state_key = str(data["rows"][0]["state_key"])
        if (policy_id, seed_index, state_key) in excluded:
            continue
        group_id = str(data["rows"][0]["group_id"])
        rows = data["rows"]
        groups[group_id] = {
            "group_id": group_id,
            "policy_id": policy_id,
            "task_id": str(data["rows"][0]["task_id"]),
            "suite": str(data["rows"][0]["suite"]),
            "state_key": state_key,
            "seed_index": seed_index,
            "rows": rows,
        }

    strategies: dict[str, list[dict]] = defaultdict(list)
    for group in groups.values():
        members = sorted(group["rows"], key=lambda b: int(b["elapsed_source_steps"]))
        boundaries = boundary_map(members)
        for name, outcome in policy_outcomes(group, boundaries).items():
            strategies[name].append(outcome)

    STRATEGY_ORDER = [
        "CONTINUE_SOURCE", "ENTER_OFT@t0", "CONTINUE_TO_t16_THEN_OFT",
        "privileged_success_oracle", "privileged_cost_aware_early_oracle",
    ]
    baselines = {name: 1.0 for name in STRATEGY_ORDER}

    def success_rate(rows: list[dict]) -> float:
        return float(np.mean([r["success"] for r in rows])) if rows else float("nan")

    def mean_teacher(rows: list[dict]) -> float:
        return float(np.mean([r["oft_teacher_steps"] for r in rows])) if rows else float("nan")

    # Global per-strategy (pooled) stats
    pooled: dict[str, dict] = {}
    for name in STRATEGY_ORDER:
        rows = strategies[name]
        if not rows:
            continue
        n = len(rows)
        succ = success_rate(rows)
        base_succ = success_rate(strategies["ENTER_OFT@t0"])
        base_steps = mean_teacher(strategies["ENTER_OFT@t0"])
        steps = mean_teacher(rows)
        harm = float(np.mean([(not r["entered"]) and (not r["source_success"]) and r["rescuable_t0"]
                              for r in rows]))
        rescue = float(np.mean([r["entered"] and r["success"] and (not r["source_success"])
                                for r in rows]))
        pooled[name] = {
            "n_groups": n,
            "success_rate": succ,
            "success_gap_vs_enter_oft_t0": succ - base_succ,
            "mean_oft_teacher_steps": steps,
            "oft_savings_vs_enter_oft_t0": 1.0 - steps / base_steps if base_steps > 0 else float("nan"),
            "absolute_paired_harm_rate": harm,
            "rescue_rate": rescue,
        }

    # Per-VLA and per-suite breakdown
    per_policy: dict[str, dict] = {}
    per_suite: dict[str, dict] = {}
    for name in STRATEGY_ORDER:
        for outcome in strategies[name]:
            per_policy.setdefault(outcome["policy_id"], {}).setdefault(name, []).append(outcome)
            per_suite.setdefault(outcome["suite"], {}).setdefault(name, []).append(outcome)

    def policy_summary(grouped: dict[str, list[dict]]) -> dict:
        by_policy: dict[str, dict] = {}
        for policy, strat_rows in grouped.items():
            base_succ = success_rate(strat_rows.get("ENTER_OFT@t0", []))
            base_steps = mean_teacher(strat_rows.get("ENTER_OFT@t0", []))
            cost_oracle = strat_rows.get("privileged_cost_aware_early_oracle", [])
            succ_t0 = success_rate(strat_rows.get("ENTER_OFT@t0", []))
            succ_cost = success_rate(cost_oracle)
            steps_cost = mean_teacher(cost_oracle)
            div_groups = set()
            rescuable_early = set()
            only_t0 = set()
            for o in strat_rows.get("ENTER_OFT@t0", []):
                g = o["group_id"]
                if o["rescuable_t0"] and not o["source_success"]:
                    rescuable_early.add(g)
                if o["rescuable_t0"] and not o["rescuable_t16"] and not o["source_success"]:
                    only_t0.add(g)
                # decision divergence: any two strategies disagree
                agreement = set()
                for strat in STRATEGY_ORDER:
                    for other in strat_rows.get(strat, []):
                        if other["group_id"] == g:
                            agreement.add(other["entered"])
                if len(agreement) > 1:
                    div_groups.add(g)
            tasks = {o["task_id"] for o in strat_rows.get("ENTER_OFT@t0", [])}
            suites = {o["suite"] for o in strat_rows.get("ENTER_OFT@t0", [])}
            n = len(strat_rows.get("ENTER_OFT@t0", []))
            # per-suite rescuable / divergence breakdown (per policy)
            per_suite_detail: dict[str, dict] = {}
            suite_tasks: dict[str, set] = defaultdict(set)
            suite_rescuable: dict[str, set] = defaultdict(set)
            suite_div: dict[str, set] = defaultdict(set)
            for o in strat_rows.get("ENTER_OFT@t0", []):
                suite = o["suite"]
                suite_tasks[suite].add(o["task_id"])
                if o["rescuable_t0"] and not o["source_success"]:
                    suite_rescuable[suite].add(o["group_id"])
                agreement = set()
                for strat in STRATEGY_ORDER:
                    for other in strat_rows.get(strat, []):
                        if other["group_id"] == o["group_id"]:
                            agreement.add(other["entered"])
                if len(agreement) > 1:
                    suite_div[suite].add(o["group_id"])
            for suite in sorted(suites):
                per_suite_detail[suite] = {
                    "n_groups": sum(1 for o in strat_rows.get("ENTER_OFT@t0", [])
                                    if o["suite"] == suite),
                    "n_tasks": len(suite_tasks.get(suite, set())),
                    "n_source_fail_early_rescuable": len(suite_rescuable.get(suite, set())),
                    "n_decision_divergence": len(suite_div.get(suite, set())),
                }
            by_policy[policy] = {
                "n_groups": n,
                "n_tasks": len(tasks),
                "n_suites": len(suites),
                "suites": sorted(suites),
                "per_suite": per_suite_detail,
                "success_rate_t0": succ_t0,
                "success_gap_cost_oracle_vs_t0": succ_cost - succ_t0,
                "oft_savings_cost_oracle": 1.0 - steps_cost / base_steps if base_steps > 0 else float("nan"),
                "n_decision_divergence_groups": len(div_groups),
                "n_source_fail_early_rescuable_groups": len(rescuable_early),
                "n_only_t0_rescuable_groups": len(only_t0),
                "only_t0_fraction_of_rescuable": (len(only_t0) / max(1, len(rescuable_early))),
            }
        return by_policy

    policy_stats = policy_summary(per_policy)
    suite_stats = policy_summary(per_suite)

    # task-cluster bootstrap for the cost-aware oracle gap/harm/savings
    bootstrap: dict[str, dict] = {}
    for policy in policy_stats:
        rows = strategies["privileged_cost_aware_early_oracle"]
        by_task: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["policy_id"] == policy:
                by_task[r["task_id"]].append(r)
        tasks = list(by_task)
        rng = random.Random(args.bootstrap_seed + hash(policy) % 1000)
        series = {"success_gap": [], "absolute_paired_harm_rate": [], "savings": []}
        t0_rows = [r for r in strategies["ENTER_OFT@t0"] if r["policy_id"] == policy]
        base_succ = success_rate(t0_rows)
        base_steps = mean_teacher(t0_rows)
        for _ in range(args.bootstrap_iterations):
            sampled: list[dict] = []
            for _ in range(len(tasks)):
                sampled.extend(by_task[rng.choice(tasks)])
            if not sampled:
                continue
            succ = success_rate(sampled)
            steps = mean_teacher(sampled)
            harm = float(np.mean([(not r["entered"]) and (not r["source_success"]) and r["rescuable_t0"]
                                  for r in sampled]))
            series["success_gap"].append(succ - base_succ)
            series["absolute_paired_harm_rate"].append(harm)
            series["savings"].append(1.0 - steps / base_steps if base_steps > 0 else 0.0)
        bootstrap[policy] = {}
        for key, values in series.items():
            values = np.asarray(values, dtype=float)
            bootstrap[policy][key] = {
                "mean": float(values.mean()),
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
            }

    # Per-VLA gate
    gates: dict[str, dict] = {}
    for policy, stats in policy_stats.items():
        gates[policy] = {
            "cost_oracle_success_gap_ge_m5pp": stats["success_gap_cost_oracle_vs_t0"] >= -0.05,
            "oft_savings_ge_30pct": stats["oft_savings_cost_oracle"] >= 0.30,
            "n_decision_divergence_groups_ge_20": stats["n_decision_divergence_groups"] >= 20,
            "n_source_fail_early_rescuable_ge_10": stats["n_source_fail_early_rescuable_groups"] >= 10,
            "covers_4_suites_and_12_tasks": (stats["n_suites"] >= 4 and stats["n_tasks"] >= 12),
            "passed": all([
                stats["success_gap_cost_oracle_vs_t0"] >= -0.05,
                stats["oft_savings_cost_oracle"] >= 0.30,
                stats["n_decision_divergence_groups"] >= 20,
                stats["n_source_fail_early_rescuable_groups"] >= 10,
                stats["n_suites"] >= 4 and stats["n_tasks"] >= 12,
            ]),
        }

    report = {
        "schema_version": "rase-r6c1-early-window-audit/v1",
        "scientific_scope": ("model-free early-window OPPORTUNITY audit (not separability); "
                             "separability requires the R6-C.1C task-held-out probe"),
        "input_root": str(args.input_root.resolve()),
        "n_groups": len(groups),
        "n_states": len({g["state_key"] for g in groups.values()}),
        "n_tasks": len({g["task_id"] for g in groups.values()}),
        "exclusions": str(args.exclusions.resolve()) if args.exclusions is not None else None,
        "boundaries_available": sorted({int(e) for g in groups.values() for e in
                                        [r["elapsed_source_steps"] for r in g["rows"]]}),
        "pooled_strategies": pooled,
        "policy_stats": policy_stats,
        "suite_stats": suite_stats,
        "task_cluster_bootstrap_95": bootstrap,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "n_groups": report["n_groups"], "n_tasks": report["n_tasks"],
        "boundaries_available": report["boundaries_available"],
        "pooled_strategies": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                  for kk, vv in v.items()} for k, v in pooled.items()},
        "policy_stats": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                             for kk, vv in v.items()} for k, v in policy_stats.items()},
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
