#!/usr/bin/env python3
"""Monte Carlo calibration for the sequential NGC labeling protocol."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _simulate_candidate(
    rng,
    p: float,
    *,
    n_first: int,
    n_total: int,
    threshold: float,
    alpha_first: float,
    alpha_final: float,
    sidedness: str,
):
    from rase.collect.adaptive import sequential_adaptive_sample

    outcomes = (rng.random() < p for _ in range(n_total))
    iterator = iter(outcomes)

    def rollout(_index: int) -> bool:
        return bool(next(iterator))

    return sequential_adaptive_sample(
        rollout,
        threshold=threshold,
        n_first=n_first,
        n_total=n_total,
        alpha_first=alpha_first,
        alpha_final=alpha_final,
        sidedness=sidedness,
    )


def calibrate(config: dict, *, replications: int, seed: int = 0) -> dict:
    import random

    from rase.collect.adaptive import triage

    adaptive = config.get("adaptive", {})
    n_first = int(adaptive.get("first_stage_rollouts", 6))
    n_total = int(adaptive.get("total_rollouts", 20))
    threshold = float(adaptive.get("threshold", 0.5))
    alpha_first = float(adaptive.get("alpha_first", 0.01))
    alpha_final = float(adaptive.get("alpha_final", 0.04))
    sidedness = str(adaptive.get("sidedness", "one-sided"))
    set_a_min = int(adaptive.get("set_a_min_good_candidates", 3))
    k = int(config.get("candidates", {}).get("k", 8))

    grid = [i / 20 for i in range(21)]
    rng = random.Random(seed)
    per_p = []
    for p in grid:
        early = 0
        set_c = 0
        set_a = 0
        uncertain = 0
        trials = 0
        for _ in range(replications):
            estimates = [
                _simulate_candidate(
                    rng,
                    p,
                    n_first=n_first,
                    n_total=n_total,
                    threshold=threshold,
                    alpha_first=alpha_first,
                    alpha_final=alpha_final,
                    sidedness=sidedness,
                )
                for _ in range(k)
            ]
            trials += sum(item.trials for item in estimates) / k
            early += sum(1 for item in estimates if item.stopped_early) / k
            label = triage(
                estimates, threshold=threshold, set_a_min_good=set_a_min
            ).value
            set_c += label == "C"
            set_a += label == "A"
            uncertain += label == "uncertain"
        per_p.append(
            {
                "p": p,
                "set_c_rate": set_c / replications,
                "set_a_rate": set_a / replications,
                "uncertain_rate": uncertain / replications,
                "mean_trials": trials / replications,
                "early_stop_rate": early / replications,
            }
        )

    # False-positive Set C: true p > threshold but labeled C.
    fp_rows = [row for row in per_p if row["p"] > threshold]
    max_fp = max((row["set_c_rate"] for row in fp_rows), default=0.0)
    # Power at strong NGC (p=0.05) and mild NGC (p=0.15)
    power = {
        row["p"]: row["set_c_rate"]
        for row in per_p
        if abs(row["p"] - 0.05) < 1e-12 or abs(row["p"] - 0.15) < 1e-12
    }
    passed = max_fp <= 0.05 + 1e-12
    return {
        "replications": replications,
        "seed": seed,
        "protocol": {
            "n_first": n_first,
            "n_total": n_total,
            "threshold": threshold,
            "alpha_first": alpha_first,
            "alpha_final": alpha_final,
            "sidedness": sidedness,
            "k": k,
            "set_a_min_good": set_a_min,
        },
        "max_set_c_false_positive": max_fp,
        "gate_set_c_fp_le_0.05": passed,
        "set_c_power": power,
        "grid": per_p,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    summary = calibrate(config, replications=args.replications, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in (
        "max_set_c_false_positive",
        "gate_set_c_fp_le_0.05",
        "set_c_power",
        "protocol",
    )}, indent=2, sort_keys=True))
    return 0 if summary["gate_set_c_fp_le_0.05"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
