#!/usr/bin/env python3
"""R4: Result grading with McNemar paired test and hierarchical bootstrap.

Input: eval JSON files from eval_recovery_lora.py (one per variant x training_seed).
Output: analysis JSON with graded decision (CONFIRMED / SIGNAL / NO-SIGNAL).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_eval(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"results": [], "n_total": 0, "n_success": 0}


def _group_by_training_seed(results: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for r in results:
        ts = r.get("training_seed", 0)
        groups.setdefault(ts, []).append(r)
    return groups


def _group_by_task(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        tk = r.get("task_id", "unknown")
        groups.setdefault(tk, []).append(r)
    return groups


def _success_rate(results: list[dict[str, Any]]) -> float:
    n = max(1, len(results))
    return sum(1 for r in results if r.get("success", False)) / n


def _mcnemar_paired(paired: list[dict[str, Any]]) -> dict[str, Any]:
    """McNemar test for paired binary outcomes."""
    n_00 = sum(1 for p in paired if not p["b2_success"] and not p["b3_success"])
    n_01 = sum(1 for p in paired if not p["b2_success"] and p["b3_success"])
    n_10 = sum(1 for p in paired if p["b2_success"] and not p["b3_success"])
    n_11 = sum(1 for p in paired if p["b2_success"] and p["b3_success"])
    n_discordant = n_01 + n_10
    if n_discordant < 5:
        return {"test": "mcnemar", "discordant": n_discordant, "reliable": False,
                "n_00": n_00, "n_01": n_01, "n_10": n_10, "n_11": n_11}
    chi2 = ((abs(n_01 - n_10) - 1) ** 2) / max(1, n_01 + n_10)
    # No scipy — manual p-value via chi2(1)
    # Using normal approximation
    if n_discordant == 0:
        p_value = 1.0
    else:
        z = (n_01 - n_10) / math.sqrt(max(1, n_01 + n_10))
        # Two-tailed normal CDF approximation
        abs_z = abs(z)
        p_value = erfc_approx(abs_z / math.sqrt(2))
    return {"test": "mcnemar", "chi2": chi2, "p_value": p_value,
            "n_discordant": n_discordant, "reliable": n_discordant >= 5,
            "n_00": n_00, "n_01": n_01, "n_10": n_10, "n_11": n_11}


def erfc_approx(x: float) -> float:
    """Simple approximation of erfc for z-score to p-value conversion."""
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    sign = 1 if x >= 0 else -1
    t = 1.0 / (1.0 + p * abs(x))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return y if sign >= 0 else 2.0 - y


def _hierarchical_bootstrap(
    paired: list[dict[str, Any]],
    n_bootstrap: int = 5_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Cluster bootstrap by training seed -> task -> episode seed."""
    by_seed = _group_by_training_seed(paired)
    seed_keys = list(by_seed.keys())
    if not seed_keys:
        return {"ci_lower": None, "ci_upper": None, "n_bootstrap": n_bootstrap}

    rng = random.Random(42)
    diffs: list[float] = []
    for _ in range(n_bootstrap):
        sampled_seeds = rng.choices(seed_keys, k=len(seed_keys))
        total_b3 = 0
        total_b2 = 0
        n = 0
        for seed_k in sampled_seeds:
            seed_data = by_seed[seed_k]
            by_task = _group_by_task(seed_data)
            sampled_tasks = rng.choices(list(by_task.keys()), k=len(by_task))
            for task_k in sampled_tasks:
                task_data = by_task[task_k]
                sampled_eps = rng.choices(task_data, k=len(task_data))
                for ep in sampled_eps:
                    total_b3 += 1 if ep.get("b3_success", False) else 0
                    total_b2 += 1 if ep.get("b2_success", False) else 0
                    n += 1
        if n > 0:
            diffs.append((total_b3 - total_b2) / n)
    diffs.sort()
    alpha = 1.0 - confidence
    lo_idx = int(alpha / 2 * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap)
    return {"ci_lower": diffs[max(0, lo_idx)], "ci_upper": diffs[min(n_bootstrap - 1, hi_idx)],
            "mean_diff": float(np.mean(diffs)), "n_bootstrap": n_bootstrap,
            "confidence": confidence}


def _pair_results(b2_results: list[dict[str, Any]], b3_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair results by (training_seed, task_id, seed)."""
    b2_map: dict[tuple[int, str, int], dict[str, Any]] = {}
    for r in b2_results:
        key = (r.get("training_seed", 0), str(r.get("task_id", "")), r.get("seed", 0))
        b2_map[key] = r

    paired: list[dict[str, Any]] = []
    for r in b3_results:
        key = (r.get("training_seed", 0), str(r.get("task_id", "")), r.get("seed", 0))
        b2_match = b2_map.get(key)
        if b2_match:
            paired.append({
                "training_seed": key[0],
                "task_id": key[1],
                "seed": key[2],
                "b2_success": b2_match.get("success", False),
                "b3_success": r.get("success", False),
            })
    return paired


def _grade_result(
    diff: float, ci_lower: float | None, n_seeds_improved: int, n_seeds: int,
    n_tasks_reproducible: int, n_tasks: int,
) -> tuple[str, str]:
    if n_seeds_improved >= max(2, n_seeds * 2 // 3) and (ci_lower is not None and ci_lower > 0) and n_tasks_reproducible >= max(4, n_tasks // 2):
        return "RECOVERY-DISTILLATION-CONFIRMED", "B3 improves over B2 on sufficient seeds, CI>0, task coverage OK"
    elif n_seeds_improved >= 1 and diff > 0:
        return "RECOVERY-DISTILLATION-SIGNAL", "some reproducible recovery exists but CI or task coverage insufficient"
    else:
        return "NO-SIGNAL", "no stable improvement over B2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True, help="Directory with eval_*.json files")
    parser.add_argument("--b2-label", default="B2")
    parser.add_argument("--b3-label", default="B3")
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect eval files
    b2_files = sorted(args.eval_dir.glob(f"eval_{args.b2_label}_seed*.json"))
    b3_files = sorted(args.eval_dir.glob(f"eval_{args.b3_label}_seed*.json"))

    if not b2_files or not b3_files:
        print(f"WARNING: missing eval files. B2: {len(b2_files)}, B3: {len(b3_files)}")
        gate = {"grade": "NO-SIGNAL", "message": "missing eval files",
                "b2_files": len(b2_files), "b3_files": len(b3_files)}
        (output_dir / "analysis_result.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1

    b2_all: list[dict[str, Any]] = []
    b3_all: list[dict[str, Any]] = []
    for f in b2_files:
        b2_all.extend(_load_eval(f).get("results", []))
    for f in b3_files:
        b3_all.extend(_load_eval(f).get("results", []))

    paired = _pair_results(b2_all, b3_all)

    if len(paired) < 10:
        print(f"WARNING: only {len(paired)} paired points")
        gate = {"grade": "NO-SIGNAL", "message": f"insufficient paired data ({len(paired)} points)",
                "n_paired": len(paired)}
        (output_dir / "analysis_result.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1

    # Per-seed analysis
    by_seed = _group_by_training_seed(paired)
    seed_summary: dict[int, dict[str, Any]] = {}
    n_seeds_improved = 0
    for seed_k, seed_data in sorted(by_seed.items()):
        b3_rate = sum(1 for p in seed_data if p["b3_success"]) / max(1, len(seed_data))
        b2_rate = sum(1 for p in seed_data if p["b2_success"]) / max(1, len(seed_data))
        improved = b3_rate > b2_rate
        if improved:
            n_seeds_improved += 1
        seed_summary[seed_k] = {"b2_success_rate": b2_rate, "b3_success_rate": b3_rate,
                                 "n": len(seed_data), "improved": improved}

    # Per-task analysis
    by_task = _group_by_task(paired)
    n_tasks_reproducible = 0
    task_summary: dict[str, dict[str, Any]] = {}
    for task_k, task_data in sorted(by_task.items()):
        b3_rate = sum(1 for p in task_data if p["b3_success"]) / max(1, len(task_data))
        b2_rate = sum(1 for p in task_data if p["b2_success"]) / max(1, len(task_data))
        reproducible = b3_rate > b2_rate
        if reproducible:
            n_tasks_reproducible += 1
        task_summary[task_k] = {"b2_success_rate": b2_rate, "b3_success_rate": b3_rate,
                                 "n": len(task_data), "reproducible": reproducible}

    # McNemar test on all pairs
    mcnemar = _mcnemar_paired(paired)

    # Hierarchical bootstrap
    bootstrap = _hierarchical_bootstrap(paired, n_bootstrap=args.bootstrap,
                                         confidence=args.confidence)

    overall_b3 = sum(1 for p in paired if p["b3_success"]) / max(1, len(paired))
    overall_b2 = sum(1 for p in paired if p["b2_success"]) / max(1, len(paired))
    overall_diff = overall_b3 - overall_b2

    grade, message = _grade_result(
        overall_diff, bootstrap.get("ci_lower"), n_seeds_improved, len(by_seed),
        n_tasks_reproducible, len(by_task),
    )

    analysis = {
        "grade": grade,
        "message": message,
        "overall": {
            "b2_success_rate": overall_b2, "b3_success_rate": overall_b3,
            "diff": overall_diff, "n_paired": len(paired),
        },
        "mcnemar": mcnemar,
        "bootstrap": bootstrap,
        "per_seed": {str(k): v for k, v in seed_summary.items()},
        "per_task": task_summary,
        "n_seeds_improved": n_seeds_improved,
        "n_total_seeds": len(by_seed),
        "n_tasks_reproducible": n_tasks_reproducible,
        "n_total_tasks": len(by_task),
    }

    result_path = output_dir / "analysis_result.json"
    result_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"GRADE: {grade}")
    print(f"{'='*60}")
    print(f"  B2 overall: {overall_b2:.3f}")
    print(f"  B3 overall: {overall_b3:.3f}")
    print(f"  Diff: {overall_diff:+.3f}")
    print(f"  CI: [{bootstrap.get('ci_lower', 'N/A')}, {bootstrap.get('ci_upper', 'N/A')}]")
    print(f"  Seeds improved: {n_seeds_improved}/{len(by_seed)}")
    print(f"  Tasks reproducible: {n_tasks_reproducible}/{len(by_task)}")
    print(f"  McNemar p: {mcnemar.get('p_value', 'N/A')}")
    print(f"  Message: {message}")
    print(f"  Output: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
