#!/usr/bin/env python3
"""Audit W9C clean probe: task names + preregistered suite SR floors."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rase.backends.libero_clean import assert_clean_task_name, load_clean_task_names
from rase.eval.collapse import CollapseError

_PERTURB_RE = re.compile(
    r"(_table_\d+|_view_\d+|_tb_\d+|_light_\d+|_noise_\d+|"
    r"_robot_\d+|_background_\d+|_language_\d+|_add_\d+)($|\.)"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    floors = dict((config.get("probe") or {}).get("preregistered_suite_sr_floor") or {})
    mean_floor = float((config.get("probe") or {}).get("preregistered_mean_sr_floor", 0.35))
    catalog = load_clean_task_names()

    # Prefer scheduled/probe episode rows if present; else outcomes only.
    by_suite_outcomes: dict[str, list[str]] = defaultdict(list)
    task_names: dict[str, set[str]] = defaultdict(set)
    bad_names: list[str] = []

    pool_dir = Path(str(config["collection"]["output_dir"]))
    meta_paths = sorted(pool_dir.glob("*/*/meta.json")) if pool_dir.is_dir() else []
    # StatePool layout varies; also accept flat **/meta.json
    if not meta_paths:
        meta_paths = sorted(pool_dir.glob("**/meta.json")) if pool_dir.is_dir() else []

    for meta_path in meta_paths:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        suite = str(meta.get("suite", ""))
        outcome = str(meta.get("episode_outcome", ""))
        name = str(meta.get("clean_task_name") or meta.get("bddl_stem") or "")
        flavor = str(meta.get("libero_flavor") or "")
        if suite and outcome:
            # Count each episode once via episode_id
            by_suite_outcomes[suite].append(
                json.dumps(
                    {
                        "episode_id": meta.get("episode_id"),
                        "outcome": outcome,
                        "name": name,
                        "flavor": flavor,
                        "task_id": meta.get("task_id"),
                    },
                    sort_keys=True,
                )
            )
        if name:
            try:
                assert_clean_task_name(name)
            except CollapseError as exc:
                bad_names.append(f"{name}: {exc}")
            if _PERTURB_RE.search(name):
                bad_names.append(name)
            task_names[suite].add(name)

    # Deduplicate episode rows
    suite_stats: dict[str, dict[str, float | int | list[str]]] = {}
    for suite, encoded in by_suite_outcomes.items():
        unique = [json.loads(item) for item in sorted(set(encoded))]
        n = len(unique)
        n_success = sum(1 for row in unique if row["outcome"] == "success")
        sr = (n_success / n) if n else 0.0
        names = sorted({row["name"] for row in unique if row["name"]})
        suite_stats[suite] = {
            "n_episodes": n,
            "n_success": n_success,
            "success_rate": sr,
            "task_names": names,
            "n_distinct_task_names": len(names),
        }

    # Fallback to summary.outcomes if pool empty (should not pass)
    if not suite_stats:
        outcomes = dict(summary.get("outcomes") or {})
        suite_stats["ALL"] = {
            "n_episodes": int(sum(outcomes.values())),
            "n_success": int(outcomes.get("success", 0)),
            "success_rate": (
                float(outcomes.get("success", 0)) / float(sum(outcomes.values()) or 1)
            ),
            "task_names": [],
            "n_distinct_task_names": 0,
        }

    failures: list[str] = []
    if bad_names:
        failures.append(f"perturbed/Plus variant task names: {bad_names[:5]}")

    for suite, floor in floors.items():
        stats = suite_stats.get(suite)
        if stats is None:
            failures.append(f"missing suite {suite} in probe outcomes")
            continue
        if int(stats["n_episodes"]) < 20:
            failures.append(
                f"{suite} has {stats['n_episodes']} episodes (<20 preregistered)"
            )
        if float(stats["success_rate"]) < float(floor):
            failures.append(
                f"{suite} SR {stats['success_rate']:.3f} < floor {float(floor):.3f}"
            )
        if suite in {"Object", "Goal"} and float(stats["success_rate"]) < 0.05:
            failures.append(f"{suite} SR near zero — task identity still wrong")
        # Expect diverse clean names when enough episodes
        if int(stats["n_episodes"]) >= 10 and int(stats["n_distinct_task_names"]) < 2:
            failures.append(f"{suite} produced <2 distinct clean task names")

    suite_srs = [
        float(stats["success_rate"])
        for suite, stats in suite_stats.items()
        if suite in floors
    ]
    mean_sr = sum(suite_srs) / len(suite_srs) if suite_srs else 0.0
    if suite_srs and mean_sr < mean_floor:
        failures.append(f"mean suite SR {mean_sr:.3f} < floor {mean_floor:.3f}")

    # Official name coverage check when names present
    for suite, stats in suite_stats.items():
        suite_key = {
            "Spatial": "libero_spatial",
            "Object": "libero_object",
            "Goal": "libero_goal",
            "Long": "libero_10",
        }.get(suite)
        if not suite_key:
            continue
        official = set(catalog[suite_key])
        observed = set(stats["task_names"])
        unknown = sorted(observed - official)
        if unknown:
            failures.append(f"{suite} unknown clean names: {unknown[:3]}")

    payload = {
        "protocol": "W9C-clean-alignment-probe/v1",
        "summary_path": str(args.summary),
        "suite_stats": suite_stats,
        "mean_suite_sr": mean_sr,
        "floors": floors,
        "mean_floor": mean_floor,
        "pass": not failures,
        "failures": failures,
        "n_meta_files": len(meta_paths),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": payload["pass"], "mean_suite_sr": mean_sr, "failures": failures}, indent=2))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
