#!/usr/bin/env python3
"""Analyze OFT model-pair opportunity matrix.

Parses per-task success from the official EVAL-*.txt logs (pure text written
by log_message; the combined stdout log truncates task names).  Files are
mapped to (model, suite) combos by run note + creation time, in the fixed
run order of the matrix/cross scripts:

  rase_oft_matrix (mtime order): oft_spatial@libero_spatial, oft_object@libero_object
  rase_oft_cross  (mtime order): oft_spatial@libero_object,  oft_object@libero_spatial

Outputs per-task success matrix and opportunity-window statistics for the
source/fallback pair (A=oft_spatial, B=oft_object) over the mixed domain
(libero_spatial + libero_object):
  - strict advantage counts (A wins / B wins / ties)
  - oracle mean = mean_i max(A_i, B_i); single-best; headroom (pp)
  - heterogeneity rate: fraction of tasks with |A-B| >= threshold
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

SUITE_LINE = re.compile(r"Task suite: (\S+)")
TASK_LINE = re.compile(r"Task: (.+?)\s*$")
SUCCESS_LINE = re.compile(r"Success: (True|False)")

# note -> ordered (model, suite) pairs, in script run order
RUN_ORDER: dict[str, list[tuple[str, str]]] = {
    "rase_oft_matrix": [
        ("oft_spatial", "libero_spatial"),
        ("oft_object", "libero_object"),
    ],
    "rase_oft_cross": [
        ("oft_spatial", "libero_object"),
        ("oft_object", "libero_spatial"),
    ],
}


def parse_eval_file(path: Path) -> dict:
    text = path.read_text()
    suite = None
    m = SUITE_LINE.search(text)
    if m:
        suite = m.group(1)
    blocks: list[tuple[str, list[str]]] = []
    cur: tuple[str, list[str]] | None = None
    for line in text.splitlines():
        m = TASK_LINE.match(line)
        if m:
            if cur:
                blocks.append(cur)
            cur = (m.group(1).strip(), [])
            continue
        if cur is not None:
            cur[1].append(line)
    if cur:
        blocks.append(cur)
    tasks = []
    for name, body in blocks:
        n_ep = n_ok = 0
        for ln in body:
            sm = SUCCESS_LINE.search(ln)
            if sm:
                n_ep += 1
                n_ok += int(sm.group(1) == "True")
        if n_ep:
            tasks.append({"task": name, "success": n_ok, "episodes": n_ep})
    return {"suite": suite, "tasks": tasks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_note: dict[str, list[Path]] = {}
    for path in sorted(args.logs_dir.glob("EVAL-*.txt"), key=lambda p: p.stat().st_mtime):
        name = path.name
        note = name.split("--")[-1].replace(".txt", "") if "--" in name else ""
        if note in RUN_ORDER:
            by_note.setdefault(note, []).append(path)

    combos: dict[str, dict] = {}
    for note, order in RUN_ORDER.items():
        files = by_note.get(note, [])
        if len(files) != len(order):
            print(f"WARN: {note}: expected {len(order)} logs, found {len(files)}: "
                  f"{[f.name for f in files]}")
        for path, (model, suite) in zip(files, order):
            parsed = parse_eval_file(path)
            if not parsed["tasks"]:
                print(f"WARN: no tasks parsed from {path.name}")
                continue
            combos[f"{model}@{suite}"] = parsed["tasks"]
            print(f"parsed {path.name} -> {model}@{suite} "
                  f"({len(parsed['tasks'])} tasks)")

    if not combos:
        print("no combos parsed")
        return 1

    matrix: dict[str, dict] = {}
    suite_of: dict[str, str] = {}
    for key, tasks in combos.items():
        model, suite = key.split("@", 1)
        for t in tasks:
            name = t["task"]
            rate = t["success"] / t["episodes"]
            matrix.setdefault(name, {})[key] = rate
            suite_of.setdefault(name, suite)
            if suite_of[name] != suite:
                print(f"WARN: task {name!r} seen in both {suite_of[name]} and {suite}")

    keys = sorted(combos)
    rows = [{"task": name, "suite": suite_of[name],
             **{k: matrix[name].get(k) for k in keys}} for name in matrix]

    a_key, b_key = "oft_spatial", "oft_object"
    stats = {"a_wins": 0, "b_wins": 0, "ties": 0, "incomplete": 0, "tasks": []}
    for name, cells in matrix.items():
        suite = suite_of[name]
        sa = cells.get(f"{a_key}@{suite}")
        sb = cells.get(f"{b_key}@{suite}")
        if sa is None or sb is None:
            stats["incomplete"] += 1
            continue
        d = round(sa - sb, 4)
        stats["tasks"].append({"task": name, "suite": suite, "A": sa, "B": sb, "diff": d})
        if d > 0:
            stats["a_wins"] += 1
        elif d < 0:
            stats["b_wins"] += 1
        else:
            stats["ties"] += 1

    n = len(stats["tasks"])
    if n == 0:
        print("no matched tasks")
        return 2
    a_rates = [t["A"] for t in stats["tasks"]]
    b_rates = [t["B"] for t in stats["tasks"]]
    oracle = statistics.fmean(max(t["A"], t["B"]) for t in stats["tasks"])
    single_best = max(statistics.fmean(a_rates), statistics.fmean(b_rates))

    def het(thr: float) -> float:
        return sum(1 for t in stats["tasks"] if abs(t["diff"]) >= thr) / n

    report = {
        "combos_parsed": keys,
        "n_tasks": n,
        "mean_A": statistics.fmean(a_rates),
        "mean_B": statistics.fmean(b_rates),
        "single_best_mean": single_best,
        "oracle_mean": oracle,
        "oracle_headroom_pp": round((oracle - single_best) * 100, 2),
        "a_wins": stats["a_wins"],
        "b_wins": stats["b_wins"],
        "ties": stats["ties"],
        "incomplete_tasks": stats["incomplete"],
        "hetero_rate_ge_0.2": round(het(0.2), 3),
        "hetero_rate_ge_0.3": round(het(0.3), 3),
        "hetero_rate_ge_0.5": round(het(0.5), 3),
        "per_task": stats["tasks"],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
