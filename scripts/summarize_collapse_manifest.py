#!/usr/bin/env python3
"""Summarize a collapse ResultManifest into tables (stdout + optional JSON)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _pc(metrics: dict[str, Any] | None) -> float | None:
    if not metrics:
        return None
    if metrics.get("pc_success") is not None:
        return float(metrics["pc_success"])
    if metrics.get("successes") is not None and metrics.get("episodes"):
        return 100.0 * float(metrics["successes"]) / float(metrics["episodes"])
    return None


def _parse_row(key: str, record: dict[str, Any]) -> dict[str, Any]:
    task = record.get("task") or {}
    parts = key.split(":")
    suite = str(task.get("suite") or (parts[0] if parts else "?"))
    dim = str(task.get("dimension") or (parts[2] if len(parts) > 2 else "?"))
    level = task.get("difficulty")
    if level is None and len(parts) > 3 and parts[3].startswith("L"):
        try:
            level = int(parts[3][1:])
        except ValueError:
            level = None
    return {
        "key": key,
        "suite": suite,
        "dimension": dim,
        "level": int(level) if level is not None else None,
        "status": record.get("status"),
        "pc_success": _pc(record.get("metrics")),
        "error": record.get("error"),
    }


def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = [_parse_row(k, r) for k, r in manifest.get("results", {}).items()]
    completed = [r for r in rows if r["status"] == "completed" and r["pc_success"] is not None]
    skipped = [r for r in rows if r["status"] == "skipped"]
    other = [r for r in rows if r["status"] not in ("completed", "skipped")]

    def agg(items: list[dict[str, Any]]) -> dict[str, Any]:
        pcs = [float(r["pc_success"]) for r in items]
        return {
            "n": len(pcs),
            "mean_pc_success": (sum(pcs) / len(pcs)) if pcs else None,
            "n_success_tasks": sum(1 for x in pcs if x > 0),
        }

    by_dim: dict[str, list] = defaultdict(list)
    by_level: dict[str, list] = defaultdict(list)
    by_suite: dict[str, list] = defaultdict(list)
    by_cell: dict[str, list] = defaultdict(list)
    for row in completed:
        by_dim[row["dimension"]].append(row)
        if row["level"] is not None:
            by_level[f"L{row['level']}"].append(row)
            by_cell[f"{row['dimension']}:L{row['level']}"].append(row)
        by_suite[row["suite"]].append(row)

    return {
        "counts": {
            "total": len(rows),
            "completed": len(completed),
            "skipped": len(skipped),
            "other": len(other),
        },
        "overall": agg(completed),
        "by_dimension": {k: agg(v) for k, v in sorted(by_dim.items())},
        "by_level": {k: agg(v) for k, v in sorted(by_level.items())},
        "by_suite": {k: agg(v) for k, v in sorted(by_suite.items())},
        "by_dimension_level": {k: agg(v) for k, v in sorted(by_cell.items())},
        "skipped_keys": [r["key"] for r in skipped],
        "success_keys": [r["key"] for r in completed if float(r["pc_success"]) > 0],
    }


def _print_human(summary: dict[str, Any]) -> None:
    c = summary["counts"]
    o = summary["overall"]
    print(
        f"total={c['total']} completed={c['completed']} "
        f"skipped={c['skipped']} other={c['other']}"
    )
    if o["mean_pc_success"] is not None:
        print(
            f"overall mean_pc_success={o['mean_pc_success']:.3f}% "
            f"({o['n_success_tasks']}/{o['n']})"
        )
    print("\nby_dimension")
    for k, v in summary["by_dimension"].items():
        print(f"  {k}: {v['mean_pc_success']:.3f}% ({v['n_success_tasks']}/{v['n']})")
    print("\nby_level")
    for k, v in summary["by_level"].items():
        print(f"  {k}: {v['mean_pc_success']:.3f}% ({v['n_success_tasks']}/{v['n']})")
    print("\nby_suite")
    for k, v in summary["by_suite"].items():
        print(f"  {k}: {v['mean_pc_success']:.3f}% ({v['n_success_tasks']}/{v['n']})")
    if summary["skipped_keys"]:
        print("\nskipped")
        for key in summary["skipped_keys"]:
            print(f"  {key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default="runs/collapse_full_nas10/manifest.json",
        help="Path to collapse manifest.json",
    )
    parser.add_argument("--json-out", help="Optional path to write summary JSON")
    args = parser.parse_args(argv)
    path = Path(args.manifest)
    if not path.is_file():
        print(f"error: manifest not found: {path}", file=sys.stderr)
        return 2
    summary = summarize(json.loads(path.read_text(encoding="utf-8")))
    _print_human(summary)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
