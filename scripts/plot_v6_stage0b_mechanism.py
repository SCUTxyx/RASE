#!/usr/bin/env python3
"""Plot the preregistered V6 mechanism curve: AC_R vs perturbation severity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_report(token: str) -> tuple[float, Path]:
    severity, separator, raw_path = token.partition("=")
    if not separator:
        raise ValueError(f"report must use severity=path, got {token!r}")
    return float(severity), Path(raw_path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", required=True,
        help="one or more severity=Stage0_analysis.json entries",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for token in args.report:
        severity, path = parse_report(token)
        artifact = json.loads(path.read_text(encoding="utf-8"))
        summary = artifact.get("summary") or {}
        bootstrap = artifact.get("bootstrap") or {}
        ac = summary.get("ac_refresh")
        ci = bootstrap.get("ac_refresh_ci95")
        if ac is None:
            raise ValueError(f"{path} has no auditable AC_R")
        rows.append(
            {
                "severity": severity,
                "ac_refresh": float(ac),
                "ci95": ci,
                "n_roots": int(summary.get("n_roots", 0)),
                "report": str(path),
                "analysis_gate": (artifact.get("gate") or {}).get("decision"),
            }
        )
    rows.sort(key=lambda row: row["severity"])
    # Lazy import: Stage-0 collection/analysis remain usable in a minimal env.
    import matplotlib.pyplot as plt

    x = [row["severity"] for row in rows]
    y = [row["ac_refresh"] for row in rows]
    lower = [
        max(0.0, value - float(row["ci95"][0])) if isinstance(row["ci95"], list) else 0.0
        for row, value in zip(rows, y, strict=True)
    ]
    upper = [
        max(0.0, float(row["ci95"][1]) - value) if isinstance(row["ci95"], list) else 0.0
        for row, value in zip(rows, y, strict=True)
    ]
    figure, axis = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.errorbar(x, y, yerr=[lower, upper], marker="o", capsize=3, linewidth=1.5)
    axis.set_xlabel("position perturbation severity")
    axis.set_ylabel(r"$AC_R$ (refresh opportunity)")
    axis.set_title("V6 mechanism curve: perturbation creates or removes arbitration opportunity")
    axis.grid(alpha=0.25)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=180)
    plt.close(figure)
    atomic_json(args.output_json.resolve(), {
        "schema_version": "rase-v6-stage0b-mechanism/v1",
        "estimand": "AC_R using mean R-new K=1 estimate; no best-of-K selection",
        "rows": rows,
    })
    print(f"wrote {args.output_png} and {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
