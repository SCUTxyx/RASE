#!/usr/bin/env python3
"""vFinal figures: opportunity plot + phase diagram.

Opportunity plot: every observed domain cell as (source success, fallback
success) with heterogeneous rate as marker size/color.  Phase diagram:
conceptual regions (ceiling / fallback-dominant / all-fail / Goldilocks).

No new rollouts: all data points come from frozen results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # (label, source_success, fallback_success, heterogeneous_rate, color)
    points = [
        ("LIBERO K5 (pi0fast@8)", 0.583, 0.951, 0.0093, "#d62728"),
        ("LIBERO K3 (pi0fast@8)", 0.583, 0.958, 0.0140, "#d62728"),
        ("LIBERO conf (pi0fast@16)", 0.55, 0.95, 0.0208, "#d62728"),
        ("LIBERO conf (pi05@8/16)", 0.972, 0.972, 0.0208, "#ff7f0e"),
        ("libero_90 (pi0fast)", 0.0, 0.0, 0.0, "#9467bd"),
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    for label, source, fallback, hetero, color in points:
        size = 120 + hetero * 4000
        ax.scatter(source, fallback, s=size, c=color, alpha=0.75, edgecolors="black", zorder=5)
        ax.annotate(label, (source, fallback), textcoords="offset points",
                    xytext=(8, 8), fontsize=9)
    # Goldilocks regime
    ax.add_patch(plt.Rectangle((0.15, 0.10), 0.70, 0.55, fill=True,
                               alpha=0.12, color="green"))
    ax.text(0.50, 0.37, "Goldilocks regime\n(required by RASE)", ha="center",
            fontsize=11, color="darkgreen", style="italic")
    # region labels
    ax.text(0.95, 0.97, "ceiling", ha="right", fontsize=10, color="gray")
    ax.text(0.03, 0.97, "fallback\ndomination", ha="left", fontsize=10, color="gray")
    ax.text(0.03, 0.05, "all-fail floor", ha="left", fontsize=10, color="gray")
    ax.set_xlabel("source success rate (continue/requery)")
    ax.set_ylabel("fallback (corrective) success rate")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("RASE observed domains vs required Goldilocks regime\n"
                 "(marker size = heterogeneous rate)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figure1_opportunity_plot.png", dpi=150)
    plt.close(fig)

    # Phase diagram (conceptual)
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    regions = [
        (0.0, 0.10, 0.0, 1.0, "Region III\nall-fail floor\n(libero_90)", "#9467bd", 0.20),
        (0.10, 0.90, 0.80, 1.0, "Region II\nfallback domination\n(LIBERO)", "#d62728", 0.20),
        (0.10, 0.90, 0.0, 0.80, "Region IV\nGoldilocks regime\n(comparative-advantage\nheterogeneous)", "#2ca02c", 0.16),
        (0.90, 1.0, 0.0, 1.0, "Region I\nall-success ceiling\n(pi0.5)", "#ff7f0e", 0.20),
    ]
    for x0, x1, y0, y1, label, color, alpha in regions:
        ax2.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True,
                                    alpha=alpha, color=color))
        ax2.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center",
                 fontsize=10, fontweight="bold")
    # observed points
    for label, source, fallback, hetero, color in points:
        ax2.scatter(source, fallback, s=90, c="black", marker="x", zorder=6)
    ax2.set_xlabel("source competence")
    ax2.set_ylabel("corrective-policy dominance (fallback success)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("RASE applicability phase diagram")
    fig2.tight_layout()
    fig2.savefig(out / "figure2_phase_diagram.png", dpi=150)
    plt.close(fig2)
    print("figures written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
