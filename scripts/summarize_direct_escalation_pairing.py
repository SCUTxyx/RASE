#!/usr/bin/env python3
"""Pair W7 prefix-portfolio outcomes with W8 direct OFT outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _render(result: dict) -> str:
    prefix = result["prefix_oft_portfolio"]
    direct = result["direct_oft"]
    smol = result["smol_portfolio"]
    pairs = result["prefix_direct_pair_counts"]
    lines = [
        "# W8 direct escalation paired analysis",
        "",
        "## Headline",
        "",
        f"- Smol portfolio: {smol['hits']}/{smol['trials']} ({smol['rate']:.1%})",
        f"- W7 prefix OFT portfolio: {prefix['hits']}/{prefix['trials']} ({prefix['rate']:.1%})",
        f"- W8 direct OFT: {direct['hits']}/{direct['trials']} ({direct['rate']:.1%})",
        "- Prefix/direct overlap: "
        f"both={pairs['both_success']}, prefix-only={pairs['portfolio_only']}, "
        f"direct-only={pairs['direct_only']}, neither={pairs['both_fail']}",
        "- Exact McNemar p (prefix portfolio vs direct): "
        f"{result['prefix_direct_mcnemar_exact_p_two_sided']}",
        "- Exact McNemar p (Smol vs direct): "
        f"{result['direct_vs_smol_mcnemar_exact_p_two_sided']}",
        "- Direct minus prefix risk difference: "
        f"{result['direct_minus_prefix_risk_difference']:.4f}",
        "",
        "## Per suite",
        "",
        "| suite | n | both | prefix only | direct only | neither | prefix hits | direct hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_suite"]:
        lines.append(
            f"| {row['suite']} | {row['n_states']} | {row['both_success']} | "
            f"{row['portfolio_only']} | {row['direct_only']} | {row['both_fail']} | "
            f"{row['portfolio_hits']} | {row['direct_hits']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in result["interpretation"].items())
    lines.extend(["", "## Constraints", ""])
    lines.extend(f"- {warning}" for warning in result["warnings"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--direct-summary", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    from rase.collect.escalation_analysis import aggregate_direct_escalation_pairing

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.direct_summary
    ]
    result = aggregate_direct_escalation_pairing(matrix, summaries)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(_render(result), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "n_states", "smol_portfolio", "prefix_oft_portfolio",
        "direct_oft", "prefix_direct_pair_counts",
        "prefix_direct_mcnemar_exact_p_two_sided",
        "direct_vs_smol_mcnemar_exact_p_two_sided",
    )}, indent=2), flush=True)
    print(f"WROTE {args.output_json}", flush=True)
    print(f"WROTE {args.output_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
