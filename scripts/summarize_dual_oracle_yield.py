#!/usr/bin/env python3
"""Aggregate Wilson-triaged SmolVLA and deterministic one-shot OFT results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rase.collect.dual_oracle import aggregate_dual_oracle  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pool_metadata(pool_root: Path) -> dict[str, dict[str, Any]]:
    manifest = _load(pool_root / "manifest.json")
    metadata: dict[str, dict[str, Any]] = {}
    for key, entry in (manifest.get("states") or {}).items():
        meta_path = pool_root / entry["path"] / "meta.json"
        if meta_path.is_file():
            metadata[str(key)] = _load(meta_path)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smolvla-summary", type=Path, required=True)
    parser.add_argument(
        "--oft-summary",
        action="append",
        default=[],
        metavar="SUITE=PATH",
        help="Repeatable: libero_spatial=runs/.../summary.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="State pool root for causal yield table meta join",
    )
    parser.add_argument(
        "--causal-out",
        type=Path,
        default=None,
        help="Write perturbation→NGC yield JSON (requires --pool)",
    )
    parser.add_argument(
        "--causal-markdown",
        type=Path,
        default=None,
        help="Optional markdown render of the causal yield table",
    )
    parser.add_argument(
        "--ngc-oracle",
        choices=("smolvla", "oft", "both"),
        default="smolvla",
        help="Causal outcome(s): SmolVLA NGC and/or OFT portfolio-unrecovered",
    )
    args = parser.parse_args()

    smol = _load(args.smolvla_summary.resolve())
    oft_payloads: list[tuple[str, dict[str, Any]]] = []
    for item in args.oft_summary:
        if "=" not in item:
            raise SystemExit(f"expected SUITE=PATH, got {item!r}")
        suite, path_s = item.split("=", 1)
        payload = _load(Path(path_s).resolve())
        oft_payloads.append((suite, payload))

    pool_meta = _pool_metadata(args.pool.resolve()) if args.pool is not None else None
    out = aggregate_dual_oracle(smol, oft_payloads, pool_meta=pool_meta)
    out["sources"] = {
        "smolvla_summary": str(args.smolvla_summary.resolve()),
        "oft_summaries": list(args.oft_summary),
        "pool": str(args.pool.resolve()) if args.pool is not None else None,
    }
    n = out["n_states"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in (
        "n_states", "deterministic_candidate_hits", "candidate_hit_rate",
        "portfolio_recovered_states", "portfolio_coverage",
    )}, indent=2), flush=True)
    print(f"WROTE {args.output}", flush=True)

    if args.markdown is not None:
        candidate_hits = (
            f"{out['deterministic_candidate_hits']}/"
            f"{out['deterministic_candidate_trials']}"
        )
        portfolio_fraction = (
            f"{out['portfolio_recovered_states']}/"
            f"{out['portfolio_evaluable_states']}"
        )
        coverage_ci = out["portfolio_coverage_wilson_95"]
        lines = [
            "# Dual-oracle yield",
            "",
            "| metric | value |",
            "|---|---|",
            f"| n_states | {n} |",
            f"| deterministic candidate hits | {candidate_hits} |",
            f"| candidate hit rate | {out['candidate_hit_rate']} |",
            f"| portfolio coverage | {out['portfolio_coverage']} ({portfolio_fraction}) |",
            f"| state-level Wilson 95% CI | "
            f"[{coverage_ci['lower']}, {coverage_ci['upper']}] |",
            "",
            "OFT is deterministic one-shot verification. Candidate hit rate is not a "
            "success-probability estimate and cannot certify Wilson Set A/B.",
            "",
            "## Cross labels",
            "",
            "| label | states |",
            "|---|---:|",
            *[
                f"| {label} | {count} |"
                for label, count in out["cross_label_counts"].items()
            ],
            "",
            "## Per suite OFT raw",
            "",
            "| suite | successes/trials | states |",
            "|---|---|---|",
        ]
        for suite, info in sorted(out["oft_raw_by_suite"].items()):
            lines.append(
                f"| {suite} | {info['successes']}/{info['trials']} | {info['n_states']} |"
            )
        lines.append("")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"WROTE {args.markdown}", flush=True)

    if args.causal_out is not None or args.causal_markdown is not None:
        if args.pool is None:
            raise SystemExit("--causal-out/--causal-markdown require --pool")
        from rase.collect.causal_analysis import (
            build_dual_yield_tables,
            build_yield_table,
            write_yield_table,
            yield_table_markdown,
        )
        from rase.collect.state_pool import StatePool

        pool_root = args.pool.resolve()
        pool = StatePool(pool_root)
        if args.ngc_oracle == "both":
            table = build_dual_yield_tables(out, pool)
            out["causal_yield"] = table
        else:
            table = build_yield_table(out, pool, ngc_oracle=args.ngc_oracle)
            out["causal_yield"] = {
                key: table[key]
                for key in (
                    "ngc_oracle",
                    "outcome",
                    "n_states",
                    "n_ngc",
                    "yield",
                    "wilson_lower",
                    "wilson_upper",
                    "warnings",
                )
            }
        # Refresh dual-oracle summary with causal headline.
        args.output.write_text(
            json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.causal_out is not None:
            write_yield_table(table, args.causal_out.resolve())
            print(f"WROTE {args.causal_out}", flush=True)
        if args.causal_markdown is not None:
            md_path = args.causal_markdown.resolve()
            md_path.parent.mkdir(parents=True, exist_ok=True)
            if args.ngc_oracle == "both":
                markdown = "\n".join(
                    yield_table_markdown(item) for item in table.values()
                )
            else:
                markdown = yield_table_markdown(table)
            md_path.write_text(markdown, encoding="utf-8")
            print(f"WROTE {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
