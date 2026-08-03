"""Perturbation yield tables for SmolVLA NGC and OFT portfolio failure."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rase.collect.adaptive import wilson_interval
from rase.collect.state_pool import StatePool
from rase.collect.stratified_sample import SUITE_ALIASES


def _load_meta(pool: StatePool, state_key: str) -> dict[str, Any]:
    entry = pool.manifest()["states"].get(state_key)
    if entry is None:
        raise KeyError(f"state_key not in pool manifest: {state_key}")
    meta_path = pool.root / entry["path"] / "meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _cell_key(meta: Mapping[str, Any]) -> tuple[str, str, int]:
    suite = SUITE_ALIASES.get(str(meta.get("suite", "")), str(meta.get("suite", "")))
    dim = str(meta.get("perturb_dim", ""))
    level = int(meta.get("level", 0))
    return suite, dim, level


def _wilson_or_none(successes: int, trials: int) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    return wilson_interval(successes, trials)


def build_yield_table(
    dual_oracle_summary: Mapping[str, Any],
    pool: StatePool,
    *,
    ngc_oracle: str = "smolvla",
) -> dict[str, Any]:
    """Build a causal table while retaining the original public API.

    SmolVLA NGC means a Wilson Set C label. OFT failure means an evaluated
    deterministic portfolio with no successful candidate. Unknown outcomes are
    excluded rather than silently counted as failures.
    """
    if ngc_oracle not in {"smolvla", "oft"}:
        raise ValueError("ngc_oracle must be 'smolvla' or 'oft'")
    buckets: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "ngc": 0}
    )
    missing_meta = 0
    excluded_unknown = 0
    for row in dual_oracle_summary.get("per_state") or []:
        recoverable: bool | None
        if ngc_oracle == "smolvla" and "set_label_smolvla" in row:
            label = row.get("set_label_smolvla")
            if label == "C":
                recoverable = False
            elif label in {"A", "B"}:
                recoverable = True
            else:
                recoverable = None
        else:
            value = row.get(f"recoverable_{ngc_oracle}")
            recoverable = value if isinstance(value, bool) else None
        if recoverable is None:
            excluded_unknown += 1
            continue
        key = str(row["state_key"])
        try:
            meta = _load_meta(pool, key)
        except KeyError:
            missing_meta += 1
            continue
        cell = _cell_key(meta)
        buckets[cell]["n"] += 1
        if not recoverable:
            buckets[cell]["ngc"] += 1

    rows: list[dict[str, Any]] = []
    for (suite, dim, level), counts in sorted(buckets.items()):
        n = int(counts["n"])
        ngc = int(counts["ngc"])
        lo, hi = _wilson_or_none(ngc, n)
        rows.append(
            {
                "suite": suite,
                "dim": dim,
                "level": level,
                "n": n,
                "ngc": ngc,
                "yield": (ngc / n) if n else None,
                "wilson_lower": lo,
                "wilson_upper": hi,
            }
        )

    total_n = sum(int(r["n"]) for r in rows)
    total_ngc = sum(int(r["ngc"]) for r in rows)
    lo, hi = _wilson_or_none(total_ngc, total_n)
    outcome_name = (
        "smolvla_ngc" if ngc_oracle == "smolvla" else "oft_portfolio_unrecovered"
    )
    warnings = [
        f"small cell ({row['suite']}/{row['dim']}/L{row['level']}): "
        f"n={row['n']} < 10"
        for row in rows
        if int(row["n"]) < 10
    ]
    if ngc_oracle == "oft":
        warnings.append(
            "OFT candidates within a state may be non-independent; inference and "
            "Wilson intervals use state-level portfolio outcomes only."
        )
    if excluded_unknown:
        warnings.append(f"excluded {excluded_unknown} state(s) with unknown oracle outcome")
    if missing_meta:
        warnings.append(f"excluded {missing_meta} state(s) missing pool metadata")
    return {
        "ngc_oracle": ngc_oracle,
        "outcome": outcome_name,
        "n_states": total_n,
        "n_ngc": total_ngc,
        "yield": (total_ngc / total_n) if total_n else None,
        "wilson_lower": lo,
        "wilson_upper": hi,
        "missing_meta": missing_meta,
        "excluded_unknown": excluded_unknown,
        "warnings": warnings,
        "cells": rows,
    }


def build_dual_yield_tables(
    dual_oracle_summary: Mapping[str, Any], pool: StatePool
) -> dict[str, dict[str, Any]]:
    """Build both semantically distinct causal outcomes."""
    return {
        "smolvla_ngc": build_yield_table(
            dual_oracle_summary, pool, ngc_oracle="smolvla"
        ),
        "oft_portfolio_unrecovered": build_yield_table(
            dual_oracle_summary, pool, ngc_oracle="oft"
        ),
    }


def yield_table_markdown(table: Mapping[str, Any]) -> str:
    """Render a compact markdown table for progress notes."""
    lines = [
        f"# Perturbation → {table.get('outcome', 'NGC')} yield",
        "",
        f"- NGC oracle: `{table.get('ngc_oracle')}`",
        f"- Overall yield: {table.get('yield')} "
        f"({table.get('n_ngc')}/{table.get('n_states')})",
        f"- Wilson 95% CI: [{table.get('wilson_lower')}, {table.get('wilson_upper')}]",
        *[f"- Warning: {warning}" for warning in table.get("warnings") or []],
        "",
        "| suite | dim | level | n | ngc | yield | wilson_lo | wilson_hi |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.get("cells") or []:
        y = row.get("yield")
        y_s = f"{y:.4f}" if isinstance(y, float) else "—"
        lo = row.get("wilson_lower")
        hi = row.get("wilson_upper")
        lo_s = f"{lo:.4f}" if isinstance(lo, float) else "—"
        hi_s = f"{hi:.4f}" if isinstance(hi, float) else "—"
        lines.append(
            f"| {row['suite']} | {row['dim']} | {row['level']} | "
            f"{row['n']} | {row['ngc']} | {y_s} | {lo_s} | {hi_s} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_yield_table(table: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def attach_pool_meta(
    dual_oracle_summary: Mapping[str, Any],
    pool: StatePool,
    *,
    keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return per_state rows enriched with suite/dim/level from the pool."""
    out: list[dict[str, Any]] = []
    wanted = set(keys) if keys is not None else None
    for row in dual_oracle_summary.get("per_state") or []:
        key = str(row["state_key"])
        if wanted is not None and key not in wanted:
            continue
        enriched = dict(row)
        try:
            meta = _load_meta(pool, key)
            suite, dim, level = _cell_key(meta)
            enriched["suite"] = suite
            enriched["perturb_dim"] = dim
            enriched["level"] = level
            enriched["t0"] = meta.get("t0", meta.get("step", meta.get("timestep")))
        except KeyError:
            enriched["suite"] = enriched.get("suite")
            enriched["perturb_dim"] = None
            enriched["level"] = None
        out.append(enriched)
    return out
