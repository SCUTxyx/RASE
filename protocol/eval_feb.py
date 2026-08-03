"""FEB evaluation protocol for NGC-Plus decision logs.

Zero heavy dependencies (stdlib only). Input: a list of per-episode decision
records. Output: FEB, broken-success, net-success, clean-regret.

Schema (each episode dict):
  - set_label: "A" | "B" | "C" | "uncertain"
  - chose_from_candidates: bool  (m(s) ∈ A(s); forced execution of a candidate)
  - task_success: bool
  - oracle_recoverable: bool     (max_i r̂(s,a_i) ≥ τ)
  - used_fallback: bool          (ROLLBACK/RESAMPLE/REPLAN/WAIT/ABSTAIN path)

Definitions (design report §3.2 / §5):
  - FEB (Set C): mean(chose_from_candidates | set_label == C)
  - net-success (Set C): mean(task_success | set_label == C)
  - broken-success: mean(oracle_recoverable ∧ used_fallback ∧ ¬task_success)
    over episodes in the requested scope (default: all labeled episodes)
  - clean-regret (Set A∪B): mean(used_fallback ∧ ¬task_success |
    set_label ∈ {A,B} ∧ oracle_recoverable)
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "feb-protocol/v2"
SET_C = "C"
SET_AB = frozenset({"A", "B"})


@dataclass(frozen=True)
class FebMetrics:
    protocol_version: str
    n_episodes: int
    n_set_c: int
    n_set_ab: int
    feb: float | None
    net_success: float | None
    broken_success: float | None
    clean_regret: float | None
    feb_wilson_95: dict[str, float | int | None]
    net_success_wilson_95: dict[str, float | int | None]
    broken_success_wilson_95: dict[str, float | int | None]
    clean_regret_wilson_95: dict[str, float | int | None]
    n_feb_hits: int
    n_net_success: int
    n_broken_success: int
    n_broken_success_denom: int
    n_clean_regret_hits: int
    n_clean_regret_denom: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise TypeError(f"{field} must be bool, got {type(value).__name__}")


def _mean(hits: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return hits / denom


def _wilson(hits: int, denom: int, z: float = 1.959963984540054) -> dict[str, float | int | None]:
    """Wilson score interval with explicit numerator and denominator."""
    if denom <= 0:
        return {"lower": None, "upper": None, "hits": hits, "n": denom}
    p = hits / denom
    z2 = z * z
    scale = 1 + z2 / denom
    center = (p + z2 / (2 * denom)) / scale
    radius = z * math.sqrt((p * (1 - p) + z2 / (4 * denom)) / denom) / scale
    return {
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
        "hits": hits,
        "n": denom,
    }


def evaluate_feb(
    episodes: Sequence[Mapping[str, Any]],
    *,
    broken_scope: str = "all",
) -> FebMetrics:
    """Compute FEB-family metrics from a decision log."""
    if broken_scope not in {"all", "set_c", "set_ab"}:
        raise ValueError("broken_scope must be 'all', 'set_c', or 'set_ab'")

    n_set_c = n_set_ab = 0
    n_feb_hits = n_net_success = 0
    n_broken_success = n_broken_denom = 0
    n_clean_hits = n_clean_denom = 0

    for i, ep in enumerate(episodes):
        try:
            label = str(ep["set_label"])
            chose = _as_bool(ep["chose_from_candidates"], "chose_from_candidates")
            success = _as_bool(ep["task_success"], "task_success")
            recoverable = _as_bool(ep["oracle_recoverable"], "oracle_recoverable")
            used_fb = _as_bool(ep["used_fallback"], "used_fallback")
        except KeyError as exc:
            raise KeyError(f"episode[{i}] missing field {exc.args[0]}") from exc

        in_c = label == SET_C
        in_ab = label in SET_AB
        if in_c:
            n_set_c += 1
            if chose:
                n_feb_hits += 1
            if success:
                n_net_success += 1
        if in_ab:
            n_set_ab += 1

        broken_event = recoverable and used_fb and (not success)
        if broken_scope == "all":
            n_broken_denom += 1
            if broken_event:
                n_broken_success += 1
        elif broken_scope == "set_c" and in_c:
            n_broken_denom += 1
            if broken_event:
                n_broken_success += 1
        elif broken_scope == "set_ab" and in_ab:
            n_broken_denom += 1
            if broken_event:
                n_broken_success += 1

        if in_ab and recoverable:
            n_clean_denom += 1
            if used_fb and (not success):
                n_clean_hits += 1

    return FebMetrics(
        protocol_version=PROTOCOL_VERSION,
        n_episodes=len(episodes),
        n_set_c=n_set_c,
        n_set_ab=n_set_ab,
        feb=_mean(n_feb_hits, n_set_c),
        net_success=_mean(n_net_success, n_set_c),
        broken_success=_mean(n_broken_success, n_broken_denom),
        clean_regret=_mean(n_clean_hits, n_clean_denom),
        feb_wilson_95=_wilson(n_feb_hits, n_set_c),
        net_success_wilson_95=_wilson(n_net_success, n_set_c),
        broken_success_wilson_95=_wilson(n_broken_success, n_broken_denom),
        clean_regret_wilson_95=_wilson(n_clean_hits, n_clean_denom),
        n_feb_hits=n_feb_hits,
        n_net_success=n_net_success,
        n_broken_success=n_broken_success,
        n_broken_success_denom=n_broken_denom,
        n_clean_regret_hits=n_clean_hits,
        n_clean_regret_denom=n_clean_denom,
    )


def load_decision_log(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(x) for x in payload]
    if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        return [dict(x) for x in payload["episodes"]]
    raise ValueError("decision log must be a list or {episodes:[...]}")


def metrics_nearly_equal(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    tol: float = 1e-12,
) -> bool:
    """Compare metric dicts for golden tests (None-aware, float-tolerant)."""
    def equal(va: Any, vb: Any) -> bool:
        if isinstance(va, Mapping) and isinstance(vb, Mapping):
            keys = set(va) | set(vb)
            return all(key in va and key in vb and equal(va[key], vb[key]) for key in keys)
        if isinstance(va, float) or isinstance(vb, float):
            if va is None or vb is None:
                return va == vb
            return math.isclose(float(va), float(vb), rel_tol=0.0, abs_tol=tol)
        return va == vb

    return equal(a, b)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--broken-scope",
        choices=("all", "set_c", "set_ab"),
        default="all",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    metrics = evaluate_feb(
        load_decision_log(args.log.resolve()),
        broken_scope=args.broken_scope,
    )
    payload = metrics.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"WROTE {args.output}", flush=True)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
