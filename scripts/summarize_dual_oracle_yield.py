#!/usr/bin/env python3
"""Aggregate SmolVLA primary + suite-matched OFT verify into state-level yield.

Emits Y_Smol / Y_OFT / C_div plus per-candidate recoverable_* flags for later
fallback GT (does not implement fallback executors).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_successes(state: dict[str, Any]) -> tuple[int, int]:
    succ = sum(int(c.get("successes", 0)) for c in state.get("candidates") or [])
    trials = sum(int(c.get("trials", 0)) for c in state.get("candidates") or [])
    return succ, trials


def _per_candidate_flags(
    state: dict[str, Any], *, oracle: str
) -> list[dict[str, Any]]:
    out = []
    for idx, cand in enumerate(state.get("candidates") or []):
        succ = int(cand.get("successes", 0))
        trials = int(cand.get("trials", 0))
        row = {
            "state_key": state["state_key"],
            "candidate_id": idx,
            "oracle": oracle,
            "successes": succ,
            "trials": trials,
            "recoverable_smolvla": None,
            "recoverable_oft": None,
        }
        if oracle == "smolvla":
            row["recoverable_smolvla"] = succ > 0
        elif oracle == "oft":
            row["recoverable_oft"] = succ > 0
        out.append(row)
    return out


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
    args = parser.parse_args()

    smol = _load(args.smolvla_summary.resolve())
    smol_by_key = {s["state_key"]: s for s in smol.get("per_state") or []}

    oft_by_key: dict[str, dict[str, Any]] = {}
    oft_suite_of: dict[str, str] = {}
    suite_raw: dict[str, dict[str, int]] = {}
    for item in args.oft_summary:
        if "=" not in item:
            raise SystemExit(f"expected SUITE=PATH, got {item!r}")
        suite, path_s = item.split("=", 1)
        payload = _load(Path(path_s).resolve())
        succ_tot = trials_tot = 0
        for st in payload.get("per_state") or []:
            key = st["state_key"]
            oft_by_key[key] = st
            oft_suite_of[key] = suite
            s, t = _state_successes(st)
            succ_tot += s
            trials_tot += t
        suite_raw[suite] = {
            "successes": succ_tot,
            "trials": trials_tot,
            "n_states": len(payload.get("per_state") or []),
            "rollouts_this_process": int(payload.get("rollouts_this_process") or 0),
        }

    all_keys = sorted(set(smol_by_key) | set(oft_by_key))
    per_state: list[dict[str, Any]] = []
    cand_rows: list[dict[str, Any]] = []
    n_smol_rec = n_oft_rec = n_div = 0

    for key in all_keys:
        sm = smol_by_key.get(key)
        ot = oft_by_key.get(key)
        sm_s = sm_t = 0
        ot_s = ot_t = 0
        if sm is not None:
            sm_s, sm_t = _state_successes(sm)
            cand_rows.extend(_per_candidate_flags(sm, oracle="smolvla"))
        if ot is not None:
            ot_s, ot_t = _state_successes(ot)
            cand_rows.extend(_per_candidate_flags(ot, oracle="oft"))
        rec_smol = sm_s > 0
        rec_oft = ot_s > 0
        divergent = rec_oft and not rec_smol
        if rec_smol:
            n_smol_rec += 1
        if rec_oft:
            n_oft_rec += 1
        if divergent:
            n_div += 1
        # Merge candidate-level GT when both oracles present for same cand id.
        if sm is not None and ot is not None:
            for idx, (sc, oc) in enumerate(
                zip(sm.get("candidates") or [], ot.get("candidates") or [])
            ):
                cand_rows.append(
                    {
                        "state_key": key,
                        "candidate_id": idx,
                        "oracle": "merged",
                        "successes_smolvla": int(sc.get("successes", 0)),
                        "trials_smolvla": int(sc.get("trials", 0)),
                        "successes_oft": int(oc.get("successes", 0)),
                        "trials_oft": int(oc.get("trials", 0)),
                        "recoverable_smolvla": int(sc.get("successes", 0)) > 0,
                        "recoverable_oft": int(oc.get("successes", 0)) > 0,
                    }
                )
        per_state.append(
            {
                "state_key": key,
                "suite": oft_suite_of.get(key),
                "set_label_smolvla": (sm or {}).get("set_label"),
                "smolvla_successes": sm_s,
                "smolvla_trials": sm_t,
                "oft_successes": ot_s,
                "oft_trials": ot_t,
                "recoverable_smolvla": rec_smol,
                "recoverable_oft": rec_oft,
                "divergent_oft_only": divergent,
            }
        )

    n = len(all_keys)
    out = {
        "n_states": n,
        "Y_Smol": (n_smol_rec / n) if n else 0.0,
        "Y_OFT": (n_oft_rec / n) if n else 0.0,
        "C_div": (n_div / n) if n else 0.0,
        "n_recoverable_smolvla": n_smol_rec,
        "n_recoverable_oft": n_oft_rec,
        "n_divergent_oft_only": n_div,
        "smolvla_raw": {
            "successes": sum(r["smolvla_successes"] for r in per_state),
            "trials": sum(r["smolvla_trials"] for r in per_state),
            "label_counts": smol.get("label_counts"),
            "rollouts_this_process": smol.get("rollouts_this_process"),
        },
        "oft_raw_by_suite": suite_raw,
        "per_state": per_state,
        "per_candidate_gt": [r for r in cand_rows if r.get("oracle") == "merged"],
        "sources": {
            "smolvla_summary": str(args.smolvla_summary.resolve()),
            "oft_summaries": list(args.oft_summary),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in (
        "n_states", "Y_Smol", "Y_OFT", "C_div",
        "n_recoverable_smolvla", "n_recoverable_oft", "n_divergent_oft_only",
    )}, indent=2), flush=True)
    print(f"WROTE {args.output}", flush=True)

    if args.markdown is not None:
        lines = [
            "# Dual-oracle yield",
            "",
            f"| metric | value |",
            f"|---|---|",
            f"| n_states | {n} |",
            f"| Y_Smol | {out['Y_Smol']:.4f} ({n_smol_rec}/{n}) |",
            f"| Y_OFT | {out['Y_OFT']:.4f} ({n_oft_rec}/{n}) |",
            f"| C_div | {out['C_div']:.4f} ({n_div}/{n}) |",
            "",
            "## Per suite OFT raw",
            "",
            "| suite | successes/trials | states |",
            "|---|---|---|",
        ]
        for suite, info in sorted(suite_raw.items()):
            lines.append(
                f"| {suite} | {info['successes']}/{info['trials']} | {info['n_states']} |"
            )
        lines.append("")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"WROTE {args.markdown}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
