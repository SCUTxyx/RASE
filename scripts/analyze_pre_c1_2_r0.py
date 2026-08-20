#!/usr/bin/env python3
"""Analyze R0 diagnostics and emit branch decision (never auto-unlocks legacy E3/E4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rase.adapt.pre_c1_2 import load_protocol_lock, r0_decision_from_diagnostics


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coverage_ok(
    *,
    recoverability: dict[str, Any],
    locked_keys: list[str],
    ks: list[int],
) -> dict[str, Any]:
    anchors = set(recoverability.get("coverage", {}).get("anchors") or [])
    missing = [k for k in locked_keys if k not in anchors]
    curves = dict(recoverability.get("curves") or {})
    base = {int(k): v for k, v in dict(curves.get("base") or {}).items()}
    adapted = {int(k): v for k, v in dict(curves.get("adapted") or {}).items()}
    missing_k = []
    for k in ks:
        if int(k) == 0:
            continue
        if int(k) not in base or int(k) not in adapted:
            missing_k.append(int(k))
    ok = not missing and not missing_k and recoverability.get("R_oft_k0") is not None
    return {
        "ok": ok,
        "missing_anchors": missing,
        "missing_k": missing_k,
        "n_anchors": len(anchors),
        "n_locked": len(locked_keys),
    }


def _progress_md(
    decision: dict[str, Any],
    *,
    coverage: dict[str, Any],
    teacher_forced: dict[str, Any],
    recoverability: dict[str, Any],
) -> str:
    m = decision["metrics"]
    lines = [
        "# PRE-C1.2 R0 Decision",
        "",
        f"- branch: `{decision['branch']}`",
        f"- rationale: {decision['rationale']}",
        f"- legacy E3/E4 allowed: `{decision['legacy_e3_e4_allowed']}`",
        f"- capacity ladder allowed: `{decision['capacity_ladder_allowed']}`",
        f"- coverage ok: `{coverage['ok']}` ({coverage['n_anchors']}/{coverage['n_locked']} anchors)",
        "",
        "## Teacher-forced",
        "",
        f"- original adapted/base loss: `{m['tf_original_adapted_loss']:.4f}` / `{m['tf_original_base_loss']:.4f}` good=`{m['tf_original_good']}`",
        f"- R1 query adapted/base loss: `{m['tf_query_adapted_loss']:.4f}` / `{m['tf_query_base_loss']:.4f}` good=`{m['tf_query_good']}`",
        "",
        "## Recoverability",
        "",
        f"- R(OFT,0)={m['R_oft_0']:.3f}",
        f"- R(base,1)={m['R_base_1']:.3f} R(adapted,1)={m['R_adapted_1']:.3f}",
        f"- R(base,4)={m['R_base_4']:.3f} R(adapted,4)={m['R_adapted_4']:.3f}",
        f"- decay_fast=`{m['decay_fast']}`",
        f"- P(OFT success|student query)=`{m['p_oft_success_student_query']:.3f}`",
        "",
        "## Next actions",
        "",
    ]
    for action in decision["next_actions"]:
        lines.append(f"- `{action}`")
    lines.extend(
        [
            "",
            "## Curves",
            "",
            "```json",
            json.dumps(recoverability.get("curves") or {}, indent=2, sort_keys=True),
            "```",
            "",
            "## Teacher-forced buckets",
            "",
            "```json",
            json.dumps(
                {
                    "original_c1_1": teacher_forced.get("original_c1_1"),
                    "r1_student_query": teacher_forced.get("r1_student_query"),
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_protocol_lock.yaml"),
    )
    parser.add_argument(
        "--teacher-forced-json",
        type=Path,
        default=Path("runs/rase_pre_c1_2_r0_teacher_forced_v1.json"),
    )
    parser.add_argument(
        "--recoverability-json",
        type=Path,
        default=Path("runs/rase_pre_c1_2_r0_recoverability_v1/summary.json"),
    )
    parser.add_argument(
        "--dagger-qc-json",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_dagger_global_qc_r1.json"),
    )
    parser.add_argument(
        "--state-keys-json",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/rase_pre_c1_2_r0_decision_v1.json"),
    )
    parser.add_argument(
        "--progress-md",
        type=Path,
        default=Path("progress/2026-08-05_pre_c1_2_r0_decision.md"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Emit a provisional decision even if coverage is incomplete.",
    )
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    locked = list(json.loads(args.state_keys_json.read_text(encoding="utf-8"))["state_keys"])
    ks = list((lock.get("r0") or {}).get("k_env_steps") or [0, 1, 2, 4, 8, 16])

    if not args.teacher_forced_json.is_file():
        raise SystemExit(f"missing teacher-forced summary: {args.teacher_forced_json}")
    if not args.recoverability_json.is_file():
        raise SystemExit(f"missing recoverability summary: {args.recoverability_json}")

    teacher_forced = json.loads(args.teacher_forced_json.read_text(encoding="utf-8"))
    recoverability = json.loads(args.recoverability_json.read_text(encoding="utf-8"))
    dagger_qc = (
        json.loads(args.dagger_qc_json.read_text(encoding="utf-8"))
        if args.dagger_qc_json.is_file()
        else {}
    )
    coverage = _coverage_ok(recoverability=recoverability, locked_keys=locked, ks=ks)
    if not coverage["ok"] and not args.allow_incomplete:
        payload = {
            "schema_version": "rase-pre-c1-2-r0-decision/v1",
            "blocked": True,
            "reason": "incomplete R0 coverage; refusing automatic training decision",
            "coverage": coverage,
            "legacy_e3_e4_allowed": False,
        }
        _write(args.output.resolve(), payload)
        print(json.dumps(payload, sort_keys=True))
        print("PRE_C1_2_R0_DECISION_BLOCKED incomplete coverage", flush=True)
        return 3

    decision = r0_decision_from_diagnostics(
        teacher_forced=teacher_forced,
        recoverability=recoverability,
        dagger_qc=dagger_qc,
    )
    decision["coverage"] = coverage
    decision["provisional"] = not coverage["ok"]
    decision["inputs"] = {
        "teacher_forced_json": str(args.teacher_forced_json),
        "recoverability_json": str(args.recoverability_json),
        "dagger_qc_json": str(args.dagger_qc_json) if args.dagger_qc_json.is_file() else None,
    }
    decision["protocol_revision"] = dict(lock.get("revision") or {})
    _write(args.output.resolve(), decision)
    _write(
        args.progress_md.resolve(),
        _progress_md(
            decision,
            coverage=coverage,
            teacher_forced=teacher_forced,
            recoverability=recoverability,
        ),
    )
    print(json.dumps(decision, sort_keys=True))
    print(f"PRE_C1_2_R0_DECISION_DONE output={args.output} branch={decision['branch']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
