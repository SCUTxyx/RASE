#!/usr/bin/env python3
"""Global DAgger Round-1 QC (aggregate root run summaries; never suite-local dagger_qc.json)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rase.adapt.pre_c1_2 import aggregate_dagger_global_qc, load_protocol_lock


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_run_payloads(dagger_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(dagger_dir.glob("*.json")):
        if path.name in {"dagger_qc.json", "run_manifest.json"}:
            continue
        if path.name.endswith("_qc.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == "rase-pre-c1-2-dagger-run/v1":
            payloads.append(payload)
    return payloads


def _progress_md(qc: dict[str, Any]) -> str:
    lines = [
        "# PRE-C1.2 DAgger Round 1 Global QC",
        "",
        f"- anchors covered: `{qc['anchors_covered']}/{qc['anchors_locked']}`",
        f"- OFT queries: `{qc['n_oft_queries']}`",
        f"- successful teacher queries: `{qc['n_successful_teacher_queries']}`",
        f"- P(OFT success | student query): `{qc['p_oft_success_given_student_query']:.4f}`",
        f"- accepted rows: `{qc['n_accepted_rows']}`",
        f"- median teacher recovery length: `{qc['median_teacher_recovery_length']:.1f}`",
        f"- meets per-anchor Round1 minimum (all): `{qc['meets_round1_minimum_all_anchors']}`",
        f"- failed teacher JSON count (not in BC): `{qc['failed_teacher_json_count']}`",
        "",
        "## Source / offset",
        "",
        f"- sources: `{json.dumps(qc['source_counts'], sort_keys=True)}`",
        f"- offsets: `{json.dumps(qc['offset_counts'], sort_keys=True)}`",
        "",
        "## Successful queries by trigger",
        "",
    ]
    for trigger, stats in dict(qc.get("by_trigger_successful_queries") or {}).items():
        lines.append(
            f"- `{trigger}`: successful_queries={stats['successful_queries']} "
            f"accepted_rows={stats['accepted_rows']}"
        )
    lines.extend(["", "## Per-anchor minimums", ""])
    for anchor, stats in dict(qc.get("per_anchor") or {}).items():
        lines.append(
            f"- `{anchor[:16]}…`: seeds={stats['n_seeds']} "
            f"unique_q={stats['unique_student_query_states']} "
            f"success_relabel={stats['successful_teacher_relabels']} "
            f"near_chunks={stats['accepted_query_near_chunks']} "
            f"ok={stats['meets_round1_minimum']} triggers={stats.get('triggers')}"
        )
    if qc.get("missing_anchors"):
        lines.extend(["", "## Missing anchors", ""])
        for key in qc["missing_anchors"]:
            lines.append(f"- `{key}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_protocol_lock.yaml"),
    )
    parser.add_argument(
        "--dagger-dir",
        type=Path,
        default=Path("runs/rase_pre_c1_2_dagger_r1_v1"),
    )
    parser.add_argument(
        "--state-keys-json",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json"),
    )
    parser.add_argument("--dataset-jsonl", type=Path, default=None)
    parser.add_argument("--splits-json", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_dagger_global_qc_r1.json"),
    )
    parser.add_argument(
        "--progress-md",
        type=Path,
        default=Path("progress/2026-08-05_pre_c1_2_dagger_r1_global_qc.md"),
    )
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    keys_payload = json.loads(args.state_keys_json.read_text(encoding="utf-8"))
    locked = list(keys_payload.get("state_keys") or [])
    dagger_dir = args.dagger_dir.resolve()
    payloads = _load_run_payloads(dagger_dir)
    failed_teacher = len(list((dagger_dir / "failed_teacher").glob("*.json"))) if (
        dagger_dir / "failed_teacher"
    ).is_dir() else 0
    qc = aggregate_dagger_global_qc(
        payloads,
        locked_state_keys=locked,
        seeds_per_anchor=int(lock["dagger_round_1_minimum"].get("seeds_per_anchor", 5)),
        mins=dict(lock["dagger_round_1_minimum"]),
        failed_teacher_count=failed_teacher,
    )

    if args.dataset_jsonl and args.dataset_jsonl.is_file():
        n_lines = sum(1 for line in args.dataset_jsonl.read_text(encoding="utf-8").splitlines() if line.strip())
        qc["dataset_jsonl"] = str(args.dataset_jsonl)
        qc["dataset_n_rows"] = n_lines
    if args.splits_json and args.splits_json.is_file():
        splits = json.loads(args.splits_json.read_text(encoding="utf-8"))
        qc["splits"] = {
            "n_train_episodes": len(splits.get("train_episodes") or []),
            "n_val_episodes": len(splits.get("val_episodes") or []),
            "n_train_rows": splits.get("n_train_rows"),
            "n_val_rows": splits.get("n_val_rows"),
            "leakage_episode_overlap": splits.get("leakage_episode_overlap"),
        }

    qc["protocol_revision"] = dict(lock.get("revision") or {})
    _write(args.output.resolve(), qc)
    _write(args.progress_md.resolve(), _progress_md(qc))
    print(json.dumps({k: qc[k] for k in qc if k != "per_anchor"}, sort_keys=True))
    print(f"PRE_C1_2_DAGGER_GLOBAL_QC_DONE output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
