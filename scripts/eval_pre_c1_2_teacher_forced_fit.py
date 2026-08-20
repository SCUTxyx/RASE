#!/usr/bin/env python3
"""R0-A: teacher-forced native flow fit for base vs current PRE-C1.1 adapter."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rase.adapt.pre_c1_2 import load_protocol_lock, native_flow_forward_weighted
from rase.adapt.recovery_lora import load_lora_onto_policy, set_adapter_enabled
from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.collect.state_pool import StatePool
from train_smolvla_recovery_lora import _cache_or_build_batch, _load_jsonl  # type: ignore


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _row_bucket(row: dict[str, Any]) -> str:
    source = str(row.get("source") or "")
    role = str(row.get("dataset_role") or "")
    if bool(row.get("clean_flag")) or source == "clean_retention" or role == "clean_retention":
        return "clean"
    if source == "student_query_state":
        return "r1_student_query"
    if source == "teacher_suffix_after_student_query":
        return "r1_teacher_suffix"
    if source == "original_recovery" or role == "original_recovery":
        return "original_c1_1"
    # Legacy PRE-C1.1 rows without explicit source.
    if not source and not bool(row.get("clean_flag")):
        return "original_c1_1"
    return "other"


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "base_loss_full": None,
            "adapted_loss_full": None,
            "adapted_minus_base": None,
        }
    base = [float(r["base"]["loss_full"]) for r in rows]
    adapted = [float(r["adapted"]["loss_full"]) for r in rows]
    out = {
        "n": len(rows),
        "base_loss_full": _mean(base),
        "adapted_loss_full": _mean(adapted),
        "adapted_minus_base": _mean(adapted) - _mean(base),
        "base_loss_prefix_2": _mean([float(r["base"]["loss_prefix_2"]) for r in rows]),
        "adapted_loss_prefix_2": _mean([float(r["adapted"]["loss_prefix_2"]) for r in rows]),
        "base_loss_prefix_4": _mean([float(r["base"]["loss_prefix_4"]) for r in rows]),
        "adapted_loss_prefix_4": _mean([float(r["adapted"]["loss_prefix_4"]) for r in rows]),
        "base_loss_tail": _mean([float(r["base"]["loss_tail"]) for r in rows]),
        "adapted_loss_tail": _mean([float(r["adapted"]["loss_tail"]) for r in rows]),
    }
    for key in (
        "prefix_translation_error",
        "prefix_rotation_error",
        "prefix_gripper_error",
    ):
        if key in rows[0]["adapted"]:
            out[f"base_{key}"] = _mean([float(r["base"].get(key, float("nan"))) for r in rows])
            out[f"adapted_{key}"] = _mean(
                [float(r["adapted"].get(key, float("nan"))) for r in rows]
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_protocol_lock.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collect_pre_c0_deviation_pilot24.json"),
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=Path("runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl"),
    )
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=Path("runs/rase_pre_c1_2_distill_dataset_r1_v1.benchmark-splits.json"),
    )
    parser.add_argument(
        "--original-dataset-jsonl",
        type=Path,
        default=Path("runs/rase_pre_c1_1_distill_dataset_v1.jsonl"),
        help="Used if R1 merged dataset is missing or for explicit original-state coverage.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("runs/rase_pre_c1_1_lora_train_v1/adapter_final"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("runs/rase_pre_c1_2_tensor_cache_r0_tf_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/rase_pre_c1_2_r0_teacher_forced_v1.json"),
    )
    parser.add_argument(
        "--row-jsonl",
        type=Path,
        default=Path("runs/rase_pre_c1_2_r0_teacher_forced_rows_v1.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    adapter = dict(cfg.get("adapter_config") or {})
    pool = StatePool(Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve())

    rows: list[dict[str, Any]] = []
    if args.dataset_jsonl.is_file():
        rows.extend(_load_jsonl(args.dataset_jsonl.resolve()))
    elif args.original_dataset_jsonl.is_file():
        rows.extend(_load_jsonl(args.original_dataset_jsonl.resolve()))
    else:
        raise SystemExit("no dataset available for teacher-forced fit")

    # Ensure original C1.1 rows are present even if R1 merge omitted some.
    if args.original_dataset_jsonl.is_file() and args.dataset_jsonl.is_file():
        have = {str(r.get("sample_id") or r.get("chunk_path") or id(r)) for r in rows}
        for row in _load_jsonl(args.original_dataset_jsonl.resolve()):
            key = str(row.get("sample_id") or row.get("chunk_path") or "")
            if key and key not in have:
                rows.append({**row, "source": row.get("source") or (
                    "clean_retention" if row.get("clean_flag") else "original_recovery"
                )})

    split_map: dict[str, str] = {}
    if args.splits_json.is_file():
        splits = json.loads(args.splits_json.read_text(encoding="utf-8"))
        for ep in splits.get("train_episodes") or []:
            split_map[str(ep)] = "train"
        for ep in splits.get("val_episodes") or []:
            split_map[str(ep)] = "val"
    for row in rows:
        if "episode_id" not in row or row["episode_id"] is None:
            row["episode_id"] = str(row.get("anchor_id") or row.get("state_key") or row.get("sample_id"))
        row["_split"] = split_map.get(str(row["episode_id"]), "unsplit")
        row["_bucket"] = _row_bucket(row)

    # Prefer successful teacher rows only for recovery buckets.
    rows = [
        r
        for r in rows
        if r["_bucket"] == "clean" or bool(r.get("teacher_rollout_success", True))
    ]
    if args.smoke:
        # Keep a few from each primary bucket.
        picked: list[dict[str, Any]] = []
        for bucket in ("original_c1_1", "r1_student_query", "clean"):
            bucket_rows = [r for r in rows if r["_bucket"] == bucket][:4]
            picked.extend(bucket_rows)
        rows = picked or rows[:8]
    if args.limit:
        rows = rows[: args.limit]

    bundle = load_smolvla_policy_bundle(
        Path(adapter.get("policy_path") or "ckpts/smolvla_libero"),
        device=str(adapter.get("device", "cuda")),
        num_steps=int(adapter.get("num_steps", 10)),
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=Path(adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct"),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    handle = load_lora_onto_policy(bundle["policy"], str(args.adapter_dir.resolve()))
    bundle["policy"] = handle.policy
    policy = handle.policy
    policy.eval()

    detail_rows: list[dict[str, Any]] = []
    cache_dir = args.cache_dir.resolve()
    with torch.no_grad():
        for idx, row in enumerate(rows):
            try:
                batch = _cache_or_build_batch(
                    cache_dir=cache_dir,
                    pool=pool,
                    bundle=bundle,
                    row=row,
                    libero_plus_root=adapter.get("libero_plus_root"),
                    observation_height=int(adapter.get("observation_height", 360)),
                    observation_width=int(adapter.get("observation_width", 360)),
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic row skip
                detail_rows.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "bucket": row["_bucket"],
                        "split": row["_split"],
                        "error": str(exc),
                    }
                )
                continue

            arm_metrics: dict[str, dict[str, float]] = {}
            for arm_name, enabled in (("base", False), ("adapted", True)):
                set_adapter_enabled(handle, enabled)
                _loss, metrics = native_flow_forward_weighted(
                    policy, batch, enable_weighting=False
                )
                arm_metrics[arm_name] = {k: float(v) for k, v in metrics.items()}
            detail = {
                "sample_id": row.get("sample_id"),
                "state_key": row.get("state_key") or row.get("anchor_id") or row.get("failure_key"),
                "episode_id": row.get("episode_id"),
                "suite": row.get("suite"),
                "source": row.get("source"),
                "dataset_role": row.get("dataset_role"),
                "offset_from_student_state": row.get("offset_from_student_state"),
                "query_trigger": row.get("query_trigger"),
                "bucket": row["_bucket"],
                "split": row["_split"],
                "base": arm_metrics["base"],
                "adapted": arm_metrics["adapted"],
                "adapted_minus_base_loss_full": float(
                    arm_metrics["adapted"]["loss_full"] - arm_metrics["base"]["loss_full"]
                ),
            }
            detail_rows.append(detail)
            if (idx + 1) % 25 == 0:
                print(f"TF progress {idx+1}/{len(rows)}", flush=True)

    ok_rows = [r for r in detail_rows if "error" not in r]
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        by_bucket[str(row["bucket"])].append(row)
        by_split_bucket[f"{row['split']}:{row['bucket']}"].append(row)

    summary = {
        "schema_version": "rase-pre-c1-2-r0-teacher-forced/v1",
        "adapter_dir": str(args.adapter_dir),
        "n_rows_requested": len(rows),
        "n_rows_scored": len(ok_rows),
        "n_rows_error": sum(1 for r in detail_rows if "error" in r),
        "original_c1_1": _summarize(by_bucket.get("original_c1_1", [])),
        "r1_student_query": _summarize(by_bucket.get("r1_student_query", [])),
        "r1_teacher_suffix": _summarize(by_bucket.get("r1_teacher_suffix", [])),
        "clean": _summarize(by_bucket.get("clean", [])),
        "by_split_bucket": {k: _summarize(v) for k, v in sorted(by_split_bucket.items())},
        "protocol_revision": dict(lock.get("revision") or {}),
        "primary_metric": "native_flow_loss",
        "auxiliary_sampled_action_mse": False,
    }
    # Convenience aliases used by decision helper.
    summary["original"] = summary["original_c1_1"]
    summary["query"] = summary["r1_student_query"]

    args.row_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.row_jsonl.open("w", encoding="utf-8") as handle_out:
        for row in detail_rows:
            handle_out.write(json.dumps(row, sort_keys=True) + "\n")
    _write(args.output.resolve(), summary)
    print(json.dumps({k: summary[k] for k in summary if k != "by_split_bucket"}, sort_keys=True))
    print(f"PRE_C1_2_R0_TEACHER_FORCED_DONE output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
