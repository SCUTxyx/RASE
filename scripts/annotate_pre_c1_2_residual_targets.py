#!/usr/bin/env python3
"""Annotate revised dataset with base action chunks and residual targets Δa = a_OFT - a_base.

This prepares residual correction learning without changing the deploy-time OFT ban.
Does not train; writes an annotated JSONL + NPZ sidecars.
"""

from __future__ import annotations

import argparse
import json
import sys
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

from rase.adapt.pre_c1_2 import load_protocol_lock
from rase.collect.forked_rollout import InProcessSmolVLAContinuation, load_smolvla_policy_bundle
from rase.collect.state_pool import StatePool
from train_smolvla_recovery_lora import _load_jsonl, _unpack_chunk_observation  # type: ignore


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        "--input-jsonl",
        type=Path,
        default=Path("runs/rase_pre_c1_2_revised_dataset_r1_v1.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("runs/rase_pre_c1_2_revised_dataset_r1_residual_v1.jsonl"),
    )
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=Path("runs/rase_pre_c1_2_residual_targets_r1_v1"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=2026080405)
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    adapter = dict(cfg.get("adapter_config") or {})
    rows = _load_jsonl(args.input_jsonl.resolve())
    rows = [
        r
        for r in rows
        if str(r.get("source")) == "student_query_state"
        and int(r.get("offset_from_student_state", 0) or 0) == 0
        and bool(r.get("teacher_rollout_success", True))
    ]
    if args.smoke:
        rows = rows[:4]
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
    # Base only (no recovery LoRA) for nominal proposal.
    cont = InProcessSmolVLAContinuation(
        bundle,
        temperature=float(adapter.get("continuation_temperature", 0.5)),
        seed=int(args.seed),
    )
    sidecar_dir = args.sidecar_dir.resolve()
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        chunk_path = Path(str(row.get("chunk_path") or ""))
        if not chunk_path.is_file():
            continue
        packed = dict(np.load(str(chunk_path), allow_pickle=False))
        if "oft_action_chunk" not in packed:
            continue
        oft = np.asarray(packed["oft_action_chunk"], dtype=np.float32)
        obs = _unpack_chunk_observation(packed)
        task = str(obs.pop("task", "") or row.get("task") or "")
        cont.reset()
        # Predict a base action chunk of matching horizon by repeated act on same obs
        # (open-loop nominal proposal at the query state).
        base_actions = []
        for _ in range(int(oft.shape[0])):
            action = np.asarray(cont.act(obs, task=task), dtype=np.float32).reshape(-1)
            base_actions.append(action)
            # Keep same observation: residual target is state-conditioned correction direction.
        base = np.stack(base_actions, axis=0)
        if base.shape != oft.shape:
            # Truncate to common prefix.
            t = min(base.shape[0], oft.shape[0])
            base = base[:t]
            oft_use = oft[:t]
        else:
            oft_use = oft
        residual = oft_use - base
        sample_id = str(row.get("sample_id") or f"residual_{idx}")
        side = sidecar_dir / f"{sample_id.replace('/', '_')}.npz"
        np.savez_compressed(
            side,
            base_action_chunk=base.astype(np.float32),
            oft_action_chunk=oft_use.astype(np.float32),
            residual_action_chunk=residual.astype(np.float32),
        )
        out_rows.append(
            {
                **row,
                "residual_target_path": str(side),
                "residual_form": "a_oft - a_base",
                "residual_clip_suggested": True,
                "dataset_role": "student_state_recovery_residual",
            }
        )
        if (idx + 1) % 20 == 0:
            print(f"residual annotate {idx+1}/{len(rows)}", flush=True)

    out = args.output_jsonl.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema_version": "rase-pre-c1-2-residual-annotate/v1",
        "n_input_query_rows": len(rows),
        "n_annotated": len(out_rows),
        "output_jsonl": str(out),
        "sidecar_dir": str(sidecar_dir),
        "protocol_revision": dict(lock.get("revision") or {}),
        "note": "Use with revised short-horizon training; runtime OFT still forbidden.",
    }
    _write(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, sort_keys=True))
    print(f"PRE_C1_2_RESIDUAL_ANNOTATE_DONE output={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
