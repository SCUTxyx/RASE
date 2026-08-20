#!/usr/bin/env python3
"""Build root-level E3 residual supervision from exact-root recovery traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def canonical_instruction(value: str) -> str:
    """Remove LIBERO-plus camera/init suffix while retaining task semantics."""
    text = " ".join(str(value).lower().strip().split())
    return re.sub(r"\s+view(?:\s+-?\d+){5}\s+initstate\s+\d+\s*$", "", text)


def language_hash(value: str, dimensions: int = 64) -> np.ndarray:
    if dimensions < 1:
        raise ValueError("language hash dimension must be positive")
    output = np.zeros(dimensions, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", canonical_instruction(value))
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        output[index] += sign
    norm = float(np.linalg.norm(output))
    return output / norm if norm > 0 else output


def load_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--viability-audit", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument(
        "--source-chunk-dir",
        type=Path,
        help="optional exact-root Smol replan chunks; required when horizon differs from saved suffix",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=24)
    parser.add_argument("--language-dim", type=int, default=64)
    args = parser.parse_args()
    if args.horizon < 1 or args.image_size < 4:
        raise ValueError("invalid horizon or image size")

    from rase.collect.candidates import load_artifact
    from rase.collect.state_pool import StatePool
    from rase.interventions.decision_context import strict_continue_suffix

    protocol = read_json(args.protocol.resolve())
    audit = read_json(args.viability_audit.resolve())
    if audit.get("decision") != "PASS":
        raise ValueError("E3-V viability audit must PASS before dataset construction")
    source_summary = read_json(args.source_summary.resolve())
    source_outcomes = {
        str(row["state_key"]): bool(row["continue_smol_active_chunk"])
        for row in source_summary.get("per_pair") or []
    }
    audit_rows = {str(row["state_key"]): row for row in audit.get("per_root") or []}
    pool = StatePool(Path(protocol["pool"]).resolve())
    manifest = pool.manifest()
    all_keys = sorted(source_outcomes)
    if len(all_keys) != int(source_summary.get("n_states", len(all_keys))):
        raise ValueError("source summary has duplicate or missing state outcomes")

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for state_key in all_keys:
        loaded = pool.read_state(state_key, load_observations=False)
        if args.source_chunk_dir:
            source_artifact = load_artifact(args.source_chunk_dir.resolve() / f"{state_key}.npz")
            source_actions = np.asarray(source_artifact.actions, dtype=np.float32)
            if source_actions.ndim != 3 or source_actions.shape[0] != 1 or source_actions.shape[2] != 7:
                raise ValueError(f"{state_key}: invalid source chunk artifact {source_actions.shape}")
            source = source_actions[0, : args.horizon].copy()
        else:
            source = np.asarray(strict_continue_suffix(loaded.controller_state), dtype=np.float32)
        if source.shape != (args.horizon, 7):
            raise ValueError(f"{state_key}: source suffix {source.shape} != {(args.horizon, 7)}")
        source_success = source_outcomes[state_key]
        if source_success:
            target = source.copy()
            supervision = "identity_source_success"
        else:
            viability = audit_rows.get(state_key)
            if viability is None or not bool(viability.get("success")):
                excluded.append(
                    {
                        "state_key": state_key,
                        "task_id": loaded.metadata.task_id,
                        "reason": "source_and_reference_failure_no_action_target",
                    }
                )
                continue
            artifact = load_artifact(args.trajectory_dir.resolve() / f"{state_key}.npz")
            actions = np.asarray(artifact.actions, dtype=np.float32)
            if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 7:
                raise ValueError(f"{state_key}: invalid teacher trajectory {actions.shape}")
            if actions.shape[1] < args.horizon:
                raise ValueError(f"{state_key}: teacher trajectory shorter than horizon")
            target = actions[0, : args.horizon].copy()
            supervision = "successful_exact_root_oft_prefix"

        state_path = pool.root / manifest["states"][state_key]["path"]
        instruction = str(loaded.metadata.instruction)
        rows.append(
            {
                "state_key": state_key,
                "task_id": loaded.metadata.task_id,
                "suite": loaded.metadata.suite,
                "group_id": f"{loaded.metadata.suite}|{canonical_instruction(instruction)}",
                "instruction": instruction,
                "supervision": supervision,
                "source_success": source_success,
                "proprio": np.asarray(loaded.proprio, dtype=np.float32),
                "source": source,
                "target": target,
                "delta": target - source,
                "language": language_hash(instruction, args.language_dim),
                "agentview": load_rgb(state_path / "obs_agentview.png", args.image_size),
                "wrist": load_rgb(state_path / "obs_wrist.png", args.image_size),
            }
        )

    if not rows:
        raise ValueError("no supervised residual examples")
    arrays = {
        "state_key": np.asarray([row["state_key"] for row in rows]),
        "task_id": np.asarray([row["task_id"] for row in rows]),
        "suite": np.asarray([row["suite"] for row in rows]),
        "group_id": np.asarray([row["group_id"] for row in rows]),
        "supervision": np.asarray([row["supervision"] for row in rows]),
        "source_success": np.asarray([row["source_success"] for row in rows], dtype=np.bool_),
        "proprio": np.stack([row["proprio"] for row in rows]).astype(np.float32),
        "source_action": np.stack([row["source"] for row in rows]).astype(np.float32),
        "target_action": np.stack([row["target"] for row in rows]).astype(np.float32),
        "delta_target": np.stack([row["delta"] for row in rows]).astype(np.float32),
        "language_hash": np.stack([row["language"] for row in rows]).astype(np.float32),
        "agentview": np.stack([row["agentview"] for row in rows]),
        "wrist": np.stack([row["wrist"] for row in rows]),
    }
    if not all(np.isfinite(value).all() for key, value in arrays.items() if value.dtype.kind in "f"):
        raise ValueError("dataset contains non-finite numeric values")
    atomic_save_npz(args.output.resolve(), **arrays)

    delta = arrays["delta_target"]
    correction = arrays["supervision"] == "successful_exact_root_oft_prefix"
    correction_delta = np.abs(delta[correction])
    metadata = {
        "schema_version": "rase-e3-residual-root-dataset/v1",
        "status": "complete",
        "scientific_scope": "development_only; group-CV model selection then independent-root E3-U",
        "protocol_sha256": protocol.get("protocol_sha256"),
        "viability_audit": str(args.viability_audit.resolve()),
        "source_summary": str(args.source_summary.resolve()),
        "source_chunk_dir": str(args.source_chunk_dir.resolve()) if args.source_chunk_dir else None,
        "dataset": str(args.output.resolve()),
        "horizon": args.horizon,
        "image_size": args.image_size,
        "language_dim": args.language_dim,
        "n_examples": len(rows),
        "n_correction": int(correction.sum()),
        "n_identity_retention": int((~correction).sum()),
        "n_excluded_unrecoverable": len(excluded),
        "n_groups": len(set(arrays["group_id"].tolist())),
        "n_tasks": len(set(arrays["task_id"].tolist())),
        "suite_counts": dict(sorted(Counter(arrays["suite"].tolist()).items())),
        "supervision_counts": dict(sorted(Counter(arrays["supervision"].tolist()).items())),
        "delta_abs_quantiles": {
            str(q): [float(value) for value in np.quantile(correction_delta, q, axis=(0, 1))]
            for q in (0.5, 0.9, 0.95, 1.0)
        },
        "excluded": excluded,
        "training_rule": (
            "select capacity by canonical-instruction GroupKFold on this development set; "
            "train final model on all supervised roots; do not report CV as held-out system evidence"
        ),
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: metadata[key] for key in ("n_examples", "n_correction", "n_identity_retention", "n_excluded_unrecoverable", "n_groups")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
