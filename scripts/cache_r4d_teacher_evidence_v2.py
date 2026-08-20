#!/usr/bin/env python3
"""Milestone 3a: Teacher evidence cache.

Runs the frozen V-JEPA 2-AC encoder over trajectory windows and caches
pooled latents + K-step latent deltas for student-action and OFT-action
rollouts.  Produces a per-fold evidence cache consumed by the
LightRiskStudent distillation trainer.

When the V-JEPA teacher is unavailable (e.g. checkpoint too large for the
GPU budget), this script falls back to producing an empty cache with a
`teacher_unavailable` flag, which the distillation trainer interprets as
"hard-label-only mode".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _frame_from_b64(b64: str) -> np.ndarray:
    import base64
    import io

    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), np.uint8)


def compute_evidence(
    *,
    frames: list[np.ndarray],
    student_chunk: np.ndarray,
    oft_chunk: np.ndarray,
    encoder: Any,
    k: int,
) -> dict[str, Any]:
    """Compute teacher evidence for one window.

    Returns pooled latents for the context window, the delta between
    student-rollout and OFT-rollout pooled latents after K steps, and the
    teacher's predicted action-conditioned latent improvement over the
    persistence baseline (empty on failure).
    """
    if encoder is None:
        return {}
    try:
        import torch

        pooled = encoder.pooled_latent(frames)  # (D,)
        student_deltas = encoder.predict_k_step(frames, student_chunk, k=k)  # (K, D)
        oft_deltas = encoder.predict_k_step(frames, oft_chunk, k=k)  # (K, D)
        if len(student_deltas) == 0 or len(oft_deltas) == 0:
            return {"latent": pooled, "note": "empty rollout"}
        # Average over the K steps
        student_mean = student_deltas.mean(axis=0)
        oft_mean = oft_deltas.mean(axis=0)
        # Improvement of action-conditioned prediction over persistence is
        # measured by the normalized delta magnitude difference.
        student_norm = float(np.linalg.norm(student_mean))
        oft_norm = float(np.linalg.norm(oft_mean))
        return {
            "latent": pooled,
            "student_delta": student_mean,
            "oft_delta": oft_mean,
            "delta_magnitude": {"student": student_norm, "oft": oft_norm},
        }
    except Exception as exc:  # teacher failure must not block the cache build
        return {"error": str(exc)}


def compute_evidence_fast(
    *,
    frames: list[np.ndarray],
    student_chunk: np.ndarray,
    oft_chunk: np.ndarray,
    encoder: Any,
    k: int,
) -> dict[str, Any]:
    """Teacher evidence for one window with a single encoder forward.

    Encodes the frame stack once, then rolls out the action-conditioned
    predictor from the cached latent.  Returns pooled latent, per-VLA deltas,
    and delta magnitudes.
    """
    if encoder is None:
        return {}
    try:
        latent = encoder.encode_stack(frames)  # (1, N, D)
        pooled = latent.mean(dim=1).squeeze(0).float().cpu().numpy()  # (D,)
        out: dict[str, Any] = {"latent": pooled}

        # When the dataset only contains OFT actions (no per-VLA student chunk),
        # student and OFT rollouts are identical — run the rollout once.
        if np.array_equal(student_chunk, oft_chunk):
            deltas = encoder.rollout_from_latent(latent, oft_chunk, k=k)  # (K, D)
            if len(deltas) == 0:
                return {**out, "note": "empty rollout"}
            mean = deltas.mean(axis=0)
            out.update({
                "student_delta": mean,
                "oft_delta": mean,
                "delta_magnitude": {
                    "student": float(np.linalg.norm(mean)),
                    "oft": float(np.linalg.norm(mean)),
                },
            })
        else:
            s_deltas = encoder.rollout_from_latent(latent, student_chunk, k=k)
            o_deltas = encoder.rollout_from_latent(latent, oft_chunk, k=k)
            if len(s_deltas) == 0 or len(o_deltas) == 0:
                return {**out, "note": "empty rollout"}
            s_mean = s_deltas.mean(axis=0)
            o_mean = o_deltas.mean(axis=0)
            out.update({
                "student_delta": s_mean,
                "oft_delta": o_mean,
                "delta_magnitude": {
                    "student": float(np.linalg.norm(s_mean)),
                    "oft": float(np.linalg.norm(o_mean)),
                },
            })
        return out
    except Exception as exc:  # teacher failure must not block the cache build
        return {"error": str(exc)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True,
                   help="worldmodel_windows jsonl (Milestone 2 output)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--boundary-dataset",
        type=Path,
        default=None,
        help=("Optional boundary_transitions.jsonl. When supplied, emit exactly one "
              "teacher record per boundary and use its Student action chunk; the "
              "nearest causal video window supplies frames and the OFT chunk."),
    )
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--teacher-ckpt", type=Path,
                   help="V-JEPA checkpoint dir; if absent, empty cache")
    args = p.parse_args()

    window_rows = read_jsonl(args.dataset)
    rows: list[dict[str, Any]] = []
    if args.boundary_dataset is not None:
        boundaries = read_jsonl(args.boundary_dataset)
        by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for window_row in window_rows:
            by_state[str(window_row["state_key"])].append(window_row)
        for state_windows in by_state.values():
            state_windows.sort(key=lambda r: int(r.get("window_start", 0)))

        unmatched = 0
        for boundary in boundaries:
            candidates = by_state.get(str(boundary["state_key"]), [])
            if not candidates:
                unmatched += 1
                continue
            elapsed = int(boundary.get("elapsed_oft_steps", 0))
            causal = [w for w in candidates if int(w.get("window_start", 0)) <= elapsed]
            chosen = causal[-1] if causal else candidates[0]
            rows.append({"window": chosen, "boundary": boundary})
        if unmatched:
            print(f"WARNING: {unmatched}/{len(boundaries)} boundaries have no video window",
                  file=sys.stderr)
    else:
        rows = [{"window": row, "boundary": None} for row in window_rows]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    encoder = None
    teacher_unavailable = False
    if args.teacher_ckpt and args.teacher_ckpt.is_dir():
        try:
            from rase.world_models.vjepa2_adapter import VJEPA2ACEncoder

            encoder = VJEPA2ACEncoder(args.teacher_ckpt, device=args.device)
            encoder.load()
            print(f"teacher loaded: {args.teacher_ckpt}", flush=True)
        except Exception as exc:
            print(f"WARNING teacher load failed: {exc}", file=sys.stderr)
            encoder = None
            teacher_unavailable = True
    else:
        teacher_unavailable = True
        print("teacher checkpoint not given; producing empty evidence cache")

    from rase.world_models.action_adapter import create_default_libero_adapter

    action_adapter = create_default_libero_adapter()
    cache: list[dict[str, Any]] = []
    distinct_action_rows = 0
    t0 = time.time()
    for i, item in enumerate(rows, start=1):
        row = item["window"]
        boundary = item["boundary"]
        frames = []
        try:
            frames = [_frame_from_b64(b) for b in row["frames_b64"]]
        except Exception as exc:
            print(f"row {i}: frame decode failed: {exc}", file=sys.stderr)
            continue
        if boundary is None:
            student_raw = row.get("student_action_chunk", row["action_chunk"])
        else:
            student_raw = boundary["student_action_chunk"]
        student_chunk = action_adapter.transform_chunk(
            np.asarray(student_raw, np.float32).reshape(-1, 7)
        )
        oft_chunk = action_adapter.transform_chunk(
            np.asarray(row["action_chunk"], np.float32).reshape(-1, 7)
        )
        action_l2 = float(np.linalg.norm(
            student_chunk[0] - oft_chunk[0]
        ))
        distinct_action_rows += int(action_l2 > 1e-8)
        ev = compute_evidence_fast(
            frames=frames,
            student_chunk=student_chunk,
            oft_chunk=oft_chunk,
            encoder=encoder,
            k=args.k,
        )
        entry = {
            "state_key": row["state_key"],
            "task_id": (boundary or row).get("task_id"),
            "suite": (boundary or row).get("suite"),
            "window_index": row["window_index"],
            "window_start": int(row.get("window_start", 0)),
            "elapsed_oft_steps": (
                int(boundary.get("elapsed_oft_steps", 0)) if boundary is not None else None
            ),
            "n_frames": len(frames),
            "k": args.k,
            "action_adapter_hash": action_adapter.adapter_hash,
            "student_oft_first_action_l2": action_l2,
        }
        entry.update(ev)
        if "latent" in ev:
            # Store latent as float list for JSON-serializable cache
            entry["latent"] = np.asarray(ev["latent"]).tolist()
            if "student_delta" in ev:
                entry["student_delta"] = np.asarray(ev["student_delta"]).tolist()
                entry["oft_delta"] = np.asarray(ev["oft_delta"]).tolist()
        cache.append(entry)
        if i % 50 == 0:
            print(f"cache: {i}/{len(rows)} elapsed={time.time() - t0:.1f}s", flush=True)

    report = {
        "schema_version": "rase-pre-c0-r4d-teacher-evidence/v1",
        "n_rows": len(cache),
        "n_distinct_action_rows": distinct_action_rows,
        "k": args.k,
        "teacher_unavailable": teacher_unavailable,
        "teacher_ckpt": str(args.teacher_ckpt) if args.teacher_ckpt else None,
        "source": str(args.dataset.resolve()),
        "boundary_source": (
            str(args.boundary_dataset.resolve()) if args.boundary_dataset else None
        ),
        "action_adapter_hash": action_adapter.adapter_hash,
        "cache_sha256": hashlib.sha256(
            json.dumps(cache, sort_keys=True).encode()
        ).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "evidence_cache.jsonl").write_text(
        "".join(json.dumps(c, sort_keys=True) + "\n" for c in cache)
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
