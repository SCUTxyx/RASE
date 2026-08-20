#!/usr/bin/env python3
"""R6-D pre-registered world-model residual/disagreement feature cache.

The only allowed world-model experiment after the R6-C gate (per protocol
`r6b1_dynamic_boundary_protocol_v1.json` -> `world_model_lock`): for each B1.2
boundary, restore the frozen pool snapshot, run the *source* policy for a few
extra steps to collect time-aligned real next frames, and compute

- pooled latent ``z_t`` of the boundary frame;
- ``K``-step action-conditioned latent rollouts for K in {1, 4, 8} starting from
  ``z_t`` conditioned on the recorded source actions, giving predicted latents
  ``z_{t+1}, z_{t+4}, z_{t+8}``;
- multi-step prediction residual: mean squared deviation between the predicted
  pooled latent and the *real* pooled latent of the aligned real frames;
- ensemble disagreement: variance across the K horizon predictors of their
  K-step delta directions (a cheap, non-Bayesian disagreement proxy).

These features are additional inputs to the R6-C risk baseline only; they never
replace the original action/state/history features (latent replacement was
already rejected in R4-D).  The cache is a pre-registration: whether to keep the
world-model arm is decided solely by the frozen state-level Pareto comparison in
`eval_r6d_wm_ablation.py`.

Real-frame alignment (preregistration table, `Real frames` row):
for each boundary we replay the recorded ``source_action_trace`` deterministically
from the frozen pool snapshot.  The B1.2 source rollout is side-effect-free and
parity-gated, so the same initial snapshot plus the same recorded action sequence
reproduces the identical simulator states; agent-view frames at sim steps
``t+1..t+8`` are therefore exactly the real frames the source policy saw.  The
boundary frame is additionally checked against the frozen B1.2 npz frame
(pixel-identical up to one LSB; a larger divergence is a hard error).

Usage (mirrors collect_r6b1_dynamic_boundaries.py resource needs):
    python scripts/cache_r6d_wm_features.py \
        --initial-keys runs/rase_ui_phase1a_replacement48_initial_keys_v2.json \
        --policy-id pi0fast_libero --seed-index 0 \
        --input-root runs/pre_c0_r6/r6b1_b1p2_v1 \
        --output runs/pre_c0_r6/r6d_wm_features_v1 \
        --teacher-ckpt /root/autodl-tmp/vjepa2 \
        --k 1 4 8

Env: LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus and the MUJOCO EGL
variables used by the B1.2 collector.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SIZE = 96
MAX_EXTRA_STEPS = 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def resize_chw(value: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)


def agent_view_hwc(observation: dict) -> np.ndarray:
    """Extract the agent-view RGB frame (H, W, C) from a batched Gym obs."""
    pixels = observation["pixels"]
    return np.asarray(pixels["image"][0], dtype=np.uint8)


def find_boundary_rows(input_root: Path, policy_id: str, seed_index: int) -> list[dict]:
    """Collect boundary metadata rows for a policy/seed across all suites."""
    rows: list[dict] = []
    for path in sorted(glob.glob(str(input_root / "suite_*" / policy_id / f"seed_{seed_index}" / "*__seed*.json"))):
        if Path(path).name == "report.json":
            continue
        data = json.loads(Path(path).read_text())
        npz_path = Path(data["npz"])
        for position, boundary in enumerate(data["rows"]):
            boundary["npz"] = str(npz_path.resolve())
            boundary["boundary_position"] = position
            rows.append(boundary)
    return rows


def replay_source_frames(
    pool: object,
    state_key: str,
    trace: np.ndarray,
    *,
    libero_plus_root: str | None,
    max_steps: int,
) -> list[np.ndarray]:
    """Deterministically replay the recorded source action trace from the pool
    snapshot and return the agent-view frame at every sim step.

    Mirrors the collector's side-effect-free Stage-1 source rollout: the pool
    snapshot is restored once, and only env stepping (plus one observation read
    before the loop) touches the environment.  ``max_steps`` caps the replay at
    the largest boundary elapsed + MAX_EXTRA_STEPS.
    """
    from rase.collect.forked_rollout import FixedActionContinuation, restore_pool_state
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env

    main = restore_pool_state(pool, state_key, libero_plus_root=libero_plus_root)
    frames: list[np.ndarray] = []
    try:
        main.forkable.restore(main.snapshot, check_task_fingerprint=main.check_task_fingerprint)
        single = main.handle.vector_env.envs[0]
        vector_env = main.handle.vector_env
        horizon = int(getattr(single, "_max_episode_steps", 600))
        instruction = str(getattr(single, "task_description", "") or main.loaded.metadata.instruction)
        observation = observation_from_libero_env(single)
        frames.append(agent_view_hwc(observation))
        replay = FixedActionContinuation(trace)
        replay.reset()
        elapsed = 0
        while current_timestep(main.handle.control_env) < horizon:
            if elapsed >= len(trace) or elapsed >= max_steps:
                break
            action = replay.act(observation, task=instruction)
            observation, _, term, trunc, info = vector_env.step(as_batched_action(action))
            elapsed += 1
            frames.append(agent_view_hwc(observation))
            terminal = bool(np.asarray(term).reshape(-1)[0]) or bool(
                np.asarray(trunc).reshape(-1)[0]
            )
            if terminal:
                break
        return frames
    finally:
        main.close()


def compute_wm_features(
    *,
    encoder: object,
    boundary_frame: np.ndarray,
    action_trace: np.ndarray,
    source_actions: np.ndarray,
    elapsed: int,
    ks: list[int],
    real_frames: dict[int, np.ndarray],
) -> dict:
    """Compute the pre-registered world-model features for one boundary.

    ``real_frames[k]`` is the agent-view HWC frame at sim step ``elapsed + k``
    for every k that the replayed source episode actually reached.
    """
    try:
        latent = encoder.encode_stack([boundary_frame])  # (1, N, D)
        z_t = latent.mean(dim=1).squeeze(0).float().cpu().numpy()  # (D,)
        # Recorded source actions after the boundary for conditioning.  Use the
        # single boundary action if the trace does not extend beyond elapsed.
        after = action_trace[elapsed:] if len(action_trace) > elapsed else source_actions[:1]
        if len(after) == 0:
            after = source_actions[:1]
        k_max = max(ks)
        chunk = after[:k_max] if len(after) >= k_max else np.pad(
            after, ((0, k_max - len(after)), (0, 0)), mode="edge")
        deltas = encoder.rollout_from_latent(latent, chunk, k=k_max)  # (k_max, D)

        predicted = {}
        disagreement = {}
        residual = {}
        residual_mse = {}
        real_latents = {}
        for k in ks:
            if len(deltas) >= k:
                predicted_vec = z_t + deltas[:k].sum(axis=0)
                predicted[str(k)] = predicted_vec.tolist()
            else:
                predicted[str(k)] = None
                predicted_vec = None
            if k in real_frames:
                real_z = encoder.pooled_latent([real_frames[k]])  # (D,)
                real_latents[str(k)] = real_z.tolist()
                if predicted_vec is not None:
                    se = (predicted_vec - real_z) ** 2
                    residual[str(k)] = se.tolist()
                    residual_mse[str(k)] = float(se.mean())
                else:
                    residual[str(k)] = None
            else:
                residual[str(k)] = None
        # Disagreement proxy: variance of per-step delta directions across the
        # full rollout (how uncertain the K-step forecast is internally).
        norms = np.linalg.norm(deltas, axis=1) + 1e-9
        directions = deltas / norms[:, None]
        disagreement["delta_direction_var"] = float(np.mean(
            np.var(directions, axis=0)))
        disagreement["delta_magnitude_var"] = float(np.var(norms))
        return {
            "latent_z_t": z_t.tolist(),
            "predicted_latents": predicted,
            "disagreement": disagreement,
            "residual": residual,
            "residual_mse": residual_mse,
            "real_latents": real_latents,
            "k_values": ks,
            "action_l2": float(np.linalg.norm(source_actions[0])),
            "real_frames_available": {str(k): k in real_frames for k in ks},
        }
    except Exception as exc:  # world-model failure must not block caching
        return {"error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="B1.2 collection output root (contains suite_*)")
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path, required=True,
                        help="V-JEPA 2-AC repo root (e.g. /root/autodl-tmp/vjepa2)")
    parser.add_argument("--initial-keys", type=Path, required=True,
                        help="frozen initial-keys manifest (locates the StatePool)")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--exclusions", type=Path, default=None,
                        help="frozen exclusion manifest of known nondeterministic groups")
    args = parser.parse_args()

    excluded: set[tuple[str, int, str]] = set()
    if args.exclusions is not None:
        payload = json.loads(args.exclusions.read_text())
        for entry in payload["excluded"]:
            policy, seed, state = entry
            excluded.add((str(policy), int(seed), str(state)))

    rows = find_boundary_rows(args.input_root, args.policy_id, args.seed_index)
    rows = [row for row in rows
            if (str(row["policy_id"]), int(row["seed_index"]), str(row["state_key"])) not in excluded]
    rows.sort(key=lambda r: (r["state_key"], int(r["elapsed_source_steps"])))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise SystemExit(f"no B1.2 boundaries for {args.policy_id} seed {args.seed_index}")

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.state_pool import StatePool
    from rase.world_models.action_adapter import create_default_libero_adapter
    from rase.world_models.vjepa2_adapter import VJEPA2ACEncoder

    ensure_libero_plus_paths(os.environ.get("LIBERO_PLUS_ROOT"))
    _patch_lerobot_init_states()
    keys_payload = read_json(args.initial_keys)
    pool = StatePool(Path(str(keys_payload["pool"])).resolve())

    adapter = create_default_libero_adapter()
    encoder = VJEPA2ACEncoder(args.teacher_ckpt, device=args.device)
    encoder.load()

    # Group rows by state so one deterministic replay serves all its boundaries.
    by_state: dict[str, list[dict]] = {}
    for boundary in rows:
        by_state.setdefault(boundary["state_key"], []).append(boundary)

    cache: list[dict] = []
    n_parity_mismatch = 0
    started = time.perf_counter()
    processed = 0
    for state_key, state_rows in by_state.items():
        npz = np.load(state_rows[0]["npz"])
        trace = np.asarray(npz["source_action_trace"], dtype=np.float32)
        max_elapsed = max(int(row["elapsed_source_steps"]) for row in state_rows)
        replay_max = max_elapsed + MAX_EXTRA_STEPS
        frames = replay_source_frames(
            pool, state_key, trace, libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
            max_steps=replay_max,
        )
        for boundary in state_rows:
            elapsed = int(boundary["elapsed_source_steps"])
            position = int(boundary["boundary_position"])
            npz_frame_chw = np.asarray(npz["image"][position][0], dtype=np.uint8)  # (3, 96, 96)
            if elapsed < len(frames):
                replay_frame_chw = resize_chw(frames[elapsed], IMAGE_SIZE)
                max_diff = int(np.abs(replay_frame_chw.astype(int) - npz_frame_chw.astype(int)).max())
                boundary_frame = frames[elapsed]
            else:
                max_diff = None
                # Episode ended before the boundary frame; fall back to the frozen npz frame.
                boundary_frame = np.asarray(npz_frame_chw, dtype=np.uint8).transpose(1, 2, 0)
            if max_diff is not None and max_diff > 1:
                n_parity_mismatch += 1
                print(f"WM cache parity MISMATCH state={state_key} elapsed={elapsed} max_diff={max_diff}",
                      flush=True)
            if n_parity_mismatch:
                raise SystemExit(
                    f"R6D cache: {n_parity_mismatch} boundary replay frames differ from the "
                    "frozen B1.2 npz (>1 LSB). Real-frame alignment is broken; refusing to "
                    "write a non-compliant WM cache.")
            real_frames: dict[int, np.ndarray] = {}
            for k in sorted(set(args.k)):
                step = elapsed + k
                if step < len(frames):
                    real_frames[k] = frames[step]
            source_actions = np.asarray(
                npz["source_action"][position], dtype=np.float32)
            if source_actions.ndim == 1:
                source_actions = source_actions.reshape(1, -1)
            features = compute_wm_features(
                encoder=encoder, boundary_frame=boundary_frame, action_trace=trace,
                source_actions=source_actions, elapsed=elapsed,
                ks=sorted(set(args.k)), real_frames=real_frames,
            )
            entry = {
                "state_key": boundary["state_key"],
                "policy_id": boundary["policy_id"],
                "seed_index": boundary["seed_index"],
                "group_id": boundary["group_id"],
                "task_id": boundary["task_id"],
                "suite": boundary["suite"],
                "elapsed_source_steps": elapsed,
                "source_final_success": bool(boundary["source_final_success"]),
                "persistent_success": bool(boundary["persistent_success_if_enter_now"]),
                "persistent_teacher_steps": float(boundary["persistent_teacher_steps_if_enter_now"] or 0.0),
                "action_adapter_hash": adapter.adapter_hash,
                "replay_parity_max_diff": max_diff,
            }
            entry.update(features)
            cache.append(entry)
        processed += 1
        if processed % 10 == 0:
            print(f"WM cache states={processed}/{len(by_state)} rows={len(cache)} "
                  f"at {time.perf_counter() - started:.0f}s", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "rase-r6d-wm-residual-disagreement/v1",
        "policy_id": args.policy_id, "seed_index": args.seed_index,
        "n_rows": len(cache),
        "n_states": len(by_state),
        "n_excluded_groups": len(excluded),
        "exclusions": str(args.exclusions.resolve()) if args.exclusions is not None else None,
        "k_values": sorted(set(args.k)),
        "teacher_ckpt": str(args.teacher_ckpt.resolve()),
        "input_root": str(args.input_root.resolve()),
        "real_frame_alignment": "recorded-trace deterministic replay",
        "replay_parity_mismatches": n_parity_mismatch,
        "teacher_unavailable": False,
        "cache_sha256": hashlib.sha256(json.dumps(cache, sort_keys=True).encode()).hexdigest(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "wm_features.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cache))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if n_parity_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
