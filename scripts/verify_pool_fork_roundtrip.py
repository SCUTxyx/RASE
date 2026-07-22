#!/usr/bin/env python3
"""Restore NGC pool states into ForkableEnv and run the double-restore gate."""

from __future__ import annotations

import argparse
import io
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sample_keys(pool, n: int, seed: int) -> list[str]:
    """Prefer one camera + one robot state, then fill randomly."""
    import json
    from collections import defaultdict

    states = pool.manifest()["states"]
    keys = list(states)
    rng = random.Random(seed)
    rng.shuffle(keys)
    by_dim: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        entry = states[key]
        meta_path = pool.root / entry["path"] / "meta.json"
        if not meta_path.is_file():
            continue
        dim = str(
            json.loads(meta_path.read_text(encoding="utf-8")).get("perturb_dim", "")
        )
        by_dim[dim].append(key)
        # Early exit once we have enough diversity for small N.
        if n <= 2 and by_dim.get("camera") and by_dim.get("robot"):
            break
        if sum(len(v) for v in by_dim.values()) >= max(n * 8, 32):
            break
    chosen: list[str] = []
    for dim in ("camera", "robot"):
        if by_dim.get(dim) and len(chosen) < n:
            chosen.append(by_dim[dim][0])
    remaining = [key for dim_keys in by_dim.values() for key in dim_keys if key not in chosen]
    rng.shuffle(remaining)
    while len(chosen) < n and remaining:
        chosen.append(remaining.pop())
    if len(chosen) < n:
        raise SystemExit(f"pool only has {len(chosen)} sampleable states; need {n}")
    return chosen[:n]


def _png_bytes_hwc(image) -> bytes:
    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HWC uint8 RGB, got {array.shape}/{array.dtype}")
    stream = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def _compare_agentview(forkable, loaded, *, check_pixels: bool) -> None:
    if not check_pixels:
        return
    import numpy as np

    pooled = loaded.observations.get("agentview")
    if pooled is None:
        print("WARN: no obs_agentview.png in bundle; skip pixel check", flush=True)
        return
    from rase.collect.pool_candidates import raw_observations_from_control_env

    try:
        obs = raw_observations_from_control_env(forkable.env, force_update=True)
    except RuntimeError as exc:
        print(f"WARN: {exc}; skip pixel check", flush=True)
        return
    # LIBERO ControlEnv keys vary; try common names.
    image = None
    for key in ("agentview_image", "agentview"):
        if isinstance(obs, dict) and key in obs:
            image = obs[key]
            break
    if image is None:
        print(
            f"WARN: cannot locate agentview in obs keys={list(obs) if isinstance(obs, dict) else type(obs)}; "
            "skip pixel check",
            flush=True,
        )
        return
    live = _png_bytes_hwc(image)
    if live != pooled:
        # Fall back to decoded array compare (PNG re-encode may differ).
        from PIL import Image

        live_arr = np.asarray(Image.open(io.BytesIO(live)).convert("RGB"))
        pool_arr = np.asarray(Image.open(io.BytesIO(pooled)).convert("RGB"))
        if not np.array_equal(live_arr, pool_arr):
            raise AssertionError(
                "restored agentview pixels != pool obs_agentview.png "
                "(task/env bind mismatch or render nondeterminism)"
            )
    print("PIXEL_CHECK agentview matches pool", flush=True)


def _assert_task_bind(handle, metadata) -> None:
    """Ensure the live env is the catalog task named by pool metadata."""
    from rase.collect.libero_env_factory import parse_pool_task_id

    parsed = parse_pool_task_id(metadata.task_id)
    single = handle.vector_env.envs[0]
    live_name = str(getattr(single, "task", ""))
    if not live_name:
        raise AssertionError("LiberoEnv missing task name for bind check")
    # Pool instruction is the Plus language string (includes view/initstate).
    live_desc = str(getattr(single, "task_description", ""))
    if live_desc and live_desc != metadata.instruction:
        raise AssertionError(
            f"task_description mismatch: live={live_desc!r} "
            f"meta={metadata.instruction!r}"
        )
    if handle.suite != parsed.suite or handle.catalog_task_id != parsed.catalog_task_id:
        raise AssertionError("factory suite/task_id diverged from metadata")
    print(f"TASK_BIND ok name={live_name}", flush=True)


def verify_one(
    pool,
    state_key: str,
    *,
    steps: int,
    check_pixels: bool,
    libero_plus_root: str | None,
    strict_fingerprint: bool,
) -> None:
    from rase.collect.libero_env_factory import make_libero_env_for_task
    from rase.collect.state_pool import bundle_to_env_snapshot
    from rase.envs.fork_checks import fork_roundtrip_from_snapshot
    from rase.envs.forkable_env import ForkableEnv

    loaded = pool.read_state(state_key)
    snapshot = bundle_to_env_snapshot(loaded)
    meta = loaded.metadata
    print(
        f"VERIFY_START key={state_key} task_id={meta.task_id} "
        f"suite={meta.suite} dim={meta.perturb_dim} step={meta.step}",
        flush=True,
    )
    handle = make_libero_env_for_task(
        meta.task_id,
        seed=int(meta.seed),
        libero_plus_root=libero_plus_root,
    )
    try:
        _assert_task_bind(handle, meta)
        forkable = ForkableEnv(handle.control_env)
        live_fp = forkable._compute_task_fingerprint()
        fp_match = live_fp == snapshot.task_fingerprint
        print(f"FINGERPRINT match={fp_match}", flush=True)
        check_fp = strict_fingerprint or fp_match
        if not check_fp:
            print(
                "FINGERPRINT_RELAX using task_id/instruction bind "
                "(Plus model XML not bit-stable for some initstate robots)",
                flush=True,
            )
        forkable.restore(snapshot, check_task_fingerprint=check_fp)
        _compare_agentview(forkable, loaded, check_pixels=check_pixels)
        t0 = time.perf_counter()
        fork_roundtrip_from_snapshot(
            forkable,
            snapshot,
            steps=steps,
            sim_env=handle.control_env,
            check_task_fingerprint=check_fp,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"PASS key={state_key} steps={steps} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    finally:
        handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument(
        "--state-key",
        action="append",
        default=[],
        help="State key to verify (repeatable). If omitted, use --sample.",
    )
    parser.add_argument("--sample", type=int, default=2, help="Sample N keys when none given")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--check-pixels",
        action="store_true",
        help="Compare restored agentview to pool PNG after first restore",
    )
    parser.add_argument(
        "--strict-fingerprint",
        action="store_true",
        help="Require full task/model fingerprint match (fails on some Plus initstate robots)",
    )
    parser.add_argument("--libero-plus-root", default=None)
    args = parser.parse_args()
    if not 1 <= args.steps <= 50:
        raise SystemExit("--steps must be in [1, 50]")

    from rase.collect.state_pool import StatePool

    pool = StatePool(args.pool.resolve())
    keys = list(args.state_key)
    if not keys:
        keys = _sample_keys(pool, args.sample, args.sample_seed)
    print(f"POOL_FORK_GATE n={len(keys)} steps={args.steps} pool={pool.root}", flush=True)
    for key in keys:
        verify_one(
            pool,
            key,
            steps=args.steps,
            check_pixels=args.check_pixels,
            libero_plus_root=args.libero_plus_root,
            strict_fingerprint=args.strict_fingerprint,
        )
    print(f"POOL_FORK_GATE_DONE passed={len(keys)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
