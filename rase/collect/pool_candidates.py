"""Restore pool states and emit NGC candidate artifacts."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rase.collect.candidates import (
    K_DEFAULT,
    CandidateArtifact,
    generate_candidates,
    load_artifact,
    save_artifact,
)
from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.smolvla_candidate_policy import SmolVLACandidatePolicy
from rase.collect.state_pool import StatePool, bundle_to_env_snapshot
from rase.envs.forkable_env import ForkableEnv


def sample_pool_keys(pool: StatePool, n: int, seed: int) -> list[str]:
    """Prefer one camera + one robot state, then fill randomly."""
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
        dim = str(json.loads(meta_path.read_text(encoding="utf-8")).get("perturb_dim", ""))
        by_dim[dim].append(key)
        if n <= 2 and by_dim.get("camera") and by_dim.get("robot"):
            break
        if sum(len(values) for values in by_dim.values()) >= max(n * 8, 32):
            break
    chosen: list[str] = []
    for dim in ("camera", "robot"):
        if by_dim.get(dim) and len(chosen) < n:
            chosen.append(by_dim[dim][0])
    remaining = [
        key for dim_keys in by_dim.values() for key in dim_keys if key not in chosen
    ]
    rng.shuffle(remaining)
    while len(chosen) < n and remaining:
        chosen.append(remaining.pop())
    if len(chosen) < n:
        raise ValueError(f"pool only has {len(chosen)} sampleable states; need {n}")
    return chosen[:n]


def raw_observations_from_control_env(control_env: Any, *, force_update: bool = True) -> Any:
    """Read robosuite observations from a LIBERO ``ControlEnv`` / OffScreen wrapper.

    LeRobot's ``LiberoEnv._env`` is ``OffScreenRenderEnv``, which does **not**
    expose ``_get_observations``; that method lives on the inner task env
    (``control_env.env``). After ``ForkableEnv.restore``, pass
    ``force_update=True`` so camera/proprio refresh from the restored sim.
    """
    if hasattr(control_env, "_get_observations"):
        try:
            return control_env._get_observations(force_update=force_update)
        except TypeError:
            return control_env._get_observations()
    task_env = getattr(control_env, "env", None)
    if task_env is not None and hasattr(task_env, "_get_observations"):
        return task_env._get_observations(force_update=force_update)
    raise RuntimeError(
        "cannot read observations: neither ControlEnv nor task env exposes "
        f"_get_observations (type={type(control_env)!r})"
    )


def _batch_array(value: Any) -> Any:
    """Add a leading batch dim to nested NumPy leaves (SyncVectorEnv layout)."""
    if isinstance(value, Mapping):
        return {key: _batch_array(item) for key, item in value.items()}
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"cannot batch non-numeric observation leaf: {type(value)!r}")
    return array[None, ...]


def batch_single_gym_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a single-env LiberoEnv obs into the batched dict processors expect.

    ``LiberoProcessorStep._quat2axisangle`` requires ``eef.quat`` shape ``(B, 4)``.
    Images already get an implicit batch in ``preprocess_observation`` when
    ``ndim==3``, but nested ``robot_state`` arrays do not.
    """
    batched: dict[str, Any] = {}
    for key, value in observation.items():
        if key == "task":
            batched[key] = value
            continue
        batched[key] = _batch_array(value)
    return batched


def observation_from_libero_env(libero_env: Any) -> dict[str, Any]:
    """Build a batched Gym observation from the live ControlEnv (B=1)."""
    raw = raw_observations_from_control_env(libero_env._env, force_update=True)
    formatted = libero_env._format_raw_obs(raw)
    task = str(getattr(libero_env, "task_description", "") or "")
    if not task:
        raise RuntimeError("LiberoEnv missing task_description for candidate obs")
    return batch_single_gym_observation({**formatted, "task": task})


def candidate_base_seed(
    metadata_seed: int, state_key: str, base_seed: int, *, k: int = K_DEFAULT
) -> int:
    """Per-state seed offset so K candidates stay isolated across states.

    NumPy's legacy ``np.random.seed`` only accepts ``[0, 2**32 - 1]``. Mixing
    ``base_seed + metadata.seed + state_key[:8]`` routinely overflows that, so
    fold into a range that still leaves headroom for the K=8 offsets
    (``base + 0..7``) used by ``generate_candidates``.
    """
    digest = int(state_key.replace("sp1_", "")[:8], 16) if "sp1_" in state_key else 0
    mixed = int(base_seed) + int(metadata_seed) + digest
    if k < 2 or k >= 2**32:
        raise ValueError("k must be within [2, 2**32)")
    return int(mixed % (2**32 - k))


@dataclass(frozen=True)
class CandidateRunResult:
    state_key: str
    path: Path
    artifact: CandidateArtifact
    skipped: bool


def generate_for_state(
    pool: StatePool,
    state_key: str,
    policy: SmolVLACandidatePolicy,
    *,
    output_dir: Path,
    temperature: float,
    base_seed: int,
    policy_hash: str,
    k: int = K_DEFAULT,
    libero_plus_root: str | None = None,
    strict_fingerprint: bool = False,
    force: bool = False,
    observation_height: int = 360,
    observation_width: int = 360,
) -> CandidateRunResult:
    """Restore one pool state, sample K chunks, write ``{state_key}.npz``."""
    target = Path(output_dir) / f"{state_key}.npz"
    if target.is_file() and not force:
        existing = load_artifact(target)
        if (
            existing.metadata.policy_hash == policy_hash
            and abs(existing.metadata.temperature - temperature) < 1e-12
            and existing.actions.shape[0] == k
        ):
            return CandidateRunResult(state_key, target, existing, skipped=True)

    loaded = pool.read_state(state_key)
    snapshot = bundle_to_env_snapshot(loaded)
    meta = loaded.metadata
    flavor = (
        str(getattr(meta, "libero_flavor", None) or "")
        or ("clean" if meta.perturb_dim == "clean" else "plus")
    )
    handle = make_libero_env_for_task(
        meta.task_id,
        init_state_id=meta.init_state_id if meta.init_state_id is not None else 0,
        seed=int(meta.seed),
        observation_height=observation_height,
        observation_width=observation_width,
        libero_plus_root=libero_plus_root,
        libero_flavor=flavor,  # type: ignore[arg-type]
    )
    try:
        live_desc = str(getattr(handle.vector_env.envs[0], "task_description", ""))
        if live_desc and live_desc != meta.instruction:
            raise AssertionError(
                f"task_description mismatch: live={live_desc!r} meta={meta.instruction!r}"
            )
        forkable = ForkableEnv(handle.control_env)
        live_fp = forkable._compute_task_fingerprint()
        check_fp = strict_fingerprint or live_fp == snapshot.task_fingerprint
        forkable.restore(snapshot, check_task_fingerprint=check_fp)
        observation = observation_from_libero_env(handle.vector_env.envs[0])
        artifact = generate_candidates(
            policy,
            observation,
            k=k,
            temperature=temperature,
            base_seed=candidate_base_seed(meta.seed, state_key, base_seed, k=k),
            policy_hash=policy_hash,
        )
        save_artifact(target, artifact)
        return CandidateRunResult(state_key, target, artifact, skipped=False)
    finally:
        handle.close()


def diversity_summary(artifacts: Sequence[CandidateArtifact]) -> dict[str, Any]:
    """Aggregate endpoint/chunk diversity for a pilot acceptance check."""
    if not artifacts:
        return {
            "n": 0,
            "mean_endpoint_l2": 0.0,
            "min_endpoint_l2": 0.0,
            "max_endpoint_l2": 0.0,
            "mean_chunk_l2": 0.0,
        }
    endpoints = [item.metadata.diversity.mean_pairwise_endpoint_l2 for item in artifacts]
    mins = [item.metadata.diversity.min_pairwise_endpoint_l2 for item in artifacts]
    maxs = [item.metadata.diversity.max_pairwise_endpoint_l2 for item in artifacts]
    chunks = [item.metadata.diversity.mean_pairwise_chunk_l2 for item in artifacts]
    return {
        "n": len(artifacts),
        "mean_endpoint_l2": float(np.mean(endpoints)),
        "min_endpoint_l2": float(np.min(mins)),
        "max_endpoint_l2": float(np.max(maxs)),
        "mean_chunk_l2": float(np.mean(chunks)),
        "per_state": [
            {
                "mean_pairwise_endpoint_l2": item.metadata.diversity.mean_pairwise_endpoint_l2,
                "min_pairwise_endpoint_l2": item.metadata.diversity.min_pairwise_endpoint_l2,
                "mean_pairwise_chunk_l2": item.metadata.diversity.mean_pairwise_chunk_l2,
                "shape": list(item.metadata.shape),
                "temperature": item.metadata.temperature,
            }
            for item in artifacts
        ],
    }


def write_summary(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
