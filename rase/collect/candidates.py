"""Candidate generation and portable ``.npz`` artifacts.

The policy is deliberately duck-typed so this module can run in either the
SmolVLA or OFT environment without importing either heavy dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

K_DEFAULT = 8
ACTION_DIM = 7
ARTIFACT_VERSION = 1


class CandidatePolicy(Protocol):
    def reset(self) -> None: ...

    def sample_chunk(self, observation: Any, *, temperature: float) -> Any: ...


@dataclass(frozen=True)
class DiversityMetrics:
    mean_pairwise_endpoint_l2: float
    min_pairwise_endpoint_l2: float
    max_pairwise_endpoint_l2: float
    mean_pairwise_chunk_l2: float


@dataclass(frozen=True)
class CandidateMetadata:
    version: int
    seeds: tuple[int, ...]
    temperature: float
    policy_hash: str
    shape: tuple[int, int, int]
    diversity: DiversityMetrics


@dataclass(frozen=True)
class CandidateArtifact:
    actions: np.ndarray
    metadata: CandidateMetadata


def seed_everything(seed: int) -> None:
    """Seed lightweight RNGs; torch is seeded only when already imported."""
    random.seed(seed)
    np.random.seed(seed)
    import sys

    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if getattr(torch, "cuda", None) is not None:
            torch.cuda.manual_seed_all(seed)


def policy_fingerprint(policy: object, explicit_hash: str | None = None) -> str:
    """Return a stable provenance hash without serializing model weights."""
    if explicit_hash:
        return explicit_hash
    identity = {
        "module": type(policy).__module__,
        "class": type(policy).__qualname__,
        "revision": getattr(policy, "revision", None),
        "checkpoint": str(getattr(policy, "checkpoint", "")),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def diversity_metrics(actions: np.ndarray) -> DiversityMetrics:
    actions = validate_actions(actions, require_k=False)
    if actions.shape[0] < 2:
        return DiversityMetrics(0.0, 0.0, 0.0, 0.0)

    # The cumulative translational delta is the action chunk's implied endpoint.
    endpoints = actions[:, :, :3].sum(axis=1, dtype=np.float64)
    endpoint_distances: list[float] = []
    chunk_distances: list[float] = []
    for i in range(actions.shape[0]):
        for j in range(i + 1, actions.shape[0]):
            endpoint_distances.append(float(np.linalg.norm(endpoints[i] - endpoints[j])))
            chunk_distances.append(
                float(np.linalg.norm(actions[i].astype(np.float64) - actions[j]))
            )
    return DiversityMetrics(
        mean_pairwise_endpoint_l2=float(np.mean(endpoint_distances)),
        min_pairwise_endpoint_l2=float(np.min(endpoint_distances)),
        max_pairwise_endpoint_l2=float(np.max(endpoint_distances)),
        mean_pairwise_chunk_l2=float(np.mean(chunk_distances)),
    )


def validate_actions(actions: Any, *, require_k: bool = True) -> np.ndarray:
    array = np.asarray(actions)
    if array.ndim != 3:
        raise ValueError(f"candidate actions must have shape [K,T,7], got {array.shape}")
    if require_k and array.shape[0] != K_DEFAULT:
        raise ValueError(f"candidate artifact requires K={K_DEFAULT}, got {array.shape[0]}")
    if array.shape[1] < 1 or array.shape[2] != ACTION_DIM:
        raise ValueError(f"candidate actions must have shape [K,T,7], got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError("candidate actions must be finite numeric values")
    return np.ascontiguousarray(array)


def make_artifact(
    actions: Any,
    *,
    seeds: Sequence[int],
    temperature: float,
    policy_hash: str,
) -> CandidateArtifact:
    array = validate_actions(actions, require_k=False)
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if len(normalized_seeds) != array.shape[0]:
        raise ValueError(
            "candidate seed count must match the candidate axis: "
            f"seeds={len(normalized_seeds)} candidates={array.shape[0]}"
        )
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("candidate seeds must be unique")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not policy_hash:
        raise ValueError("policy_hash must be non-empty")
    metadata = CandidateMetadata(
        version=ARTIFACT_VERSION,
        seeds=normalized_seeds,
        temperature=float(temperature),
        policy_hash=str(policy_hash),
        shape=tuple(int(value) for value in array.shape),
        diversity=diversity_metrics(array),
    )
    return CandidateArtifact(array, metadata)


def generate_candidates(
    policy: CandidatePolicy,
    observation: Any,
    *,
    k: int = K_DEFAULT,
    temperature: float = 0.7,
    base_seed: int = 0,
    policy_hash: str | None = None,
    seed_fn: Callable[[int], None] = seed_everything,
) -> CandidateArtifact:
    """Generate K same-profile samples with isolated, identical policy resets."""
    if k < 2:
        raise ValueError("candidate generation requires k >= 2")
    seeds = tuple(base_seed + index for index in range(k))
    chunks = []
    for seed in seeds:
        policy.reset()
        seed_fn(seed)
        chunks.append(np.asarray(policy.sample_chunk(observation, temperature=temperature)))
    return make_artifact(
        np.stack(chunks),
        seeds=seeds,
        temperature=temperature,
        policy_hash=policy_fingerprint(policy, policy_hash),
    )


def _metadata_dict(metadata: CandidateMetadata) -> dict[str, Any]:
    value = asdict(metadata)
    value["seeds"] = list(metadata.seeds)
    value["shape"] = list(metadata.shape)
    return value


def save_artifact(path: str | os.PathLike[str], artifact: CandidateArtifact) -> None:
    """Atomically save an artifact. Loading never requires pickle."""
    actions = validate_actions(artifact.actions, require_k=False)
    if tuple(actions.shape) != artifact.metadata.shape:
        raise ValueError("metadata shape does not match candidate actions")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(
        _metadata_dict(artifact.metadata), sort_keys=True, separators=(",", ":")
    )
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(
                handle,
                actions=actions,
                seeds=np.asarray(artifact.metadata.seeds, dtype=np.int64),
                temperature=np.asarray(artifact.metadata.temperature, dtype=np.float64),
                policy_hash=np.asarray(artifact.metadata.policy_hash),
                diversity=np.asarray(
                    json.dumps(
                        asdict(artifact.metadata.diversity),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
                format_version=np.asarray(ARTIFACT_VERSION, dtype=np.int64),
                metadata=np.asarray(metadata_json),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_artifact(path: str | os.PathLike[str]) -> CandidateArtifact:
    with np.load(path, allow_pickle=False) as data:
        actions = validate_actions(data["actions"], require_k=False)
        raw = json.loads(str(data["metadata"].item()))
        if "format_version" in data and int(data["format_version"].item()) != raw.get(
            "version"
        ):
            raise ValueError("corrupt artifact: version fields disagree")
        if "seeds" in data and tuple(data["seeds"].tolist()) != tuple(raw["seeds"]):
            raise ValueError("corrupt artifact: seed fields disagree")
        if "temperature" in data and float(data["temperature"].item()) != float(
            raw["temperature"]
        ):
            raise ValueError("corrupt artifact: temperature fields disagree")
        if "policy_hash" in data and str(data["policy_hash"].item()) != raw["policy_hash"]:
            raise ValueError("corrupt artifact: policy hash fields disagree")
    if raw.get("version") != ARTIFACT_VERSION:
        raise ValueError(f"unsupported candidate artifact version: {raw.get('version')}")
    diversity = DiversityMetrics(**raw["diversity"])
    metadata = CandidateMetadata(
        version=int(raw["version"]),
        seeds=tuple(int(value) for value in raw["seeds"]),
        temperature=float(raw["temperature"]),
        policy_hash=str(raw["policy_hash"]),
        shape=tuple(int(value) for value in raw["shape"]),
        diversity=diversity,
    )
    artifact = CandidateArtifact(actions, metadata)
    # Reuse construction validation while preserving stored metrics.
    make_artifact(
        actions,
        seeds=metadata.seeds,
        temperature=metadata.temperature,
        policy_hash=metadata.policy_hash,
    )
    if metadata.shape != tuple(actions.shape):
        raise ValueError("corrupt artifact: metadata shape does not match actions")
    return artifact
