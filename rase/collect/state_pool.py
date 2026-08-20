"""Crash-safe, checksummed storage for NGC Step 1 states."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .schema import SCHEMA_VERSION, StateMetadata, canonical_json, validate_state_key

MANIFEST_VERSION = "ngc-state-pool-manifest/v1"
_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class WriteResult:
    state_key: str
    path: Path
    created: bool


def retain_snapshot(
    episode_outcome: str,
    episode_id: str,
    step: int,
    seed: int,
    success_fraction: float = 0.20,
) -> bool:
    """Keep every failure snapshot and a stable hash sample of successes."""
    if episode_outcome == "failure":
        return True
    if episode_outcome != "success":
        raise ValueError("episode_outcome must be 'success' or 'failure'")
    if not 0.0 <= success_fraction <= 1.0:
        raise ValueError("success_fraction must be in [0, 1]")
    token = f"retention/v1:{seed}:{episode_id}:{step}".encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
    return value < success_fraction


def snapshot_steps(action_chunks: int, cadence: int = 2) -> range:
    if action_chunks < 0:
        raise ValueError("action_chunks must be non-negative")
    if cadence <= 0:
        raise ValueError("cadence must be positive")
    return range(0, action_chunks, cadence)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, canonical_json(value) + b"\n")


def _json_safe(value: Any) -> Any:
    """Encode snapshot metadata without pickle while preserving array types."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("object-dtype arrays are forbidden in state snapshots")
        return {
            "__ndarray__": value.tolist(),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [_json_safe(item) for item in value]}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("snapshot mappings must use string keys")
        return {key: _json_safe(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"unsupported snapshot value type: {type(value).__qualname__}")


def _json_unsafe(value: Any) -> Any:
    """Inverse of ``_json_safe`` for pool controller/rng JSON blobs."""
    if isinstance(value, list):
        return [_json_unsafe(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__ndarray__", "dtype", "shape"}:
            array = np.asarray(value["__ndarray__"], dtype=np.dtype(value["dtype"]))
            return array.reshape(tuple(value["shape"]))
        if set(value) == {"__tuple__"}:
            return tuple(_json_unsafe(item) for item in value["__tuple__"])
        return {key: _json_unsafe(item) for key, item in value.items()}
    return value


def _json_bytes(value: Any) -> np.ndarray:
    return np.frombuffer(canonical_json(_json_safe(value)), dtype=np.uint8)


def _decode_json_blob(blob: np.ndarray) -> Any:
    raw = np.asarray(blob, dtype=np.uint8).tobytes()
    return _json_unsafe(json.loads(raw.decode("utf-8")))


@dataclass(frozen=True)
class LoadedState:
    """One verified pool bundle decoded for restore / inspection."""

    state_key: str
    metadata: StateMetadata
    path: Path
    sim_state: np.ndarray
    controller_state: Mapping[str, Any]
    rng_state: Any
    observations: Mapping[str, bytes]
    proprio: np.ndarray


def bundle_to_env_snapshot(loaded: LoadedState):
    """Rebuild a ForkableEnv ``EnvSnapshot`` from a pool bundle."""
    from rase.envs.snapshot import EnvSnapshot, SNAPSHOT_VERSION

    controller = dict(loaded.controller_state)
    fmt = controller.get("snapshot_format")
    if fmt != "rase.forkable_env/v1":
        raise ValueError(
            f"unsupported controller snapshot_format {fmt!r} for {loaded.state_key}"
        )
    fingerprint = controller.get("task_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"missing task_fingerprint for {loaded.state_key}")
    for field in ("env_counters", "robots", "observables", "obs_cache"):
        if field not in controller:
            raise ValueError(f"controller_state missing {field!r} for {loaded.state_key}")
    # StatePool v1 stores a flattened MuJoCo ``sim_state`` but not the raw
    # ``mujoco_data`` block used by full ForkableEnv snapshots.  Newer writers
    # may nevertheless include runtime controller caches in ``robots``.  That
    # hybrid payload cannot be restored: ForkableEnv correctly chooses its
    # legacy controller schema when raw mujoco_data is absent.  Canonicalize
    # only the in-memory compatibility view by dropping caches that are
    # recomputed before ``run_controller``; stored bytes/checksums stay intact.
    controller_cache_fields = {
        "ee_pos", "ee_ori_mat", "ee_pos_vel", "ee_ori_vel",
        "joint_pos", "joint_vel", "J_pos", "J_ori", "J_full", "mass_matrix",
    }
    robots = copy.deepcopy(controller["robots"])
    for robot in robots:
        state = robot.get("controller")
        if isinstance(state, dict):
            for field in controller_cache_fields:
                state.pop(field, None)
    return EnvSnapshot(
        task_fingerprint=fingerprint,
        version=SNAPSHOT_VERSION,
        payload={
            "sim_state": np.asarray(loaded.sim_state).copy(),
            "env_counters": controller["env_counters"],
            "robots": robots,
            "observables": controller["observables"],
            "obs_cache": controller["obs_cache"],
            "rng": loaded.rng_state,
        },
    )


class StatePool:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.lock_path = self.root / ".manifest.lock"
        self.root.mkdir(parents=True, exist_ok=True)

    def state_path(
        self, state_key: str, metadata: StateMetadata | None = None
    ) -> Path:
        validate_state_key(state_key)
        if metadata is not None:
            if metadata.state_key != state_key:
                raise ValueError("metadata does not match state key")
            for label, value in (
                ("task_id", metadata.task_id),
                ("episode_id", metadata.episode_id),
            ):
                if not _PATH_COMPONENT.fullmatch(value):
                    raise ValueError(f"{label} is not a safe path component: {value!r}")
            return (
                self.root
                / metadata.task_id
                / metadata.episode_id
                / f"{metadata.step:06d}"
            )
        manifest = self.manifest()
        entry = manifest["states"].get(state_key)
        if entry is None:
            raise KeyError(f"state key is not present in the manifest: {state_key}")
        return self.root / entry["path"]

    def write_state(
        self,
        metadata: StateMetadata,
        *,
        sim_state: np.ndarray,
        controller_state: Any,
        rng_state: Any,
        observations: Mapping[str, bytes],
        proprio: np.ndarray,
    ) -> WriteResult:
        """Atomically publish one complete state; identical retries are no-ops."""
        metadata.validate()
        key = metadata.state_key
        destination = self.state_path(key, metadata)
        previous = self.manifest()["states"].get(key)
        # Resume against whatever path the manifest already recorded (layout may change).
        existing_dir: Path | None = None
        if previous is not None:
            candidate = self.root / previous["path"]
            if candidate.exists():
                existing_dir = candidate
        if existing_dir is None and destination.exists():
            existing_dir = destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=destination.parent))
        try:
            self._stage_state(
                stage,
                metadata,
                sim_state,
                controller_state,
                rng_state,
                observations,
                proprio,
            )
            staged_checksums = self._load_checksums(stage)
            if existing_dir is not None:
                existing_checksums = self.verify_state(key, path=existing_dir)
                if existing_checksums != staged_checksums:
                    raise FileExistsError(
                        f"state key collision or non-idempotent retry for {key}"
                    )
                shutil.rmtree(stage)
                self._update_manifest(
                    key, metadata, existing_checksums, path=existing_dir
                )
                return WriteResult(key, existing_dir, False)
            try:
                os.rename(stage, destination)
            except FileExistsError:
                existing_checksums = self.verify_state(key, path=destination)
                if existing_checksums != staged_checksums:
                    raise FileExistsError(
                        f"concurrent state key collision for {key}"
                    )
                shutil.rmtree(stage)
                self._update_manifest(
                    key, metadata, existing_checksums, path=destination
                )
                return WriteResult(key, destination, False)
            self._fsync_dir(destination.parent)
            self._update_manifest(key, metadata, staged_checksums, path=destination)
            return WriteResult(key, destination, True)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def _stage_state(
        self,
        stage: Path,
        metadata: StateMetadata,
        sim_state: np.ndarray,
        controller_state: Any,
        rng_state: Any,
        observations: Mapping[str, bytes],
        proprio: np.ndarray,
    ) -> None:
        if not observations:
            raise ValueError("at least one observation image is required")
        with (stage / "sim_state.npz").open("wb") as stream:
            np.savez_compressed(
                stream,
                sim_state=np.asarray(sim_state),
                controller_state_json=_json_bytes(controller_state),
                rng_state_json=_json_bytes(rng_state),
            )
            stream.flush()
            os.fsync(stream.fileno())
        with (stage / "proprio.npy").open("wb") as stream:
            np.save(stream, np.asarray(proprio), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        for camera, image in sorted(observations.items()):
            if (
                not camera
                or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in camera)
                or not isinstance(image, bytes)
            ):
                raise ValueError("observation names must be safe lowercase names and values bytes")
            _write_bytes(stage / f"obs_{camera}.png", image)
        _write_json(stage / "meta.json", metadata.to_dict())
        files = {
            path.name: _sha256(path)
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        bundle = hashlib.sha256(canonical_json(files)).hexdigest()
        _write_json(
            stage / "checksums.json",
            {"algorithm": "sha256", "files": files, "bundle_sha256": bundle},
        )
        self._fsync_dir(stage)

    def _load_checksums(self, state_dir: Path) -> dict[str, Any]:
        with (state_dir / "checksums.json").open(encoding="utf-8") as stream:
            return json.load(stream)

    def verify_state(
        self, state_key: str, *, path: Path | None = None
    ) -> dict[str, Any]:
        state_dir = path or self.state_path(state_key)
        checksums = self._load_checksums(state_dir)
        if checksums.get("algorithm") != "sha256":
            raise ValueError(f"unsupported checksum algorithm for {state_key}")
        files = checksums.get("files", {})
        for name, expected in files.items():
            path = state_dir / name
            if not path.is_file() or _sha256(path) != expected:
                raise IOError(f"checksum mismatch for {state_key}/{name}")
        bundle = hashlib.sha256(canonical_json(files)).hexdigest()
        if bundle != checksums.get("bundle_sha256"):
            raise IOError(f"bundle checksum mismatch for {state_key}")
        metadata = StateMetadata.from_dict(
            json.loads((state_dir / "meta.json").read_text(encoding="utf-8"))
        )
        if metadata.state_key != state_key:
            raise IOError(f"metadata key mismatch for {state_key}")
        return checksums

    def read_state(self, state_key: str, *, load_observations: bool = True) -> LoadedState:
        """Verify checksums and decode one published bundle for restore."""
        validate_state_key(state_key)
        state_dir = self.state_path(state_key)
        self.verify_state(state_key, path=state_dir)
        with np.load(state_dir / "sim_state.npz", allow_pickle=False) as archive:
            sim_state = np.asarray(archive["sim_state"]).copy()
            controller_state = _decode_json_blob(archive["controller_state_json"])
            rng_state = _decode_json_blob(archive["rng_state_json"])
        if not isinstance(controller_state, Mapping):
            raise ValueError(f"controller_state must be a mapping for {state_key}")
        proprio = np.load(state_dir / "proprio.npy", allow_pickle=False)
        metadata = StateMetadata.from_dict(
            json.loads((state_dir / "meta.json").read_text(encoding="utf-8"))
        )
        observations: dict[str, bytes] = {}
        if load_observations:
            for path in sorted(state_dir.glob("obs_*.png")):
                camera = path.name[len("obs_") : -len(".png")]
                observations[camera] = path.read_bytes()
        return LoadedState(
            state_key=state_key,
            metadata=metadata,
            path=state_dir,
            sim_state=sim_state,
            controller_state=controller_state,
            rng_state=rng_state,
            observations=observations,
            proprio=np.asarray(proprio).copy(),
        )

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"manifest_version": MANIFEST_VERSION, "schema_version": SCHEMA_VERSION, "states": {}}
        with self.manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError("unsupported manifest version")
        return manifest

    @contextlib.contextmanager
    def _manifest_lock(self):
        import fcntl

        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _update_manifest(
        self,
        key: str,
        metadata: StateMetadata,
        checksums: Mapping[str, Any],
        *,
        path: Path | None = None,
    ) -> None:
        with self._manifest_lock():
            manifest = self.manifest()
            resolved = path or self.state_path(key, metadata)
            entry = {
                "path": str(resolved.relative_to(self.root)),
                "bundle_sha256": checksums["bundle_sha256"],
                "task_id": metadata.task_id,
                "episode_id": metadata.episode_id,
                "step": metadata.step,
                "outcome": metadata.episode_outcome,
            }
            if metadata.init_state_id is not None:
                entry["init_state_id"] = metadata.init_state_id
            previous = manifest["states"].get(key)
            if previous is not None and previous != entry:
                same_payload = (
                    previous.get("bundle_sha256") == entry["bundle_sha256"]
                    and previous.get("task_id") == entry["task_id"]
                    and previous.get("episode_id") == entry["episode_id"]
                    and previous.get("step") == entry["step"]
                    and previous.get("outcome") == entry["outcome"]
                    and previous.get("init_state_id") == entry.get("init_state_id")
                )
                if not same_payload:
                    raise FileExistsError(f"conflicting manifest entry for {key}")
                # Keep the on-disk path already recorded when content matches.
                entry = previous
            manifest["states"][key] = entry
            fd, temporary = tempfile.mkstemp(prefix=".manifest.", dir=self.root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(canonical_json(manifest) + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.manifest_path)
                self._fsync_dir(self.root)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
