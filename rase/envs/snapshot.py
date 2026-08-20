"""Versioned, pickle-free snapshots for forkable simulation environments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_FORMAT = "rase.forkable_env"
SNAPSHOT_VERSION = 2


class SnapshotError(RuntimeError):
    """Raised when a snapshot is malformed or incompatible."""


def _numpy():
    # Keep importing rase.envs.snapshot usable in orchestration processes that
    # have not loaded the numerical / simulator stack yet.
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised outside test env
        raise SnapshotError("NumPy is required to read or write snapshots") from exc
    return np


def _paths(path: os.PathLike[str] | str) -> tuple[Path, Path]:
    path = Path(path)
    if path.suffix in {".json", ".npz"}:
        path = path.with_suffix("")
    return path.with_suffix(".json"), path.with_suffix(".npz")


def _encode(value: Any, arrays: dict[str, Any]) -> Any:
    np = _numpy()
    if isinstance(value, np.ndarray):
        key = f"a{len(arrays):06d}"
        if value.dtype.hasobject:
            raise SnapshotError(f"object-dtype array is forbidden at {key}")
        arrays[key] = np.ascontiguousarray(value)
        return {"__array__": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_encode(item, arrays) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise SnapshotError("snapshot mappings must have string keys")
        return {key: _encode(item, arrays) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise SnapshotError(f"unsupported snapshot value type: {type(value).__module__}.{type(value).__qualname__}")


def _decode(value: Any, arrays: Mapping[str, Any]) -> Any:
    if isinstance(value, list):
        return [_decode(item, arrays) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__array__"}:
            key = value["__array__"]
            if key not in arrays:
                raise SnapshotError(f"snapshot references missing array {key!r}")
            return arrays[key].copy()
        if set(value) == {"__tuple__"}:
            return tuple(_decode(item, arrays) for item in value["__tuple__"])
        return {key: _decode(item, arrays) for key, item in value.items()}
    return value


def _array_digest(array: Any) -> str:
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


@dataclass(frozen=True)
class EnvSnapshot:
    """An in-memory environment snapshot.

    ``payload`` is deliberately constrained to JSON values, tuples, and
    non-object NumPy arrays so persistence can never invoke pickle.
    """

    task_fingerprint: str
    payload: Mapping[str, Any]
    version: int = SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if self.version != SNAPSHOT_VERSION:
            raise SnapshotError(
                f"unsupported snapshot version {self.version}; expected {SNAPSHOT_VERSION}"
            )
        if not isinstance(self.task_fingerprint, str) or not self.task_fingerprint:
            raise SnapshotError("task_fingerprint must be a non-empty string")
        if not isinstance(self.payload, Mapping):
            raise SnapshotError("payload must be a mapping")

    def save(self, path: os.PathLike[str] | str) -> tuple[Path, Path]:
        """Atomically write sibling ``.json`` and ``.npz`` files."""

        np = _numpy()
        json_path, npz_path = _paths(path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {}
        encoded_payload = _encode(dict(self.payload), arrays)
        manifest = {
            "format": SNAPSHOT_FORMAT,
            "version": self.version,
            "task_fingerprint": self.task_fingerprint,
            "payload": encoded_payload,
            "arrays": {
                key: {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "sha256": _array_digest(array),
                }
                for key, array in arrays.items()
            },
        }

        json_tmp: Path | None = None
        npz_tmp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=npz_path.parent, prefix=f".{npz_path.name}.", delete=False
            ) as handle:
                npz_tmp = Path(handle.name)
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=json_path.parent,
                prefix=f".{json_path.name}.",
                delete=False,
            ) as handle:
                json_tmp = Path(handle.name)
                json.dump(manifest, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(npz_tmp, npz_path)
            os.replace(json_tmp, json_path)
        finally:
            for temporary in (json_tmp, npz_tmp):
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return json_path, npz_path

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "EnvSnapshot":
        """Load and validate a sibling JSON/NPZ snapshot pair."""

        np = _numpy()
        json_path, npz_path = _paths(path)
        try:
            manifest = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"cannot read snapshot manifest {json_path}: {exc}") from exc

        required = {"format", "version", "task_fingerprint", "payload", "arrays"}
        if set(manifest) != required:
            raise SnapshotError(
                f"manifest keys differ: expected {sorted(required)}, got {sorted(manifest)}"
            )
        if manifest["format"] != SNAPSHOT_FORMAT:
            raise SnapshotError(f"unexpected snapshot format {manifest['format']!r}")
        if manifest["version"] != SNAPSHOT_VERSION:
            raise SnapshotError(
                f"unsupported snapshot version {manifest['version']}; expected {SNAPSHOT_VERSION}"
            )

        expected_arrays = manifest["arrays"]
        if not isinstance(expected_arrays, dict):
            raise SnapshotError("manifest arrays entry must be an object")
        try:
            with np.load(npz_path, allow_pickle=False) as archive:
                if set(archive.files) != set(expected_arrays):
                    raise SnapshotError("NPZ members do not match the JSON manifest")
                arrays = {}
                for key in archive.files:
                    array = archive[key]
                    spec = expected_arrays[key]
                    if (
                        set(spec) != {"dtype", "shape", "sha256"}
                        or array.dtype.str != spec["dtype"]
                        or list(array.shape) != spec["shape"]
                        or _array_digest(array) != spec["sha256"]
                    ):
                        raise SnapshotError(f"array {key!r} failed manifest validation")
                    arrays[key] = array.copy()
        except (OSError, ValueError) as exc:
            raise SnapshotError(f"cannot read snapshot arrays {npz_path}: {exc}") from exc

        return cls(
            version=manifest["version"],
            task_fingerprint=manifest["task_fingerprint"],
            payload=_decode(manifest["payload"], arrays),
        )
