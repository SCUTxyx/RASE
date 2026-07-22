"""Small disk-backed scheduler for week-long rollout collection.

Every unit is keyed by ``(state, candidate, rollout)``. Results and retry
metadata are written atomically, so constructing a new scheduler on the same
directory resumes without a separate database.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEDULER_VERSION = 1


@dataclass(frozen=True, order=True)
class RolloutKey:
    state: str
    candidate: int
    rollout: int

    def __post_init__(self) -> None:
        if not self.state:
            raise ValueError("state key must be non-empty")
        if self.candidate < 0 or self.rollout < 0:
            raise ValueError("candidate and rollout indices must be non-negative")

    @property
    def state_digest(self) -> str:
        return hashlib.sha256(self.state.encode("utf-8")).hexdigest()[:24]

    @property
    def identity(self) -> str:
        return f"{self.state}\0{self.candidate}\0{self.rollout}"


# A less domain-specific alias is convenient for callers and older scripts.
TaskKey = RolloutKey


@dataclass(frozen=True)
class Claim:
    key: RolloutKey
    worker: str
    claimed_at: float
    lease_seconds: float


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


class DiskRolloutScheduler:
    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_attempts: int = 3,
        lease_seconds: float = 3600.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.root = Path(root)
        self.max_attempts = int(max_attempts)
        self.lease_seconds = float(lease_seconds)
        for name in ("results", "attempts", "claims"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def _path(self, section: str, key: RolloutKey) -> Path:
        filename = f"c{key.candidate:03d}-r{key.rollout:03d}.json"
        return self.root / section / key.state_digest / filename

    def _validate_key(self, stored: Mapping[str, Any], key: RolloutKey) -> None:
        raw = stored.get("key")
        expected = asdict(key)
        if raw != expected:
            raise ValueError(
                f"key digest collision/corrupt scheduler record: {raw!r} != {expected!r}"
            )

    def is_complete(self, key: RolloutKey) -> bool:
        path = self._path("results", key)
        if not path.exists():
            return False
        self._validate_key(_read_json(path), key)
        return True

    def result(self, key: RolloutKey) -> dict[str, Any] | None:
        path = self._path("results", key)
        if not path.exists():
            return None
        record = _read_json(path)
        self._validate_key(record, key)
        return record

    def attempts(self, key: RolloutKey) -> int:
        path = self._path("attempts", key)
        if not path.exists():
            return 0
        record = _read_json(path)
        self._validate_key(record, key)
        return int(record["attempts"])

    def can_retry(self, key: RolloutKey) -> bool:
        return not self.is_complete(key) and self.attempts(key) < self.max_attempts

    def _release_stale_claim(
        self, key: RolloutKey, worker: str, now: float
    ) -> bool:
        """Return True if another worker still holds a live lease."""
        path = self._path("claims", key)
        if not path.exists():
            return False
        record = _read_json(path)
        self._validate_key(record, key)
        expired = now >= float(record["claimed_at"]) + float(record["lease_seconds"])
        # Same worker may reclaim after a crash before the lease expires.
        same_worker = str(record.get("worker", "")) == worker
        if expired or same_worker:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return False
        return True

    def claim(
        self, key: RolloutKey, worker: str, *, now: float | None = None
    ) -> Claim | None:
        """Atomically lease work; returns None if complete, leased, or exhausted."""
        if not worker:
            raise ValueError("worker must be non-empty")
        now = time.time() if now is None else float(now)
        if not self.can_retry(key):
            return None
        if self._release_stale_claim(key, worker, now):
            return None
        claim = Claim(key, worker, now, self.lease_seconds)
        path = self._path("claims", key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SCHEDULER_VERSION,
            "key": asdict(key),
            "worker": worker,
            "claimed_at": now,
            "lease_seconds": self.lease_seconds,
        }
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            # Lost the race; treat as active foreign lease.
            return None
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        return claim

    def complete(
        self,
        key: RolloutKey,
        result: Mapping[str, Any],
        *,
        worker: str | None = None,
    ) -> dict[str, Any]:
        """Commit once. Repeated completion returns the original record."""
        existing = self.result(key)
        if existing is not None:
            return existing
        claim_path = self._path("claims", key)
        if worker is not None and claim_path.exists():
            claim = _read_json(claim_path)
            self._validate_key(claim, key)
            if claim.get("worker") != worker:
                raise PermissionError("rollout is leased by another worker")
        record = {
            "version": SCHEDULER_VERSION,
            "key": asdict(key),
            "result": dict(result),
            "completed_at": time.time(),
        }
        _atomic_json(self._path("results", key), record)
        try:
            claim_path.unlink()
        except FileNotFoundError:
            pass
        return record

    def fail(
        self,
        key: RolloutKey,
        error: str,
        *,
        worker: str | None = None,
    ) -> int:
        """Record a failed attempt and release its lease for a later retry."""
        if self.is_complete(key):
            return self.attempts(key)
        path = self._path("attempts", key)
        count = self.attempts(key) + 1
        _atomic_json(
            path,
            {
                "version": SCHEDULER_VERSION,
                "key": asdict(key),
                "attempts": count,
                "last_error": str(error),
                "failed_at": time.time(),
                "worker": worker,
            },
        )
        claim_path = self._path("claims", key)
        if worker is None or not claim_path.exists():
            try:
                claim_path.unlink()
            except FileNotFoundError:
                pass
        else:
            claim = _read_json(claim_path)
            if claim.get("worker") == worker:
                claim_path.unlink()
        return count

    def _foreign_claim_active(self, key: RolloutKey, now: float) -> bool:
        """True when an unexpired claim exists (pending scan has no worker id)."""
        path = self._path("claims", key)
        if not path.exists():
            return False
        record = _read_json(path)
        self._validate_key(record, key)
        expired = now >= float(record["claimed_at"]) + float(record["lease_seconds"])
        if expired:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return False
        return True

    def pending(self, keys: Iterable[RolloutKey]) -> Iterator[RolloutKey]:
        """Yield resumable keys in input order, skipping persisted results."""
        now = time.time()
        for key in keys:
            if self.can_retry(key) and not self._foreign_claim_active(key, now):
                yield key

    def completed_keys(self) -> Iterator[RolloutKey]:
        """Scan durable records, allowing resume diagnostics without a manifest."""
        for path in sorted((self.root / "results").glob("*/*.json")):
            record = _read_json(path)
            try:
                raw = record["key"]
                key = RolloutKey(
                    state=str(raw["state"]),
                    candidate=int(raw["candidate"]),
                    rollout=int(raw["rollout"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"corrupt scheduler record: {path}") from exc
            self._validate_key(record, key)
            yield key


Scheduler = DiskRolloutScheduler
