"""Versioned on-disk schema for NGC Step 1 state snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

SCHEMA_VERSION = "ngc-state-pool/v1"
STATE_KEY_VERSION = "sp1"
_KEY_RE = re.compile(r"^sp1_[0-9a-f]{32}$")


def canonical_json(value: Any) -> bytes:
    """Serialize JSON data identically across processes and platforms."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


@dataclass(frozen=True)
class StateMetadata:
    task_id: str
    instruction: str
    suite: str
    episode_id: str
    step: int
    perturb_dim: str
    perturb_sub: str
    level: int
    episode_outcome: str
    seed: int
    snapshot_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        required = {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "suite": self.suite,
            "episode_id": self.episode_id,
            "perturb_dim": self.perturb_dim,
            "perturb_sub": self.perturb_sub,
        }
        empty = [name for name, value in required.items() if not value]
        if empty:
            raise ValueError(f"metadata fields must be non-empty: {', '.join(empty)}")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.level not in range(1, 6):
            raise ValueError("level must be in [1, 5]")
        if self.episode_outcome not in {"success", "failure"}:
            raise ValueError("episode_outcome must be 'success' or 'failure'")
        if self.snapshot_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported snapshot_version {self.snapshot_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def identity(self) -> dict[str, Any]:
        """Fields that identify an environment state independent of annotations."""
        return {
            "state_key_version": STATE_KEY_VERSION,
            "snapshot_version": self.snapshot_version,
            "task_id": self.task_id,
            "suite": self.suite,
            "episode_id": self.episode_id,
            "step": self.step,
            "perturb_dim": self.perturb_dim,
            "perturb_sub": self.perturb_sub,
            "level": self.level,
            "seed": self.seed,
        }

    @property
    def state_key(self) -> str:
        self.validate()
        digest = hashlib.sha256(canonical_json(self.identity())).hexdigest()[:32]
        return f"{STATE_KEY_VERSION}_{digest}"

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["state_key"] = self.state_key
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateMetadata":
        fields = {
            name: value[name]
            for name in cls.__dataclass_fields__
            if name in value
        }
        metadata = cls(**fields)
        metadata.validate()
        supplied_key = value.get("state_key")
        if supplied_key is not None and supplied_key != metadata.state_key:
            raise ValueError("state_key does not match versioned metadata identity")
        return metadata


def validate_state_key(state_key: str) -> None:
    if not _KEY_RE.fullmatch(state_key):
        raise ValueError(f"invalid state key: {state_key!r}")
