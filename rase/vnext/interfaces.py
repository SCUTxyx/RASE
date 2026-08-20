"""Runtime protocols for benchmark and policy adapters used by RASE vNext."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np

from .schema import (
    CanonicalActionToken, CanonicalObservation, CanonicalRobotSpec,
    CorrectionProfile, PolicyDescriptor, SeedLedger,
)


class BenchmarkAdapter(Protocol):
    @property
    def robot_spec(self) -> CanonicalRobotSpec: ...

    def reset(self) -> Mapping[str, Any]: ...

    def observation_to_canonical(
        self, observation: Mapping[str, Any], *, task_text: str, timestamp_s: float
    ) -> CanonicalObservation: ...

    def success(self, info: Mapping[str, Any]) -> bool: ...

    def snapshot(self) -> Any: ...

    def restore(self, snapshot: Any) -> None: ...

    def execute(self, action: CanonicalActionToken) -> list[tuple[Any, ...]]: ...


class PolicyAdapter(Protocol):
    @property
    def descriptor(self) -> PolicyDescriptor: ...

    def propose(self, observation: CanonicalObservation) -> CanonicalActionToken: ...

    def raw_to_canonical(self, value: np.ndarray) -> CanonicalActionToken: ...

    def canonical_to_raw(self, token: CanonicalActionToken) -> np.ndarray: ...


class CorrectionOperator(Protocol):
    """One semantic correction operator; resample candidates stay internal."""

    @property
    def profile(self) -> CorrectionProfile: ...

    def propose(
        self, observation: CanonicalObservation, *, seed_ledger: SeedLedger
    ) -> CanonicalActionToken | None: ...
