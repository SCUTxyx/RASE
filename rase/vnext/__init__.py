"""RASE vNext deployment contracts and preregistered offline audits."""

from .schema import (
    CanonicalActionToken,
    CanonicalObservation,
    CanonicalRobotSpec,
    CorrectionKind,
    CorrectionProfile,
    PolicyDescriptor,
    SeedLedger,
    operator_mask,
)
from .discovery import build_discovery_manifest, select_discovery_roots
from .opportunity import audit_opportunity

__all__ = [
    "CanonicalActionToken",
    "CanonicalObservation",
    "CanonicalRobotSpec",
    "CorrectionKind",
    "CorrectionProfile",
    "PolicyDescriptor",
    "SeedLedger",
    "operator_mask",
    "build_discovery_manifest",
    "select_discovery_roots",
    "audit_opportunity",
]
