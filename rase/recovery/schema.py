"""Data schema validation for Route C recovery samples.

Defines the canonical schema version and required fields for
recovery training samples so that collector, trainer, and evaluator
can detect version mismatches and legacy zero-feature data early.
"""

from __future__ import annotations

from typing import Any

# ── Schema identity ──────────────────────────────────────────────────

DATA_SCHEMA_VERSION = "route-c-recovery/v2"

REQUIRED_SAMPLE_FIELDS = {
    "delta_target",
    "student_action",
    "obs_features",
    "history_before",
    "history_mask",
    "step_index",
    "episode_id",
}

OPTIONAL_SAMPLE_FIELDS = {
    "teacher_action",
    "reward",
    "done",
    "success",
    "phase",
}

# ── Version metadata to embed in episode files ──────────────────────

VERSION_METADATA_KEYS = {
    "schema_version",
    "feature_pipeline_version",
    "feature_extractor_sha",
    "history_window",
    "proprio_dim",
    "action_dim",
    "obs_feature_dim",
}


# ── Exceptions ──────────────────────────────────────────────────────

class RecoveryDatasetSchemaError(ValueError):
    """Raised when a recovery sample does not conform to the expected schema."""


# ── Validation ──────────────────────────────────────────────────────

def validate_recovery_sample(sample: dict[str, Any]) -> None:
    """Validate a single recovery training sample.

    Raises ``RecoveryDatasetSchemaError`` on missing or zero fields.
    """
    missing = REQUIRED_SAMPLE_FIELDS - set(sample.keys())
    if missing:
        raise RecoveryDatasetSchemaError(
            f"Recovery sample missing required fields: {sorted(missing)}"
        )

    # --- obs_features must be non-empty and non-all-zero (production) ---
    import numpy as np

    obs_feat = np.asarray(sample["obs_features"])
    if obs_feat.size == 0:
        raise RecoveryDatasetSchemaError(
            "obs_features is empty; legacy zero-feature samples cannot be used "
            "for training. If this is intentional for diagnostics, pass "
            "--allow-legacy-zero-features."
        )
    if np.count_nonzero(obs_feat) == 0:
        raise RecoveryDatasetSchemaError(
            "obs_features is all zeros; legacy zero-feature samples cannot be "
            "used for production training. Pass --allow-legacy-zero-features "
            "for diagnostic runs."
        )

    # --- history must be present and contain real values ---
    hist = np.asarray(sample["history_before"])
    if hist.size == 0:
        raise RecoveryDatasetSchemaError("history_before is empty.")

    # delta_target and student_action must be finite
    for field in ("delta_target", "student_action"):
        val = np.asarray(sample[field])
        if not np.all(np.isfinite(val)):
            raise RecoveryDatasetSchemaError(
                f"{field} contains NaN or Inf values"
            )


def is_legacy_zero_feature_sample(sample: dict[str, Any]) -> bool:
    """Return True if this sample appears to be a legacy zero-feature sample."""
    import numpy as np

    if "obs_features" not in sample:
        return True
    obs_feat = np.asarray(sample["obs_features"])
    if obs_feat.size == 0:
        return True
    if np.count_nonzero(obs_feat) == 0:
        # Heuristic: also check that history is present
        if "history_before" not in sample:
            return True
        hist = np.asarray(sample["history_before"])
        if hist.size == 0 or np.count_nonzero(hist) == 0:
            return True
    return False


def validate_episode_metadata(episode: dict[str, Any]) -> None:
    """Validate that an episode file declares its schema version and pipeline identity."""
    if episode.get("schema_version") != DATA_SCHEMA_VERSION:
        raise RecoveryDatasetSchemaError(
            f"Episode schema version is {episode.get('schema_version')!r}, "
            f"expected {DATA_SCHEMA_VERSION!r}. "
            "Run scripts/migrate_route_c_v1_to_v2.py to migrate legacy data."
        )

    missing_meta = VERSION_METADATA_KEYS - set(episode.keys())
    if missing_meta:
        raise RecoveryDatasetSchemaError(
            f"Episode missing version metadata keys: {sorted(missing_meta)}"
        )


def make_version_metadata(
    pipeline_version: str,
    extractor_sha: str,
    history_window: int = 8,
    proprio_dim: int = 8,
    action_dim: int = 7,
    obs_feature_dim: int = 144,
) -> dict[str, Any]:
    """Return a dict of version metadata to embed in every episode file."""
    return {
        "schema_version": DATA_SCHEMA_VERSION,
        "feature_pipeline_version": pipeline_version,
        "feature_extractor_sha": extractor_sha,
        "history_window": history_window,
        "proprio_dim": proprio_dim,
        "action_dim": action_dim,
        "obs_feature_dim": obs_feature_dim,
    }
