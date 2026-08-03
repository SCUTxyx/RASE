"""Small deployment-time features extracted from stored observations."""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import numpy as np


def _image_features(name: str, payload: bytes) -> dict[str, float]:
    from PIL import Image

    image = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.float32)
    image /= 255.0
    gray = image.mean(axis=2)
    prefix = f"image_{name}"
    result = {
        f"{prefix}_mean": float(image.mean()),
        f"{prefix}_std": float(image.std()),
        f"{prefix}_edge_x": float(np.abs(np.diff(gray, axis=1)).mean()),
        f"{prefix}_edge_y": float(np.abs(np.diff(gray, axis=0)).mean()),
    }
    for channel, label in enumerate(("r", "g", "b")):
        result[f"{prefix}_{label}_mean"] = float(image[..., channel].mean())
        result[f"{prefix}_{label}_std"] = float(image[..., channel].std())
    return result


def extract_deployable_features(
    *,
    observations: Mapping[str, bytes],
    proprio: np.ndarray,
    t0: int,
) -> dict[str, float]:
    """Return features available from the current robot observation only.

    Perturbation dimension, level, episode outcome, and oracle outcomes are
    intentionally absent. They may remain dataset annotations for reporting,
    but using them as model inputs would leak benchmark labels at deployment.
    """
    values = np.asarray(proprio, dtype=np.float32).reshape(-1)
    result = {
        "t0": float(t0),
        "proprio_l1": float(np.abs(values).sum()),
        "proprio_l2": float(np.linalg.norm(values)),
        "proprio_mean": float(values.mean()) if values.size else 0.0,
        "proprio_std": float(values.std()) if values.size else 0.0,
    }
    for index, value in enumerate(values[:16]):
        result[f"proprio_{index:02d}"] = float(value)
    for name, payload in sorted(observations.items()):
        result.update(_image_features(str(name), payload))
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("non-finite deployable feature")
    return result


def feature_artifact(
    rows: Mapping[str, Mapping[str, Any]], *, source_pool: str
) -> dict[str, Any]:
    return {
        "schema_version": "rase-selector-features/v1",
        "source_pool": source_pool,
        "n_states": len(rows),
        "features_by_state": {key: dict(value) for key, value in sorted(rows.items())},
        "forbidden_inputs": [
            "cohort",
            "episode_outcome",
            "perturb_dim",
            "perturb_sub",
            "level",
            "policy_outcome",
        ],
    }
