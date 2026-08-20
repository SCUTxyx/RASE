"""R6-D world-model auxiliary feature vector construction.

Turns one R6-D cache row (a boundary of a trajectory group) into the fixed-size
feature vector appended to the R6-C baseline inputs.  Pre-registered in
`progress/2026-08-10_r6d_wm_ablation_preregistration.md`:

- residual per k in K={1,4,8} (predicted minus real pooled latent), down-sampled
  to a small fixed dimension by mean-pooling over chunks;
- disagreement direction variance and magnitude variance;
- optionally the pooled latent ``z_t`` (mean-pooled).

The cache script stores JSON lists; this module is the single place that
converts them into a deterministic float32 vector so the baseline trainer and
the Pareto evaluator cannot disagree on the feature layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _pooled(values: list[float], out_dim: int) -> np.ndarray:
    """Deterministic chunk-mean pooling of a vector to ``out_dim`` floats."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(out_dim, dtype=np.float32)
    if arr.size < out_dim:
        return np.pad(arr, (0, out_dim - arr.size))
    edges = np.linspace(0, arr.size, out_dim + 1).astype(int)
    out = np.asarray([arr[edges[i]:edges[i + 1]].mean() for i in range(out_dim)],
                     dtype=np.float32)
    return out


def feature_vector(row: dict[str, Any], latent_dim: int = 16,
                   residual_dim: int = 16) -> np.ndarray | None:
    """Build the fixed-size WM feature vector for one cache row.

    Layout is deterministic: one ``residual_dim`` block per k in ``k_values``
    (per-dim squared error between the predicted and the real time-aligned
    pooled latent; zero-filled only when the source episode ended before
    ``t + k`` and no real frame exists), then the two disagreement scalars,
    then the pooled latent.  All caches produced for a given protocol must
    therefore share the same width.
    """
    if "error" in row or "latent_z_t" not in row:
        return None
    ks = sorted(int(k) for k in row.get("k_values", []))
    if not ks:
        ks = sorted(int(key) for key in row.get("residual", {}) if key.isdigit())
    residuals = row.get("residual", {})
    parts: list[np.ndarray] = []
    for k in ks:
        value = residuals.get(str(k)) if isinstance(residuals, dict) else None
        if isinstance(value, list) and value:
            parts.append(_pooled(value, residual_dim))
        else:
            parts.append(np.zeros(residual_dim, dtype=np.float32))
    disagreement = row.get("disagreement", {})
    parts.append(np.asarray([
        float(disagreement.get("delta_direction_var", 0.0)),
        float(disagreement.get("delta_magnitude_var", 0.0)),
    ], dtype=np.float32))
    parts.append(_pooled(row["latent_z_t"], latent_dim))
    if not parts:
        return None
    return np.concatenate(parts).astype(np.float32)
