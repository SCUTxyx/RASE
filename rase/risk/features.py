"""Feature extraction for candidate-conditioned risk scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

FAMILY_TO_ID = {
    "current_suffix": 0,
    "strict_resample": 1,
    "fresh_replan": 2,
    "receding_horizon": 3,
}


def action_chunk_features(action_tensor: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Compact finite features from a [T,7] action chunk."""

    actions = np.asarray(action_tensor, dtype=np.float64)
    if actions.size == 0:
        return np.zeros(12, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"action_tensor must be [T,7], got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
    delta = np.diff(actions, axis=0) if len(actions) > 1 else np.zeros((1, 7))
    return np.asarray(
        [
            float(actions.shape[0]),
            float(np.mean(np.linalg.norm(actions[:, :3], axis=1))),
            float(np.mean(np.linalg.norm(actions[:, 3:6], axis=1))),
            float(np.mean(np.abs(actions[:, 6]))),
            float(np.std(actions[:, :3])),
            float(np.std(actions[:, 3:6])),
            float(np.mean(np.linalg.norm(delta[:, :3], axis=1))),
            float(np.mean(np.linalg.norm(delta[:, 3:6], axis=1))),
            float(np.max(np.linalg.norm(actions, axis=1))),
            float(np.mean(actions[:, 0])),
            float(np.mean(actions[:, 1])),
            float(np.mean(actions[:, 2])),
        ],
        dtype=np.float64,
    )


def history_only_features(row: Mapping[str, Any]) -> np.ndarray:
    """State-level features that ignore the candidate action chunk."""

    stage = str(row.get("stage") or "")
    cell = str(row.get("cell") or "")
    suite = str(row.get("suite") or "")
    suites = ("Spatial", "Object", "Goal", "Long")
    cells = ("clean:L0", "camera:L1", "robot:L1")
    stages = ("T1", "T3")
    return np.asarray(
        [
            float(stages.index(stage) if stage in stages else -1),
            float(cells.index(cell) if cell in cells else -1),
            float(suites.index(suite) if suite in suites else -1),
            float(bool(row.get("source_episode_outcome") == "success")),
        ],
        dtype=np.float64,
    )


def export_candidate_rows(
    state_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten corrective rollout arms into candidate-level supervised rows."""

    exported: list[dict[str, Any]] = []
    for state in state_rows:
        history = history_only_features(state)
        for arm in state.get("arms") or []:
            family = str(arm.get("family") or "")
            if family not in FAMILY_TO_ID:
                continue
            actions = arm.get("action_tensor") or []
            action_feat = action_chunk_features(actions)
            family_onehot = np.zeros(len(FAMILY_TO_ID), dtype=np.float64)
            family_onehot[FAMILY_TO_ID[family]] = 1.0
            horizon = arm.get("execution_horizon")
            horizon_feat = np.asarray(
                [
                    float(horizon) if horizon is not None else 0.0,
                    1.0 if horizon is not None else 0.0,
                ],
                dtype=np.float64,
            )
            candidate_x = np.concatenate([history, family_onehot, horizon_feat, action_feat])
            history_x = np.concatenate([history, family_onehot, horizon_feat])
            exported.append(
                {
                    "state_key": state.get("state_key"),
                    "episode_id": state.get("episode_id"),
                    "task_id": state.get("task_id"),
                    "suite": state.get("suite"),
                    "cell": state.get("cell"),
                    "stage": state.get("stage"),
                    "family": family,
                    "arm_name": arm.get("arm_name"),
                    "success": bool(arm.get("success")),
                    "x_candidate": candidate_x.tolist(),
                    "x_history": history_x.tolist(),
                    "action_l2": float(np.linalg.norm(action_feat)),
                }
            )
    return exported
