from __future__ import annotations

import json
from pathlib import Path

from rase.collect.pre_c0 import (
    episode_cluster_bootstrap_natural_headroom,
    horizon_decomposition,
    leave_one_task_out_natural_direction,
)
from scripts.audit_pre_c0_collection_integrity import audit_collection
from scripts.freeze_pre_c0_48_state_manifest import build_manifest


class _FakePool:
    def __init__(self, root: Path, states: dict):
        self.root = root
        self._states = states

    def manifest(self):
        return {"states": self._states}

    def read_state(self, state_key, *, load_observations=True):
        del load_observations
        entry = self._states[state_key]

        class Meta:
            seed = 123

        class Loaded:
            metadata = Meta()
            controller_state = {
                "decision_context": {
                    "schema_version": "v2",
                    "public_observation_history": [],
                    "public_proprio_history": [],
                    "public_action_history": [],
                    "active_action_suffix": [],
                }
            }

        Loaded.state_key = state_key
        Loaded.metadata.seed = 1000 + hash(entry["episode_id"]) % 1000
        return Loaded()


def test_audit_collection_accepts_complete_summary_without_exit_marker(tmp_path: Path):
    design = {
        "n_episodes": 2,
        "design_sha256": "abc",
        "records": [
            {"episode_id": "ep-0", "concrete_task_id": "t0"},
            {"episode_id": "ep-1", "concrete_task_id": "t1"},
        ],
    }
    summary = {
        "episode_metrics": [
            {"episode_id": "ep-0", "outcome": "failure"},
            {"episode_id": "ep-1", "outcome": "success"},
        ]
    }
    states = {
        "k0": {"episode_id": "ep-0", "step": 0, "task_id": "t0"},
        "k1": {"episode_id": "ep-1", "step": 0, "task_id": "t1"},
    }
    log = tmp_path / "collect.log"
    log.write_text(
        "COLLECT_EPISODE_DONE index=0\nCOLLECT_EPISODE_DONE index=1\nWROTE summary\n",
        encoding="utf-8",
    )
    report = audit_collection(
        design=design,
        summary=summary,
        pool=_FakePool(tmp_path, states),
        collect_log=log,
        sample_decision_context=2,
    )
    # expected_episodes_24 fails for this tiny fixture; focus on completion logic helpers.
    assert report["collect_log"]["exit_marker_present"] is False
    assert report["collect_log"]["collect_episode_done_count"] == 2
    assert report["completed_trajectories"] == 2
    assert not report["episodes_without_snapshots"]


def test_build_manifest_selects_t1_t3_and_counts_48():
    episodes = []
    selected = []
    for index in range(24):
        episode_id = f"ep-{index:02d}"
        stages = {}
        for stage, step in (("T0", 1), ("T1", 2), ("T2", 3), ("T3", 4), ("T4", 5)):
            key = f"{episode_id}-{stage}"
            stages[stage] = {"name": stage, "state_key": key, "step": step, "index": step - 1}
            selected.append(
                {
                    "episode_id": episode_id,
                    "task_id": f"task{index}",
                    "suite": "Spatial",
                    "cell": "clean:L0",
                    "stage": stage,
                    "state_key": key,
                    "step": step,
                }
            )
        episodes.append(
            {
                "episode_id": episode_id,
                "logical_task_id": f"task{index}",
                "task_id": f"concrete{index}",
                "suite": "Spatial",
                "cell": "clean:L0",
                "reliability": {"reliable": True},
                "temporal_fallback": False,
                "stages": stages,
            }
        )
    payload = {
        "schema_version": "rase-pre-c0-deviation-keys/v1",
        "selected_states": selected,
        "episodes": episodes,
        "reliability_summary": {"reliable_rate": 1.0},
    }
    manifest = build_manifest(payload)
    assert manifest["n_states"] == 48
    assert manifest["n_episodes"] == 24
    assert not manifest["missing_episode_stage_pairs"]
    assert {row["stage"] for row in manifest["records"]} == {"T1", "T3"}


def test_bootstrap_and_horizon_helpers_are_deterministic():
    rows = []
    for index in range(8):
        rows.append(
            {
                "state_key": f"k{index}",
                "episode_id": f"ep{index // 2}",
                "task_id": f"task{index // 2}",
                "suite": "Spatial",
                "stage": "T1" if index % 2 == 0 else "T3",
                "family_success": {
                    "current_suffix": False,
                    "strict_resample": index % 4 == 0,
                    "fresh_replan": False,
                    "receding_horizon": index % 3 == 0,
                },
                "arms": [
                    {
                        "family": "receding_horizon",
                        "execution_horizon": horizon,
                        "success": horizon == 2 and index % 3 == 0,
                    }
                    for horizon in (1, 2, 4)
                ],
            }
        )
    first = episode_cluster_bootstrap_natural_headroom(rows, replicates=200, seed=7)
    second = episode_cluster_bootstrap_natural_headroom(rows, replicates=200, seed=7)
    assert first == second
    assert "ci95_pp" in first
    horizon = horizon_decomposition(rows)
    assert set(horizon["per_horizon_successes"]) == {"H1", "H2", "H4"}
    loto = leave_one_task_out_natural_direction(rows)
    assert loto["n_folds"] == 4
