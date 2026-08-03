import numpy as np
import pytest

from rase.collect.schema import StateMetadata
from rase.collect.state_pool import StatePool
from scripts.analyze_replacement_audit import analyze
from scripts.export_initial_replacement_keys import freeze_initial_keys


def _oft_row(key: str, episode: str, success: bool) -> dict:
    return {
        "state_key": key,
        "episode_id": episode,
        "direct_oft_success": success,
        "result": {
            "success": success,
            "env_steps": 10,
            "elapsed_s": 1.0,
            "oracle_predict_calls": 2,
            "oracle_predict_elapsed_s": 0.2,
        },
    }


def test_reset_export_distinguishes_policy_step_from_libero_reset_steps(tmp_path) -> None:
    pool = StatePool(tmp_path / "pool")
    metadata = StateMetadata(
        task_id="libero_goal_000001",
        instruction="do the task",
        suite="Goal",
        episode_id="ep0",
        step=0,
        perturb_dim="clean",
        perturb_sub="none",
        level=0,
        episode_outcome="success",
        seed=7,
        libero_flavor="clean",
    )
    pool.write_state(
        metadata,
        sim_state=np.zeros(2, dtype=np.float64),
        controller_state={"env_counters": {"timestep": 10}},
        rng_state={},
        observations={"agentview": b"test-png-bytes"},
        proprio=np.zeros(8, dtype=np.float32),
    )
    design = {
        "schema_version": "rase-independent-factorial-design/v1",
        "status": "ready",
        "n_requests": 1,
        "records": [
            {
                "request_index": 0,
                "episode_id": "ep0",
                "task_id": "libero_goal_000001",
                "suite": "Goal",
                "dimension": "clean",
                "level": 0,
            }
        ],
    }
    result = freeze_initial_keys(
        pool.root, design, expected_reset_simulator_timestep=10
    )
    assert result["reset_semantics"] == {
        "snapshot_policy_step": 0,
        "source_actions_before_snapshot": 0,
        "expected_post_reset_simulator_timestep": 10,
        "note": (
            "LIBERO reset performs internal initialization steps; the simulator "
            "counter is therefore nonzero before either policy acts."
        ),
    }
    assert result["records"][0]["snapshot_simulator_timestep"] == 10
    with pytest.raises(ValueError, match="post-reset simulator timestep"):
        freeze_initial_keys(
            pool.root, design, expected_reset_simulator_timestep=0
        )


def test_replacement_audit_joins_exact_episodes_and_flags_replacement_risk() -> None:
    records = []
    handoff_rows = []
    for index in range(4):
        records.append(
            {
                "state_key": f"k{index}",
                "task_id": f"task{index}",
                "episode_id": f"ep{index}",
                "suite": "Spatial" if index < 2 else "Goal",
                "perturbation_dimension": "clean" if index % 2 == 0 else "camera",
                "perturbation_level": 0 if index % 2 == 0 else 1,
                "source_only_success": index == 0,
            }
        )
        handoff_rows.append(
            {
                "task_id": f"task{index}",
                "episode_id": f"ep{index}",
                "direct_oft_success": index < 3,
            }
        )
    keys = {
        "artifact_version": "rase-replacement-initial-keys/v1",
        "selection_uses_outcomes": False,
        "n_states": 4,
        "state_keys_sha256": "checksum",
        "records": records,
    }
    source_summary = {
        "episode_metrics": [
            {
                "env_steps": 10,
                "policy_select_calls": 10,
                "policy_select_elapsed_s": 0.5,
                "episode_wall_s": 1.5,
            }
            for _ in range(4)
        ]
    }
    spatial = {
        "schema_version": "rase-oft-direct-escalation/v1",
        "status": "complete",
        "suite": "libero_spatial",
        "state_keys_sha256": "checksum",
        "per_state": [
            _oft_row("k0", "ep0", True),
            _oft_row("k1", "ep1", True),
        ],
    }
    goal = {
        "schema_version": "rase-oft-direct-escalation/v1",
        "status": "complete",
        "suite": "libero_goal",
        "state_keys_sha256": "checksum",
        "per_state": [
            _oft_row("k2", "ep2", True),
            _oft_row("k3", "ep3", False),
        ],
    }
    handoff = {
        "schema_version": "rase-deferred-switch-analysis/v1",
        "timing": {"direct_oft": {"policy_ms_per_env_step": 10.0}},
        "three_operator": {"per_state": handoff_rows},
    }
    result = analyze(
        keys,
        source_summary,
        [("libero_spatial", spatial), ("libero_goal", goal)],
        handoff,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )
    assert result["overall_successes"] == {
        "source_only_success": 1,
        "oft_only_success": 3,
        "source_to_oft_success": 3,
    }
    assert result["source_vs_oft_only_quadrants"] == {
        "rescue": 2,
        "harm": 0,
        "redundant": 1,
        "unsupported": 1,
    }
    assert result["replacement_gate"]["status"] == (
        "replacement_risk_high_cost_audit_required"
    )
    assert result["timing"]["source_only_full_episode"][
        "policy_ms_per_env_step"
    ] == 50.0


def test_replacement_audit_retains_framing_for_cross_suite_source_unique_wins() -> None:
    records = [
        {
            "state_key": f"k{index}",
            "task_id": f"task{index}",
            "episode_id": f"ep{index}",
            "suite": suite,
            "perturbation_dimension": "clean",
            "perturbation_level": 0,
            "source_only_success": True,
        }
        for index, suite in enumerate(("Spatial", "Goal"))
    ]
    keys = {
        "artifact_version": "rase-replacement-initial-keys/v1",
        "selection_uses_outcomes": False,
        "n_states": 2,
        "state_keys_sha256": "checksum",
        "records": records,
    }
    source_summary = {
        "episode_metrics": [
            {
                "env_steps": 1,
                "policy_select_calls": 1,
                "policy_select_elapsed_s": 0.01,
                "episode_wall_s": 0.1,
            }
            for _ in range(2)
        ]
    }
    summaries = []
    handoff_rows = []
    for index, suite in enumerate(("libero_spatial", "libero_goal")):
        summaries.append(
            (
                suite,
                {
                    "schema_version": "rase-oft-direct-escalation/v1",
                    "status": "complete",
                    "suite": suite,
                    "state_keys_sha256": "checksum",
                    "per_state": [_oft_row(f"k{index}", f"ep{index}", False)],
                },
            )
        )
        handoff_rows.append(
            {
                "task_id": f"task{index}",
                "episode_id": f"ep{index}",
                "direct_oft_success": False,
            }
        )
    result = analyze(
        keys,
        source_summary,
        summaries,
        {
            "schema_version": "rase-deferred-switch-analysis/v1",
            "three_operator": {"per_state": handoff_rows},
        },
        bootstrap_replicates=20,
        bootstrap_seed=1,
    )
    assert result["replacement_gate"]["status"] == "recovery_framing_signal"
