from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_intervention_matrix.py"
    spec = importlib.util.spec_from_file_location("analyze_intervention_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_analysis_reports_unique_winners_groups_and_cluster_bootstrap():
    operators = ["continue_source", "replan_source", "switch_target"]
    patterns = [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ]
    snapshots = [
        {
            "snapshot_id": f"sp1_{index:032x}",
            "task_id": f"task-{index // 2}",
            "episode_id": f"episode-{index // 2}",
            "suite": "Spatial" if index < 2 else "Object",
            "step": index % 2,
            "perturbation": {"dimension": "clean", "level": 0},
        }
        for index in range(4)
    ]
    outcomes = [
        {
            "snapshot_id": snapshot["snapshot_id"],
            "operator_id": operator_id,
            "observed": True,
            "proxy": False,
            "success": success,
        }
        for snapshot, pattern in zip(snapshots, patterns)
        for operator_id, success in zip(operators, pattern)
    ]
    result = _module().analyze_matrix(
        snapshots,
        outcomes,
        operators,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )
    assert result["overall"]["same_state_oracle_success_rate"] == 0.75
    assert result["overall"]["best_fixed_success_rate"] == 0.25
    assert result["overall"]["oracle_minus_best_fixed"] == 0.5
    assert result["overall"]["unique_winner_counts"] == {
        "continue_source": 1,
        "replan_source": 1,
        "switch_target": 1,
    }
    assert result["by_group"]["suite"]["Spatial"]["n_states"] == 2
    assert result["by_group"]["dimension_level"]["clean:L0"]["n_states"] == 4
    assert result["episode_cluster_bootstrap"]["n_episode_clusters"] == 2
    assert result["pairwise_vs_continue"]["replan_source"] == {
        "continue_only_states": 1,
        "operator_only_states": 1,
        "tied_states": 2,
        "mcnemar_exact_p": pytest.approx(1.0),
    }


def test_write_json_normalizes_numpy_scalars(tmp_path: Path):
    output = tmp_path / "analysis.json"

    _module()._write_json(output, {"count": np.int64(3), "rate": np.float64(0.5)})

    assert json.loads(output.read_text()) == {"count": 3, "rate": 0.5}
