import pytest

from scripts.build_pre_b_safe_handback_dataset import build_dataset
from scripts.train_safe_handback_baselines import evaluate


def test_build_dataset_requires_gate_pass():
    audit = {
        "gate_pass": False,
        "status": "not_ready",
        "per_state": [],
    }
    with pytest.raises(ValueError, match="refusing"):
        build_dataset(audit)


def test_build_and_evaluate_baselines():
    audit = {
        "gate_pass": True,
        "status": "pre_a3_gate_pass",
        "per_state": [
            {
                "state_key": "s0",
                "task_id": "t0",
                "suite": "Goal",
                "cell": "camera:L1",
                "split": "train",
                "outcomes": {"0": False, "8": False, "32": True},
                "base_success": False,
                "direct_oft_success": True,
            },
            {
                "state_key": "s1",
                "task_id": "t1",
                "suite": "Spatial",
                "cell": "robot:L1",
                "split": "test",
                "outcomes": {"0": True, "8": True, "32": False},
                "base_success": True,
                "direct_oft_success": True,
            },
        ],
    }
    dataset = build_dataset(audit)
    assert dataset["n_rows"] == 4
    result = evaluate(dataset)
    assert result["world_model_used"] is False
    assert result["best_fixed_duration"]["h"] in {8, 32}
