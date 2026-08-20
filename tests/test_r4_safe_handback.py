import numpy as np
import torch

from scripts.train_r4_safe_handback_world_model import (
    SafeHandbackWorldModel,
    _state_policy,
    _tensorize,
    build_arrays,
    grouped_task_folds,
    inner_task_split,
    objective,
    validate_rows,
)


def _row(state: str, task: str, h: int, handback: bool) -> dict:
    latent = (np.arange(8, dtype=np.float32) + h / 10).tolist()
    return {
        "state_key": state,
        "task_id": task,
        "suite": "Spatial" if task.endswith("0") else "Goal",
        "elapsed_oft_steps": h,
        "simulator_timestep": 10 + h,
        "horizon": 200,
        "latent": latent,
        "proprio": np.linspace(-1, 1, 8).tolist(),
        "student_action": np.zeros(7).tolist(),
        "oft_action": np.ones(7).tolist(),
        "student_action_chunk": np.zeros((10, 7)).tolist(),
        "next_latent_student": (np.asarray(latent) + 0.2).tolist(),
        "next_latent_oft": (np.asarray(latent) + 0.1).tolist(),
        "student_step_terminal": False,
        "oft_step_terminal": False,
        "success_if_handback_now": handback,
        "success_if_continue_oft": True,
        "remaining_teacher_steps": 80 - h,
        "persistent_executed_oft_steps": 80,
    }


def test_arrays_model_and_task_folds():
    rows = [
        _row(f"s{i}", f"task{i % 3}", h, h >= 32)
        for i in range(6)
        for h in (0, 32)
    ]
    dims = validate_rows(rows)
    assert dims["student_chunk"] == 70
    arrays, _ = build_arrays(rows)
    model = SafeHandbackWorldModel(
        arrays["state"].shape[1], arrays["decision"].shape[1],
        arrays["delta"].shape[2], arrays["transition"].shape[2], 16,
    )
    tensors = _tensorize(arrays, "cpu")
    pred = model(tensors["state"], tensors["decision"], tensors["transition"])
    assert pred["delta"].shape == tensors["delta"].shape
    assert torch.isfinite(objective(pred, tensors, 1.0))
    folds = grouped_task_folds(rows, 3)
    assert len(folds) == 3
    for fold in folds:
        train_tasks = {row["task_id"] for row in fold["train"]}
        val_tasks = {row["task_id"] for row in fold["val"]}
        assert train_tasks.isdisjoint(val_tasks)
    inner_fit, calibration, calibration_tasks = inner_task_split(rows, rotation=1)
    assert calibration_tasks
    assert {row["task_id"] for row in inner_fit}.isdisjoint(
        {row["task_id"] for row in calibration}
    )
    assert {row["task_id"] for row in calibration} == set(calibration_tasks)


def test_conservative_policy_stops_at_safe_boundary():
    rows = [_row("s0", "task0", 0, False), _row("s0", "task0", 32, True)]
    handback = np.asarray([0.1, 0.95], np.float32)
    persistent = np.asarray([0.95, 0.95], np.float32)
    zero = np.zeros(2, np.float32)
    result = _state_policy(
        rows, handback, persistent, zero, zero,
        threshold=0.9, z=1.64, cost_credit=0.2,
    )
    assert result["success_rate"] == 1.0
    assert result["decisions"][0]["boundary"] == 32
    assert result["oft_step_savings_fraction"] == 0.6
