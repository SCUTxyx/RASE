from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from rase.risk.light_risk_student import RecoverabilityHazardStudent


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_r8b_local_hazard_probe", ROOT / "scripts" / "train_r8b_local_hazard_probe.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DummyEncoder(torch.nn.Module):
    output_dim = 12

    def forward(self, image: torch.Tensor, proprio: torch.Tensor,
                text_embed: torch.Tensor | None = None) -> torch.Tensor:
        del image, text_embed
        return torch.cat([proprio, torch.zeros(len(proprio), 4, device=proprio.device)], dim=-1)


def test_recoverability_hazard_model_shapes() -> None:
    model = RecoverabilityHazardStudent(DummyEncoder(), n_policies=2)
    output = model(
        torch.zeros(3, 2, 3, 8, 8), torch.zeros(3, 8), torch.zeros(3, 20),
        torch.zeros(3, 28), torch.zeros(3, 2), torch.zeros(3, 256),
        policy_index=torch.tensor([0, 1, 0]),
    )
    assert output["current_recoverable_logit"].shape == (3,)
    assert output["next_recoverable_logit"].shape == (3,)
    assert output["loss_hazard_logit"].shape == (3,)


def test_build_transitions_excludes_ambiguous_counts() -> None:
    rows = 6
    source = {
        "group_id": np.asarray(["a"] * 3 + ["b"] * 3),
        "elapsed_source_steps": np.asarray([0, 8, 16, 0, 8, 16]),
        "persistent_trials": np.asarray([2, 2, 2, 2, 2, 2]),
        "persistent_successes": np.asarray([2, 0, 2, 2, 1, 0]),
    }
    for key, value in {
        "image": np.zeros((rows, 2, 3, 2, 2), np.uint8),
        "proprio": np.zeros((rows, 8), np.float32),
        "action_summary": np.zeros((rows, 20), np.float32),
        "history": np.zeros((rows, 28), np.float32),
        "language_hash": np.zeros((rows, 256), np.float32),
        "state_key": np.asarray([f"s{i}" for i in range(rows)]),
        "task_id": np.asarray(["t"] * rows),
        "suite": np.asarray(["Spatial"] * rows),
        "cohort_role": np.asarray(["natural"] * rows),
        "policy_id": np.asarray(["pi05_libero"] * rows),
        "policy_index": np.zeros(rows, np.int64),
    }.items():
        source[key] = value
    result = MODULE.build_transitions(source)
    assert result["group_id"].tolist() == ["a", "a"]
    assert result["loss_hazard"].tolist() == [1.0, 0.0]
    assert result["elapsed_source_steps"].tolist() == [0, 8]
