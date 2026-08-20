from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rase.risk.probabilistic_handback_student import ProbabilisticHandbackStudent
from scripts.train_r5_probabilistic_handback_oof import (
    controller_records,
    sha256,
    task_folds,
    validate_protocol_summary,
)
from scripts.collect_r4_boundary_transitions_v3 import repeat_seeds_for_boundary
from scripts.summarize_r5_probabilistic_boundaries import missing_boundary_is_violation
from scripts.analyze_r5_b24_decision import minimum_all_success_trials_for_wilson_lcb


class IdentityEncoder(nn.Module):
    output_dim = 4

    def forward(self, image: torch.Tensor, proprio: torch.Tensor, text_embed=None) -> torch.Tensor:
        del proprio, text_embed
        return image


def boundary_row(step: int, probability: float = 1.0) -> dict:
    return {
        "state_key": "s0",
        "task_id": "t0",
        "elapsed_oft_steps": step,
        "success_if_continue_oft": True,
        "persistent_executed_oft_steps": 100,
        "handback_success_probability": probability,
        "success_if_handback_now": probability == 1.0,
    }


def test_two_boundary_dwell_rejects_isolated_crossing() -> None:
    rows = [boundary_row(step) for step in (0, 16, 64)]
    records = controller_records(rows, np.asarray([0.9, 0.1, 0.9]), threshold=0.8, dwell=2)
    assert records[0]["handback"] is False


def test_two_boundary_dwell_hands_back_on_second_crossing() -> None:
    rows = [boundary_row(step) for step in (0, 16, 64)]
    records = controller_records(rows, np.asarray([0.9, 0.85, 0.1]), threshold=0.8, dwell=2)
    assert records[0]["handback"] is True
    assert records[0]["elapsed_oft_steps"] == 16


def test_task_folds_are_disjoint_and_cover_tasks() -> None:
    rows = [{"task_id": f"t{task}", "row": row} for task in range(8) for row in range(2)]
    folds = task_folds(rows, folds=4, seed=17)
    validation = [task for fold in folds for task in fold["validation_tasks"]]
    assert sorted(validation) == [f"t{task}" for task in range(8)]
    for fold in folds:
        assert {row["task_id"] for row in fold["train"]}.isdisjoint(fold["validation_tasks"])


def test_cost_quantiles_are_nonnegative_and_ordered() -> None:
    model = ProbabilisticHandbackStudent(
        IdentityEncoder(), action_dim=20, history_dim=8, fused_dim=16,
        head_hidden=16, n_cost_quantiles=3, dropout=0.0,
    )
    output = model(
        torch.randn(3, 4), torch.randn(3, 8),
        torch.randn(3, 20), torch.randn(3, 20), torch.randn(3, 4, 6),
    )["remaining_cost_quantiles"]
    assert torch.all(output >= 0.0)
    assert torch.all(output[:, 1:] >= output[:, :-1])


def test_paired_repeat_seeds_are_common_across_boundaries() -> None:
    def fake_seed(state: str, elapsed: int, repeat: int, *, salt: int) -> int:
        return len(state) * 1_000_000 + elapsed * 10_000 + repeat * 100 + salt % 100

    left = repeat_seeds_for_boundary(
        state_key="state", elapsed=0, repeats=5, shared_seed=7,
        paired_across_boundaries=True, rollout_seed_fn=fake_seed,
    )
    right = repeat_seeds_for_boundary(
        state_key="state", elapsed=128, repeats=5, shared_seed=7,
        paired_across_boundaries=True, rollout_seed_fn=fake_seed,
    )
    legacy = repeat_seeds_for_boundary(
        state_key="state", elapsed=128, repeats=5, shared_seed=7,
        paired_across_boundaries=False, rollout_seed_fn=fake_seed,
    )
    assert left == right
    assert legacy != left
    assert len(set(left)) == 5


def test_training_interlock_requires_ready_opportunity_gate(tmp_path) -> None:
    dataset = tmp_path / "boundaries.jsonl"
    dataset.write_text("{}\n")
    protocol = {
        "protocol_gate_status": "ready",
        "source_dataset": str(dataset.resolve()),
        "source_dataset_sha256": sha256(dataset),
        "probability_opportunity_gate_status": "fail",
        "probability_opportunity_gate_reasons": ["finite states below gate"],
    }
    try:
        validate_protocol_summary(protocol, dataset=dataset, require_opportunity_ready=True)
    except ValueError as error:
        assert "opportunity gate is closed" in str(error)
    else:
        raise AssertionError("closed opportunity gate did not block training")

    protocol["probability_opportunity_gate_status"] = "ready"
    validate_protocol_summary(protocol, dataset=dataset, require_opportunity_ready=True)


def test_boundary_at_terminal_step_is_not_a_reachable_decision() -> None:
    assert missing_boundary_is_violation(96, 128)
    assert not missing_boundary_is_violation(128, 128)
    assert not missing_boundary_is_violation(128, 96)
    assert missing_boundary_is_violation(128, -1)


def test_zero_failure_repeat_requirements_match_wilson_formula() -> None:
    z = 1.6448536269514722
    assert minimum_all_success_trials_for_wilson_lcb(0.8, z) == 11
    assert minimum_all_success_trials_for_wilson_lcb(0.9, z) == 25
    assert minimum_all_success_trials_for_wilson_lcb(0.95, z) == 52
