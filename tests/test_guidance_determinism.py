import numpy as np
import pytest

from rase.guidance.flow_guidance import apply_guidance_update, iterative_guidance
from rase.guidance.privileged_recovery_score import (
    RecoveryScoreWeights,
    TransitionSignals,
    score_transition,
    select_best_of_k,
)


def test_structured_score_retains_weighted_components():
    score = score_transition(
        TransitionSignals(
            progress_delta=2.0,
            grasp_stability=0.5,
            collision_harm=0.25,
            irreversible=True,
        ),
        RecoveryScoreWeights(
            progress=3.0,
            grasp_stability=4.0,
            collision_harm=8.0,
            irreversible_penalty=20.0,
        ),
    )
    assert score.progress == 6.0
    assert score.grasp_stability == 2.0
    assert score.collision_harm == -2.0
    assert score.irreversible_penalty == -20.0
    assert score.total == -14.0
    assert score.irreversible


def test_best_of_k_scores_all_candidates_and_uses_stable_tie_break():
    candidates = [
        TransitionSignals(1.0, 0.0),
        TransitionSignals(1.0, 0.0),
        TransitionSignals(0.0, 0.0),
        TransitionSignals(1.0, 0.0),
    ]
    first = select_best_of_k(candidates, k=4, candidate_ids=["z", "a", "q", "b"])
    second = select_best_of_k(candidates, k=4, candidate_ids=["z", "a", "q", "b"])
    assert first == second
    assert first.index == 1
    assert first.candidate_id == "a"
    assert first.evaluated_count == 4
    assert len(first.scores) == 4


def test_best_of_k_enforces_matched_compute_and_valid_signals():
    transitions = [TransitionSignals(0.0, 0.0)] * 3
    with pytest.raises(ValueError, match="exactly k=4"):
        select_best_of_k(transitions, k=4)
    with pytest.raises(ValueError, match="finite"):
        score_transition(TransitionSignals(np.nan, 0.0))
    with pytest.raises(ValueError, match="non-negative"):
        score_transition(TransitionSignals(0.0, 0.0, collision_harm=-1.0))


def test_guidance_update_is_bitwise_deterministic():
    base = np.arange(21, dtype=np.float64).reshape(3, 7) / 100.0
    direction = np.linspace(-1.0, 1.0, 21).reshape(3, 7)
    kwargs = {
        "step_size": 0.3,
        "action_low": np.full(7, -0.5),
        "action_high": np.full(7, 0.5),
        "trust_region_radius": 0.2,
        "max_guidance_norm": 0.4,
    }
    first = apply_guidance_update(base, direction, **kwargs)
    second = apply_guidance_update(base, direction, **kwargs)
    np.testing.assert_array_equal(first.actions, second.actions)
    assert first == second


def test_iterative_guidance_is_deterministic_and_callback_is_generic():
    def direction(actions, step):
        return np.cos(actions + step)

    kwargs = {
        "num_steps": 4,
        "step_size": 0.1,
        "action_low": -0.5,
        "action_high": 0.5,
        "trust_region_radius": 0.25,
        "max_guidance_norm": 0.3,
    }
    first = iterative_guidance(np.zeros((2, 7)), direction, **kwargs)
    second = iterative_guidance(np.zeros((2, 7)), direction, **kwargs)
    np.testing.assert_array_equal(first.actions, second.actions)
    assert first.reason is None


def test_callback_nan_falls_back_to_original_bounded_actions():
    base = np.full((2, 7), 0.2)

    def bad_direction(actions, step):
        del actions, step
        return np.full((2, 7), np.nan)

    result = iterative_guidance(
        base,
        bad_direction,
        num_steps=2,
        step_size=0.1,
        action_low=-0.1,
        action_high=0.1,
        trust_region_radius=0.2,
        max_guidance_norm=1.0,
    )
    assert result.used_fallback
    np.testing.assert_array_equal(result.actions, np.full((2, 7), 0.1))
