import numpy as np
import pytest

from rase.guidance.flow_guidance import (
    apply_guidance_update,
    iterative_guidance,
    project_to_trust_region,
)


def test_projection_respects_global_trust_region():
    reference = np.zeros((3, 7), dtype=np.float64)
    proposed = np.full((3, 7), 10.0)
    projected = project_to_trust_region(proposed, reference, radius=0.25)
    assert np.linalg.norm(projected - reference) == pytest.approx(0.25)
    np.testing.assert_allclose(
        projected / np.linalg.norm(projected),
        proposed / np.linalg.norm(proposed),
    )


def test_guidance_clips_norm_trust_region_and_per_dimension_bounds():
    base = np.zeros((2, 7), dtype=np.float64)
    low = np.array([-0.04, -1, -1, -1, -1, -1, -0.2])
    high = np.array([0.04, 1, 1, 1, 1, 1, 0.2])
    result = apply_guidance_update(
        base,
        np.full_like(base, 100.0),
        step_size=2.0,
        action_low=low,
        action_high=high,
        trust_region_radius=0.3,
        max_guidance_norm=0.2,
    )
    assert not result.used_fallback
    assert result.applied_guidance_norm == pytest.approx(0.2)
    assert result.update_norm <= 0.3 + 1e-12
    assert np.all(result.actions >= low)
    assert np.all(result.actions <= high)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_guidance_returns_bounded_fallback(bad):
    base = np.full((2, 7), 0.5)
    fallback = np.full((2, 7), -0.5)
    guidance = np.zeros((2, 7))
    guidance[0, 0] = bad
    result = apply_guidance_update(
        base,
        guidance,
        step_size=1.0,
        action_low=-0.1,
        action_high=0.1,
        trust_region_radius=0.5,
        max_guidance_norm=1.0,
        fallback_actions=fallback,
    )
    assert result.used_fallback
    assert result.reason == "non_finite_guidance"
    np.testing.assert_array_equal(result.actions, np.full((2, 7), -0.1))
    assert np.all(np.isfinite(result.actions))


def test_non_finite_base_can_use_explicit_finite_fallback():
    base = np.zeros((1, 7))
    base[0, 2] = np.nan
    result = apply_guidance_update(
        base,
        np.ones((1, 7)),
        step_size=1.0,
        action_low=np.full(7, -1.0),
        action_high=np.full(7, 1.0),
        trust_region_radius=0.2,
        max_guidance_norm=0.2,
        fallback_actions=np.zeros((1, 7)),
    )
    assert result.used_fallback
    assert result.reason == "non_finite_base"
    np.testing.assert_array_equal(result.actions, np.zeros((1, 7)))


def test_iterative_updates_remain_near_original_not_previous_step():
    base = np.zeros((2, 7))
    result = iterative_guidance(
        base,
        lambda actions, step: np.ones_like(actions),
        num_steps=20,
        step_size=1.0,
        action_low=-1.0,
        action_high=1.0,
        trust_region_radius=0.15,
        max_guidance_norm=1.0,
    )
    assert not result.used_fallback
    assert np.linalg.norm(result.actions - base) <= 0.15 + 1e-12


def test_rejects_wrong_action_shape_and_invalid_safety_configuration():
    with pytest.raises(ValueError, match=r"\[T,7\]"):
        project_to_trust_region(np.zeros((2, 6)), np.zeros((2, 6)), 1.0)
    with pytest.raises(ValueError, match="lower bound"):
        apply_guidance_update(
            np.zeros((1, 7)),
            np.zeros((1, 7)),
            step_size=1.0,
            action_low=np.ones(7),
            action_high=np.zeros(7),
            trust_region_radius=1.0,
            max_guidance_norm=1.0,
        )
