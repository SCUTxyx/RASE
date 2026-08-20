from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rase.risk.light_risk_student import SourceRiskStudent


class DummyEncoder(nn.Module):
    output_dim = 12

    def forward(self, image, proprio, text_embed=None):
        del image, text_embed
        return torch.cat([proprio, torch.zeros(proprio.shape[0], 4)], dim=-1)


class ImageShapeEncoder(nn.Module):
    output_dim = 12

    def forward(self, image, proprio, text_embed=None):
        del text_embed
        assert image.dim() == 4
        return torch.cat([proprio, torch.zeros(proprio.shape[0], 4)], dim=-1)


def test_source_risk_student_is_single_target_and_additive() -> None:
    model = SourceRiskStudent(
        DummyEncoder(), action_dim=20, n_members=3,
        native_feature_dim=5, wm_feature_dim=7,
    )
    out = model(
        torch.zeros(4, 2, 3, 96, 96), torch.zeros(4, 8), torch.zeros(4, 20),
        native_features=torch.zeros(4, 5), wm_features=torch.zeros(4, 7),
    )
    assert out["source_failure"].shape == (3, 4)
    assert out["source_failure_logit"].shape == (3, 4)
    assert out["risk_embedding"].shape[:2] == (3, 4)
    assert set(out) == {"source_failure", "source_failure_logit", "risk_embedding"}


def test_source_risk_student_rejects_missing_optional_stream() -> None:
    model = SourceRiskStudent(DummyEncoder(), wm_feature_dim=7)
    with pytest.raises(ValueError, match="wm_features"):
        model(torch.zeros(2, 2, 3, 96, 96), torch.zeros(2, 8), torch.zeros(2, 20))


def test_source_risk_student_encodes_two_camera_views_separately() -> None:
    model = SourceRiskStudent(ImageShapeEncoder(), n_members=2)
    out = model(
        torch.zeros(4, 2, 3, 96, 96),
        torch.zeros(4, 8),
        torch.zeros(4, 20),
    )
    assert out["source_failure"].shape == (2, 4)


def test_source_risk_student_seen_policy_embedding_is_optional() -> None:
    model = SourceRiskStudent(ImageShapeEncoder(), n_policies=3)
    out = model(
        torch.zeros(4, 2, 3, 96, 96), torch.zeros(4, 8), torch.zeros(4, 20),
        policy_index=torch.tensor([0, 1, 2, 1]),
    )
    assert out["source_failure"].shape == (1, 4)
    with pytest.raises(ValueError, match="policy_index"):
        model(torch.zeros(4, 2, 3, 96, 96), torch.zeros(4, 8), torch.zeros(4, 20))
