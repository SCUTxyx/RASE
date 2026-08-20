from __future__ import annotations

import torch

from rase.risk.probabilistic_losses import (
    beta_binomial_nll,
    beta_mean,
    binomial_nll_from_logits,
    quantile_pinball_loss,
)


def test_binomial_prefers_matching_probability() -> None:
    successes = torch.tensor([3.0])
    trials = torch.tensor([5.0])
    matching = torch.logit(torch.tensor([0.6]))
    wrong = torch.logit(torch.tensor([0.1]))
    assert binomial_nll_from_logits(matching, successes, trials) < binomial_nll_from_logits(wrong, successes, trials)


def test_beta_binomial_is_finite_and_differentiable() -> None:
    alpha_raw = torch.tensor([0.2], requires_grad=True)
    beta_raw = torch.tensor([-0.1], requires_grad=True)
    loss = beta_binomial_nll(alpha_raw, beta_raw, torch.tensor([3.0]), torch.tensor([5.0]))
    assert torch.isfinite(loss)
    loss.backward()
    assert alpha_raw.grad is not None and torch.isfinite(alpha_raw.grad).all()
    assert beta_raw.grad is not None and torch.isfinite(beta_raw.grad).all()


def test_beta_mean_is_probability() -> None:
    value = beta_mean(torch.tensor([-3.0, 0.0, 3.0]), torch.zeros(3))
    assert torch.all((value > 0.0) & (value < 1.0))


def test_quantile_pinball_prefers_exact_target() -> None:
    quantiles = torch.tensor([0.1, 0.5, 0.9])
    target = torch.tensor([2.0])
    exact = torch.tensor([[2.0, 2.0, 2.0]])
    wrong = torch.tensor([[0.0, 0.0, 0.0]])
    assert quantile_pinball_loss(exact, target, quantiles) < quantile_pinball_loss(wrong, target, quantiles)
