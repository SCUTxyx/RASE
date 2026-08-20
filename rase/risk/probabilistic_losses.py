"""Losses for repeated Bernoulli handback outcomes."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def binomial_nll_from_logits(
    logits: Tensor,
    successes: Tensor,
    trials: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Binomial negative log likelihood, including the combinatorial term."""
    logits, successes, trials = torch.broadcast_tensors(logits, successes, trials)
    if torch.any(trials <= 0):
        raise ValueError("trials must be positive")
    if torch.any(successes < 0) or torch.any(successes > trials):
        raise ValueError("successes must lie in [0, trials]")
    log_combination = (
        torch.lgamma(trials + 1.0)
        - torch.lgamma(successes + 1.0)
        - torch.lgamma(trials - successes + 1.0)
    )
    log_likelihood = (
        log_combination
        - successes * F.softplus(-logits)
        - (trials - successes) * F.softplus(logits)
    )
    loss = -log_likelihood
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def beta_binomial_nll(
    alpha_raw: Tensor,
    beta_raw: Tensor,
    successes: Tensor,
    trials: Tensor,
    *,
    minimum_concentration: float = 1e-4,
    reduction: str = "mean",
) -> Tensor:
    """Beta-binomial NLL for overdispersed repeated handback outcomes."""
    alpha = F.softplus(alpha_raw) + minimum_concentration
    beta = F.softplus(beta_raw) + minimum_concentration
    alpha, beta, successes, trials = torch.broadcast_tensors(
        alpha, beta, successes, trials
    )
    if torch.any(trials <= 0):
        raise ValueError("trials must be positive")
    if torch.any(successes < 0) or torch.any(successes > trials):
        raise ValueError("successes must lie in [0, trials]")
    failures = trials - successes
    log_combination = (
        torch.lgamma(trials + 1.0)
        - torch.lgamma(successes + 1.0)
        - torch.lgamma(failures + 1.0)
    )
    log_beta_ratio = (
        torch.lgamma(successes + alpha)
        + torch.lgamma(failures + beta)
        - torch.lgamma(trials + alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + torch.lgamma(alpha + beta)
    )
    loss = -(log_combination + log_beta_ratio)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def beta_mean(alpha_raw: Tensor, beta_raw: Tensor, *, minimum_concentration: float = 1e-4) -> Tensor:
    alpha = F.softplus(alpha_raw) + minimum_concentration
    beta = F.softplus(beta_raw) + minimum_concentration
    return alpha / (alpha + beta)


def quantile_pinball_loss(
    predictions: Tensor,
    target: Tensor,
    quantiles: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Pinball loss for ordered remaining-cost quantiles."""
    while target.ndim < predictions.ndim:
        target = target.unsqueeze(-1)
    error = target - predictions
    loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unsupported reduction: {reduction}")
