"""Wilson intervals, two-stage rollout sampling, and NGC triage."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

Z_95 = 1.959963984540054
PROTOCOL_LEGACY_TWOSIDED = "wilson-twosided-v0"
PROTOCOL_SEQUENTIAL_ONESIDED_V1 = "wilson-onesided-alpha-spend-v1"


@dataclass(frozen=True)
class WilsonEstimate:
    successes: int
    trials: int
    rate: float
    lower: float
    upper: float
    protocol_version: str = PROTOCOL_LEGACY_TWOSIDED
    sidedness: str = "two-sided"
    alpha: float | None = None
    stopped_early: bool = False


class SetLabel(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    UNCERTAIN = "uncertain"


def z_from_alpha(alpha: float, *, sidedness: str = "two-sided") -> float:
    """Normal critical value for a Wilson bound at level ``alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if sidedness not in {"one-sided", "two-sided"}:
        raise ValueError("sidedness must be 'one-sided' or 'two-sided'")
    # Inverse erf approximation for Phi^{-1}(p); accurate enough for CI z.
    target = 1.0 - alpha if sidedness == "one-sided" else 1.0 - alpha / 2.0
    return _norm_ppf(target)


def _norm_ppf(p: float) -> float:
    """Approximate standard normal quantile (Acklam's algorithm)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def wilson_interval(
    successes: int, trials: int, *, z: float = Z_95
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval (95% by default)."""
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise TypeError("successes and trials must be integers")
    if int(successes) != successes or int(trials) != trials:
        raise TypeError("successes and trials must be integers")
    successes, trials = int(successes), int(trials)
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    if successes == 0:
        lower_endpoint = 0.0
    else:
        lower_endpoint = None
    if successes == trials:
        upper_endpoint = 1.0
    else:
        upper_endpoint = None
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials * trials)
        )
        / denominator
    )
    lower = lower_endpoint if lower_endpoint is not None else max(0.0, center - radius)
    upper = upper_endpoint if upper_endpoint is not None else min(1.0, center + radius)
    return lower, upper


def estimate(
    successes: int,
    trials: int,
    *,
    z: float = Z_95,
    protocol_version: str = PROTOCOL_LEGACY_TWOSIDED,
    sidedness: str = "two-sided",
    alpha: float | None = None,
    stopped_early: bool = False,
) -> WilsonEstimate:
    lower, upper = wilson_interval(successes, trials, z=z)
    return WilsonEstimate(
        successes,
        trials,
        successes / trials,
        lower,
        upper,
        protocol_version=protocol_version,
        sidedness=sidedness,
        alpha=alpha,
        stopped_early=stopped_early,
    )


def adaptive_sample(
    rollout: Callable[[int], bool],
    *,
    threshold: float = 0.5,
    n_first: int = 3,
    n_total: int = 10,
) -> WilsonEstimate:
    """Legacy two-sided 3→10 estimator (kept for protocol fidelity tests).

    The callback receives the zero-based rollout index, enabling deterministic
    per-rollout seeding and idempotent scheduler keys.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    if n_first <= 0 or n_total < n_first:
        raise ValueError("require 0 < n_first <= n_total")
    successes = sum(bool(rollout(index)) for index in range(n_first))
    current = estimate(successes, n_first)
    if current.upper < threshold or current.lower > threshold:
        return current
    for index in range(n_first, n_total):
        successes += bool(rollout(index))
    return estimate(successes, n_total)


def sequential_adaptive_sample(
    rollout: Callable[[int], bool],
    *,
    threshold: float = 0.5,
    n_first: int = 6,
    n_total: int = 20,
    alpha_first: float = 0.01,
    alpha_final: float = 0.04,
    sidedness: str = "one-sided",
    protocol_version: str = PROTOCOL_SEQUENTIAL_ONESIDED_V1,
) -> WilsonEstimate:
    """Two-look alpha-spending Wilson sampler for formal NGC labeling.

    Stage-1 uses ``alpha_first``; stage-2 uses ``alpha_final``. For one-sided
    Set-C-oriented decisions the relevant bound is the upper Wilson endpoint
    (bad) or lower endpoint (good). Early stop only when the interval excludes
    ``threshold`` at the stage-specific alpha.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    if n_first <= 0 or n_total < n_first:
        raise ValueError("require 0 < n_first <= n_total")
    if alpha_first <= 0 or alpha_final <= 0 or alpha_first + alpha_final > 0.05 + 1e-12:
        raise ValueError("require alpha_first>0, alpha_final>0, sum <= 0.05")

    z1 = z_from_alpha(alpha_first, sidedness=sidedness)
    successes = sum(bool(rollout(index)) for index in range(n_first))
    stage1 = estimate(
        successes,
        n_first,
        z=z1,
        protocol_version=protocol_version,
        sidedness=sidedness,
        alpha=alpha_first,
        stopped_early=True,
    )
    if stage1.upper < threshold or stage1.lower > threshold:
        return stage1

    for index in range(n_first, n_total):
        successes += bool(rollout(index))
    z2 = z_from_alpha(alpha_final, sidedness=sidedness)
    return estimate(
        successes,
        n_total,
        z=z2,
        protocol_version=protocol_version,
        sidedness=sidedness,
        alpha=alpha_final,
        stopped_early=False,
    )


def triage(
    candidates: Sequence[WilsonEstimate],
    *,
    threshold: float = 0.5,
    set_a_min_good: int = 3,
) -> SetLabel:
    """Apply the exact, conservative Set A/B/C/uncertain rules.

    C: every candidate's upper bound is below threshold.
    A: at least three candidates' lower bounds are above threshold.
    B: one or two candidates' lower bounds are above threshold.
    uncertain: no confidently good candidate, but C is not established.
    """
    if not candidates:
        raise ValueError("at least one candidate estimate is required")
    if set_a_min_good < 1:
        raise ValueError("set_a_min_good must be positive")
    if all(candidate.upper < threshold for candidate in candidates):
        return SetLabel.C
    confidently_good = sum(candidate.lower > threshold for candidate in candidates)
    if confidently_good >= set_a_min_good:
        return SetLabel.A
    if confidently_good > 0:
        return SetLabel.B
    return SetLabel.UNCERTAIN


def estimates_from_outcomes(outcomes: Iterable[Sequence[bool]]) -> list[WilsonEstimate]:
    result = []
    for candidate in outcomes:
        values = tuple(bool(value) for value in candidate)
        if not values:
            raise ValueError("each candidate needs at least one rollout")
        result.append(estimate(sum(values), len(values)))
    return result


def adaptive_r_hat(
    rollout: Callable[[int], bool],
    *,
    tau: float = 0.5,
    n1: int = 3,
    n2: int = 10,
) -> WilsonEstimate:
    """Guide-compatible spelling for the two-stage estimator."""
    return adaptive_sample(rollout, threshold=tau, n_first=n1, n_total=n2)


classify_state = triage
