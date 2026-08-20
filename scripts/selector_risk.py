#!/usr/bin/env python3
"""Stage D: uncertainty-aware risk-driven selector (no lookup, no identity).

Decision rule (per roadmap §8):

  For each candidate i: mu_i = E[V_i], sigma_i = uncertainty
    LCB_i = mu_i - beta * sigma_i
    UCB_i = mu_i + beta * sigma_i

  switch from `current` to `new` only when:
    LCB_new > UCB_current + delta
  otherwise abstain -> continue.

  Hysteresis: after a switch, stay at least `dwell` decision points unless
  risk of current exceeds `emergency` threshold.

  Optional fallback arm and abort arm complete the behaviour space:
    continue / switch / fallback / abort.

Pure numpy — unit-testable locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np


@dataclass
class SelectorConfig:
    beta: float = 1.0           # uncertainty scaling (LCB/UCB width)
    delta: float = 0.05         # required LCB_new > UCB_cur + delta
    dwell: int = 2              # min decision points after a switch
    emergency: float = 0.35     # if current mean risk drops below -> forced action
    fallback_arm: Optional[str] = None
    abort_threshold: float = 0.20  # if all arms below -> abort
    emergency_action: str = "fallback"  # what to do in emergency


@dataclass
class SelectorState:
    current: str = ""
    since_switch: int = 0
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"current": self.current, "since_switch": self.since_switch,
                "last_decisions": self.history[-20:]}


def decide(
    mu: Mapping[str, float],
    sigma: Mapping[str, float],
    cfg: SelectorConfig,
    state: SelectorState,
) -> tuple[str, dict]:
    """Return (chosen_arm, info).  mu/sigma: predicted success mean + std per
    candidate arm.  state carries the current arm and dwell counter."""
    if state.current and state.current not in mu:
        state.current = ""  # reset if arm disappeared
    if not mu:
        raise ValueError("decide requires at least one arm")

    arms = list(mu)
    mu_a = np.array([mu[a] for a in arms], dtype=np.float64)
    sigma_a = np.array([sigma.get(a, 0.05) for a in arms], dtype=np.float64)
    lcb = mu_a - cfg.beta * sigma_a
    ucb = mu_a + cfg.beta * sigma_a

    # emergency: current arm is collapsing -> fallback/abort
    emergency = False
    if state.current:
        idx_cur = arms.index(state.current)
        if mu_a[idx_cur] < cfg.emergency:
            emergency = True
            if cfg.emergency_action == "abort":
                chosen = "abort"
            elif cfg.fallback_arm is not None:
                chosen = cfg.fallback_arm
            else:
                chosen = state.current
            info = _info(arms, mu_a, sigma_a, lcb, ucb, chosen,
                         "emergency", emergency=True)
            state.history.append(info)
            state.since_switch = 0 if chosen != state.current else state.since_switch + 1
            if chosen != state.current:
                state.current = chosen
            return chosen, info

    # dwell: don't switch too soon after a switch (cold start excluded)
    if state.current and state.history and state.since_switch < cfg.dwell:
        chosen = state.current
        info = _info(arms, mu_a, sigma_a, lcb, ucb, chosen, "dwell")
        state.history.append(info)
        state.since_switch += 1
        return chosen, info

    # relative-advantage switch
    if state.current:
        idx_cur = arms.index(state.current)
        # best non-current by LCB margin over current's UCB
        best_new = None
        best_margin = -np.inf
        for i, a in enumerate(arms):
            if a == state.current:
                continue
            margin = lcb[i] - ucb[idx_cur]
            if margin > best_margin:
                best_margin = float(margin)
                best_new = a
        if best_new is not None and best_margin > cfg.delta:
            chosen = best_new
            mode = "switch"
        else:
            chosen = state.current
            mode = "abstain"
    else:
        # cold start: pick best mean
        chosen = arms[int(np.argmax(mu_a))]
        mode = "cold_start"

    # global abort if everything looks hopeless
    if cfg.abort_threshold is not None and mu_a.max() < cfg.abort_threshold:
        chosen = "abort"
        mode = "abort_all"

    info = _info(arms, mu_a, sigma_a, lcb, ucb, chosen, mode)
    state.history.append(info)
    state.since_switch = 0 if chosen != state.current else state.since_switch + 1
    state.current = chosen
    return chosen, info


def _info(arms, mu, sigma, lcb, ucb, chosen, mode, emergency=False) -> dict:
    return {
        "arms": arms,
        "mu": {a: round(float(m), 4) for a, m in zip(arms, mu)},
        "sigma": {a: round(float(s), 4) for a, s in zip(arms, sigma)},
        "lcb": {a: round(float(l), 4) for a, l in zip(arms, lcb)},
        "ucb": {a: round(float(u), 4) for a, u in zip(arms, ucb)},
        "chosen": chosen, "mode": mode, "emergency": emergency,
    }


# ---------------------------------------------------------------------------
# uncertainty estimation for a ridge risk model (simple plugin)
# ---------------------------------------------------------------------------


def ridge_uncertainty(
    X: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    weights: np.ndarray, intercept: float,
    sigma_noise: float = 0.15,
) -> np.ndarray:
    """First-order uncertainty: sigma ≈ sigma_noise * ||dlogit/dX||.

    For a standardized ridge with logistic link, the gradient magnitude w.r.t.
    the standardized features is ||weights||_2 * p(1-p); we scale by the
    feature scale to return in probability units.
    """
    Xs = (np.asarray(X, dtype=np.float64) - mean) / scale
    logit = intercept + Xs @ weights
    p = 1.0 / (1.0 + np.exp(-logit))
    grad_std = np.abs(weights) / np.abs(scale + 1e-9)  # (D,)
    pp = (p * (1 - p))[:, None] if p.ndim == 1 else p * (1 - p)
    local = np.sqrt(np.sum((grad_std[None, :] * pp) ** 2, axis=-1))
    return sigma_noise * local + 0.02  # floor


if __name__ == "__main__":
    # local self-test
    cfg = SelectorConfig(beta=1.0, delta=0.05, dwell=2)
    st = SelectorState(current="A")
    # A clearly better
    mu = {"A": 0.9, "B": 0.4}
    sigma = {"A": 0.05, "B": 0.05}
    chosen, info = decide(mu, sigma, cfg, st)
    print("A-better ->", chosen, info["mode"])
    assert chosen == "A"
    # B clearly better after reset
    st2 = SelectorState(current="A")
    mu2 = {"A": 0.4, "B": 0.9}
    chosen2, info2 = decide(mu2, sigma, cfg, st2)
    print("B-better ->", chosen2, info2["mode"])
    assert chosen2 == "B"
    # marginal -> abstain
    st3 = SelectorState(current="A")
    mu3 = {"A": 0.62, "B": 0.60}
    chosen3, info3 = decide(mu3, sigma, cfg, st3)
    print("marginal ->", chosen3, info3["mode"])
    assert chosen3 == "A" and info3["mode"] == "abstain"
    # dwell: switch then stay
    st4 = SelectorState(current="B", since_switch=0)
    _, i1 = decide(mu2, sigma, cfg, st4)
    _, i2 = decide(mu2, sigma, cfg, st4)
    print("dwell:", i1["mode"], i2["mode"])
    assert i2["mode"] == "dwell"
    # emergency
    cfg_e = SelectorConfig(emergency=0.5, emergency_action="fallback",
                           fallback_arm="F")
    st5 = SelectorState(current="A")
    mu5 = {"A": 0.3, "B": 0.6, "F": 0.5}
    c5, i5 = decide(mu5, sigma, cfg_e, st5)
    print("emergency ->", c5, i5["mode"])
    assert c5 == "F"
    print("SELECTOR LOCAL TESTS PASSED")
