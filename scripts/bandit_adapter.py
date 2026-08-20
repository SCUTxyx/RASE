#!/usr/bin/env python3
"""Stage E: contextual bandit adapter over candidate arms, initialized by the
OPD risk prior and learning only a residual online.

Design (roadmap §11-§13):
  - arms: candidate policies (A, B, C, fallback)
  - context: x_t = [risk scores, uncertainty, state embedding, task embedding]
  - prior: Q_i^0 = f_OPD(x, a_i)  (zero-shot cold start, N_testVLA = 0)
  - online: Thompson sampling over Gaussian posteriors on arm means;
    posterior mean = prior + residual accumulated from observed rewards.
  - reporting: performance(k) and cumulative regret(k) for k = 1,3,5,10
    episodes of feedback, vs random / fixed / bandit-from-scratch / oracle.

Pure numpy — unit-testable locally with a simulated reward oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class BanditConfig:
    arms: Sequence[str]
    prior_mean: Callable[[str], float] | None = None  # OPD prior per arm
    prior_weight: float = 5.0   # pseudo-count of the prior (how many fake obs)
    alpha0: float = 1.0         # beta prior a0 for Thompson (Bernoulli)
    beta0: float = 1.0
    thompson: bool = True       # else UCB
    uc_beta: float = 1.0        # UCB exploration constant
    seed: int = 7


@dataclass
class BanditState:
    counts: dict[str, int] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)
    prior: dict[str, float] = field(default_factory=dict)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(7))
    decisions: list[str] = field(default_factory=list)
    regrets: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "counts": self.counts,
            "mean_rewards": {k: self.rewards.get(k, 0.0) / max(1, self.counts.get(k, 0))
                             for k in self.counts},
            "n_decisions": len(self.decisions),
            "cumulative_regret": float(sum(self.regrets)),
        }


class OPDPlusBandit:
    """Q_i^online = Q_i^OPD + Delta_i^bandit via Thompson/UCB on residuals."""

    def __init__(self, cfg: BanditConfig) -> None:
        self.cfg = cfg
        self.state = BanditState()
        self.state.rng = np.random.default_rng(cfg.seed)
        for a in cfg.arms:
            self.state.counts[a] = 0
            self.state.rewards[a] = 0.0
            if cfg.prior_mean is not None:
                self.state.prior[a] = float(cfg.prior_mean(a))

    # -- posterior quantities ------------------------------------------------
    def _posterior(self, arm: str) -> tuple[float, float]:
        n = self.state.counts[arm]
        s = self.state.rewards[arm]
        prior_mu = self.state.prior.get(arm, 0.5)
        w = self.cfg.prior_weight
        # posterior mean of Bernoulli rate with conjugate beta prior:
        # beta(a0 + s, b0 + n - s) -> mean; effective count n + w.
        a = self.cfg.alpha0 + s
        b = self.cfg.beta0 + n - s
        # blend with OPD prior via pseudo-count weight
        mean = (a / (a + b)) if (n + self.cfg.alpha0 + self.cfg.beta0) > 0 else 0.5
        mean = (w * prior_mu + (n + self.cfg.alpha0 + self.cfg.beta0) * mean) / (
            w + n + self.cfg.alpha0 + self.cfg.beta0) if n > 0 else prior_mu
        # uncertainty shrinks with evidence
        sigma = np.sqrt(
            (mean * (1 - mean)) / max(1.0, n + self.cfg.alpha0 + self.cfg.beta0 + w))
        return float(mean), float(max(sigma, 1e-3))

    def choose(self) -> str:
        if self.cfg.thompson:
            samples = {}
            for a in self.cfg.arms:
                mu, sigma = self._posterior(a)
                samples[a] = float(self.state.rng.normal(mu, sigma))
            chosen = max(samples, key=samples.get)
        else:  # UCB
            best, best_v = None, -np.inf
            for a in self.cfg.arms:
                mu, sigma = self._posterior(a)
                v = mu + self.cfg.uc_beta * sigma
                if v > best_v:
                    best_v, best = v, a
            chosen = best
        self.state.decisions.append(chosen)
        return chosen

    def update(self, arm: str, reward: float, oracle_best: float | None = None) -> None:
        self.state.counts[arm] += 1
        self.state.rewards[arm] += float(reward)
        if oracle_best is not None:
            self.state.regrets.append(max(0.0, oracle_best - float(reward)))

    def report(self) -> dict:
        return self.state.to_dict()


# ---------------------------------------------------------------------------
# offline simulation: OPD vs bandit-from-scratch vs fixed vs random vs oracle
# ---------------------------------------------------------------------------


def simulate_episodes(
    n_episodes: int,
    arms: Sequence[str],
    true_means: dict[str, float],
    opd_prior: dict[str, float] | None,
    cfg: BanditConfig,
    oracle_means: dict[str, float] | None = None,
    rng_seed: int = 0,
) -> dict:
    """Simulate repeated episodes where the environment samples an arm-context
    (here: static per-episode Bernoulli reward per arm)."""
    rng = np.random.default_rng(rng_seed)
    oracle = oracle_means or true_means
    best_true = max(true_means.values())

    results: dict[str, dict] = {}
    # 1) OPD-only (zero-shot: never adapt)
    if opd_prior is not None:
        chosen = max(arms, key=lambda a: opd_prior[a])
        succ = sum(int(rng.random() < true_means[chosen]) for _ in range(n_episodes))
        results["opd_only"] = {"success_rate": succ / n_episodes,
                               "chosen": chosen}
    # 2) fixed best (oracle-ish fixed)
    fixed = max(arms, key=lambda a: true_means[a])
    succ = sum(int(rng.random() < true_means[fixed]) for _ in range(n_episodes))
    results["best_fixed"] = {"success_rate": succ / n_episodes, "chosen": fixed}
    # 3) random
    succ = sum(int(rng.random() < true_means[arms[rng.integers(len(arms))]])
               for _ in range(n_episodes))
    results["random"] = {"success_rate": succ / n_episodes}
    # 4) oracle
    results["oracle"] = {"success_rate": best_true}
    # 5) OPD+bandit
    b = OPDPlusBandit(BanditConfig(arms=arms, prior_mean=(
        (lambda a: opd_prior[a]) if opd_prior else None),
        seed=rng_seed))
    succ = 0
    for _ in range(n_episodes):
        a = b.choose()
        r = int(rng.random() < true_means[a])
        b.update(a, r, oracle_best=oracle.get(a))
        succ += r
    results["opd_plus_bandit"] = {
        "success_rate": succ / n_episodes, **b.report()}
    # 6) bandit from scratch (uniform prior)
    b2 = OPDPlusBandit(BanditConfig(arms=arms, prior_mean=None, seed=rng_seed))
    succ = 0
    for _ in range(n_episodes):
        a = b2.choose()
        r = int(rng.random() < true_means[a])
        b2.update(a, r, oracle_best=oracle.get(a))
        succ += r
    results["bandit_from_scratch"] = {
        "success_rate": succ / n_episodes, **b2.report()}
    return results


if __name__ == "__main__":
    arms = ["A", "B", "F"]
    true = {"A": 0.5, "B": 0.95, "F": 0.7}
    prior = {"A": 0.5, "B": 0.8, "F": 0.6}  # imperfect OPD prior (B right)
    r = simulate_episodes(60, arms, true, prior,
                          BanditConfig(arms=arms, prior_mean=lambda a: prior[a],
                                       seed=1))
    for k, v in r.items():
        if k in ("opd_only", "best_fixed", "random", "oracle"):
            print(f"{k:20s} success_rate={v['success_rate']:.3f}")
        else:
            print(f"{k:20s} success_rate={v['success_rate']:.3f} "
                  f"cum_regret={v.get('cumulative_regret', float('nan')):.2f}")
    assert r["opd_plus_bandit"]["success_rate"] > r["random"]["success_rate"]
    assert r["bandit_from_scratch"]["success_rate"] >= r["random"]["success_rate"]
    # prior helps cold start: OPD-only should beat random
    assert r["opd_only"]["success_rate"] > r["random"]["success_rate"]
    print("BANDIT LOCAL TESTS PASSED")
