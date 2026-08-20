"""Conservative safe-handback controller.

Makes CONTINUE_OFT / HAND_BACK_TO_STUDENT / ABSTAIN decisions using
ensemble LCB predictions, conformal correction, and dwell-time constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class SafeHandbackDecision(IntEnum):
    CONTINUE_OFT = 0
    HAND_BACK_TO_STUDENT = 1
    ABSTAIN = 2


@dataclass
class ControllerConfig:
    tau_success: float = 0.95
    tau_risk: float = 0.05
    tau_ood: float = 0.5
    tau_disagree: float = 0.10
    tau_dwell: int = 2
    tau_cost: float = 0.10
    lcb_z: float = 1.64
    conformal_correction: float = 0.0


class SafeHandbackController:
    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

    def decide(
        self,
        student_success_scores: np.ndarray,       # (n_members,) probabilities
        remaining_cost_q80: float,                 # predicted q80 remaining cost
        ood_score: float,                          # OOD/abstention score
        ensemble_disagreement: float,              # std across members
        elapsed_oft_steps: int,                    # current takeover elapsed
        persistent_total: int,                     # estimated total persistent steps
    ) -> SafeHandbackDecision:
        cfg = self.config

        # Ensemble statistics
        p_mean = float(np.mean(student_success_scores))
        p_std = float(np.std(student_success_scores))
        p_lcb = p_mean - cfg.lcb_z * p_std - cfg.conformal_correction
        r_ucb = 1.0 - p_lcb

        # Cost gate: hand back only if predicted remaining OFT cost is significant
        cost_ratio = remaining_cost_q80 / max(1, persistent_total)

        checks = {
            "success": p_lcb >= cfg.tau_success,
            "risk": r_ucb <= cfg.tau_risk,
            "ood": ood_score <= cfg.tau_ood,
            "disagree": ensemble_disagreement <= cfg.tau_disagree,
            "dwell": elapsed_oft_steps >= cfg.tau_dwell,
            "cost": cost_ratio >= cfg.tau_cost,
        }

        if ood_score > cfg.tau_ood or ensemble_disagreement > cfg.tau_disagree:
            return SafeHandbackDecision.ABSTAIN

        if all(checks.values()):
            return SafeHandbackDecision.HAND_BACK_TO_STUDENT

        return SafeHandbackDecision.CONTINUE_OFT

    def decision_info(
        self,
        student_success_scores: np.ndarray,
        remaining_cost_q80: float,
        ood_score: float,
        ensemble_disagreement: float,
        elapsed_oft_steps: int,
        persistent_total: int,
    ) -> dict[str, Any]:
        decision = self.decide(
            student_success_scores, remaining_cost_q80, ood_score,
            ensemble_disagreement, elapsed_oft_steps, persistent_total,
        )
        return {
            "decision": int(decision),
            "decision_name": decision.name,
            "p_mean": float(np.mean(student_success_scores)),
            "p_lcb": float(np.mean(student_success_scores)
                          - self.config.lcb_z * float(np.std(student_success_scores))
                          - self.config.conformal_correction),
            "ood_score": ood_score,
            "disagreement": ensemble_disagreement,
            "elapsed_oft_steps": elapsed_oft_steps,
        }
