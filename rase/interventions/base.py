"""Runtime protocol for executable intervention implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .schema import CostVector, Feasibility, InterventionSnapshot, OperatorSpec


@dataclass(frozen=True)
class InterventionBudget:
    max_compute_seconds: float | None = None
    max_latency_seconds: float | None = None
    max_env_steps: int | None = None
    allow_human: bool = False


@dataclass(frozen=True)
class InterventionProposal:
    operator_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)


class ExecutionTrace(Protocol):
    success: bool
    operator_completed: bool
    stop_reason: str


class InterventionOperator(Protocol):
    """Contract used by simulator and real-robot executors.

    Implementations may inspect public deployment history only. Privileged
    simulator restore state belongs to the runner, not to ``propose``.
    """

    @property
    def spec(self) -> OperatorSpec: ...

    def feasible(
        self,
        snapshot: InterventionSnapshot,
        public_history: Mapping[str, Any],
        budget: InterventionBudget,
    ) -> Feasibility: ...

    def propose(
        self,
        snapshot: InterventionSnapshot,
        public_history: Mapping[str, Any],
        budget: InterventionBudget,
    ) -> InterventionProposal: ...

    def execute(
        self,
        env: Any,
        proposal: InterventionProposal,
        *,
        continuation_seed: int,
    ) -> ExecutionTrace: ...

    def cost(self, trace: ExecutionTrace) -> CostVector: ...
