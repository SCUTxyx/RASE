"""Unified takeover event logging for Route C pipeline.

All collection and evaluation scripts write takeover events using the same
JSON schema, making cross-experiment analysis tractable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TakeoverEvent:
    """A single takeover event recorded during rollout."""

    # Event identity
    event_id: str = ""
    episode_id: str = ""

    # Timing
    trigger_step: int = 0          # episode step when stagnation first detected
    takeover_start_step: int = 0   # first step with plugin mixing
    takeover_end_step: int = 0     # last plugin-mixed step (or -1 if ongoing)
    takeover_duration: int = 0     # number of plugin-mixed steps

    # Stagnation
    stagnation_window: int = 20
    stagnation_eps: float = 1e-4
    progress_at_trigger: float = 0.0
    progress_std_at_trigger: float = 0.0

    # Plugin
    plugin_ckpt: str = ""
    g_mix_ramp: list[float] = field(default_factory=lambda: [0.0, 0.3, 0.6, 1.0])
    delta_clip: float = 0.5
    action_rate_limit: float = 0.1

    # Outcome
    handback_reason: str = ""      # "handback_progress" / "max_steps" / "episode_end"
    outcome: str = ""              # "success" / "failure" / "ongoing"
    irreversible_after: bool = False

    # Metadata
    task_id: str = ""
    suite: str = ""
    seed: int = 0
    timestamp_utc: str = ""


@dataclass
class EpisodeTakeoverLog:
    """Full takeover event log for a single episode."""

    episode_id: str = ""
    task_id: str = ""
    suite: str = ""
    seed: int = 0
    variant: str = ""              # B0 / B1 / B2 / B3 / B-direct
    plugin_ckpt: str = ""
    t0_utc: str = ""
    total_steps: int = 0
    success: bool = False
    takeover_count: int = 0
    events: list[TakeoverEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "suite": self.suite,
            "seed": self.seed,
            "variant": self.variant,
            "plugin_ckpt": self.plugin_ckpt,
            "t0_utc": self.t0_utc,
            "total_steps": self.total_steps,
            "success": self.success,
            "takeover_count": self.takeover_count,
            "events": [asdict(e) for e in self.events],
        }


def make_takeover_log(
    episode_id: str,
    task_id: str,
    suite: str,
    seed: int,
    variant: str,
    plugin_ckpt: str = "",
) -> EpisodeTakeoverLog:
    """Create a fresh takeover log for a single episode."""
    from datetime import datetime, timezone
    return EpisodeTakeoverLog(
        episode_id=episode_id,
        task_id=task_id,
        suite=suite,
        seed=seed,
        variant=variant,
        plugin_ckpt=plugin_ckpt,
        t0_utc=datetime.now(timezone.utc).isoformat(),
    )


def save_takeover_log(log: EpisodeTakeoverLog, path: str) -> None:
    """Write takeover log as a JSONLines-style record."""
    d = log.to_dict()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
