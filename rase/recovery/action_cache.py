"""Policy action-cache and history reset utilities for Route C parity audits.

Ensures that after a fork/restore, the policy does not carry over old
action chunks or history state that could cause divergence.
"""

from __future__ import annotations

from typing import Any


def reset_policy_action_cache(policy: Any) -> None:
    """Clear any pending action chunk queue in the policy.

    SmolVLA buffers a chunk of actions internally. After a snapshot restore,
    the old chunk must be discarded so the next ``select_action`` call
    generates a fresh chunk from the restored observation.
    """
    if policy is None:
        return

    # LeRobot policy action queue pattern.
    if hasattr(policy, "_action_queue"):
        policy._action_queue.clear()

    # Additional safety: reset any timestep/bookkeeping counters.
    for attr in ("_t", "_step_idx", "_chunk_idx", "_prev_action", "cur_t"):
        if hasattr(policy, attr):
            try:
                setattr(policy, attr, None)
            except Exception:
                pass


def reset_policy_history(policy: Any) -> None:
    """Reset policy observation/state history if maintained internally."""
    if policy is None:
        return
    for attr in ("obs_history", "state_history", "history", "prev_obs",
                 "cache", "h_state", "hidden"):
        if hasattr(policy, attr):
            try:
                val = getattr(policy, attr)
                if hasattr(val, "clear"):
                    val.clear()
                else:
                    setattr(policy, attr, None)
            except Exception:
                pass
