"""Route C recovery toolkit.

Modules:
  - residual_plugin: standalone feedforward delta-prediction model
  - plugin_executor: bounded takeover, safe mixing, handback logic
  - stagnation: shared stagnation detector used by all Route C scripts
  - takeover_log: unified JSON event-log schema
  - action_cache: policy action-cache and history reset utilities
"""

from rase.recovery.residual_plugin import (
    ResidualRecoveryPlugin,
    make_recovery_plugin,
    save_plugin,
    load_plugin,
)
from rase.recovery.plugin_executor import RecoveryPluginExecutor
from rase.recovery.stagnation import StagnationDetector
from rase.recovery.takeover_log import (
    TakeoverEvent,
    EpisodeTakeoverLog,
    make_takeover_log,
    save_takeover_log,
)
from rase.recovery.action_cache import reset_policy_action_cache, reset_policy_history

__all__ = [
    "ResidualRecoveryPlugin",
    "make_recovery_plugin",
    "save_plugin",
    "load_plugin",
    "RecoveryPluginExecutor",
    "StagnationDetector",
    "TakeoverEvent",
    "EpisodeTakeoverLog",
    "make_takeover_log",
    "save_takeover_log",
    "reset_policy_action_cache",
    "reset_policy_history",
]
