"""Platform-independent attention decision core.

The modules in this package do not import AstrBot. AstrBot-specific behaviour is
implemented in :mod:`astrbot_adapter` and injected through the small protocols
defined in :mod:`ports`.
"""

from .control import Actor, AttentionControlService, ControlResult
from .engine import AttentionEngine
from .models import (
    ActiveAction,
    ActiveContext,
    ActiveResult,
    ModelResult,
    PassiveDecision,
    PassiveOutcome,
    PassiveRequest,
    RecentMessage,
)
from .settings import Settings
from .store import StateStore

__all__ = [
    "ActiveAction",
    "ActiveContext",
    "ActiveResult",
    "Actor",
    "AttentionControlService",
    "AttentionEngine",
    "ControlResult",
    "ModelResult",
    "PassiveDecision",
    "PassiveOutcome",
    "PassiveRequest",
    "RecentMessage",
    "Settings",
    "StateStore",
]
