"""Domain value objects used by the attention core.

Persisted state is intentionally kept as plain dictionaries so that a plugin
restart can recover it even if a new version adds fields. Models in this module
are used at the boundary: they are built from persisted dictionaries, validated,
and converted back to dictionaries before being written to the state store.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

PASSIVE_DECISIONS = frozenset(
    {"pass", "ignore", "observe", "policy_update", "retry_later"}
)
ACTIVE_DECISIONS = frozenset({"participate", "observe", "skip", "end_focus"})
INTEREST_LEVELS = frozenset({"none", "low", "medium", "high"})
CONFIDENCE_LEVELS = frozenset({"low", "medium", "high"})
TOPIC_ACTIONS = frozenset({"keep", "update", "close"})
TOPIC_STATUSES = frozenset({"active", "closed"})
POLICY_SCOPES = frozenset({"group_user", "user", "group"})
CHECKPOINT_STATUSES = frozenset(
    {"active", "running", "paused", "completed", "cancelled"}
)

SOURCE_ADMIN = "admin"
SOURCE_MAIN_AGENT = "main_agent"
SOURCE_ATTENTION_AGENT = "attention_agent"
SOURCE_SYSTEM = "system"
SOURCE_PRIORITIES = {
    SOURCE_ADMIN: 100,
    SOURCE_MAIN_AGENT: 50,
    SOURCE_ATTENTION_AGENT: 10,
    SOURCE_SYSTEM: 1000,
}
DIRECTIVE_TYPES = frozenset(
    {
        "focus_group",
        "focus_topic",
        "ignore_user",
        "stop_focus",
        "schedule_check",
    }
)
DIRECTIVE_STATUSES = frozenset({"active", "revoked", "expired"})


@dataclass
class ModelResult:
    """One result returned by a model gateway call."""

    text: str = ""
    provider_id: str = ""
    model: str = ""
    error: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())


@dataclass
class RecentMessage:
    """A platform-neutral message used to build prompt context."""

    role: str = "user"
    sender_id: str = ""
    sender_name: str = ""
    text: str = ""
    created_at: str = ""
    created_ts: float = 0.0
    message_id: str = ""
    is_bot: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "text": self.text,
            "created_at": self.created_at,
            "created_ts": self.created_ts,
            "message_id": self.message_id,
            "is_bot": self.is_bot,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RecentMessage:
        return cls(
            role=str(raw.get("role") or "user"),
            sender_id=str(raw.get("sender_id") or ""),
            sender_name=str(raw.get("sender_name") or ""),
            text=str(raw.get("text") or ""),
            created_at=str(raw.get("created_at") or ""),
            created_ts=float(raw.get("created_ts") or 0.0),
            message_id=str(raw.get("message_id") or ""),
            is_bot=bool(raw.get("is_bot", False)),
        )


@dataclass
class PassiveRequest:
    """Normalised input for one passive attention session."""

    umo: str
    platform_id: str = ""
    group_id: str = ""
    user_id: str = ""
    sender_name: str = ""
    message_id: str = ""
    message_text: str = ""
    is_group: bool = False
    is_admin: bool = False
    is_command: bool = False
    system_prompt: str = ""
    contexts: list[dict[str, Any]] = field(default_factory=list)
    prompt: str = ""
    timestamp: str = ""

    @property
    def trigger(self) -> str:
        if self.is_group:
            return "group"
        return "private"


@dataclass
class PassiveDecision:
    """Validated passive decision output."""

    decision: str = "observe"
    interest_level: str = "low"
    reason: str = ""
    user_action: dict[str, Any] | None = None
    topic_action: str = "keep"
    confidence: str = "low"
    topic_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "interest_level": self.interest_level,
            "reason": self.reason,
            "user_action": self.user_action,
            "topic_action": self.topic_action,
            "confidence": self.confidence,
            "topic_summary": self.topic_summary,
        }


@dataclass
class PassiveOutcome:
    """What the adapter should do after a passive decision."""

    decision: str = "pass"
    block: bool = False
    reason: str = ""
    dry_run: bool = False
    model_error: str = ""


@dataclass
class ActiveAction:
    """One validated action emitted by the active attention model."""

    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    interest_level: str = ""
    topic_summary: str = ""
    topic_status: str = ""
    reply_guidance: str = ""

    @property
    def is_final(self) -> bool:
        return self.action in ACTIVE_DECISIONS


@dataclass
class ActiveResult:
    """Result of one bounded active attention loop."""

    decision: str = "observe"
    reason: str = ""
    interest_level: str = ""
    next_check_minutes: int | None = None
    reply_guidance: str = ""
    topic_summary: str = ""
    topic_status: str = ""
    effects: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    tool_calls: int = 0
    interrupted: bool = False


@dataclass
class ActiveContext:
    """Inputs available to one active attention session."""

    checkpoint: dict[str, Any]
    topic: dict[str, Any]
    window: list[RecentMessage]
    persona: str
    policy_summary: str


@dataclass
class Checkpoint:
    """Bot-level interest checkpoint."""

    checkpoint_id: str = ""
    umo: str = ""
    platform_id: str = ""
    group_id: str = ""
    scheduled_at: str = ""
    status: str = "active"
    reason: str = ""
    topic_id: str = ""
    last_result: str = ""
    created_by: str = "agent"
    revision: int = 1
    created_at: str = ""
    updated_at: str = ""
    last_run_at: str = ""
    last_error: str = ""
    run_id: str = ""
    attempts: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Checkpoint:
        known = {item.name for item in fields(cls)}
        return cls(**{key: raw[key] for key in raw if key in known})

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass
class UserPolicy:
    """A hard user or group policy."""

    policy_id: str = ""
    scope: str = "group_user"
    group_id: str = ""
    user_id: str = ""
    mode: str = "ignore"
    reason: str = ""
    expires_at: str = ""
    created_by: str = "agent"
    override: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> UserPolicy:
        known = {item.name for item in fields(cls)}
        return cls(**{key: raw[key] for key in raw if key in known})

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass
class TopicState:
    """Long-lived topic state for one session."""

    topic_id: str = ""
    umo: str = ""
    group_id: str = ""
    summary: str = ""
    last_activity_at: str = ""
    interest_level: str = "none"
    interest_source: str = ""
    focus_status: str = "idle"
    last_observation: str = ""
    has_opening: bool = False
    next_step: str = ""
    revision: int = 1
    updated_at: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TopicState:
        known = {item.name for item in fields(cls)}
        return cls(**{key: raw[key] for key in raw if key in known})

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}
