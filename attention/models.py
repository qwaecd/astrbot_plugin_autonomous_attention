"""Domain models for the session-scoped attention system.

State is persisted as plain dictionaries so a plugin upgrade can load older
documents. The dataclasses here are used at the boundary: they are built from
persisted dictionaries, validated, and converted back before being written.

Ownership rule: every long-lived resource belongs to exactly one AstrBot
``unified_msg_origin`` (UMO). ``platform_id`` and ``group_id`` are routing
metadata only; they never identify ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

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

INTEREST_STATUSES = frozenset({"watching", "paused", "closed"})
INTEREST_ACTIVE_STATUSES = frozenset({"watching", "paused"})
OPPORTUNITY_KINDS = frozenset(
    {
        "direct_request",
        "unanswered_question",
        "new_information",
        "natural_opening",
    }
)
OPPORTUNITY_STATUSES = frozenset(
    {"pending", "reserved", "consumed", "expired", "uncertain"}
)
WAKE_SOURCES = frozenset(
    {"private", "mention", "reply_to_bot", "wake_prefix", "ambient", "unknown"}
)
EXPLICIT_WAKE_SOURCES = frozenset({"private", "mention", "reply_to_bot", "wake_prefix"})

CHECKPOINT_KINDS = frozenset({"interest", "session_scan"})
CHECKPOINT_STATUSES = frozenset(
    {"active", "running", "paused", "completed", "cancelled"}
)

PASSIVE_ACTIONS = frozenset(
    {"reply", "observe", "ignore", "policy_update", "retry_later", "end_focus"}
)
ACTIVE_FINAL_ACTIONS = frozenset({"participate", "observe", "skip", "end_focus"})
ACTIVE_TOOL_ACTIONS = frozenset(
    {
        "read_messages",
        "get_interest",
        "get_topic",
        "schedule_next_check",
        "defer_check",
        "create_checkpoint",
        "update_user_policy",
        "record_interest",
        "request_main_reply",
    }
)
INTEREST_LEVELS = frozenset({"none", "low", "medium", "high"})
CONFIDENCE_LEVELS = frozenset({"low", "medium", "high"})
TOPIC_ACTIONS = frozenset({"keep", "update", "close"})
POLICY_SCOPES = frozenset({"group_user", "user", "group"})
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
    seq: int = 0
    umo: str = ""
    source: str = "event"
    aliases: list[str] = field(default_factory=list)
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
            "seq": int(self.seq or 0),
            "umo": self.umo,
            "source": self.source,
            "aliases": list(self.aliases),
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
            seq=int(raw.get("seq") or 0),
            umo=str(raw.get("umo") or ""),
            source=str(raw.get("source") or "event"),
            aliases=list(raw.get("aliases") or []),
            is_bot=bool(raw.get("is_bot", False)),
        )


@dataclass
class Interest:
    """A session-owned long-running interest."""

    interest_id: str = ""
    umo: str = ""
    platform_id: str = ""
    group_id: str = ""
    summary: str = ""
    status: str = "watching"
    source: str = SOURCE_ATTENTION_AGENT
    priority: int = 10
    directive_id: str = ""
    evidence_message_ids: list[str] = field(default_factory=list)
    wait_for: str = ""
    last_external_evidence_at: str = ""
    expires_at: str = ""
    remaining_checks: int = 3
    no_progress_count: int = 0
    last_evaluated_seq: int = 0
    created_at: str = ""
    updated_at: str = ""
    revision: int = 1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Interest:
        known = {item.name for item in fields(cls)}
        data = {key: raw[key] for key in raw if key in known}
        if "evidence_message_ids" in data and not isinstance(
            data["evidence_message_ids"], list
        ):
            data["evidence_message_ids"] = []
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["evidence_message_ids"] = list(self.evidence_message_ids or [])
        return result


@dataclass
class Opportunity:
    """One concrete chance to participate, owned by a session."""

    opportunity_id: str = ""
    umo: str = ""
    interest_id: str = ""
    kind: str = "natural_opening"
    anchor_message_id: str = ""
    anchor_created_at: str = ""
    evidence_message_ids: list[str] = field(default_factory=list)
    expected_contribution: str = ""
    status: str = "pending"
    expires_at: str = ""
    reservation_id: str = ""
    reserved_at: str = ""
    consumed_at: str = ""
    outcome: str = ""
    created_at: str = ""
    updated_at: str = ""
    revision: int = 1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Opportunity:
        known = {item.name for item in fields(cls)}
        data = {key: raw[key] for key in raw if key in known}
        if "evidence_message_ids" in data and not isinstance(
            data["evidence_message_ids"], list
        ):
            data["evidence_message_ids"] = []
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["evidence_message_ids"] = list(self.evidence_message_ids or [])
        return result


@dataclass
class SessionState:
    """Per-UMO progress and autonomous participation accounting."""

    umo: str = ""
    platform_id: str = ""
    group_id: str = ""
    message_ids: dict[str, int] = field(default_factory=dict)
    history_shapes: list[dict[str, Any]] = field(default_factory=list)
    handled_before_ts: float = 0.0
    received_messages: int = 0
    external_message_seq: int = 0
    last_decision_seq: int = 0
    last_decision_at: str = ""
    next_evaluation_at: str = ""
    last_autonomous_participation_at: str = ""
    quota_date: str = ""
    autonomous_attempts: int = 0
    pending_reply_reservation: str = ""
    revision: int = 1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SessionState:
        known = {item.name for item in fields(cls)}
        return cls(**{key: raw[key] for key in raw if key in known})

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass
class Checkpoint:
    """A scheduled attention task, either interest-bound or session scan."""

    checkpoint_id: str = ""
    umo: str = ""
    platform_id: str = ""
    group_id: str = ""
    interest_id: str = ""
    kind: str = "interest"
    scheduled_at: str = ""
    status: str = "active"
    reason: str = ""
    last_result: str = ""
    source: str = SOURCE_ATTENTION_AGENT
    priority: int = 10
    directive_id: str = ""
    created_by_session: str = ""
    expires_at: str = ""
    remaining_checks: int = 3
    no_progress_count: int = 0
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
class PassiveRequest:
    """Normalised input for one passive attention decision."""

    umo: str
    platform_id: str = ""
    group_id: str = ""
    user_id: str = ""
    sender_name: str = ""
    message_id: str = ""
    message_seq: int = 0
    message_text: str = ""
    wake_source: str = "unknown"
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
    """Validated passive model suggestion."""

    suggested_action: str = "observe"
    interest_level: str = "low"
    reason: str = ""
    confidence: str = "low"
    opportunity: dict[str, Any] = field(default_factory=dict)
    interest_update: dict[str, Any] = field(default_factory=dict)
    user_action: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggested_action": self.suggested_action,
            "interest_level": self.interest_level,
            "reason": self.reason,
            "confidence": self.confidence,
            "opportunity": self.opportunity,
            "interest_update": self.interest_update,
            "user_action": self.user_action,
        }


@dataclass
class PassiveOutcome:
    """What the adapter should do after a passive decision."""

    decision: str = "observe"
    block: bool = True
    reason: str = ""
    code: str = ""
    dry_run: bool = False
    model_error: str = ""
    opportunity_id: str = ""
    reservation_id: str = ""


@dataclass
class ActiveAction:
    """One validated action emitted by the active attention model."""

    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    interest_level: str = ""
    opportunity: dict[str, Any] = field(default_factory=dict)
    interest_update: dict[str, Any] = field(default_factory=dict)
    reply_guidance: str = ""

    @property
    def is_final(self) -> bool:
        return self.action in ACTIVE_FINAL_ACTIONS


@dataclass
class ActiveResult:
    """Result of one bounded active attention loop."""

    decision: str = "observe"
    reason: str = ""
    interest_level: str = ""
    next_check_minutes: int | None = None
    reply_guidance: str = ""
    opportunity: dict[str, Any] = field(default_factory=dict)
    interest_update: dict[str, Any] = field(default_factory=dict)
    evaluated_interest_ids: list[str] = field(default_factory=list)
    effects: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    tool_calls: int = 0
    interrupted: bool = False


@dataclass
class ActiveContext:
    """Inputs available to one active attention session."""

    checkpoint: dict[str, Any]
    interest: dict[str, Any] | None
    interests: list[dict[str, Any]]
    window: list[RecentMessage]
    persona: str
    policy_summary: str
    evaluated_seq: int = 0
    forced: bool = False


@dataclass
class ParticipationApproval:
    """A reserved, code-approved participation attempt."""

    approved: bool
    code: str = ""
    message: str = ""
    opportunity: dict[str, Any] | None = None
    reservation_id: str = ""
    interest_id: str = ""
    wake_source: str = "unknown"
    direct_request: bool = False
    reason: str = ""
