"""Unified attention control service.

All reads and writes for session-owned resources go through this service. It
validates the operator, the session scope, source priority and resource
lifecycle before mutating the state store. Low-priority operators can never
overwrite higher-priority resources.

The control layer, not the model, decides:

* whether an interest may be admitted or replaced;
* whether a checkpoint may exist and when it is due;
* whether an opportunity is a valid, unconsumed anchor;
* whether an autonomous participation attempt may consume quota.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .models import (
    CHECKPOINT_KINDS,
    DIRECT_PASS_WAKE_SOURCES,
    DIRECTIVE_TYPES,
    INTEREST_ACTIVE_STATUSES,
    OPPORTUNITY_KINDS,
    SOURCE_ADMIN,
    SOURCE_ATTENTION_AGENT,
    SOURCE_MAIN_AGENT,
    SOURCE_PRIORITIES,
    SOURCE_SYSTEM,
    Interest,
    Opportunity,
    ParticipationApproval,
    UserPolicy,
)
from .policy import PolicyManager
from .settings import Settings
from .store import StateStore
from .timeutil import (
    add_minutes,
    in_quiet_hours,
    parse_iso,
    quiet_hours_end,
    to_iso,
)

PASSIVE_SCOPE_CHOICES = frozenset(
    {"session", "interests", "checkpoints", "directives", "recent", "all"}
)


@dataclass(frozen=True)
class Actor:
    """The real operator behind a control request.

    ``session`` is the trusted UMO of the event; model-provided text is never
    used to select a target session.
    """

    source: str = SOURCE_ATTENTION_AGENT
    user_id: str = ""
    session: str = ""
    group_id: str = ""
    platform_id: str = ""
    is_admin: bool = False


@dataclass
class ControlResult:
    """Structured result returned to the main model or an internal caller."""

    ok: bool
    code: str = ""
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "ok": self.ok,
                "code": self.code,
                "message": self.message,
                "data": self.data,
            },
            ensure_ascii=False,
        )


def _clip(value: Any, limit: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _priority(source: str) -> int:
    return int(SOURCE_PRIORITIES.get(source, 0))


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class AttentionControlService:
    """Validate and write session-scoped attention state."""

    def __init__(
        self,
        store: StateStore,
        settings: Settings,
        policy: PolicyManager,
    ) -> None:
        self.store = store
        self.settings = settings
        self.policy = policy

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings
        self.policy.update_settings(settings)

    def now(self) -> datetime:
        return self.policy.now()

    # ------------------------------------------------------------------
    # Public async entry points (main-model tools)
    # ------------------------------------------------------------------

    async def query(
        self,
        actor: Actor,
        *,
        scope: str = "session",
    ) -> ControlResult:
        async with self.store.lock:
            return self.query_locked(self.store.state, actor, scope=scope)

    async def set_intent(
        self,
        actor: Actor,
        *,
        intent_type: str,
        target: str = "",
        reason: str = "",
        expires_in_minutes: int | None = None,
        minutes: int | None = None,
        interest_id: str = "",
    ) -> ControlResult:
        async with self.store.lock:
            result = self.set_intent_locked(
                self.store.state,
                actor,
                intent_type=intent_type,
                target=target,
                reason=reason,
                expires_in_minutes=expires_in_minutes,
                minutes=minutes,
                interest_id=interest_id,
            )
            if result.ok:
                self._flush_locked()
            return result

    async def ignore_user(
        self,
        actor: Actor,
        *,
        target_user: str,
        reason: str = "",
        expires_in_minutes: int | None = None,
        scope: str = "group_user",
    ) -> ControlResult:
        async with self.store.lock:
            result = self.ignore_user_locked(
                self.store.state,
                actor,
                target_user=target_user,
                reason=reason,
                expires_in_minutes=expires_in_minutes,
                scope=scope,
            )
            if result.ok:
                self._flush_locked()
            return result

    async def clear_user_policy(
        self,
        actor: Actor,
        *,
        target_user: str = "",
        policy_id: str = "",
    ) -> ControlResult:
        async with self.store.lock:
            result = self.clear_user_policy_locked(
                self.store.state,
                actor,
                target_user=target_user,
                policy_id=policy_id,
            )
            if result.ok:
                self._flush_locked()
            return result

    async def focus_topic(
        self,
        actor: Actor,
        *,
        topic: str,
        reason: str = "",
        expires_in_minutes: int | None = None,
    ) -> ControlResult:
        async with self.store.lock:
            result = self.set_intent_locked(
                self.store.state,
                actor,
                intent_type="focus_topic",
                target=topic,
                reason=reason,
                expires_in_minutes=expires_in_minutes,
            )
            if result.ok:
                self._flush_locked()
            return result

    async def stop_focus(
        self,
        actor: Actor,
        *,
        target: str = "",
    ) -> ControlResult:
        async with self.store.lock:
            result = self.stop_focus_locked(
                self.store.state,
                actor,
                target=target,
            )
            if result.ok:
                self._flush_locked()
            return result

    async def close_interest(self, actor: Actor, interest_id: str) -> ControlResult:
        async with self.store.lock:
            result = self.close_interest_locked(
                self.store.state,
                actor,
                interest_id=interest_id,
                reason="closed by operator",
            )
            if result.ok:
                self._flush_locked()
            return result

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        scope: str = "session",
    ) -> ControlResult:
        scope = str(scope or "session").strip().lower()
        if scope not in PASSIVE_SCOPE_CHOICES:
            return ControlResult(
                ok=False,
                code="invalid_scope",
                message=(
                    "scope 必须是 session、interests、checkpoints、directives、"
                    "recent 或 all。"
                ),
            )
        moment = self.now()
        self.expire_interests_locked(state, moment)
        self.policy.expire_policies(state, moment)
        umo = actor.session
        if not umo:
            return ControlResult(
                ok=False,
                code="missing_session",
                message="缺少可信会话标识。",
            )
        session = self.policy.ensure_session(state, umo, moment)
        data: dict[str, Any] = {
            "session": {
                "umo": umo,
                "group_id": actor.group_id or "",
                "group_allowed": self.policy.group_allowed(
                    actor.group_id,
                    umo,
                    actor.platform_id,
                ),
                "external_message_seq": int(session["external_message_seq"]),
                "last_decision_seq": int(session["last_decision_seq"]),
                "last_decision_at": session["last_decision_at"],
                "autonomous_attempts": int(session["autonomous_attempts"]),
                "autonomous_attempt_limit": (
                    self.settings.session_autonomous_reply_budget
                ),
                "global_autonomous_attempts": int(
                    self.policy.ensure_budget(state, moment)["autonomous_attempts"]
                ),
                "global_autonomous_attempt_limit": (
                    self.settings.daily_autonomous_reply_budget
                ),
            }
        }
        if scope in {"session", "interests", "all"}:
            interests = self.session_interests_locked(state, umo)
            data["interests"] = [self._interest_view(item) for item in interests]
        if scope in {"session", "checkpoints", "all"}:
            checkpoints = self.session_checkpoints_locked(state, umo)
            data["checkpoints"] = [self._checkpoint_view(item) for item in checkpoints]
        if scope in {"session", "directives", "all"}:
            data["directives"] = [
                self._directive_view(item)
                for item in self.session_directives_locked(state, umo)
            ]
        if scope in {"session", "recent", "all"}:
            decisions = [
                item
                for item in state.get("decisions", [])
                if isinstance(item, dict) and item.get("umo") == umo
            ][-3:]
            data["recent"] = [self._decision_view(item) for item in decisions]
        return ControlResult(
            ok=True,
            code="query_ok",
            message="已返回当前会话的注意力状态。",
            data=data,
        )

    def session_interests_locked(
        self,
        state: dict[str, Any],
        umo: str,
        *,
        include_closed: bool = False,
    ) -> list[dict[str, Any]]:
        self.expire_interests_locked(state)
        result: list[dict[str, Any]] = []
        for raw in (state.get("interests") or {}).values():
            if not isinstance(raw, dict) or raw.get("umo") != umo:
                continue
            if raw.get("quarantine"):
                continue
            if not include_closed and raw.get("status") == "closed":
                continue
            result.append(dict(raw))
        result.sort(
            key=lambda item: (
                -int(item.get("priority") or 0),
                str(item.get("updated_at") or ""),
            )
        )
        return result

    def session_checkpoints_locked(
        self,
        state: dict[str, Any],
        umo: str,
        *,
        include_finished: bool = False,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for raw in (state.get("checkpoints") or {}).values():
            if not isinstance(raw, dict) or raw.get("umo") != umo:
                continue
            if raw.get("quarantine"):
                continue
            if not include_finished and raw.get("status") not in {
                "active",
                "running",
                "paused",
            }:
                continue
            result.append(dict(raw))
        result.sort(key=lambda item: str(item.get("scheduled_at") or ""))
        return result

    def session_directives_locked(
        self,
        state: dict[str, Any],
        umo: str,
    ) -> list[dict[str, Any]]:
        self.expire_directives_locked(state)
        result: list[dict[str, Any]] = []
        for raw in (state.get("directives") or {}).values():
            if not isinstance(raw, dict) or raw.get("status") != "active":
                continue
            if raw.get("created_by_session") != umo:
                continue
            result.append(dict(raw))
        result.sort(
            key=lambda item: (
                -int(item.get("priority") or 0),
                str(item.get("updated_at") or ""),
            )
        )
        return result

    @staticmethod
    def _interest_view(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "interest_id": raw.get("interest_id", ""),
            "summary": _clip(raw.get("summary"), 300),
            "status": raw.get("status", ""),
            "source": raw.get("source", ""),
            "priority": raw.get("priority", 0),
            "wait_for": _clip(raw.get("wait_for"), 200),
            "expires_at": raw.get("expires_at", ""),
            "remaining_checks": raw.get("remaining_checks", 0),
            "no_progress_count": raw.get("no_progress_count", 0),
            "evidence_count": len(raw.get("evidence_message_ids") or []),
        }

    @staticmethod
    def _checkpoint_view(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "checkpoint_id": raw.get("checkpoint_id", ""),
            "interest_id": raw.get("interest_id", ""),
            "kind": raw.get("kind", ""),
            "status": raw.get("status", ""),
            "scheduled_at": raw.get("scheduled_at", ""),
            "expires_at": raw.get("expires_at", ""),
            "remaining_checks": raw.get("remaining_checks", 0),
            "reason": _clip(raw.get("reason"), 120),
            "last_result": raw.get("last_result", ""),
        }

    @staticmethod
    def _directive_view(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "directive_id": raw.get("directive_id", ""),
            "type": raw.get("type", ""),
            "target": _clip(raw.get("target"), 120),
            "source": raw.get("source", ""),
            "priority": raw.get("priority", 0),
            "expires_at": raw.get("expires_at", ""),
            "reason": _clip(raw.get("reason"), 120),
        }

    @staticmethod
    def _decision_view(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": raw.get("timestamp", ""),
            "kind": raw.get("kind", ""),
            "decision": raw.get("code_decision") or raw.get("decision", ""),
            "suggested_action": raw.get("suggested_action", ""),
            "code": raw.get("rejection_code", ""),
            "interest_id": raw.get("interest_id", ""),
            "anchor_message_id": raw.get("anchor_message_id", ""),
            "reason": _clip(raw.get("reason"), 100),
        }

    # ------------------------------------------------------------------
    # Interests
    # ------------------------------------------------------------------

    def admit_interest_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        summary: str,
        wait_for: str,
        evidence_message_ids: list[str],
        source: str = SOURCE_ATTENTION_AGENT,
        expires_in_minutes: int | None = None,
        interest_id: str = "",
        priority_level: int = 0,
        moment: datetime | None = None,
    ) -> ControlResult:
        """Admit or update one session-owned interest.

        Args:
            state: Mutable state dictionary.
            actor: Trusted operator; ``actor.session`` is the owner.
            summary: Bounded interest summary.
            wait_for: Concrete condition to wait for.
            evidence_message_ids: External message IDs that justify the interest.
            source: ``main_agent`` or ``attention_agent``.
            expires_in_minutes: Requested TTL; clamped by configuration.
            interest_id: Existing interest to update, if any.
            priority_level: Discrete 0-2 evidence level generated by code.
            moment: Current time, injectable for tests.

        Returns:
            A control result whose data contains the stored interest.
        """

        moment = moment or self.now()
        umo = actor.session
        if not umo:
            return ControlResult(
                ok=False, code="missing_session", message="缺少可信会话标识。"
            )
        if self._owner_mismatch(umo, actor.platform_id, actor.group_id):
            return ControlResult(
                ok=False, code="scope_mismatch", message="Session route mismatch."
            )
        self.expire_interests_locked(state, moment)
        binding = self.policy.ensure_session(state, umo, moment)
        if (
            binding.get("group_id")
            and actor.group_id
            and binding["group_id"] != actor.group_id
        ):
            return ControlResult(
                ok=False, code="scope_mismatch", message="会话路由冲突。"
            )
        if actor.platform_id:
            binding["platform_id"] = actor.platform_id
        if actor.group_id:
            binding["group_id"] = actor.group_id
        summary = _clip(summary, 300)
        wait_for = _clip(wait_for, 300)
        if self._owner_mismatch(umo, actor.platform_id, actor.group_id):
            return ControlResult(
                ok=False,
                code="scope_mismatch",
                message="会话、平台与群号不一致，不能创建兴趣。",
            )
        if not summary:
            return ControlResult(
                ok=False, code="missing_summary", message="兴趣摘要不能为空。"
            )
        if not wait_for:
            return ControlResult(
                ok=False,
                code="missing_wait_for",
                message="新兴趣必须提供具体的等待条件。",
            )
        evidence = self._valid_evidence_locked(
            state,
            umo,
            evidence_message_ids,
        )
        if not evidence:
            return ControlResult(
                ok=False,
                code="missing_evidence",
                message="新兴趣必须引用本会话的外部证据消息。",
            )
        if not self._actor_can_write_source(actor, source):
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="当前操作者不能写入该来源的兴趣。",
            )
        expires_at = self._resolve_expiry(
            actor,
            # Interests keep their own TTL even when the intent limit is short:
            # the intent budget bounds directives, not interest lifetime.
            max_minutes=max(
                self.settings.main_agent_max_intent_minutes,
                self.settings.interest_ttl_minutes,
            ),
            requested_minutes=expires_in_minutes,
            allow_permanent=False,
            default_minutes=self.settings.interest_ttl_minutes,
        )
        if expires_at is None:
            expires_at = add_minutes(
                moment, max(1, int(self.settings.interest_ttl_minutes))
            )
        if source == SOURCE_ATTENTION_AGENT:
            expires_at = min(
                expires_at,
                add_minutes(moment, max(1, int(self.settings.interest_ttl_minutes))),
            )

        existing: dict[str, Any] | None = None
        if interest_id:
            raw = (state.get("interests") or {}).get(interest_id)
            if not isinstance(raw, dict) or raw.get("umo") != umo:
                return ControlResult(
                    ok=False,
                    code="interest_not_found",
                    message="当前会话不存在该兴趣。",
                )
            if raw.get("status") == "closed" or raw.get("quarantine"):
                return ControlResult(
                    ok=False,
                    code="interest_expired",
                    message="Interest is closed or quarantined.",
                )
            existing = raw
        else:
            for raw in (state.get("interests") or {}).values():
                if not isinstance(raw, dict):
                    continue
                if raw.get("umo") != umo:
                    continue
                if raw.get("status") == "closed":
                    continue
                if str(raw.get("summary") or "").casefold() == summary.casefold():
                    existing = raw
                    break

        priority = _priority(source) + max(0, min(2, int(priority_level)))
        if existing is not None:
            existing_priority = int(existing.get("priority") or 0)
            if existing_priority > priority and not actor.is_admin:
                return ControlResult(
                    ok=False,
                    code="higher_priority_interest_exists",
                    message="存在更高优先级的兴趣，当前请求不能覆盖。",
                )
            if existing.get("source") == SOURCE_ADMIN and not actor.is_admin:
                return ControlResult(
                    ok=False,
                    code="admin_interest_protected",
                    message="管理员兴趣不能被覆盖。",
                )
            existing_evidence = [
                str(item) for item in existing.get("evidence_message_ids") or []
            ]
            fresh_ids = {
                str(item.get("message_id") or "")
                for item in (state.get("recent_messages") or {}).get(umo, [])
                if isinstance(item, dict)
                and int(item.get("seq") or 0)
                > int(existing.get("last_evaluated_seq") or 0)
            }
            new_evidence = [
                item
                for item in evidence
                if item not in existing_evidence and item in fresh_ids
            ]
            existing.update(
                {
                    "summary": summary,
                    "wait_for": wait_for,
                    "expires_at": to_iso(expires_at)
                    if new_evidence
                    else str(existing.get("expires_at") or ""),
                    "remaining_checks": max(
                        int(existing.get("remaining_checks") or 0),
                        self.settings.interest_check_limit,
                    )
                    if new_evidence
                    else int(existing.get("remaining_checks") or 0),
                    "priority": max(existing_priority, priority),
                    "source": source
                    if priority >= existing_priority
                    else existing.get("source"),
                    "evidence_message_ids": self._merge_evidence(
                        existing_evidence,
                        evidence,
                    ),
                    "last_external_evidence_at": to_iso(moment)
                    if new_evidence
                    else str(existing.get("last_external_evidence_at") or ""),
                    "updated_at": to_iso(moment),
                    "revision": int(existing.get("revision") or 1) + 1,
                }
            )
            if new_evidence:
                existing["no_progress_count"] = 0
            return ControlResult(
                ok=True,
                code="interest_updated",
                message="已有兴趣已更新。",
                data={"interest": dict(existing)},
            )

        active = [
            item
            for item in self.session_interests_locked(state, umo)
            if item.get("status") in INTEREST_ACTIVE_STATUSES
        ]
        if len(active) >= self.settings.max_interests_per_session:
            replaceable = self._weakest_replacement_locked(active)
            if replaceable is None or int(replaceable.get("priority") or 0) >= priority:
                return ControlResult(
                    ok=False,
                    code="interest_capacity_exhausted",
                    message="本会话兴趣槽位已满，且没有可替换的低优先级兴趣。",
                    data={"active": len(active)},
                )
            self.close_interest_locked(
                state,
                Actor(
                    source=SOURCE_SYSTEM,
                    session=umo,
                    group_id=actor.group_id,
                    platform_id=actor.platform_id,
                    is_admin=True,
                ),
                interest_id=str(replaceable["interest_id"]),
                reason="replaced by higher-priority interest",
                moment=moment,
            )

        interest = Interest(
            interest_id=_new_id("i"),
            umo=umo,
            platform_id=actor.platform_id,
            group_id=actor.group_id,
            summary=summary,
            status="watching",
            source=source,
            priority=priority,
            directive_id="",
            evidence_message_ids=evidence,
            wait_for=wait_for,
            last_external_evidence_at=to_iso(moment),
            expires_at=to_iso(expires_at),
            remaining_checks=self.settings.interest_check_limit,
            no_progress_count=0,
            last_evaluated_seq=0,
            created_at=to_iso(moment),
            updated_at=to_iso(moment),
            revision=1,
        ).to_dict()
        state.setdefault("interests", {})[interest["interest_id"]] = interest
        return ControlResult(
            ok=True,
            code="interest_admitted",
            message="兴趣已准入。",
            data={"interest": dict(interest)},
        )

    def close_interest_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        interest_id: str,
        reason: str,
        moment: datetime | None = None,
    ) -> ControlResult:
        moment = moment or self.now()
        raw = (state.get("interests") or {}).get(interest_id)
        if not isinstance(raw, dict) or raw.get("umo") != actor.session:
            return ControlResult(
                ok=False,
                code="interest_not_found",
                message="当前会话不存在该兴趣。",
            )
        if raw.get("source") == SOURCE_ADMIN and not actor.is_admin:
            return ControlResult(
                ok=False,
                code="admin_interest_protected",
                message="管理员兴趣不能被普通操作关闭。",
            )
        if (
            raw.get("source") == SOURCE_MAIN_AGENT
            and actor.source == SOURCE_ATTENTION_AGENT
            and not actor.is_admin
        ):
            return ControlResult(
                ok=False,
                code="main_agent_interest_protected",
                message="注意力 Agent 不能关闭主 Agent 的兴趣。",
            )
        raw["status"] = "closed"
        raw["updated_at"] = to_iso(moment)
        raw["revision"] = int(raw.get("revision") or 1) + 1
        raw["close_reason"] = _clip(reason, 120)
        cancelled: list[str] = []
        for checkpoint in (state.get("checkpoints") or {}).values():
            if not isinstance(checkpoint, dict):
                continue
            if checkpoint.get("umo") != actor.session:
                continue
            if checkpoint.get("interest_id") != interest_id:
                continue
            if checkpoint.get("status") in {"active", "running", "paused"}:
                checkpoint["status"] = (
                    "completed" if reason == "end_focus" else "cancelled"
                )
                checkpoint["scheduled_at"] = ""
                checkpoint["last_result"] = (
                    "end_focus" if reason == "end_focus" else "interest_closed"
                )
                checkpoint["updated_at"] = to_iso(moment)
                checkpoint["revision"] = int(checkpoint.get("revision") or 1) + 1
                cancelled.append(str(checkpoint.get("checkpoint_id") or ""))
        for opportunity in (state.get("opportunities") or {}).values():
            if not isinstance(opportunity, dict):
                continue
            if opportunity.get("umo") != actor.session:
                continue
            if opportunity.get("interest_id") != interest_id:
                continue
            if opportunity.get("status") == "pending":
                opportunity["status"] = "expired"
                opportunity["outcome"] = "interest_closed"
                opportunity["updated_at"] = to_iso(moment)
                opportunity["revision"] = int(opportunity.get("revision") or 1) + 1
        return ControlResult(
            ok=True,
            code="interest_closed",
            message="兴趣已关闭。",
            data={
                "interest_id": interest_id,
                "cancelled_checkpoints": cancelled,
            },
        )

    def touch_interest_evidence_locked(
        self,
        state: dict[str, Any],
        interest_id: str,
        evidence_message_ids: list[str],
        *,
        moment: datetime | None = None,
        reset_checks: bool = True,
    ) -> bool:
        """Refresh an interest after semantically relevant external evidence."""

        moment = moment or self.now()
        raw = (state.get("interests") or {}).get(interest_id)
        if not isinstance(raw, dict) or raw.get("status") == "closed":
            return False
        evidence = self._valid_evidence_locked(
            state, str(raw.get("umo") or ""), evidence_message_ids
        )
        known = set(raw.get("evidence_message_ids") or [])
        progress = int(raw.get("last_evaluated_seq") or 0)
        fresh = {
            str(item.get("message_id") or "")
            for item in (state.get("recent_messages") or {}).get(raw.get("umo"), [])
            if isinstance(item, dict) and int(item.get("seq") or 0) > progress
        }
        evidence = [item for item in evidence if item not in known and item in fresh]
        expires = parse_iso(raw.get("expires_at"), self.settings.tz)
        if not evidence or expires is None or expires <= moment:
            return False
        raw["evidence_message_ids"] = self._merge_evidence(
            raw.get("evidence_message_ids") or [],
            evidence,
        )
        raw["last_external_evidence_at"] = to_iso(moment)
        raw["no_progress_count"] = 0
        if reset_checks:
            raw["remaining_checks"] = self.settings.interest_check_limit
            raw["expires_at"] = to_iso(
                add_minutes(
                    moment,
                    max(1, int(self.settings.interest_ttl_minutes)),
                )
            )
        raw["updated_at"] = to_iso(moment)
        raw["revision"] = int(raw.get("revision") or 1) + 1
        return True

    def _weakest_replacement_locked(
        self,
        active: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        replaceable = [item for item in active if item.get("source") != SOURCE_ADMIN]
        if not replaceable:
            return None
        replaceable.sort(
            key=lambda item: (
                int(item.get("priority") or 0),
                -int(item.get("no_progress_count") or 0),
                str(item.get("last_external_evidence_at") or ""),
                str(item.get("interest_id") or ""),
            )
        )
        return replaceable[0]

    @staticmethod
    def _merge_evidence(existing: list[Any], incoming: list[str]) -> list[str]:
        merged: list[str] = []
        for item in [*existing, *incoming]:
            text = str(item).strip()
            if text and text not in merged:
                merged.append(text)
        return merged[-50:]

    def _valid_evidence_locked(
        self,
        state: dict[str, Any],
        umo: str,
        evidence_message_ids: list[str],
    ) -> list[str]:
        window = {
            str(item.get("message_id") or "")
            for item in (state.get("recent_messages") or {}).get(umo, [])
            if isinstance(item, dict)
            and not item.get("is_bot")
            and int(item.get("seq") or 0) > 0
            and parse_iso(item.get("created_at"), self.settings.tz) is not None
            and parse_iso(item.get("created_at"), self.settings.tz) <= self.now()
            and parse_iso(item.get("created_at"), self.settings.tz)
            > self.now() - timedelta(minutes=self.settings.interest_ttl_minutes)
        }
        result: list[str] = []
        for item in evidence_message_ids or []:
            text = str(item).strip()
            if text and text in window and text not in result:
                result.append(text)
        return result[-20:]

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def create_checkpoint_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        umo: str,
        platform_id: str,
        group_id: str,
        kind: str,
        scheduled_at: datetime | None = None,
        minutes: int | None = None,
        reason: str = "",
        source: str = SOURCE_ATTENTION_AGENT,
        interest_id: str = "",
        directive_id: str = "",
        moment: datetime | None = None,
    ) -> ControlResult:
        """Create or merge a checkpoint inside one session.

        An interest-bound checkpoint is unique per ``(umo, interest_id)`` while
        unfinished. A session scan is unique per ``umo``.
        """

        moment = moment or self.now()
        if not umo:
            return ControlResult(
                ok=False, code="missing_session", message="检查点必须属于一个会话。"
            )
        self.expire_interests_locked(state, moment)
        if actor.session != umo or not self._actor_can_write_source(actor, source):
            return ControlResult(
                ok=False,
                code="scope_mismatch",
                message="检查点必须使用可信操作者的会话和来源。",
            )
        if source == SOURCE_ATTENTION_AGENT and kind == "session_scan":
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="注意力模型不能创建会话扫描器。",
            )
        if self._owner_mismatch(umo, platform_id, group_id):
            return ControlResult(
                ok=False,
                code="scope_mismatch",
                message="会话、平台与群号不一致，不能创建检查点。",
            )
        if not group_id:
            return ControlResult(
                ok=False,
                code="missing_group",
                message="主动检查目前只支持群聊会话。",
            )
        if not self.policy.group_allowed(group_id, umo, platform_id):
            return ControlResult(
                ok=False,
                code="group_not_allowed",
                message="目标群不在群聊白名单中。",
            )
        kind = str(kind or "").strip()
        if kind not in CHECKPOINT_KINDS:
            return ControlResult(
                ok=False, code="invalid_kind", message="不支持的检查点类型。"
            )
        if kind == "interest":
            interest = (state.get("interests") or {}).get(interest_id)
            if (
                not isinstance(interest, dict)
                or interest.get("umo") != umo
                or interest.get("status") == "closed"
            ):
                return ControlResult(
                    ok=False,
                    code="interest_not_found",
                    message="检查点必须引用本会话中仍然有效的兴趣。",
                )
            expires_at = parse_iso(interest.get("expires_at"), self.settings.tz)
            remaining_checks = int(interest.get("remaining_checks") or 0)
        else:
            interest = None
            interest_id = ""
            expires_at = add_minutes(
                moment, max(1, int(self.settings.interest_ttl_minutes))
            )
            remaining_checks = self.settings.interest_check_limit

        if scheduled_at is None:
            scheduled_at = self._next_schedule(
                moment,
                minutes,
                expires_at,
            )
        else:
            scheduled_at = self._clamp_schedule(moment, scheduled_at, expires_at)
        if expires_at is not None and scheduled_at >= expires_at:
            return ControlResult(
                ok=False,
                code="interest_expired",
                message="检查点计划时间已经晚于资源到期时间。",
            )

        for raw in (state.get("checkpoints") or {}).values():
            if not isinstance(raw, dict) or raw.get("umo") != umo:
                continue
            if raw.get("status") not in {"active", "running", "paused"}:
                continue
            same_interest = kind == "interest" and raw.get("interest_id") == interest_id
            same_scan = kind == "session_scan" and raw.get("kind") == "session_scan"
            if not (same_interest or same_scan):
                continue
            if int(raw.get("priority") or 0) > _priority(source) and not actor.is_admin:
                return ControlResult(
                    ok=False,
                    code="higher_priority_checkpoint_exists",
                    message="A protected checkpoint already exists.",
                )
            raw["scheduled_at"] = to_iso(scheduled_at)
            raw["reason"] = _clip(reason or raw.get("reason"), 200)
            raw["updated_at"] = to_iso(moment)
            raw["revision"] = int(raw.get("revision") or 1) + 1
            return ControlResult(
                ok=True,
                code="checkpoint_merged",
                message="检查点已合并到本会话现有任务。",
                data={"checkpoint": dict(raw), "merged": True},
            )

        checkpoint = {
            "checkpoint_id": _new_id("cp"),
            "umo": umo,
            "platform_id": platform_id,
            "group_id": group_id,
            "interest_id": interest_id,
            "kind": kind,
            "scheduled_at": to_iso(scheduled_at),
            "status": "active",
            "reason": _clip(reason, 200),
            "last_result": "",
            "source": source,
            "priority": _priority(source),
            "directive_id": directive_id,
            "created_by_session": actor.session,
            "expires_at": to_iso(expires_at) if expires_at else "",
            "remaining_checks": max(0, int(remaining_checks)),
            "no_progress_count": int((interest or {}).get("no_progress_count") or 0),
            "revision": 1,
            "created_at": to_iso(moment),
            "updated_at": to_iso(moment),
            "last_run_at": "",
            "last_error": "",
            "run_id": "",
            "attempts": 0,
        }
        state.setdefault("checkpoints", {})[checkpoint["checkpoint_id"]] = checkpoint
        self._record_decision_locked(
            state,
            umo=umo,
            kind="control",
            trigger=actor.source,
            code_decision="checkpoint_saved",
            reason=reason or "checkpoint saved",
            interest_id=interest_id,
            checkpoint_id=checkpoint["checkpoint_id"],
        )
        return ControlResult(
            ok=True,
            code="checkpoint_saved",
            message="检查点已创建。",
            data={"checkpoint": dict(checkpoint), "merged": False},
        )

    def claim_checkpoint_locked(
        self,
        state: dict[str, Any],
        checkpoint_id: str,
        *,
        moment: datetime | None = None,
        force: bool = False,
    ) -> tuple[dict[str, Any] | None, str]:
        """Claim one checkpoint for execution under the store lock."""

        moment = moment or self.now()
        raw = (state.get("checkpoints") or {}).get(checkpoint_id)
        if not isinstance(raw, dict):
            return None, "checkpoint_not_found"
        if raw.get("quarantine"):
            return None, "scope_mismatch"
        if raw.get("status") == "running":
            return None, "checkpoint_running"
        if any(
            isinstance(other, dict)
            and other.get("umo") == raw.get("umo")
            and other.get("status") == "running"
            for other in (state.get("checkpoints") or {}).values()
        ):
            return None, "session_busy"
        if raw.get("status") not in {"active", "paused"}:
            return None, "checkpoint_status"
        umo = str(raw.get("umo") or "")
        if not self.policy.group_allowed(
            str(raw.get("group_id") or ""),
            umo,
            str(raw.get("platform_id") or ""),
        ):
            return None, "group_not_allowed"
        expires_at = parse_iso(raw.get("expires_at"), self.settings.tz)
        if expires_at is None or expires_at <= moment:
            raw["status"] = "cancelled"
            raw["last_result"] = "expired"
            raw["updated_at"] = to_iso(moment)
            raw["revision"] = int(raw.get("revision") or 1) + 1
            return None, "checkpoint_expired"
        if int(raw.get("remaining_checks") or 0) <= 0:
            raw["status"] = "completed"
            raw["last_result"] = "checks_exhausted"
            raw["updated_at"] = to_iso(moment)
            raw["revision"] = int(raw.get("revision") or 1) + 1
            return None, "checks_exhausted"
        if raw.get("kind") == "interest":
            interest = (state.get("interests") or {}).get(raw.get("interest_id"))
            if (
                not isinstance(interest, dict)
                or interest.get("status") != "watching"
                or interest.get("quarantine")
                or interest.get("umo") != umo
            ):
                raw["status"] = "cancelled"
                raw["last_result"] = "interest_closed"
                raw["updated_at"] = to_iso(moment)
                raw["revision"] = int(raw.get("revision") or 1) + 1
                return None, "interest_closed"
        scheduled = parse_iso(raw.get("scheduled_at"), self.settings.tz)
        if not force and scheduled is not None and scheduled > moment:
            return None, "checkpoint_not_due"
        raw["status"] = "running"
        raw["run_id"] = uuid.uuid4().hex[:12]
        raw["last_run_at"] = to_iso(moment)
        raw["attempts"] = int(raw.get("attempts") or 0) + 1
        raw["forced"] = bool(force)
        raw["revision"] = int(raw.get("revision") or 1) + 1
        raw["updated_at"] = to_iso(moment)
        return dict(raw), ""

    def finalize_checkpoint_locked(
        self,
        state: dict[str, Any],
        checkpoint_id: str,
        *,
        status: str,
        scheduled_at: str,
        last_result: str,
        last_error: str,
        moment: datetime,
        evaluated: bool = False,
        no_progress: bool = False,
    ) -> ControlResult:
        """Write the operational result of one checkpoint run."""

        raw = (state.get("checkpoints") or {}).get(checkpoint_id)
        if not isinstance(raw, dict):
            return ControlResult(
                ok=False,
                code="checkpoint_not_found",
                message="检查点不存在。",
            )
        raw["status"] = status
        raw["scheduled_at"] = scheduled_at if status in {"active", "paused"} else ""
        raw["last_result"] = last_result
        raw["last_error"] = _clip(last_error, 300)
        raw["run_id"] = ""
        raw.pop("forced", None)
        raw["updated_at"] = to_iso(moment)
        raw["revision"] = int(raw.get("revision") or 1) + 1
        interest_id = str(raw.get("interest_id") or "")
        interest = (state.get("interests") or {}).get(interest_id)
        if isinstance(interest, dict) and interest.get("umo") == raw.get("umo"):
            raw["remaining_checks"] = int(interest.get("remaining_checks") or 0)
            raw["no_progress_count"] = int(interest.get("no_progress_count") or 0)
            raw["expires_at"] = interest.get("expires_at", "")
        if evaluated:
            raw["remaining_checks"] = max(0, int(raw.get("remaining_checks") or 0) - 1)
        if no_progress:
            raw["no_progress_count"] = int(raw.get("no_progress_count") or 0) + 1
        if isinstance(interest, dict) and interest.get("umo") == raw.get("umo"):
            session = (state.get("sessions") or {}).get(str(raw.get("umo") or ""))
            session_seq = int((session or {}).get("last_decision_seq") or 0)
            if evaluated:
                interest["last_evaluated_seq"] = max(
                    int(interest.get("last_evaluated_seq") or 0),
                    session_seq,
                )
            interest["remaining_checks"] = int(raw.get("remaining_checks") or 0)
            interest["no_progress_count"] = int(raw.get("no_progress_count") or 0)
            interest["updated_at"] = to_iso(moment)
            interest["revision"] = int(interest.get("revision") or 1) + 1
        expires = parse_iso(raw.get("expires_at"), self.settings.tz)
        if (
            int(raw.get("remaining_checks") or 0) <= 0
            or int(raw.get("no_progress_count") or 0) >= self.settings.no_progress_limit
            or (expires is not None and expires <= moment)
        ):
            raw["status"] = "completed"
            raw["scheduled_at"] = ""
            if isinstance(interest, dict) and interest.get("status") != "closed":
                self.close_interest_locked(
                    state,
                    Actor(
                        source=SOURCE_SYSTEM,
                        session=str(raw.get("umo") or ""),
                        is_admin=True,
                    ),
                    interest_id=interest_id,
                    reason="lifecycle_exhausted",
                    moment=moment,
                )
        return ControlResult(
            ok=True,
            code="checkpoint_finalized",
            message="检查点结果已保存。",
            data=dict(raw),
        )

    def set_checkpoint_status_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        checkpoint_id: str,
        status: str,
        *,
        moment: datetime | None = None,
        scheduled_at: str | None = None,
    ) -> ControlResult:
        raw = (state.get("checkpoints") or {}).get(checkpoint_id)
        if not isinstance(raw, dict):
            return ControlResult(
                ok=False,
                code="checkpoint_not_found",
                message="检查点不存在。",
            )
        if raw.get("umo") != actor.session:
            return ControlResult(
                ok=False,
                code="scope_mismatch",
                message="不能修改其他会话的检查点。",
            )
        source = str(raw.get("source") or SOURCE_ATTENTION_AGENT)
        if not actor.is_admin and actor.source != SOURCE_SYSTEM:
            if source == SOURCE_ADMIN:
                return ControlResult(
                    ok=False,
                    code="admin_checkpoint_protected",
                    message="管理员创建的检查点不能被当前操作者修改。",
                )
            if (
                actor.source == SOURCE_ATTENTION_AGENT
                and source != SOURCE_ATTENTION_AGENT
            ):
                return ControlResult(
                    ok=False,
                    code="permission_denied",
                    message="注意力 Agent 不能修改主 Agent 的检查点。",
                )
        moment = moment or self.now()
        raw["status"] = status
        if status == "cancelled":
            raw["scheduled_at"] = ""
        elif scheduled_at is not None:
            raw["scheduled_at"] = scheduled_at
        raw["updated_at"] = to_iso(moment)
        raw["revision"] = int(raw.get("revision") or 1) + 1
        return ControlResult(
            ok=True,
            code="checkpoint_updated",
            message="检查点状态已更新。",
            data=dict(raw),
        )

    # ------------------------------------------------------------------
    # Opportunities and participation admission
    # ------------------------------------------------------------------

    def create_opportunity_locked(
        self,
        state: dict[str, Any],
        *,
        umo: str,
        interest_id: str,
        kind: str,
        anchor_message_id: str,
        evidence_message_ids: list[str],
        expected_contribution: str,
        moment: datetime | None = None,
        valid_ids: set[str] | None = None,
        bot_ids: set[str] | None = None,
        allow_direct: bool = False,
    ) -> ControlResult:
        """Create one pending opportunity or return the existing anchor record."""

        moment = moment or self.now()
        kind = str(kind or "").strip()
        if kind not in OPPORTUNITY_KINDS:
            return ControlResult(
                ok=False,
                code="invalid_opportunity_kind",
                message="不支持的参与机会类型。",
            )
        if kind == "direct_request" and not allow_direct:
            return ControlResult(
                ok=False,
                code="invalid_opportunity_kind",
                message="自主路径不能创建 direct_request 机会。",
            )
        expected_contribution = _clip(expected_contribution, 500)
        if not expected_contribution:
            return ControlResult(
                ok=False,
                code="missing_contribution",
                message="参与机会必须说明预计新增的具体内容。",
            )
        anchor = str(anchor_message_id or "").strip()
        if not anchor:
            return ControlResult(
                ok=False,
                code="missing_anchor",
                message="参与机会必须引用一条锚点消息。",
            )
        if valid_ids is not None and anchor not in valid_ids:
            return ControlResult(
                ok=False,
                code="invalid_anchor",
                message="锚点消息不属于本会话的可用窗口。",
            )
        if bot_ids is not None and anchor in bot_ids:
            return ControlResult(
                ok=False,
                code="bot_anchor",
                message="不能把 Bot 自己的消息当作参与锚点。",
            )
        evidence = [str(item).strip() for item in evidence_message_ids]
        external = set(self._valid_evidence_locked(state, umo, evidence))
        if any(
            not item
            or item not in external
            or (valid_ids is not None and item not in valid_ids)
            or (bot_ids is not None and item in bot_ids)
            for item in evidence
        ):
            return ControlResult(
                ok=False,
                code="invalid_evidence",
                message="证据必须来自本会话实际评估的外部消息。",
            )
        if anchor not in evidence:
            evidence.insert(0, anchor)
        evidence = self._merge_evidence([], evidence)
        interest = None
        if interest_id:
            interest = (state.get("interests") or {}).get(interest_id)
            if (
                not isinstance(interest, dict)
                or interest.get("umo") != umo
                or interest.get("status") != "watching"
                or interest.get("quarantine")
                or int(interest.get("remaining_checks") or 0) <= 0
                or parse_iso(interest.get("expires_at"), self.settings.tz) is None
                or parse_iso(interest.get("expires_at"), self.settings.tz) <= moment
            ):
                return ControlResult(
                    ok=False,
                    code="interest_not_found",
                    message="参与机会必须引用本会话中仍然有效的兴趣。",
                )
        elif kind != "direct_request":
            return ControlResult(
                ok=False,
                code="interest_required",
                message="自主参与机会必须引用一个已准入兴趣。",
            )

        for raw in (state.get("opportunities") or {}).values():
            if not isinstance(raw, dict) or raw.get("umo") != umo:
                continue
            if raw.get("anchor_message_id") != anchor:
                continue
            if raw.get("status") in {"consumed", "uncertain", "expired"}:
                return ControlResult(
                    ok=False,
                    code="opportunity_consumed",
                    message="该锚点已经处理过，不能重复参与。",
                    data={"opportunity": dict(raw)},
                )
            if raw.get("status") in {"pending", "reserved"}:
                return ControlResult(
                    ok=True,
                    code="opportunity_existing",
                    message="该锚点已有参与机会。",
                    data={"opportunity": dict(raw)},
                )

        anchor_created = None
        for item in (state.get("recent_messages") or {}).get(umo, []):
            if isinstance(item, dict) and item.get("message_id") == anchor:
                anchor_created = parse_iso(item.get("created_at"), self.settings.tz)
                break
        if anchor_created is None:
            return ControlResult(
                ok=False, code="invalid_anchor", message="锚点缺少可信消息时间。"
            )
        session = (state.get("sessions") or {}).get(umo, {})
        if anchor_created.timestamp() <= float(session.get("handled_before_ts") or 0):
            return ControlResult(
                ok=False,
                code="opportunity_consumed",
                message="该锚点已被保守去重记录覆盖。",
            )
        if anchor_created > moment:
            return ControlResult(
                ok=False,
                code="future_anchor",
                message="锚点消息时间晚于当前时间，不能作为参与证据。",
            )
        expires_at = add_minutes(
            anchor_created if anchor_created is not None else moment,
            max(1, int(self.settings.opportunity_ttl_minutes)),
        )
        if expires_at <= moment:
            return ControlResult(
                ok=False,
                code="opportunity_expired",
                message="锚点消息已超过参与机会有效期。",
            )
        opportunity = Opportunity(
            opportunity_id=_new_id("op"),
            umo=umo,
            interest_id=interest_id,
            kind=kind,
            anchor_message_id=anchor,
            anchor_created_at=to_iso(anchor_created),
            evidence_message_ids=evidence,
            expected_contribution=expected_contribution,
            status="pending",
            expires_at=to_iso(expires_at),
            created_at=to_iso(moment),
            updated_at=to_iso(moment),
            revision=1,
        ).to_dict()
        state.setdefault("opportunities", {})[opportunity["opportunity_id"]] = (
            opportunity
        )
        return ControlResult(
            ok=True,
            code="opportunity_created",
            message="参与机会已创建。",
            data={"opportunity": dict(opportunity)},
        )

    def approve_participation_locked(
        self,
        state: dict[str, Any],
        *,
        umo: str,
        wake_source: str,
        suggested_action: str,
        opportunity_payload: dict[str, Any] | None,
        moment: datetime,
        valid_ids: set[str] | None = None,
        bot_ids: set[str] | None = None,
        anchor_message_id: str = "",
        interest_id: str = "",
        interest_update: dict[str, Any] | None = None,
        consume_on_approval: bool = False,
    ) -> ParticipationApproval:
        """Unified admission for passive and active autonomous participation.

        Direct-pass wake sources (private chat, wake prefix) pass without
        consuming an autonomous quota. Mentions, replies to the bot and
        autonomous paths must present a valid, unconsumed anchor and pass the
        session/global quota checks in the same critical section.
        """

        direct_pass = wake_source in DIRECT_PASS_WAKE_SOURCES
        action = str(suggested_action or "").strip().lower()
        if direct_pass:
            if action not in {"reply", "pass"}:
                return ParticipationApproval(
                    approved=False,
                    code="explicit_action",
                    message="显式请求走直接放行，不需要兴趣模型批准。",
                    wake_source=wake_source,
                )
            anchor = str(anchor_message_id or "").strip() or str(
                (opportunity_payload or {}).get("anchor_message_id") or ""
            )
            if not anchor:
                anchor = f"explicit_{uuid.uuid4().hex[:12]}"
            existing = next(
                (
                    raw
                    for raw in (state.get("opportunities") or {}).values()
                    if isinstance(raw, dict)
                    and raw.get("umo") == umo
                    and raw.get("anchor_message_id") == anchor
                    and raw.get("status")
                    in {"pending", "reserved", "consumed", "uncertain"}
                ),
                None,
            )
            if existing is not None:
                if existing.get("status") in {"consumed", "uncertain"}:
                    return ParticipationApproval(
                        approved=False,
                        code="duplicate_event",
                        message="该锚点已经处理。",
                    )
                existing["status"] = "consumed"
                existing["outcome"] = "explicit_pass"
                existing["consumed_at"] = to_iso(moment)
                existing["updated_at"] = to_iso(moment)
                existing["revision"] = int(existing.get("revision") or 1) + 1
                return ParticipationApproval(
                    approved=True,
                    code="explicit_pass",
                    message="显式请求允许直接响应。",
                    opportunity=dict(existing),
                    reservation_id=str(existing.get("reservation_id") or ""),
                    interest_id=interest_id,
                    wake_source=wake_source,
                    direct_request=True,
                )
            opportunity = {
                "opportunity_id": _new_id("op"),
                "umo": umo,
                "interest_id": interest_id,
                "kind": "direct_request",
                "anchor_message_id": anchor,
                "anchor_created_at": next(
                    (
                        str(item.get("created_at") or to_iso(moment))
                        for item in (state.get("recent_messages") or {}).get(umo, [])
                        if isinstance(item, dict) and item.get("message_id") == anchor
                    ),
                    to_iso(moment),
                ),
                "evidence_message_ids": self._merge_evidence([], [anchor]),
                "expected_contribution": "explicit user request",
                "status": "consumed",
                "expires_at": to_iso(
                    add_minutes(
                        moment,
                        max(1, int(self.settings.opportunity_ttl_minutes)),
                    )
                ),
                "reservation_id": _new_id("rsv"),
                "reserved_at": to_iso(moment),
                "consumed_at": to_iso(moment),
                "outcome": "explicit_pass",
                "created_at": to_iso(moment),
                "updated_at": to_iso(moment),
                "revision": 1,
            }
            state.setdefault("opportunities", {})[opportunity["opportunity_id"]] = (
                opportunity
            )
            return ParticipationApproval(
                approved=True,
                code="explicit_pass",
                message="显式请求允许直接响应。",
                opportunity=opportunity,
                reservation_id=opportunity["reservation_id"],
                interest_id=interest_id,
                wake_source=wake_source,
                direct_request=True,
            )

        if action not in {"reply", "participate"}:
            return ParticipationApproval(
                approved=False,
                code="autonomous_action",
                message="自主路径只有 reply/participate 才需要参与许可。",
                wake_source=wake_source,
            )
        payload = opportunity_payload if isinstance(opportunity_payload, dict) else {}
        result = self.create_opportunity_locked(
            state,
            umo=umo,
            interest_id=str(payload.get("interest_id") or interest_id or ""),
            kind=str(payload.get("kind") or "natural_opening"),
            anchor_message_id=str(
                payload.get("anchor_message_id") or anchor_message_id or ""
            ),
            evidence_message_ids=list(payload.get("evidence_message_ids") or []),
            expected_contribution=str(payload.get("expected_contribution") or ""),
            moment=moment,
            valid_ids=valid_ids,
            bot_ids=bot_ids,
        )
        if not result.ok:
            return ParticipationApproval(
                approved=False,
                code=result.code,
                message=result.message,
                wake_source=wake_source,
            )
        opportunity = dict(result.data.get("opportunity") or {})
        if opportunity.get("status") == "reserved":
            return ParticipationApproval(
                approved=False,
                code="opportunity_pending",
                message="该参与机会已被其他路径预留。",
                opportunity=opportunity,
                wake_source=wake_source,
            )
        if opportunity.get("status") == "consumed":
            return ParticipationApproval(
                approved=False,
                code="opportunity_consumed",
                message="该参与机会已经被消费。",
                opportunity=opportunity,
                wake_source=wake_source,
            )
        allowed, reason = self.policy.can_reserve_autonomous(state, umo, moment)
        if not allowed:
            pending = state.get("opportunities", {}).get(
                opportunity.get("opportunity_id")
            )
            if isinstance(pending, dict) and pending.get("status") == "pending":
                pending["status"] = "expired"
                pending["outcome"] = reason
                pending["updated_at"] = to_iso(moment)
                pending["revision"] = int(pending.get("revision") or 1) + 1
            return ParticipationApproval(
                approved=False,
                code=reason,
                message="本会话或全局自主参与额度不足。",
                opportunity=opportunity,
                wake_source=wake_source,
            )
        reservation_id = _new_id("rsv")
        opportunity["status"] = "reserved"
        opportunity["reservation_id"] = reservation_id
        opportunity["reserved_at"] = to_iso(moment)
        opportunity["updated_at"] = to_iso(moment)
        opportunity["revision"] = int(opportunity.get("revision") or 1) + 1
        state["opportunities"][opportunity["opportunity_id"]] = opportunity
        self.policy.reserve_autonomous(state, umo, moment)
        session = self.policy.ensure_session(state, umo, moment)
        if consume_on_approval:
            opportunity["status"] = "consumed"
            opportunity["outcome"] = "approved_pass"
            opportunity["consumed_at"] = to_iso(moment)
            opportunity["updated_at"] = to_iso(moment)
            opportunity["revision"] = int(opportunity.get("revision") or 1) + 1
            state["opportunities"][opportunity["opportunity_id"]] = opportunity
            session["pending_reply_reservation"] = ""
        else:
            session["pending_reply_reservation"] = reservation_id
        session["revision"] = int(session.get("revision") or 1) + 1
        return ParticipationApproval(
            approved=True,
            code="autonomous_approved",
            message="参与机会已预留并获批。",
            opportunity=opportunity,
            reservation_id=reservation_id,
            interest_id=str(opportunity.get("interest_id") or ""),
            wake_source=wake_source,
            reason=str((interest_update or {}).get("summary") or ""),
        )

    def complete_participation_locked(
        self,
        state: dict[str, Any],
        *,
        umo: str,
        reservation_id: str,
        outcome: str,
        moment: datetime | None = None,
    ) -> None:
        """Consume or mark uncertain one reserved participation."""

        if not reservation_id:
            return
        moment = moment or self.now()
        status = (
            "uncertain"
            if outcome in {"uncertain_delivery", "restart_uncertain"}
            else "consumed"
        )
        for raw in (state.get("opportunities") or {}).values():
            if not isinstance(raw, dict) or raw.get("umo") != umo:
                continue
            if raw.get("reservation_id") != reservation_id:
                continue
            raw["status"] = status
            raw["outcome"] = outcome
            if status == "consumed":
                raw["consumed_at"] = to_iso(moment)
            raw["updated_at"] = to_iso(moment)
            raw["revision"] = int(raw.get("revision") or 1) + 1
            break
        session = self.policy.ensure_session(state, umo, moment)
        if session.get("pending_reply_reservation") == reservation_id:
            session["pending_reply_reservation"] = ""
            session["revision"] = int(session.get("revision") or 1) + 1

    # ------------------------------------------------------------------
    # Directives (focus / stop / schedule)
    # ------------------------------------------------------------------

    def set_intent_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        intent_type: str,
        target: str = "",
        reason: str = "",
        expires_in_minutes: int | None = None,
        minutes: int | None = None,
        interest_id: str = "",
    ) -> ControlResult:
        intent_type = str(intent_type or "").strip().lower()
        if intent_type not in DIRECTIVE_TYPES:
            return ControlResult(
                ok=False,
                code="invalid_intent",
                message=f"不支持的意图类型：{intent_type or '-'}",
            )
        if not actor.session:
            return ControlResult(
                ok=False, code="missing_session", message="缺少可信会话标识。"
            )
        if not self._actor_can_write_source(actor, SOURCE_MAIN_AGENT):
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="当前操作者不能写入主 Agent 意图。",
            )
        if actor.group_id and not self.policy.group_allowed(
            actor.group_id,
            actor.session,
            actor.platform_id,
        ):
            return ControlResult(
                ok=False,
                code="group_not_allowed",
                message="当前群不在 group_allowlist 中。",
            )
        if interest_id:
            selected = (state.get("interests") or {}).get(interest_id)
            if not isinstance(selected, dict) or selected.get("umo") != actor.session:
                return ControlResult(
                    ok=False,
                    code="scope_mismatch",
                    message="Interest belongs to another session.",
                )
        if intent_type == "focus_topic" and target in (state.get("interests") or {}):
            selected = state["interests"][target]
            if not isinstance(selected, dict) or selected.get("umo") != actor.session:
                return ControlResult(
                    ok=False,
                    code="scope_mismatch",
                    message="Interest belongs to another session.",
                )
            interest_id, target = target, str(selected.get("summary") or "")
        if intent_type == "ignore_user":
            return self.ignore_user_locked(
                state,
                actor,
                target_user=target,
                reason=reason,
                expires_in_minutes=expires_in_minutes,
            )
        if intent_type == "stop_focus":
            return self.stop_focus_locked(state, actor, target=target)
        if intent_type == "schedule_check":
            return self._schedule_check_locked(
                state,
                actor,
                target=target,
                reason=reason,
                minutes=minutes if minutes is not None else expires_in_minutes,
                interest_id=interest_id,
            )
        if intent_type == "focus_group":
            if not actor.is_admin:
                return ControlResult(
                    ok=False,
                    code="permission_denied",
                    message="群级关注意图需要管理员权限；普通成员可以使用 focus_topic。",
                )
            target_value = actor.session
        else:
            target_value = _clip(target, 200)
            if not target_value:
                return ControlResult(
                    ok=False,
                    code="missing_target",
                    message="focus_topic 需要提供话题关键字。",
                )
        expires_at = self._resolve_expiry(
            actor,
            max_minutes=self.settings.main_agent_max_intent_minutes,
            requested_minutes=expires_in_minutes,
            allow_permanent=False,
            default_minutes=self.settings.interest_ttl_minutes,
        )
        if expires_at is None:
            expires_at = add_minutes(
                self.now(), max(1, int(self.settings.interest_ttl_minutes))
            )
        interest = None
        if intent_type == "focus_topic":
            interest_result = self.admit_interest_locked(
                state,
                actor,
                summary=target_value,
                wait_for="Wait for relevant external progress",
                evidence_message_ids=self._latest_external_ids_locked(
                    state, actor.session, limit=3
                ),
                source=SOURCE_MAIN_AGENT,
                expires_in_minutes=expires_in_minutes,
                interest_id=interest_id,
                priority_level=2,
            )
            if not interest_result.ok:
                return interest_result
            interest = interest_result.data["interest"]
        directive = self._upsert_directive_locked(
            state,
            actor,
            directive_type=intent_type,
            target=target_value,
            reason=reason or intent_type,
            expires_at=expires_at,
        )
        data: dict[str, Any] = {"directive": directive, "umo": actor.session}
        if interest is not None:
            data["interest"] = interest
            directive["interest_id"] = interest["interest_id"]
            state["interests"][interest["interest_id"]]["directive_id"] = directive[
                "directive_id"
            ]
            if actor.group_id:
                self.create_checkpoint_locked(
                    state,
                    actor,
                    umo=actor.session,
                    platform_id=actor.platform_id,
                    group_id=actor.group_id,
                    kind="interest",
                    interest_id=interest["interest_id"],
                    source=SOURCE_MAIN_AGENT,
                    directive_id=directive["directive_id"],
                    reason=reason or "focus_topic",
                )
        return ControlResult(
            ok=True,
            code="intent_saved",
            message="主 Agent 意图已保存。",
            data=data,
        )

    def _schedule_check_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        target: str = "",
        reason: str = "",
        minutes: int | None = None,
        interest_id: str = "",
    ) -> ControlResult:
        if not actor.group_id:
            return ControlResult(
                ok=False,
                code="missing_group",
                message="schedule_check 需要在群聊中调用。",
            )
        interest = (state.get("interests") or {}).get(interest_id)
        if interest_id and (
            not isinstance(interest, dict) or interest.get("umo") != actor.session
        ):
            return ControlResult(
                ok=False,
                code="interest_not_found",
                message="当前会话不存在该兴趣。",
            )
        kind = "interest" if isinstance(interest, dict) else "session_scan"
        expires_at = self._resolve_expiry(
            actor,
            max_minutes=self.settings.main_agent_max_intent_minutes,
            requested_minutes=expires_in_minutes_from_minutes(minutes),
            allow_permanent=False,
            default_minutes=self.settings.interest_ttl_minutes,
        )
        directive = self._upsert_directive_locked(
            state,
            actor,
            directive_type="schedule_check",
            target=actor.session,
            reason=reason or "main agent schedule_check",
            expires_at=expires_at,
        )
        result = self.create_checkpoint_locked(
            state,
            actor,
            umo=actor.session,
            platform_id=actor.platform_id,
            group_id=actor.group_id,
            kind=kind,
            minutes=minutes,
            reason=reason or "main agent schedule_check",
            source=SOURCE_MAIN_AGENT,
            interest_id=interest_id if kind == "interest" else "",
            directive_id=str(directive.get("directive_id") or ""),
        )
        if not result.ok:
            directive["status"] = "revoked"
            directive["updated_at"] = to_iso(self.now())
            directive["revision"] = int(directive.get("revision") or 1) + 1
            return result
        result.data["directive"] = directive
        return result

    def stop_focus_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        target: str = "",
        moment: datetime | None = None,
    ) -> ControlResult:
        if not self._actor_can_write_source(actor, SOURCE_MAIN_AGENT):
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="当前操作者不能停止关注目标。",
            )
        moment = moment or self.now()
        target_value = str(target or "").strip()
        if target_value in {"*", "all"} and not actor.is_admin:
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="只有管理员可以停止全部会话的关注。",
            )
        for resources in (state.get("interests") or {}, state.get("checkpoints") or {}):
            resource = resources.get(target_value)
            if isinstance(resource, dict) and resource.get("umo") != actor.session:
                return ControlResult(
                    ok=False,
                    code="scope_mismatch",
                    message="Target belongs to another session.",
                )
        umos = (
            list(state.get("sessions") or {})
            if target_value in {"*", "all"} and actor.is_admin
            else [actor.session]
        )
        specific = target_value not in {"", "*", "all", actor.session, actor.group_id}
        selected = {
            str(item.get("interest_id") or "")
            for item in (state.get("interests") or {}).values()
            if isinstance(item, dict)
            and item.get("umo") in umos
            and (
                not specific
                or target_value == item.get("interest_id")
                or target_value.casefold() in str(item.get("summary") or "").casefold()
            )
            and (actor.is_admin or item.get("source") != SOURCE_ADMIN)
        }
        revoked: list[str] = []
        cancelled: list[str] = []
        closed: list[str] = []
        for interest_id in selected:
            interest = state["interests"][interest_id]
            if interest.get("status") == "closed":
                continue
            result = self.close_interest_locked(
                state,
                Actor(
                    source=actor.source,
                    session=str(interest.get("umo") or ""),
                    is_admin=actor.is_admin,
                ),
                interest_id=interest_id,
                reason="stop_focus",
                moment=moment,
            )
            if result.ok:
                closed.append(interest_id)
                cancelled.extend(result.data.get("cancelled_checkpoints") or [])
        for directive in (state.get("directives") or {}).values():
            if not isinstance(directive, dict) or directive.get("status") != "active":
                continue
            if directive.get("created_by_session") not in umos:
                continue
            if not actor.is_admin and directive.get("source") == SOURCE_ADMIN:
                continue
            if directive.get("type") not in {
                "focus_group",
                "focus_topic",
                "schedule_check",
            }:
                continue
            if (
                specific
                and directive.get("interest_id") not in selected
                and directive.get("target") != target_value
            ):
                continue
            directive["status"] = "revoked"
            directive["updated_at"] = to_iso(moment)
            directive["revision"] = int(directive.get("revision") or 1) + 1
            revoked.append(str(directive.get("directive_id") or ""))
        for checkpoint in (state.get("checkpoints") or {}).values():
            if not isinstance(checkpoint, dict) or checkpoint.get("umo") not in umos:
                continue
            if checkpoint.get("status") not in {"active", "paused", "running"}:
                continue
            if (
                specific
                and checkpoint.get("interest_id") not in selected
                and checkpoint.get("checkpoint_id") != target_value
            ):
                continue
            if not actor.is_admin and checkpoint.get("source") == SOURCE_ADMIN:
                continue
            checkpoint.update(
                status="cancelled",
                scheduled_at="",
                run_id="",
                last_result="stop_focus",
                updated_at=to_iso(moment),
                revision=int(checkpoint.get("revision") or 1) + 1,
            )
            cancelled.append(str(checkpoint.get("checkpoint_id") or ""))
        if not specific:
            for umo in umos:
                self._upsert_directive_locked(
                    state,
                    Actor(source=actor.source, session=umo, is_admin=actor.is_admin),
                    directive_type="stop_focus",
                    target=umo,
                    reason="explicit stop_focus",
                    expires_at=add_minutes(
                        moment, self.settings.main_agent_max_intent_minutes
                    ),
                )
        if not actor.is_admin:
            self._record_decision_locked(
                state,
                umo=actor.session,
                kind="control",
                trigger=actor.source,
                code_decision="focus_stopped",
                reason=f"revoked={len(revoked)} closed={len(closed)} "
                f"cancelled={len(cancelled)}",
            )
        return ControlResult(
            ok=True,
            code="focus_stopped"
            if (revoked or closed or cancelled)
            else "nothing_to_stop",
            message=(
                "已停止当前会话中由主 Agent 发起的关注。"
                if (revoked or closed or cancelled)
                else "当前会话没有可停止的主 Agent 意图。"
            ),
            data={
                "revoked_directives": revoked,
                "closed_interests": closed,
                "cancelled_checkpoints": cancelled,
            },
        )

    # ------------------------------------------------------------------
    # Policy writes (also used by engine / admin commands)
    # ------------------------------------------------------------------

    def ignore_user_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        target_user: str,
        reason: str = "",
        expires_in_minutes: int | None = None,
        scope: str = "group_user",
    ) -> ControlResult:
        target_user = str(target_user or "").strip()
        if not target_user:
            return ControlResult(
                ok=False,
                code="missing_target_user",
                message="需要提供要忽略的用户 ID。",
            )
        scope = str(scope or "group_user").strip()
        if scope not in {"group_user", "user"}:
            return ControlResult(
                ok=False,
                code="invalid_scope",
                message="ignore_user 仅支持 group_user 或 user 作用域。",
            )
        if scope == "user" and not actor.is_admin:
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="只有管理员可以设置全局用户忽略策略。",
            )
        if scope == "group_user" and not actor.group_id:
            return ControlResult(
                ok=False,
                code="missing_group",
                message="group_user 作用域需要在群聊中调用。",
            )
        if not actor.is_admin and actor.group_id:
            scope = "group_user"
        if actor.group_id and not self.policy.group_allowed(
            actor.group_id,
            actor.session,
            actor.platform_id,
        ):
            return ControlResult(
                ok=False,
                code="group_not_allowed",
                message="当前群不在 group_allowlist 中。",
            )
        expires_at = self._resolve_expiry(
            actor,
            max_minutes=self.settings.main_agent_max_policy_hours * 60,
            requested_minutes=expires_in_minutes,
            allow_permanent=actor.is_admin,
            default_minutes=self.settings.main_agent_max_policy_hours * 60,
        )
        policy_result = self.set_policy_locked(
            state,
            actor,
            scope=scope,
            group_id=actor.group_id if scope == "group_user" else "",
            user_id=target_user,
            reason=reason or "main agent ignore request",
            expires_at=expires_at,
            source=SOURCE_MAIN_AGENT,
            created_by_session=actor.session,
            allow_permanent=actor.is_admin,
        )
        if not policy_result.ok:
            return policy_result
        directive = self._upsert_directive_locked(
            state,
            actor,
            directive_type="ignore_user",
            target=target_user,
            reason=reason or "main agent ignore request",
            expires_at=expires_at,
        )
        policy_id = str(policy_result.data.get("policy_id") or "")
        if policy_id:
            raw_policy = (state.get("policies") or {}).get(policy_id)
            if isinstance(raw_policy, dict):
                raw_policy["directive_id"] = str(directive.get("directive_id") or "")
        return ControlResult(
            ok=True,
            code="user_ignored",
            message="用户忽略策略已生效，从后续消息开始生效。",
            data={"policy": policy_result.data, "directive": directive},
        )

    def clear_user_policy_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        target_user: str = "",
        policy_id: str = "",
    ) -> ControlResult:
        policies = state.get("policies") or {}
        policy: dict[str, Any] | None = None
        if policy_id:
            raw = policies.get(policy_id)
            if isinstance(raw, dict):
                policy = raw
        elif target_user:
            for raw in policies.values():
                if not isinstance(raw, dict):
                    continue
                if (
                    raw.get("user_id") == target_user
                    and raw.get("mode") == "ignore"
                    and (
                        raw.get("group_id") == actor.group_id
                        or raw.get("scope") == "user"
                    )
                ):
                    if policy is None or int(raw.get("priority") or 0) > int(
                        policy.get("priority") or 0
                    ):
                        policy = raw
        if policy is None:
            return ControlResult(
                ok=False,
                code="policy_not_found",
                message="没有找到匹配的用户策略。",
            )
        if not actor.is_admin and str(policy.get("user_id") or "") != actor.user_id:
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="普通成员只能解除针对自己的用户策略。",
            )
        removed = self.remove_policy_locked(state, actor, policy)
        if not removed.ok:
            return removed
        directive_target = str(policy.get("user_id") or "")
        if directive_target:
            for raw in state.get("directives", {}).values():
                if not isinstance(raw, dict):
                    continue
                if (
                    raw.get("type") == "ignore_user"
                    and raw.get("target") == directive_target
                    and raw.get("status") == "active"
                    and raw.get("source") == SOURCE_MAIN_AGENT
                    and (
                        actor.is_admin or raw.get("created_by_session") == actor.session
                    )
                ):
                    raw["status"] = "revoked"
                    raw["updated_at"] = to_iso(self.now())
                    raw["revision"] = int(raw.get("revision") or 1) + 1
        return ControlResult(
            ok=True,
            code="policy_cleared",
            message="用户策略已解除。",
            data={"policy_id": policy.get("policy_id", "")},
        )

    def set_policy_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        scope: str,
        group_id: str,
        user_id: str,
        reason: str,
        expires_at: datetime | None,
        source: str,
        created_by_session: str = "",
        allow_permanent: bool = False,
        directive_id: str = "",
    ) -> ControlResult:
        if not self._actor_can_write_source(actor, source):
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="当前操作者不能写入该优先级的用户策略。",
            )
        scope = str(scope or "group_user")
        if scope not in {"group_user", "user", "group"}:
            return ControlResult(
                ok=False,
                code="invalid_scope",
                message="不支持的用户策略作用域。",
            )
        if scope in {"group_user", "group"} and not group_id:
            return ControlResult(
                ok=False,
                code="missing_group",
                message="该作用域需要提供群聊。",
            )
        if scope in {"group_user", "user"} and not user_id:
            return ControlResult(
                ok=False,
                code="missing_user",
                message="该作用域需要提供用户。",
            )
        if not actor.is_admin and scope == "user":
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="只有管理员可以写入全局用户策略。",
            )
        moment = self.now()
        expires_at = self._clamp_policy_expiry(
            actor,
            source,
            expires_at,
            allow_permanent=allow_permanent,
        )
        priority = _priority(source)
        policy_id = self.policy.policy_id(scope, group_id, user_id)
        existing = (state.get("policies") or {}).get(policy_id)
        if isinstance(existing, dict):
            existing_priority = int(
                existing.get("priority") or _priority(str(existing.get("source") or ""))
            )
            if existing_priority > priority and not actor.is_admin:
                return ControlResult(
                    ok=False,
                    code="higher_priority_policy_exists",
                    message="存在更高优先级的策略，当前请求不能覆盖。",
                    data={
                        "existing_source": existing.get("source", ""),
                        "existing_priority": existing_priority,
                    },
                )
            if (
                existing.get("source") == SOURCE_ADMIN
                and source != SOURCE_ADMIN
                and not actor.is_admin
            ):
                return ControlResult(
                    ok=False,
                    code="admin_policy_exists",
                    message="存在管理员策略，主 Agent/注意力 Agent 不能覆盖。",
                )
        policy = UserPolicy(
            policy_id=policy_id,
            scope=scope,
            group_id=group_id,
            user_id=user_id,
            mode="ignore",
            reason=_clip(reason, 120),
            expires_at=to_iso(expires_at) if expires_at else "",
            created_by=source,
            override="owner" if source == SOURCE_ADMIN else "",
            created_at=(
                str(existing.get("created_at") or to_iso(moment))
                if isinstance(existing, dict)
                else to_iso(moment)
            ),
            updated_at=to_iso(moment),
        )
        raw = policy.to_dict()
        raw.update(
            {
                "source": source,
                "priority": priority,
                "created_by_session": created_by_session or actor.session,
                "directive_id": directive_id,
                "revision": int((existing or {}).get("revision") or 0) + 1,
            }
        )
        state.setdefault("policies", {})[policy_id] = raw
        self._record_decision_locked(
            state,
            umo=actor.session,
            kind="control",
            trigger=actor.source,
            code_decision="policy_saved",
            reason=reason,
            related_policy_id=policy_id,
            related_directive_id=directive_id,
        )
        return ControlResult(
            ok=True,
            code="policy_saved",
            message="用户策略已生效。",
            data={
                "policy_id": policy_id,
                "scope": scope,
                "user_id": user_id,
                "group_id": group_id,
                "source": source,
                "priority": priority,
                "expires_at": raw["expires_at"],
                "directive_id": directive_id,
            },
        )

    def remove_policy_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        policy: dict[str, Any],
    ) -> ControlResult:
        policy_id = str(policy.get("policy_id") or "")
        source = str(policy.get("source") or "")
        if not actor.is_admin:
            if source == SOURCE_ADMIN:
                return ControlResult(
                    ok=False,
                    code="admin_policy_protected",
                    message="管理员策略不能被解除。",
                )
            if (
                actor.source == SOURCE_ATTENTION_AGENT
                and source != SOURCE_ATTENTION_AGENT
            ):
                return ControlResult(
                    ok=False,
                    code="permission_denied",
                    message="注意力 Agent 只能撤销自己的临时策略。",
                )
            if actor.source == SOURCE_MAIN_AGENT and source == SOURCE_MAIN_AGENT:
                if policy.get("created_by_session") not in {"", actor.session}:
                    return ControlResult(
                        ok=False,
                        code="permission_denied",
                        message="只能撤销当前会话创建的主 Agent 策略。",
                    )
        state.setdefault("policies", {}).pop(policy_id, None)
        self._record_decision_locked(
            state,
            umo=actor.session,
            kind="control",
            trigger=actor.source,
            code_decision="policy_removed",
            reason=str(policy.get("reason") or "policy removed"),
            related_policy_id=policy_id,
        )
        return ControlResult(
            ok=True,
            code="policy_removed",
            message="策略已解除。",
            data={"policy_id": policy_id},
        )

    # ------------------------------------------------------------------
    # Expiry helpers
    # ------------------------------------------------------------------

    def expire_interests_locked(
        self,
        state: dict[str, Any],
        moment: datetime | None = None,
    ) -> int:
        """Close interests whose TTL elapsed and expire their resources."""

        moment = moment or self.now()
        expired = 0
        for raw in (state.get("interests") or {}).values():
            if not isinstance(raw, dict):
                continue
            if raw.get("status") == "closed":
                continue
            expires_at = parse_iso(raw.get("expires_at"), self.settings.tz)
            if expires_at is not None and expires_at <= moment:
                raw["status"] = "closed"
                raw["updated_at"] = to_iso(moment)
                raw["revision"] = int(raw.get("revision") or 1) + 1
                raw["close_reason"] = "expired"
                expired += 1
                for checkpoint in (state.get("checkpoints") or {}).values():
                    if (
                        isinstance(checkpoint, dict)
                        and checkpoint.get("umo") == raw.get("umo")
                        and checkpoint.get("interest_id") == raw.get("interest_id")
                        and checkpoint.get("status") in {"active", "paused", "running"}
                    ):
                        checkpoint["status"] = "cancelled"
                        checkpoint["scheduled_at"] = ""
                        checkpoint["last_result"] = "interest_expired"
                        checkpoint["updated_at"] = to_iso(moment)
                for opportunity in (state.get("opportunities") or {}).values():
                    if (
                        isinstance(opportunity, dict)
                        and opportunity.get("umo") == raw.get("umo")
                        and opportunity.get("interest_id") == raw.get("interest_id")
                        and opportunity.get("status") == "pending"
                    ):
                        opportunity["status"] = "expired"
                        opportunity["outcome"] = "interest_expired"
                        opportunity["updated_at"] = to_iso(moment)
        for checkpoint in (state.get("checkpoints") or {}).values():
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("kind") == "session_scan"
                and checkpoint.get("status") in {"active", "running", "paused"}
                and not checkpoint.get("quarantine")
            ):
                deadline = parse_iso(checkpoint.get("expires_at"), self.settings.tz)
                if deadline is None or deadline <= moment:
                    checkpoint.update(
                        status="cancelled",
                        scheduled_at="",
                        run_id="",
                        last_result="checkpoint_expired",
                        updated_at=to_iso(moment),
                    )
        for opportunity in (state.get("opportunities") or {}).values():
            if not isinstance(opportunity, dict):
                continue
            if opportunity.get("status") not in {"pending", "reserved"}:
                continue
            expires_at = parse_iso(opportunity.get("expires_at"), self.settings.tz)
            if expires_at is not None and expires_at <= moment:
                opportunity["status"] = (
                    "uncertain"
                    if opportunity.get("status") == "reserved"
                    else "expired"
                )
                opportunity["outcome"] = (
                    "reservation_expired"
                    if opportunity.get("status") == "uncertain"
                    else "ttl"
                )
                opportunity["updated_at"] = to_iso(moment)
                opportunity["revision"] = int(opportunity.get("revision") or 1) + 1
                session = (state.get("sessions") or {}).get(
                    str(opportunity.get("umo") or ""), {}
                )
                if session.get("pending_reply_reservation") == opportunity.get(
                    "reservation_id"
                ):
                    session["pending_reply_reservation"] = ""

        return expired

    def cleanup_locked(
        self,
        state: dict[str, Any],
        moment: datetime | None = None,
        *,
        retention_days: int = 7,
        max_terminal_opportunities: int = 500,
    ) -> dict[str, int]:
        """Remove old terminal resources while keeping recent dedupe tombstones.

        Args:
            state: Mutable state dictionary.
            moment: Current time, injectable for tests.
            retention_days: Age after which closed/terminal records are removed.
            max_terminal_opportunities: Maximum terminal opportunities to keep.

        Returns:
            Counts of removed interests, checkpoints and opportunities.
        """

        moment = moment or self.now()
        cutoff = moment - timedelta(days=max(1, int(retention_days)))
        removed = {"interests": 0, "checkpoints": 0, "opportunities": 0}

        for key, raw in list((state.get("interests") or {}).items()):
            if not isinstance(raw, dict) or raw.get("status") != "closed":
                continue
            updated = parse_iso(raw.get("updated_at"), self.settings.tz)
            if updated is not None and updated <= cutoff:
                state["interests"].pop(key, None)
                removed["interests"] += 1

        for key, raw in list((state.get("checkpoints") or {}).items()):
            if not isinstance(raw, dict):
                continue
            if raw.get("status") in {"active", "running", "paused"}:
                continue
            updated = parse_iso(raw.get("updated_at"), self.settings.tz)
            if updated is not None and updated <= cutoff:
                state["checkpoints"].pop(key, None)
                removed["checkpoints"] += 1

        for container, terminal_statuses in (
            ("interests", {"closed"}),
            ("checkpoints", {"completed", "cancelled"}),
        ):
            terminal_records = sorted(
                (
                    item
                    for item in state.get(container, {}).values()
                    if isinstance(item, dict)
                    and item.get("status") in terminal_statuses
                    and not item.get("quarantine")
                ),
                key=lambda item: str(item.get("updated_at") or ""),
            )
            id_field = "interest_id" if container == "interests" else "checkpoint_id"
            for item in terminal_records[:-500]:
                if (
                    state[container].pop(str(item.get(id_field) or ""), None)
                    is not None
                ):
                    removed[container] += 1
        terminal = [
            raw
            for raw in (state.get("opportunities") or {}).values()
            if isinstance(raw, dict)
            and raw.get("status") in {"consumed", "expired", "uncertain"}
        ]
        terminal.sort(key=lambda item: str(item.get("updated_at") or ""))
        keep = max(50, int(max_terminal_opportunities))
        removable = terminal[:-keep] if len(terminal) > keep else []
        for raw in terminal:
            updated = parse_iso(raw.get("updated_at"), self.settings.tz)
            if raw in removable or (updated is not None and updated <= cutoff):
                key = str(raw.get("opportunity_id") or "")
                if state["opportunities"].pop(key, None) is not None:
                    session = self.policy.ensure_session(
                        state, str(raw.get("umo") or ""), moment
                    )
                    anchor_time = parse_iso(
                        raw.get("anchor_created_at") or raw.get("created_at"),
                        self.settings.tz,
                    )
                    if anchor_time is not None:
                        session["handled_before_ts"] = max(
                            float(session.get("handled_before_ts") or 0),
                            anchor_time.timestamp(),
                        )
                    removed["opportunities"] += 1
        return removed

    def expire_directives_locked(
        self,
        state: dict[str, Any],
        moment: datetime | None = None,
    ) -> int:
        moment = moment or self.now()
        expired = 0
        for raw in (state.get("directives") or {}).values():
            if not isinstance(raw, dict) or raw.get("status") != "active":
                continue
            expires_at = parse_iso(raw.get("expires_at"), self.settings.tz)
            if expires_at is not None and expires_at <= moment:
                raw["status"] = "expired"
                raw["updated_at"] = to_iso(moment)
                raw["revision"] = int(raw.get("revision") or 1) + 1
                expired += 1
        return expired

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def focus_directive(
        self,
        state: dict[str, Any],
        *,
        umo: str,
        message_text: str = "",
        message_id: str = "",
    ) -> dict[str, Any] | None:
        """Return the strongest focus directive owned by ``umo``."""

        self.expire_directives_locked(state)
        lowered = (message_text or "").casefold()
        candidates: list[dict[str, Any]] = []
        for raw in (state.get("directives") or {}).values():
            if not isinstance(raw, dict) or raw.get("status") != "active":
                continue
            if raw.get("created_by_session") != umo:
                continue
            dtype = raw.get("type")
            if dtype == "focus_group":
                candidates.append(raw)
            elif dtype == "focus_topic":
                target = str(raw.get("target") or "").strip()
                if not target:
                    continue
                if message_id and target == message_id:
                    candidates.append(raw)
                elif target.casefold() in lowered:
                    candidates.append(raw)
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                -int(item.get("priority") or 0),
                str(item.get("updated_at") or ""),
            )
        )
        return dict(candidates[0])

    def stop_directive(
        self,
        state: dict[str, Any],
        *,
        umo: str,
    ) -> dict[str, Any] | None:
        self.expire_directives_locked(state)
        candidates: list[dict[str, Any]] = []
        for raw in (state.get("directives") or {}).values():
            if not isinstance(raw, dict) or raw.get("status") != "active":
                continue
            if raw.get("type") != "stop_focus":
                continue
            if raw.get("created_by_session") != umo:
                continue
            candidates.append(raw)
        if not candidates:
            return None
        candidates.sort(key=lambda item: -int(item.get("priority") or 0))
        return dict(candidates[0])

    def prompt_summary_locked(
        self,
        state: dict[str, Any],
        *,
        umo: str,
        user_id: str = "",
        group_id: str = "",
        platform_id: str = "",
        message_text: str = "",
        message_id: str = "",
    ) -> str:
        """Return a compact, session-scoped summary for the interest model."""

        self.expire_interests_locked(state)
        lines: list[str] = []
        policy = self.policy.matching_policy(
            state,
            group_id,
            user_id,
            umo,
            platform_id,
        )
        if policy is not None:
            lines.append(
                "有效忽略策略：来源 {source}，到期 {expires}，原因：{reason}".format(
                    source=policy.get("source") or "-",
                    expires=policy.get("expires_at") or "永久",
                    reason=_clip(policy.get("reason"), 100),
                )
            )
        interests = self.session_interests_locked(state, umo)
        if interests:
            lines.append("本会话兴趣：")
            for item in interests[:5]:
                lines.append(
                    "- {id} [{status}] {summary}；等待：{wait}；剩余检查 {checks}，"
                    "无进展 {no_progress}".format(
                        id=item.get("interest_id"),
                        status=item.get("status"),
                        summary=_clip(item.get("summary"), 120),
                        wait=_clip(item.get("wait_for"), 80) or "（无）",
                        checks=item.get("remaining_checks", 0),
                        no_progress=item.get("no_progress_count", 0),
                    )
                )
        focus = self.focus_directive(
            state,
            umo=umo,
            message_text=message_text,
            message_id=message_id,
        )
        if focus:
            lines.append(
                "主 Agent 关注意图：{type} {target}，到期 {expires}".format(
                    type=focus.get("type"),
                    target=_clip(focus.get("target"), 100),
                    expires=focus.get("expires_at") or "-",
                )
            )
        stop = self.stop_directive(state, umo=umo)
        if stop:
            lines.append("主 Agent 停止关注意图：管理员已要求停止关注。")
        if not lines:
            return "当前会话没有主 Agent 意图、兴趣或用户忽略策略。"
        lines.append("兴趣模型不得覆盖上述主 Agent 意图和管理员策略。")
        return "\n".join(lines[:10])

    # ------------------------------------------------------------------
    # Directives internals
    # ------------------------------------------------------------------

    def _upsert_directive_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        directive_type: str,
        target: str,
        reason: str,
        expires_at: datetime | None,
    ) -> dict[str, Any]:
        moment = self.now()
        source = (
            SOURCE_ADMIN
            if actor.is_admin and directive_type == "stop_focus"
            else SOURCE_MAIN_AGENT
        )
        if directive_type != "stop_focus":
            for raw in (state.get("directives") or {}).values():
                if not isinstance(raw, dict):
                    continue
                if raw.get("status") != "active" or raw.get("type") != "stop_focus":
                    continue
                if raw.get("created_by_session") != actor.session:
                    continue
                if raw.get("source") == SOURCE_ADMIN and not actor.is_admin:
                    continue
                raw["status"] = "revoked"
                raw["updated_at"] = to_iso(moment)
                raw["revision"] = int(raw.get("revision") or 1) + 1
        existing: dict[str, Any] | None = None
        for raw in (state.get("directives") or {}).values():
            if not isinstance(raw, dict) or raw.get("status") != "active":
                continue
            if (
                raw.get("type") == directive_type
                and raw.get("target") == target
                and raw.get("source") == source
                and raw.get("created_by_session") == actor.session
            ):
                existing = raw
                break
        directive_id = (
            str(existing.get("directive_id"))
            if existing
            else f"directive_{uuid.uuid4().hex[:12]}"
        )
        raw = dict(existing or {})
        raw.update(
            {
                "directive_id": directive_id,
                "type": directive_type,
                "target": _clip(target, 200),
                "priority": _priority(source),
                "reason": _clip(reason, 200),
                "source": source,
                "expires_at": to_iso(expires_at) if expires_at else "",
                "status": "active",
                "created_by_session": actor.session,
                "created_at": raw.get("created_at") or to_iso(moment),
                "updated_at": to_iso(moment),
                "revision": int(raw.get("revision") or 0) + 1,
            }
        )
        state.setdefault("directives", {})[directive_id] = raw
        return raw

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _owner_mismatch(
        self,
        umo: str,
        platform_id: str,
        group_id: str,
    ) -> bool:
        """Return whether routing metadata contradicts the UMO ownership key."""

        parts = str(umo or "").split(":", 2)
        if len(parts) != 3 or not all(parts):
            return True
        platform, message_type, identifier = parts
        if platform_id and platform != platform_id:
            return True
        # Isolated session identifiers are platform-defined and opaque. The
        # trusted Actor/event supplies the real group route independently.
        if message_type == "GroupMessage":
            binding = (self.store.state.get("sessions") or {}).get(umo, {})
            return bool(
                binding.get("group_id") and group_id and binding["group_id"] != group_id
            )
        if message_type == "FriendMessage":
            return bool(group_id)
        return False

    def _latest_external_ids_locked(
        self,
        state: dict[str, Any],
        umo: str,
        *,
        limit: int = 3,
    ) -> list[str]:
        ids: list[str] = []
        for item in (state.get("recent_messages") or {}).get(umo, []):
            if not isinstance(item, dict) or item.get("is_bot"):
                continue
            message_id = str(item.get("message_id") or "").strip()
            if message_id and message_id not in ids:
                ids.append(message_id)
        return ids[-limit:]

    def _next_schedule(
        self,
        moment: datetime,
        minutes: int | None,
        expires_at: datetime | None,
    ) -> datetime:
        delay = (
            self.settings.default_interval_minutes if minutes is None else int(minutes)
        )
        delay = max(
            self.settings.min_interval_minutes,
            min(
                self.settings.max_interval_minutes,
                self.settings.max_defer_minutes,
                delay,
            ),
        )
        desired = self._shift_out_of_quiet(moment, add_minutes(moment, delay))
        return self._clamp_schedule(moment, desired, expires_at)

    def _clamp_schedule(
        self,
        moment: datetime,
        desired: datetime,
        expires_at: datetime | None,
    ) -> datetime:
        minimum = moment + timedelta(minutes=self.settings.min_interval_minutes)
        maximum = moment + timedelta(
            minutes=min(
                self.settings.max_interval_minutes,
                self.settings.max_defer_minutes,
            )
        )
        if desired < minimum:
            desired = minimum
        if desired > maximum:
            desired = maximum
        if expires_at is not None and desired >= expires_at:
            desired = min(maximum, expires_at - timedelta(minutes=1))
            if desired < minimum:
                desired = minimum
        return desired

    def _shift_out_of_quiet(
        self,
        moment: datetime,
        desired: datetime,
    ) -> datetime:
        if not in_quiet_hours(desired, self.settings.quiet_hours):
            return desired
        end = quiet_hours_end(desired, self.settings.quiet_hours)
        if end is None:
            return desired
        candidate = end + timedelta(minutes=1)
        if candidate < moment:
            candidate = moment + timedelta(minutes=self.settings.min_interval_minutes)
        maximum = moment + timedelta(
            minutes=min(
                self.settings.max_interval_minutes,
                self.settings.max_defer_minutes,
            )
        )
        return min(candidate, maximum)

    def _resolve_expiry(
        self,
        actor: Actor,
        *,
        max_minutes: int,
        requested_minutes: int | None,
        allow_permanent: bool,
        default_minutes: int,
    ) -> datetime | None:
        moment = self.now()
        if requested_minutes is None or int(requested_minutes) <= 0:
            minutes = max(1, int(default_minutes))
        else:
            minutes = int(requested_minutes)
        if not actor.is_admin:
            minutes = min(minutes, max(1, int(max_minutes)))
        else:
            minutes = min(
                minutes,
                max(self.settings.max_auto_policy_hours * 60, int(max_minutes)),
            )
        if allow_permanent and actor.is_admin and minutes <= 0:
            return None
        return add_minutes(moment, minutes)

    def _clamp_policy_expiry(
        self,
        actor: Actor,
        source: str,
        expires_at: datetime | None,
        *,
        allow_permanent: bool,
    ) -> datetime | None:
        moment = self.now()
        if expires_at is None:
            if allow_permanent and actor.is_admin and source == SOURCE_ADMIN:
                return None
            if source == SOURCE_ATTENTION_AGENT:
                return add_minutes(moment, self.settings.max_auto_policy_hours * 60)
            if source == SOURCE_MAIN_AGENT:
                return add_minutes(
                    moment, self.settings.main_agent_max_policy_hours * 60
                )
            return None
        if source == SOURCE_ATTENTION_AGENT:
            limit = add_minutes(moment, self.settings.max_auto_policy_hours * 60)
            return min(expires_at, limit)
        if source == SOURCE_MAIN_AGENT and not actor.is_admin:
            limit = add_minutes(
                moment,
                self.settings.main_agent_max_policy_hours * 60,
            )
            return min(expires_at, limit)
        return expires_at

    def _actor_can_write_source(self, actor: Actor, source: str) -> bool:
        if actor.source == SOURCE_SYSTEM:
            return True
        if source == SOURCE_ADMIN:
            return actor.is_admin
        if source == SOURCE_MAIN_AGENT:
            return actor.is_admin or actor.source == SOURCE_MAIN_AGENT
        if source == SOURCE_ATTENTION_AGENT:
            return (
                actor.source
                in {
                    SOURCE_ATTENTION_AGENT,
                    SOURCE_MAIN_AGENT,
                    SOURCE_ADMIN,
                }
                or actor.is_admin
            )
        return False

    def _flush_locked(self) -> None:
        self.cleanup_locked(self.store.state, self.now())
        self.store.flush()

    def _record_decision_locked(
        self,
        state: dict[str, Any],
        *,
        umo: str,
        kind: str,
        trigger: str,
        code_decision: str,
        reason: str,
        suggested_action: str = "",
        code: str = "",
        interest_id: str = "",
        checkpoint_id: str = "",
        anchor_message_id: str = "",
        related_policy_id: str = "",
        related_directive_id: str = "",
        reservation_id: str = "",
        wake_source: str = "unknown",
        model_suggestion: str = "",
        message_seq: int = 0,
        cached: bool = False,
    ) -> None:
        store = self.store
        record = {
            "timestamp": to_iso(self.now()),
            "kind": kind,
            "umo": umo,
            "trigger": trigger,
            "wake_source": wake_source,
            "decision": code_decision,
            "code_decision": code_decision,
            "suggested_action": suggested_action,
            "model_suggestion": _clip(model_suggestion, 500),
            "rejection_code": code,
            "interest": "none",
            "reason": _clip(reason, self.settings.max_reason_chars),
            "source": (
                "main_agent"
                if trigger == SOURCE_MAIN_AGENT
                else "admin"
                if trigger == SOURCE_ADMIN
                else "attention_agent"
            ),
            "interest_id": interest_id,
            "checkpoint_id": checkpoint_id,
            "anchor_message_id": anchor_message_id,
            "related_policy_id": related_policy_id,
            "related_directive_id": related_directive_id,
            "reservation_id": reservation_id,
            "message_seq": int(message_seq or 0),
            "cached": bool(cached),
            "tool_calls": [],
            "checkpoint_changes": [],
            "main_model": "none",
            "send": "none",
        }
        store.record_decision(record)


def expires_in_minutes_from_minutes(minutes: int | None) -> int | None:
    """Translate a scheduling delay into a directive validity window."""

    if minutes is None:
        return None
    return max(1, int(minutes)) + 60
