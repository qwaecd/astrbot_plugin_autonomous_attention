"""Unified attention control service.

The main agent, the passive runner and the active runner all mutate policies,
directives and checkpoints through this service. It validates the operator,
scope, priority and expiry before writing to the state store. Low-priority
operators can never overwrite higher-priority state.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from astrbot.api import logger

from .models import (
    DIRECTIVE_TYPES,
    SOURCE_ADMIN,
    SOURCE_ATTENTION_AGENT,
    SOURCE_MAIN_AGENT,
    SOURCE_PRIORITIES,
    SOURCE_SYSTEM,
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
from .timeutil import (
    now as current_time,
)


@dataclass(frozen=True)
class Actor:
    """The real operator behind a control request.

    ``source`` describes the layer making the request. Permission checks also
    use the real event sender's session and admin flag, never model-supplied
    text.
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


class AttentionControlService:
    """Validate and write attention state changes."""

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
        return current_time(self.settings.tz)

    # ------------------------------------------------------------------
    # Public async entry points (main-model tools)
    # ------------------------------------------------------------------

    async def query(
        self,
        actor: Actor,
        *,
        scope: str = "session",
        target_user: str = "",
    ) -> ControlResult:
        async with self.store.lock:
            result = self.query_locked(
                self.store.state,
                actor,
                scope=scope,
                target_user=target_user,
            )
        return result

    async def set_intent(
        self,
        actor: Actor,
        *,
        intent_type: str,
        target: str = "",
        reason: str = "",
        expires_in_minutes: int | None = None,
        minutes: int | None = None,
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

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        scope: str = "session",
        target_user: str = "",
    ) -> ControlResult:
        scope = str(scope or "session").strip().lower()
        if scope not in {"session", "user", "group", "directives", "recent", "all"}:
            return ControlResult(
                ok=False,
                code="invalid_scope",
                message="scope 必须是 session、user、group、directives、recent 或 all。",
            )
        moment = self.now()
        self.expire_directives_locked(state, moment)
        group_id = actor.group_id
        umo = actor.session
        platform_id = actor.platform_id
        data: dict[str, Any] = {
            "session": {
                "umo": umo,
                "group_id": group_id or "",
                "group_allowed": self.policy.group_allowed(
                    group_id,
                    umo,
                    platform_id,
                ),
            }
        }

        if scope in {"session", "user", "all"}:
            query_user = target_user or actor.user_id
            if query_user != actor.user_id and not actor.is_admin:
                data["user_policy"] = {
                    "allowed": False,
                    "reason": "普通成员只能查询自己的忽略状态。",
                }
            else:
                policy = self.policy.matching_policy(
                    state,
                    group_id,
                    query_user,
                    umo,
                    platform_id,
                )
                data["user_policy"] = {
                    "target_user": query_user,
                    "ignored": policy is not None,
                    "source": (policy or {}).get("source", ""),
                    "priority": (policy or {}).get("priority", 0),
                    "expires_at": (policy or {}).get("expires_at", ""),
                    "reason": _clip((policy or {}).get("reason"), 120),
                }

        if scope in {"session", "group", "all"}:
            checkpoints = [
                dict(raw)
                for raw in state.get("checkpoints", {}).values()
                if isinstance(raw, dict)
                and raw.get("status") in {"active", "running", "paused"}
                and self._target_matches_group(raw, group_id, umo)
            ]
            checkpoints.sort(key=lambda item: str(item.get("scheduled_at") or ""))
            topic = self._current_topic_locked(state, umo)
            data["group"] = {
                "group_id": group_id or "",
                "blocked": self.policy.group_blocked(
                    group_id,
                    umo,
                    platform_id,
                ),
                "active_checkpoints": len(checkpoints),
                "next_check": (
                    checkpoints[0].get("scheduled_at", "") if checkpoints else ""
                ),
                "topic": {
                    "topic_id": topic.get("topic_id", ""),
                    "summary": _clip(topic.get("summary"), 300),
                    "interest_level": topic.get("interest_level", "none"),
                    "focus_status": topic.get("focus_status", "idle"),
                },
                "stop_focus": bool(
                    self.stop_directive(
                        state,
                        group_id=group_id,
                        umo=umo,
                        topic_id=str(topic.get("topic_id") or ""),
                    )
                ),
            }

        if scope in {"session", "group", "directives", "all"}:
            directives = self.active_directives_locked(
                state,
                group_id=group_id,
                umo=umo,
            )
            data["directives"] = [
                {
                    "directive_id": item.get("directive_id", ""),
                    "type": item.get("type", ""),
                    "target": item.get("target", ""),
                    "source": item.get("source", ""),
                    "priority": item.get("priority", 0),
                    "reason": _clip(item.get("reason"), 120),
                    "expires_at": item.get("expires_at", ""),
                }
                for item in directives[:10]
            ]

        if scope in {"session", "recent", "all"}:
            decisions = [
                item
                for item in state.get("decisions", [])
                if isinstance(item, dict) and item.get("umo") == umo
            ][-3:]
            data["recent"] = [
                {
                    "timestamp": item.get("timestamp", ""),
                    "source": item.get("source", ""),
                    "decision": item.get("decision", ""),
                    "interest": item.get("interest", ""),
                    "reason": _clip(item.get("reason"), 100),
                    "related_policy_id": item.get("related_policy_id", ""),
                    "related_directive_id": item.get("related_directive_id", ""),
                }
                for item in decisions
            ]

        return ControlResult(
            ok=True,
            code="query_ok",
            message="已返回当前会话范围内的注意力状态摘要。",
            data=data,
        )

    # ------------------------------------------------------------------
    # Intent / directive operations
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
    ) -> ControlResult:
        intent_type = str(intent_type or "").strip().lower()
        if intent_type not in DIRECTIVE_TYPES:
            return ControlResult(
                ok=False,
                code="invalid_intent",
                message=f"不支持的意图类型：{intent_type or '-'}",
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

        if intent_type == "focus_group" and not actor.is_admin:
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="群级关注意图需要管理员权限；普通成员可以使用 focus_topic。",
            )
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
            )

        normalized_target = self._normalize_target(
            actor,
            target,
            intent_type=intent_type,
        )
        if isinstance(normalized_target, ControlResult):
            return normalized_target
        target_value = normalized_target
        if intent_type == "focus_topic" and not target_value:
            return ControlResult(
                ok=False,
                code="missing_target",
                message="focus_topic 需要提供话题关键字或话题 ID。",
            )

        expires_at = self._resolve_expiry(
            actor,
            max_hours=self.settings.main_agent_max_intent_hours,
            requested_minutes=expires_in_minutes,
            allow_permanent=False,
        )
        directive = self._upsert_directive_locked(
            state,
            actor,
            directive_type=intent_type,
            target=target_value,
            reason=reason or intent_type,
            expires_at=expires_at,
        )
        return ControlResult(
            ok=True,
            code="intent_saved",
            message="主 Agent 意图已保存。",
            data={"directive": directive},
        )

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
            max_hours=self.settings.main_agent_max_policy_hours,
            requested_minutes=expires_in_minutes,
            allow_permanent=actor.is_admin,
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
            data={
                "policy": policy_result.data,
                "directive": directive,
            },
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

    def stop_focus_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        target: str = "",
    ) -> ControlResult:
        if not self._actor_can_write_source(actor, SOURCE_MAIN_AGENT):
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="当前操作者不能停止关注目标。",
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
        target_value = self._normalize_target(
            actor,
            target,
            intent_type="focus_group",
            allow_all=True,
        )
        if isinstance(target_value, ControlResult):
            return target_value
        moment = self.now()
        revoked: list[str] = []
        cancelled: list[str] = []
        for raw in state.get("directives", {}).values():
            if not isinstance(raw, dict) or raw.get("status") != "active":
                continue
            if raw.get("source") != SOURCE_MAIN_AGENT:
                continue
            if raw.get("type") not in {"focus_group", "focus_topic", "schedule_check"}:
                continue
            if actor.is_admin:
                if not self._directive_matches(raw, target_value, actor):
                    continue
            elif raw.get("created_by_session") != actor.session:
                continue
            raw["status"] = "revoked"
            raw["updated_at"] = to_iso(moment)
            raw["revision"] = int(raw.get("revision") or 1) + 1
            revoked.append(str(raw.get("directive_id") or ""))
        for checkpoint in state.get("checkpoints", {}).values():
            if not isinstance(checkpoint, dict):
                continue
            if checkpoint.get("status") not in {"active", "paused"}:
                continue
            source = str(checkpoint.get("source") or "")
            if actor.is_admin:
                if target_value not in {"*", "all"}:
                    target_group, target_umo, _ = self._resolve_target_group(
                        actor,
                        target_value,
                    )
                    if not (
                        checkpoint.get("umo") == target_value
                        or checkpoint.get("umo") == target_umo
                        or checkpoint.get("group_id") == target_group
                    ):
                        continue
            else:
                # Ordinary members may only stop the main-agent intents they
                # created in this session. Attention-agent checkpoints and
                # admin checkpoints are protected.
                if source != SOURCE_MAIN_AGENT:
                    continue
                if checkpoint.get("created_by_session") != actor.session:
                    continue
            checkpoint["status"] = "cancelled"
            checkpoint["scheduled_at"] = ""
            checkpoint["last_result"] = "main_agent_stop_focus"
            checkpoint["updated_at"] = to_iso(moment)
            checkpoint["revision"] = int(checkpoint.get("revision") or 1) + 1
            cancelled.append(str(checkpoint.get("checkpoint_id") or ""))

        if not actor.is_admin:
            self._record_control_decision_locked(
                state,
                actor,
                decision="focus_stopped",
                reason=f"revoked={len(revoked)} cancelled={len(cancelled)}",
            )
            return ControlResult(
                ok=True,
                code="nothing_to_stop"
                if not revoked and not cancelled
                else "focus_stopped",
                message=(
                    "当前会话没有可停止的主 Agent 意图。"
                    if not revoked and not cancelled
                    else "已停止当前会话中由主 Agent 发起的关注/检查意图。"
                ),
                data={
                    "revoked_directives": revoked,
                    "cancelled_checkpoints": cancelled,
                },
            )

        expires_at = self._resolve_expiry(
            actor,
            max_hours=self.settings.main_agent_max_intent_hours,
            requested_minutes=None,
            allow_permanent=False,
        )
        directive = self._upsert_directive_locked(
            state,
            actor,
            directive_type="stop_focus",
            target=target_value,
            reason="main agent stop_focus",
            expires_at=expires_at,
        )
        self._record_control_decision_locked(
            state,
            actor,
            decision="focus_stopped",
            reason=f"revoked={len(revoked)} cancelled={len(cancelled)}",
            related_directive_id=str(directive.get("directive_id") or ""),
        )
        return ControlResult(
            ok=True,
            code="focus_stopped",
            message="关注目标已停止。",
            data={
                "directive": directive,
                "revoked_directives": revoked,
                "cancelled_checkpoints": cancelled,
            },
        )

    def _schedule_check_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        target: str = "",
        reason: str = "",
        minutes: int | None = None,
    ) -> ControlResult:
        if not actor.group_id:
            return ControlResult(
                ok=False,
                code="missing_group",
                message="schedule_check 需要在群聊中调用。",
            )
        target_value = self._normalize_target(
            actor,
            target,
            intent_type="focus_group",
        )
        if isinstance(target_value, ControlResult):
            return target_value
        group_id, umo, platform_id = self._resolve_target_group(
            actor,
            target_value,
        )
        if not group_id or not self.policy.group_allowed(
            group_id,
            umo,
            platform_id,
        ):
            return ControlResult(
                ok=False,
                code="group_not_allowed",
                message="目标群不在 group_allowlist 中。",
            )
        delay = int(minutes or self.settings.default_interval_minutes)
        max_delay = min(
            self.settings.max_interval_minutes,
            self.settings.max_defer_minutes,
        )
        if not actor.is_admin:
            max_delay = min(
                max_delay,
                self.settings.main_agent_max_intent_hours * 60,
            )
        delay = max(self.settings.min_interval_minutes, min(max_delay, delay))
        desired = self._shift_out_of_quiet(add_minutes(self.now(), delay))
        expires_at = add_minutes(desired, 60)
        directive = self._upsert_directive_locked(
            state,
            actor,
            directive_type="schedule_check",
            target=target_value,
            reason=reason or "main agent schedule_check",
            expires_at=expires_at,
        )
        checkpoint = self._create_checkpoint_locked(
            state,
            actor,
            umo=umo,
            platform_id=platform_id,
            group_id=group_id,
            scheduled_at=desired,
            reason=reason or "main agent schedule_check",
            source=SOURCE_MAIN_AGENT,
            directive_id=str(directive.get("directive_id") or ""),
        )
        if not checkpoint.ok:
            directive["status"] = "revoked"
            directive["updated_at"] = to_iso(self.now())
            directive["revision"] = int(directive.get("revision") or 1) + 1
            return checkpoint
        return ControlResult(
            ok=True,
            code="check_scheduled",
            message="检查点已按主 Agent 意图安排。",
            data={
                "directive": directive,
                "checkpoint": checkpoint.data,
            },
        )

    # ------------------------------------------------------------------
    # Policy writes (also used by engine / admin commands)
    # ------------------------------------------------------------------

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
        if not actor.is_admin and scope == "group" and source != SOURCE_ATTENTION_AGENT:
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="只有管理员可以写入群级策略。",
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
        self._record_control_decision_locked(
            state,
            actor,
            decision="policy_saved",
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
        self._record_control_decision_locked(
            state,
            actor,
            decision="policy_removed",
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
    # Checkpoint writes (strategic)
    # ------------------------------------------------------------------

    def register_checkpoint_locked(
        self,
        state: dict[str, Any],
        checkpoint: dict[str, Any],
        *,
        actor: Actor,
        source: str,
        priority: int | None = None,
        directive_id: str = "",
    ) -> ControlResult:
        if not self._actor_can_write_source(actor, source):
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="当前操作者不能创建该来源的检查点。",
            )
        checkpoint = dict(checkpoint)
        checkpoint["source"] = source
        checkpoint["priority"] = int(priority or _priority(source))
        checkpoint["directive_id"] = directive_id
        checkpoint["created_by_session"] = actor.session
        checkpoint.setdefault("created_by", source)
        checkpoint.setdefault("revision", 1)
        checkpoint.setdefault("created_at", to_iso(self.now()))
        checkpoint.setdefault("updated_at", to_iso(self.now()))
        state.setdefault("checkpoints", {})[
            str(checkpoint.get("checkpoint_id") or "")
        ] = checkpoint
        self._record_control_decision_locked(
            state,
            actor,
            decision="checkpoint_saved",
            reason=str(checkpoint.get("reason") or "checkpoint saved"),
            related_directive_id=directive_id,
            checkpoint_id=str(checkpoint.get("checkpoint_id") or ""),
        )
        return ControlResult(
            ok=True,
            code="checkpoint_saved",
            message="检查点已创建。",
            data=checkpoint,
        )

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
    ) -> ControlResult:
        """Write the operational result of one active checkpoint run."""

        raw = (state.get("checkpoints") or {}).get(checkpoint_id)
        if not isinstance(raw, dict):
            return ControlResult(
                ok=False,
                code="checkpoint_not_found",
                message="检查点不存在。",
            )
        raw["status"] = status
        raw["scheduled_at"] = scheduled_at if status == "active" else ""
        raw["last_result"] = last_result
        raw["last_error"] = _clip(last_error, 300)
        raw["run_id"] = ""
        raw["updated_at"] = to_iso(moment)
        raw["revision"] = int(raw.get("revision") or 1) + 1
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
            if source == SOURCE_MAIN_AGENT and raw.get("created_by_session") not in {
                "",
                actor.session,
            }:
                return ControlResult(
                    ok=False,
                    code="permission_denied",
                    message="只能修改当前会话创建的主 Agent 检查点。",
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

    def _create_checkpoint_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        umo: str,
        platform_id: str,
        group_id: str,
        scheduled_at: datetime,
        reason: str,
        source: str,
        directive_id: str = "",
    ) -> ControlResult:
        if not self.policy.group_allowed(group_id, umo, platform_id):
            return ControlResult(
                ok=False,
                code="group_not_allowed",
                message="目标群不在 group_allowlist 中。",
            )
        if directive_id:
            for raw in state.get("checkpoints", {}).values():
                if not isinstance(raw, dict):
                    continue
                if (
                    raw.get("status") == "active"
                    and raw.get("umo") == umo
                    and raw.get("directive_id") == directive_id
                ):
                    raw["scheduled_at"] = to_iso(scheduled_at)
                    raw["reason"] = _clip(reason, 200)
                    raw["updated_at"] = to_iso(self.now())
                    raw["revision"] = int(raw.get("revision") or 1) + 1
                    return ControlResult(
                        ok=True,
                        code="checkpoint_saved",
                        message="检查点已更新。",
                        data=dict(raw),
                    )
        checkpoint_id = f"cp_{uuid.uuid4().hex[:12]}"
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "umo": umo,
            "platform_id": platform_id,
            "group_id": group_id,
            "scheduled_at": to_iso(scheduled_at),
            "status": "active",
            "reason": _clip(reason, 200),
            "topic_id": f"topic_{hashlib.sha1(umo.encode()).hexdigest()[:12]}",
            "last_result": "",
            "last_error": "",
            "run_id": "",
            "attempts": 0,
            "revision": 1,
            "created_at": to_iso(self.now()),
            "updated_at": to_iso(self.now()),
        }
        return self.register_checkpoint_locked(
            state,
            checkpoint,
            actor=actor,
            source=source,
            directive_id=directive_id,
        )

    # ------------------------------------------------------------------
    # Directives and summaries
    # ------------------------------------------------------------------

    def expire_directives_locked(
        self,
        state: dict[str, Any],
        moment: datetime | None = None,
    ) -> int:
        moment = moment or self.now()
        expired = 0
        for raw in state.get("directives", {}).values():
            if not isinstance(raw, dict) or raw.get("status") != "active":
                continue
            expires_at = parse_iso(raw.get("expires_at"), self.settings.tz)
            if expires_at is not None and expires_at <= moment:
                raw["status"] = "expired"
                raw["updated_at"] = to_iso(moment)
                raw["revision"] = int(raw.get("revision") or 1) + 1
                expired += 1
        return expired

    def active_directives_locked(
        self,
        state: dict[str, Any],
        *,
        group_id: str = "",
        umo: str = "",
    ) -> list[dict[str, Any]]:
        self.expire_directives_locked(state)
        result: list[dict[str, Any]] = []
        for raw in state.get("directives", {}).values():
            if not isinstance(raw, dict) or raw.get("status") != "active":
                continue
            if raw.get("type") == "stop_focus":
                if self._directive_matches_target(raw, group_id, umo, ""):
                    result.append(dict(raw))
                continue
            if raw.get("type") in {"focus_group", "schedule_check"}:
                if not group_id:
                    continue
                if self._directive_matches_target(raw, group_id, umo, ""):
                    result.append(dict(raw))
            elif raw.get("type") == "ignore_user":
                if not umo or raw.get("created_by_session") == umo:
                    result.append(dict(raw))
            elif raw.get("type") == "focus_topic":
                if not umo or raw.get("created_by_session") == umo:
                    result.append(dict(raw))
        result.sort(
            key=lambda item: (
                -int(item.get("priority") or 0),
                str(item.get("updated_at") or ""),
            )
        )
        return result

    def focus_directive(
        self,
        state: dict[str, Any],
        *,
        group_id: str,
        umo: str,
        topic_id: str = "",
        message_text: str = "",
    ) -> dict[str, Any] | None:
        self.expire_directives_locked(state)
        candidates: list[dict[str, Any]] = []
        lowered = (message_text or "").casefold()
        for raw in state.get("directives", {}).values():
            if not isinstance(raw, dict) or raw.get("status") != "active":
                continue
            dtype = raw.get("type")
            if dtype == "focus_group" and self._directive_matches_target(
                raw, group_id, umo, ""
            ):
                candidates.append(raw)
            elif dtype == "focus_topic":
                target = str(raw.get("target") or "").strip()
                if not target:
                    continue
                if target == topic_id or target.casefold() in lowered:
                    candidates.append(raw)
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                -int(item.get("priority") or 0),
                str(item.get("updated_at") or ""),
            )
        )
        stop = self.stop_directive(
            state,
            group_id=group_id,
            umo=umo,
            topic_id=topic_id,
        )
        if stop is not None and int(stop.get("priority") or 0) >= int(
            candidates[0].get("priority") or 0
        ):
            return None
        return dict(candidates[0])

    def stop_directive(
        self,
        state: dict[str, Any],
        *,
        group_id: str,
        umo: str,
        topic_id: str = "",
    ) -> dict[str, Any] | None:
        self.expire_directives_locked(state)
        candidates: list[dict[str, Any]] = []
        for raw in state.get("directives", {}).values():
            if not isinstance(raw, dict) or raw.get("status") != "active":
                continue
            if raw.get("type") != "stop_focus":
                continue
            if self._directive_matches_target(raw, group_id, umo, topic_id):
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

    def prompt_summary_locked(
        self,
        state: dict[str, Any],
        *,
        group_id: str = "",
        umo: str = "",
        user_id: str = "",
        platform_id: str = "",
        topic_id: str = "",
        message_text: str = "",
    ) -> str:
        """Return a compact directive/policy summary for the *interest* model."""

        self.expire_directives_locked(state)
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
                "有效忽略策略：来源 {source}，优先级 {priority}，到期 {expires}，原因：{reason}".format(
                    source=policy.get("source") or "-",
                    priority=policy.get("priority") or 0,
                    expires=policy.get("expires_at") or "永久",
                    reason=_clip(policy.get("reason"), 100),
                )
            )
        elif group_id:
            high_priority = []
            for raw in state.get("policies", {}).values():
                if not isinstance(raw, dict) or raw.get("mode") != "ignore":
                    continue
                if raw.get("source") not in {SOURCE_ADMIN, SOURCE_MAIN_AGENT}:
                    continue
                if not (
                    raw.get("scope") == "user"
                    or self.policy.matches_groups(
                        (str(raw.get("group_id") or ""),),
                        group_id,
                        umo,
                        platform_id,
                    )
                ):
                    continue
                high_priority.append(
                    f"{raw.get('user_id') or raw.get('group_id')}({raw.get('source')})"
                )
            if high_priority:
                lines.append("高优先级忽略策略：" + "、".join(high_priority[:8]))
        focus = self.focus_directive(
            state,
            group_id=group_id,
            umo=umo,
            topic_id=topic_id,
            message_text=message_text,
        )
        if focus:
            lines.append(
                "主 Agent 关注意图：{type} {target}，到期 {expires}，原因：{reason}".format(
                    type=focus.get("type"),
                    target=focus.get("target"),
                    expires=focus.get("expires_at") or "-",
                    reason=_clip(focus.get("reason"), 100),
                )
            )
        stop = self.stop_directive(
            state,
            group_id=group_id,
            umo=umo,
            topic_id=topic_id,
        )
        if stop:
            lines.append(
                "主 Agent 停止关注意图：{target}，到期 {expires}".format(
                    target=stop.get("target"),
                    expires=stop.get("expires_at") or "-",
                )
            )
        if not lines:
            return "当前没有主 Agent 意图或用户忽略策略。"
        lines.append("兴趣模型不得覆盖上述主 Agent 意图和管理员策略。")
        return "\n".join(lines[:8])

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
            for raw in state.get("directives", {}).values():
                if not isinstance(raw, dict):
                    continue
                if raw.get("status") != "active" or raw.get("type") != "stop_focus":
                    continue
                if raw.get("source") == SOURCE_ADMIN and not actor.is_admin:
                    continue
                stop_target = str(raw.get("target") or "")
                if stop_target in {"*", "all"} and not actor.is_admin:
                    continue
                if (
                    stop_target == target
                    or raw.get("created_by_session") == actor.session
                ):
                    raw["status"] = "revoked"
                    raw["updated_at"] = to_iso(moment)
                    raw["revision"] = int(raw.get("revision") or 1) + 1
        existing: dict[str, Any] | None = None
        for raw in state.get("directives", {}).values():
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
                "target": target,
                "priority": _priority(source),
                "reason": _clip(reason, 200),
                "source": source,
                "expires_at": to_iso(expires_at) if expires_at else "",
                "status": "active",
                "created_by_session": actor.session,
                "created_at": (raw.get("created_at") or to_iso(moment)),
                "updated_at": to_iso(moment),
                "revision": int(raw.get("revision") or 0) + 1,
            }
        )
        state.setdefault("directives", {})[directive_id] = raw
        self._record_control_decision_locked(
            state,
            actor,
            decision="directive_saved",
            reason=reason,
            related_directive_id=directive_id,
        )
        return raw

    def _directive_matches(
        self,
        directive: dict[str, Any],
        target: str,
        actor: Actor,
    ) -> bool:
        if target in {"*", "all"}:
            return True
        directive_target = str(directive.get("target") or "")
        if directive_target in {"*", "all"}:
            return True
        if directive_target == target:
            return True
        return self._directive_matches_target(
            directive,
            actor.group_id,
            actor.session,
            "",
        )

    def _directive_matches_target(
        self,
        directive: dict[str, Any],
        group_id: str,
        umo: str,
        topic_id: str,
    ) -> bool:
        target = str(directive.get("target") or "")
        if target in {"*", "all"}:
            return True
        if not target:
            return False
        if target == topic_id and topic_id:
            return True
        return self.policy.matches_groups(
            (target,),
            group_id,
            umo,
            "",
        )

    def _target_matches_group(
        self,
        raw: dict[str, Any],
        group_id: str,
        umo: str,
    ) -> bool:
        return self.policy.matches_groups(
            (str(raw.get("group_id") or raw.get("umo") or ""),),
            group_id,
            umo,
            str(raw.get("platform_id") or ""),
        )

    def _current_topic_locked(
        self,
        state: dict[str, Any],
        umo: str,
    ) -> dict[str, Any]:
        topic_id = f"topic_{hashlib.sha1(umo.encode()).hexdigest()[:12]}"
        raw = (state.get("topics") or {}).get(topic_id)
        return dict(raw) if isinstance(raw, dict) else {"topic_id": topic_id}

    def _normalize_target(
        self,
        actor: Actor,
        target: str,
        *,
        intent_type: str,
        allow_all: bool = False,
    ) -> str | ControlResult:
        target = str(target or "").strip()
        if intent_type == "focus_topic":
            if not target:
                return ControlResult(
                    ok=False,
                    code="missing_target",
                    message="focus_topic 需要提供话题关键字或话题 ID。",
                )
            return _clip(target, 200)
        if target in {"*", "all"}:
            if allow_all and actor.is_admin:
                return target
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="只有管理员可以使用全局关注目标。",
            )
        if not target:
            if actor.group_id:
                return actor.session
            return ControlResult(
                ok=False,
                code="missing_target",
                message="该意图需要提供目标群或话题。",
            )
        if not actor.is_admin:
            if actor.group_id and (
                target == actor.group_id
                or target == actor.session
                or target.endswith(f":{actor.group_id}")
            ):
                return actor.session
            return ControlResult(
                ok=False,
                code="permission_denied",
                message="普通成员只能把意图作用于当前会话。",
            )
        return target

    def _resolve_target_group(
        self,
        actor: Actor,
        target: str,
    ) -> tuple[str, str, str]:
        if not target or target in {"*", "all"} or target == actor.session:
            return actor.group_id, actor.session, actor.platform_id
        parts = target.split(":", 2)
        if len(parts) == 3 and parts[1] == "GroupMessage":
            return parts[2], target, parts[0]
        return target, f"{actor.platform_id}:GroupMessage:{target}", actor.platform_id

    def _resolve_expiry(
        self,
        actor: Actor,
        *,
        max_hours: int,
        requested_minutes: int | None,
        allow_permanent: bool,
    ) -> datetime | None:
        moment = self.now()
        if requested_minutes is None:
            return add_minutes(moment, max_hours * 60)
        try:
            minutes = int(requested_minutes)
        except (TypeError, ValueError):
            minutes = max_hours * 60
        if minutes <= 0:
            if allow_permanent and actor.is_admin:
                return None
            minutes = max_hours * 60
        if not actor.is_admin:
            minutes = min(minutes, max_hours * 60)
        else:
            minutes = min(
                minutes, max(self.settings.max_auto_policy_hours, max_hours) * 60
            )
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
                    moment,
                    self.settings.main_agent_max_policy_hours * 60,
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

    def _shift_out_of_quiet(self, desired: datetime) -> datetime:
        if not in_quiet_hours(desired, self.settings.quiet_hours):
            return desired
        end = quiet_hours_end(desired, self.settings.quiet_hours)
        if end is None:
            return desired
        return end

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
        try:
            self.store.flush()
        except Exception:
            logger.exception("Failed to persist attention control state.")

    def _record_control_decision_locked(
        self,
        state: dict[str, Any],
        actor: Actor,
        *,
        decision: str,
        reason: str,
        related_policy_id: str = "",
        related_directive_id: str = "",
        checkpoint_id: str = "",
    ) -> None:
        source = (
            actor.source
            if actor.source in {SOURCE_ADMIN, SOURCE_MAIN_AGENT}
            else "active_agent"
        )
        record = {
            "timestamp": to_iso(self.now()),
            "kind": "control",
            "umo": actor.session,
            "trigger": actor.source,
            "decision": decision,
            "interest": "none",
            "reason": _clip(reason, self.settings.max_reason_chars),
            "source": source,
            "related_policy_id": related_policy_id,
            "related_directive_id": related_directive_id,
            "checkpoint_id": checkpoint_id,
            "tool_calls": [],
            "checkpoint_changes": [],
            "main_model": "none",
            "send": "none",
        }
        decisions = state.setdefault("decisions", [])
        if not isinstance(decisions, list):
            decisions = []
            state["decisions"] = decisions
        decisions.append(record)
        limit = max(10, int(self.settings.max_decision_log))
        del decisions[: max(0, len(decisions) - limit)]
