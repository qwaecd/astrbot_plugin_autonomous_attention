"""Hard policies, budgets, cooldowns and quiet hours.

Policy decisions always win over interest-model output. The manager only mutates
the in-memory state object; the caller is responsible for holding the store lock
and flushing it.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any

from .models import UserPolicy
from .settings import Settings
from .timeutil import (
    in_quiet_hours,
    parse_iso,
    parse_quiet_hours,
    quiet_hours_end,
    timestamp,
    to_iso,
)
from .timeutil import (
    now as current_time,
)


class PolicyManager:
    """Evaluate hard policies and account for model/send budgets."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    def now(self) -> datetime:
        return current_time(self.settings.tz)

    @staticmethod
    def ensure_budget(state: dict[str, Any], moment: datetime) -> dict[str, Any]:
        """Reset daily counters when the local date changes."""

        budget = state.setdefault("budget", {})
        date_key = moment.date().isoformat()
        if budget.get("date") != date_key:
            budget.clear()
            budget.update(
                {
                    "date": date_key,
                    "decision_calls": 0,
                    "main_calls": 0,
                    "active_checks": 0,
                    "messages_sent": 0,
                }
            )
        for key in (
            "decision_calls",
            "main_calls",
            "active_checks",
            "messages_sent",
        ):
            budget[key] = int(budget.get(key) or 0)
        return budget

    def total_model_calls(self, state: dict[str, Any], moment: datetime) -> int:
        budget = self.ensure_budget(state, moment)
        return int(budget["decision_calls"]) + int(budget["main_calls"])

    def can_call_model(self, state: dict[str, Any], moment: datetime) -> bool:
        return (
            self.total_model_calls(state, moment)
            < self.settings.daily_model_call_budget
        )

    def consume_model_call(
        self,
        state: dict[str, Any],
        moment: datetime,
        kind: str,
    ) -> bool:
        """Reserve one daily model call.

        Args:
            state: Mutable state dictionary.
            moment: Current local time.
            kind: ``decision`` for the interest model or ``main`` for the main
                reply model.

        Returns:
            ``True`` when a call was reserved, ``False`` when the budget is
            exhausted.
        """

        budget = self.ensure_budget(state, moment)
        if (
            self.total_model_calls(state, moment)
            >= self.settings.daily_model_call_budget
        ):
            return False
        key = "main_calls" if kind == "main" else "decision_calls"
        budget[key] = int(budget[key]) + 1
        return True

    def can_run_active_check(
        self,
        state: dict[str, Any],
        moment: datetime,
    ) -> bool:
        budget = self.ensure_budget(state, moment)
        return int(budget["active_checks"]) < self.settings.daily_active_check_budget

    def consume_active_check(self, state: dict[str, Any], moment: datetime) -> bool:
        budget = self.ensure_budget(state, moment)
        if int(budget["active_checks"]) >= self.settings.daily_active_check_budget:
            return False
        budget["active_checks"] = int(budget["active_checks"]) + 1
        return True

    def is_quiet(self, moment: datetime) -> bool:
        return in_quiet_hours(moment, self.settings.quiet_hours)

    def quiet_end(self, moment: datetime) -> datetime | None:
        return quiet_hours_end(moment, self.settings.quiet_hours)

    @staticmethod
    def _group_candidates(group_id: str, umo: str, platform_id: str = "") -> set[str]:
        candidates = {str(group_id or "").strip(), str(umo or "").strip()}
        if group_id and platform_id:
            candidates.add(f"{platform_id}:{group_id}")
        return {candidate for candidate in candidates if candidate}

    def matches_groups(
        self,
        entries: tuple[str, ...],
        group_id: str,
        umo: str,
        platform_id: str = "",
    ) -> bool:
        if not entries:
            return False
        return not self._group_candidates(group_id, umo, platform_id).isdisjoint(
            set(entries)
        )

    def group_allowed(
        self,
        group_id: str,
        umo: str,
        platform_id: str = "",
    ) -> bool:
        """Return whether the plugin may inspect and act on this group."""

        if not group_id:
            return True
        if self.matches_groups(
            self.settings.blocked_group_ids,
            group_id,
            umo,
            platform_id,
        ):
            return False
        # The allowlist is the single boundary for observation, checkpoint
        # creation and proactive sending. An empty list selects no groups.
        if not self.settings.group_allowlist:
            return False
        return self.matches_groups(
            self.settings.group_allowlist,
            group_id,
            umo,
            platform_id,
        )

    def group_blocked(
        self,
        group_id: str,
        umo: str,
        platform_id: str = "",
    ) -> bool:
        return self.matches_groups(
            self.settings.blocked_group_ids,
            group_id,
            umo,
            platform_id,
        )

    def expire_policies(self, state: dict[str, Any], moment: datetime) -> int:
        """Remove expired policies and return the number removed."""

        policies = state.get("policies")
        if not isinstance(policies, dict):
            state["policies"] = {}
            return 0
        expired: list[str] = []
        for policy_id, raw in policies.items():
            if not isinstance(raw, dict):
                expired.append(policy_id)
                continue
            expires_at = parse_iso(raw.get("expires_at"), self.settings.tz)
            if expires_at is not None and expires_at <= moment:
                expired.append(policy_id)
        for policy_id in expired:
            policies.pop(policy_id, None)
        return len(expired)

    def matching_policy(
        self,
        state: dict[str, Any],
        group_id: str,
        user_id: str,
        umo: str,
        platform_id: str = "",
    ) -> dict[str, Any] | None:
        """Return the strongest active ignore policy matching a sender."""

        self.expire_policies(state, self.now())
        policies = state.get("policies") or {}
        if not isinstance(policies, dict):
            return None
        candidates: list[dict[str, Any]] = []
        for raw in policies.values():
            if not isinstance(raw, dict) or raw.get("mode") != "ignore":
                continue
            scope = raw.get("scope")
            policy_group = str(raw.get("group_id") or "")
            policy_user = str(raw.get("user_id") or "")
            if scope == "group_user":
                if (
                    policy_user == user_id
                    and policy_group
                    and self.matches_groups(
                        (policy_group,),
                        group_id,
                        umo,
                        platform_id,
                    )
                ):
                    candidates.append(raw)
            elif scope == "group":
                if self.matches_groups(
                    (policy_group,),
                    group_id,
                    umo,
                    platform_id,
                ):
                    candidates.append(raw)
            elif scope == "user":
                if policy_user == user_id:
                    candidates.append(raw)
        if not candidates:
            return None
        scope_priority = {"group_user": 0, "group": 1, "user": 2}
        candidates.sort(
            key=lambda item: (
                -int(item.get("priority") or 0),
                scope_priority.get(str(item.get("scope")), 9),
            )
        )
        return candidates[0]

    def is_user_blocked(
        self,
        state: dict[str, Any],
        *,
        group_id: str,
        user_id: str,
        umo: str,
        platform_id: str = "",
        is_admin: bool = False,
    ) -> dict[str, Any] | None:
        """Return the matching ignore policy, respecting owner override."""

        if is_admin and self.settings.owner_override:
            return None
        return self.matching_policy(state, group_id, user_id, umo, platform_id)

    def active_policies(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        self.expire_policies(state, self.now())
        policies = state.get("policies") or {}
        if not isinstance(policies, dict):
            return []
        result: list[dict[str, Any]] = []
        for raw in policies.values():
            if isinstance(raw, dict) and raw.get("mode") == "ignore":
                result.append(dict(raw))
        result.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return result

    def policy_summary(
        self,
        state: dict[str, Any],
        *,
        group_id: str = "",
        user_id: str = "",
        umo: str = "",
        platform_id: str = "",
    ) -> str:
        self.expire_policies(state, self.now())
        if user_id:
            policy = self.matching_policy(
                state,
                group_id,
                user_id,
                umo,
                platform_id,
            )
            if policy:
                return (
                    f"当前用户命中忽略策略（{policy.get('reason') or '无原因'}），"
                    f"到期：{policy.get('expires_at') or '永久'}。"
                )
        policies = self.active_policies(state)
        if not policies:
            return "当前没有用户或群聊忽略策略。"
        lines = []
        for item in policies[:10]:
            target = item.get("user_id") or item.get("group_id") or "全局"
            lines.append(
                f"- {item.get('scope')} {target}，到期 "
                f"{item.get('expires_at') or '永久'}，原因："
                f"{str(item.get('reason') or '')[:60]}"
            )
        return "\n".join(lines)

    def can_send(
        self,
        state: dict[str, Any],
        *,
        umo: str,
        group_id: str,
        platform_id: str,
        moment: datetime,
    ) -> tuple[bool, str]:
        """Check quiet hours, allowlist, cooldowns and daily send budget."""

        if self.is_quiet(moment):
            return False, "quiet_hours"
        if not self.group_allowed(group_id, umo, platform_id):
            return False, "group_not_allowed"
        budget = self.ensure_budget(state, moment)
        if int(budget["messages_sent"]) >= self.settings.daily_active_message_budget:
            return False, "daily_message_budget_exhausted"
        stats = state.setdefault("send_stats", {})
        if self.settings.global_send_cooldown_minutes > 0:
            last_send = parse_iso(stats.get("last_send_at"), self.settings.tz)
            if last_send is not None:
                elapsed = (moment - last_send).total_seconds()
                if elapsed < self.settings.global_send_cooldown_minutes * 60:
                    return False, "global_cooldown"
        if group_id and self.settings.group_send_cooldown_minutes > 0:
            by_umo = stats.get("last_send_by_umo")
            if isinstance(by_umo, dict):
                last_group_send = parse_iso(by_umo.get(umo), self.settings.tz)
                if last_group_send is not None:
                    elapsed = (moment - last_group_send).total_seconds()
                    if elapsed < self.settings.group_send_cooldown_minutes * 60:
                        return False, "group_cooldown"
        return True, ""

    def record_send(
        self,
        state: dict[str, Any],
        *,
        umo: str,
        group_id: str,
        moment: datetime,
        text: str,
        result: str,
    ) -> None:
        budget = self.ensure_budget(state, moment)
        budget["messages_sent"] = int(budget["messages_sent"]) + 1
        stats = state.setdefault("send_stats", {})
        stats["last_send_at"] = to_iso(moment)
        by_umo = stats.setdefault("last_send_by_umo", {})
        if isinstance(by_umo, dict):
            by_umo[umo] = to_iso(moment)
        records = state.setdefault("send_records", [])
        records.append(
            {
                "timestamp": to_iso(moment),
                "umo": umo,
                "group_id": group_id,
                "result": result,
                "message_length": len(text),
                "message_hash": hashlib.sha1(text.encode()).hexdigest()[:16],
            }
        )
        del records[: max(0, len(records) - 100)]

    def describe_budget(self, state: dict[str, Any]) -> str:
        budget = self.ensure_budget(state, self.now())
        total = int(budget["decision_calls"]) + int(budget["main_calls"])
        return (
            f"日期 {budget['date']}：模型调用 {total}/"
            f"{self.settings.daily_model_call_budget}，"
            f"主动检查 {budget['active_checks']}/"
            f"{self.settings.daily_active_check_budget}，"
            f"主动发言 {budget['messages_sent']}/"
            f"{self.settings.daily_active_message_budget}"
        )

    @staticmethod
    def policy_id(scope: str, group_id: str, user_id: str) -> str:
        return f"{scope}:{group_id}:{user_id}"

    def build_policy(
        self,
        *,
        scope: str,
        group_id: str,
        user_id: str,
        reason: str,
        expires_at: datetime | None,
        created_by: str,
        moment: datetime,
    ) -> UserPolicy:
        return UserPolicy(
            policy_id=self.policy_id(scope, group_id, user_id),
            scope=scope,
            group_id=group_id,
            user_id=user_id,
            mode="ignore",
            reason=reason,
            expires_at=to_iso(expires_at) if expires_at else "",
            created_by=created_by,
            override="owner" if created_by == "owner" else "",
            created_at=to_iso(moment),
            updated_at=to_iso(moment),
        )

    def remaining_quiet_text(self, moment: datetime) -> str:
        end = self.quiet_end(moment)
        if end is None:
            return ""
        window = parse_quiet_hours(self.settings.quiet_hours)
        if window is None:
            return ""
        minutes = max(1, int((end - moment).total_seconds() // 60))
        return f"静默时段，{minutes} 分钟后结束"

    def debug_dump(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a compact, non-sensitive policy snapshot for logs."""

        return {
            "policies": len(state.get("policies") or {}),
            "checkpoints": len(state.get("checkpoints") or {}),
            "now": to_iso(self.now()),
            "ts": timestamp(self.now()),
        }
