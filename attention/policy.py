"""Hard policies, budgets, cooldowns and quiet hours.

Policy decisions always win over interest-model output. The manager only mutates
the in-memory state object; the caller is responsible for holding the store lock
and flushing it.

Two budgets are deliberately separate:

* model/semantic-check budget: global plugin gateway calls
  (``daily_model_call_budget`` / ``daily_active_check_budget``);
* autonomous participation budget: per-session attempts plus a global cap
  (``session_autonomous_reply_budget`` / ``daily_autonomous_reply_budget``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime
from typing import Any

from .models import UserPolicy  # noqa: F401  (public typing compatibility)
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
    """Evaluate hard policies and account for model/participation budgets."""

    def __init__(
        self,
        settings: Settings,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self._clock = clock

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    def now(self) -> datetime:
        """Return the current time, optionally from an injected clock."""

        if self._clock is not None:
            value = self._clock()
            if value.tzinfo is None:
                value = value.replace(tzinfo=self.settings.tz)
            return value.astimezone(self.settings.tz)
        return current_time(self.settings.tz)

    # ------------------------------------------------------------------
    # Global plugin gateway budget
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_budget(state: dict[str, Any], moment: datetime) -> dict[str, Any]:
        """Reset global daily counters when the local date changes."""

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
                    "autonomous_attempts": 0,
                    "messages_sent": 0,
                }
            )
        for key in (
            "decision_calls",
            "main_calls",
            "active_checks",
            "autonomous_attempts",
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
        """Reserve one global gateway call.

        Args:
            state: Mutable state dictionary.
            moment: Current local time.
            kind: ``decision`` for the interest model or ``main`` for the main
                reply model.

        Returns:
            ``True`` when a call was reserved.
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

    # ------------------------------------------------------------------
    # Session state and autonomous participation quota
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_session(
        state: dict[str, Any],
        umo: str,
        moment: datetime,
    ) -> dict[str, Any]:
        """Return the mutable session state, resetting the daily quota."""

        sessions = state.setdefault("sessions", {})
        if not isinstance(sessions, dict):
            sessions = {}
            state["sessions"] = sessions
        raw = sessions.get(umo)
        if not isinstance(raw, dict):
            raw = {
                "umo": umo,
                "external_message_seq": 0,
                "last_decision_seq": 0,
                "last_decision_at": "",
                "next_evaluation_at": "",
                "last_autonomous_participation_at": str(
                    ((state.get("send_stats") or {}).get("last_send_by_umo") or {}).get(
                        umo
                    )
                    or ""
                ),
                "quota_date": "",
                "autonomous_attempts": 0,
                "pending_reply_reservation": "",
                "revision": 1,
            }
            sessions[umo] = raw
        date_key = moment.date().isoformat()
        if raw.get("quota_date") != date_key:
            raw["quota_date"] = date_key
            raw["autonomous_attempts"] = 0
        raw["autonomous_attempts"] = int(raw.get("autonomous_attempts") or 0)
        return raw

    def can_reserve_autonomous(
        self,
        state: dict[str, Any],
        umo: str,
        moment: datetime,
    ) -> tuple[bool, str]:
        """Check session and global autonomous participation resources.

        Args:
            state: Mutable state dictionary.
            umo: Session that wants to participate.
            moment: Current local time.

        Returns:
            ``(True, "")`` when a reservation may be taken, otherwise
            ``(False, stable_reason_code)``.
        """

        if (
            self.settings.session_autonomous_reply_budget <= 0
            or self.settings.daily_autonomous_reply_budget <= 0
        ):
            return False, "quota_disabled"
        session = self.ensure_session(state, umo, moment)
        if session.get("pending_reply_reservation"):
            return False, "session_busy"
        if session["autonomous_attempts"] >= (
            self.settings.session_autonomous_reply_budget
        ):
            return False, "session_quota_exhausted"
        budget = self.ensure_budget(state, moment)
        if int(budget.get("autonomous_attempts") or 0) >= (
            self.settings.daily_autonomous_reply_budget
        ):
            return False, "global_quota_exhausted"
        stats = state.setdefault("send_stats", {})
        if self.settings.global_send_cooldown_minutes > 0:
            last_global = parse_iso(
                stats.get("last_autonomous_at") or stats.get("last_send_at"),
                self.settings.tz,
            )
            if last_global is not None:
                elapsed = (moment - last_global).total_seconds()
                if elapsed < self.settings.global_send_cooldown_minutes * 60:
                    return False, "global_cooldown"
        if self.settings.session_autonomous_cooldown_minutes > 0:
            last_session = parse_iso(
                session.get("last_autonomous_participation_at")
                or ((state.get("send_stats") or {}).get("last_send_by_umo") or {}).get(
                    umo
                ),
                self.settings.tz,
            )
            if last_session is not None:
                elapsed = (moment - last_session).total_seconds()
                if elapsed < (self.settings.session_autonomous_cooldown_minutes * 60):
                    return False, "session_cooldown"
        return True, ""

    def reserve_autonomous(
        self,
        state: dict[str, Any],
        umo: str,
        moment: datetime,
    ) -> None:
        """Atomically record one approved autonomous participation attempt.

        The caller must hold the store lock and have checked
        :meth:`can_reserve_autonomous` in the same critical section.
        """

        session = self.ensure_session(state, umo, moment)
        session["autonomous_attempts"] = int(session["autonomous_attempts"]) + 1
        session["last_autonomous_participation_at"] = to_iso(moment)
        session["revision"] = int(session.get("revision") or 1) + 1
        budget = self.ensure_budget(state, moment)
        budget["autonomous_attempts"] = int(budget["autonomous_attempts"]) + 1
        stats = state.setdefault("send_stats", {})
        stats["last_autonomous_at"] = to_iso(moment)

    def activation_elapsed_seconds(
        self,
        state: dict[str, Any],
        umo: str,
        moment: datetime,
    ) -> float | None:
        """Seconds since the last approved autonomous attempt in ``umo``."""

        session = self.ensure_session(state, umo, moment)
        last = parse_iso(
            session.get("last_autonomous_participation_at"), self.settings.tz
        )
        if last is None:
            return None
        return (moment - last).total_seconds()

    # ------------------------------------------------------------------
    # Quiet hours and group boundary
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Hard user/group ignore policies
    # ------------------------------------------------------------------

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
        """Record an actual platform send result for statistics only."""

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

    @staticmethod
    def policy_id(scope: str, group_id: str, user_id: str) -> str:
        return f"{scope}:{group_id}:{user_id}"

    def describe_budget(self, state: dict[str, Any], umo: str = "") -> str:
        budget = self.ensure_budget(state, self.now())
        total = int(budget["decision_calls"]) + int(budget["main_calls"])
        parts = [
            f"日期 {budget['date']}：模型调用 {total}/"
            f"{self.settings.daily_model_call_budget}，"
            f"主动检查 {budget['active_checks']}/"
            f"{self.settings.daily_active_check_budget}，"
            f"全局自主尝试 {budget['autonomous_attempts']}/"
            f"{self.settings.daily_autonomous_reply_budget}"
        ]
        if umo:
            session = self.ensure_session(state, umo, self.now())
            parts.append(
                f"会话自主尝试 {session['autonomous_attempts']}/"
                f"{self.settings.session_autonomous_reply_budget}"
            )
        return "，".join(parts)

    def debug_dump(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a compact, non-sensitive snapshot for logs."""

        return {
            "policies": len(state.get("policies") or {}),
            "interests": len(state.get("interests") or {}),
            "checkpoints": len(state.get("checkpoints") or {}),
            "now": to_iso(self.now()),
            "ts": timestamp(self.now()),
        }

    def remaining_quiet_text(self, moment: datetime) -> str:
        end = self.quiet_end(moment)
        if end is None:
            return ""
        window = parse_quiet_hours(self.settings.quiet_hours)
        if window is None:
            return ""
        minutes = max(1, int((end - moment).total_seconds() // 60))
        return f"静默时段，{minutes} 分钟后结束"
