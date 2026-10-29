"""Attention engine: coordinates passive and active attention sessions.

Every resource is owned by one UMO. The engine never mixes state between
sessions, never calls a model while holding the store lock, and never writes a
participation result as ``active`` after its lease or evidence became stale.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import wraps
from typing import Any

from astrbot.api import logger

from .active import ActiveAttentionRunner
from .control import Actor, AttentionControlService
from .models import (
    EXPLICIT_WAKE_SOURCES,
    SOURCE_ADMIN,
    SOURCE_ATTENTION_AGENT,
    SOURCE_MAIN_AGENT,
    SOURCE_SYSTEM,
    ActiveContext,
    ModelResult,
    ParticipationApproval,
    PassiveDecision,
    PassiveOutcome,
    PassiveRequest,
    RecentMessage,
)
from .passive import PassiveAttentionRunner, PassiveRunResult
from .policy import PolicyManager
from .ports import ContextProvider, MessageSender, ModelGateway
from .prompts import (
    build_active_reply_contexts,
    build_active_reply_prompt,
    build_active_reply_system_prompt,
)
from .scheduler import AttentionScheduler
from .settings import Settings
from .store import StateStore
from .timeutil import (
    add_minutes,
    in_quiet_hours,
    parse_iso,
    quiet_hours_end,
    timestamp,
    to_iso,
)
from .validation import clip_text, sanitize_reply


def _timestamp_from_iso(value: str, tz: Any) -> float:
    parsed = parse_iso(value, tz)
    return timestamp(parsed) if parsed is not None else 0.0


def _simulated_operation(operation):
    """Use isolated business state consistently across scheduling and evaluation."""

    @wraps(operation)
    async def wrapped(self, *args, **kwargs):
        dry_run = self.settings.dry_run
        with self.store.simulation(dry_run):
            outcome = await operation(self, *args, **kwargs)
        if isinstance(outcome, PassiveOutcome) and dry_run:
            outcome.dry_run = True
            outcome.block = False
        return outcome

    return wrapped


class AttentionEngine:
    """Own state, policies, budgets and both attention paths."""

    def __init__(
        self,
        *,
        settings: Settings,
        store: StateStore,
        gateway: ModelGateway,
        context_provider: ContextProvider,
        sender: MessageSender,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.gateway = gateway
        self.context_provider = context_provider
        self.sender = sender
        self.policy = PolicyManager(settings, clock)
        self.control = AttentionControlService(store, settings, self.policy)
        self.passive_runner = PassiveAttentionRunner(settings, gateway)
        self.active_runner = ActiveAttentionRunner(settings, gateway)
        self.scheduler = AttentionScheduler(self)
        self._passive_semaphore = asyncio.Semaphore(settings.max_concurrent_decisions)
        self._passive_inflight: set[str] = set()
        self._passive_done: OrderedDict[str, PassiveOutcome] = OrderedDict()
        self._schedule_cursor = 0
        self._started = False
        self._settings_revision = 0
        self._evaluating_sessions: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load persisted state, recover interrupted work and schedule."""

        await self.store.load(
            interest_ttl_minutes=self.settings.interest_ttl_minutes,
            interest_check_limit=self.settings.interest_check_limit,
            moment=self.policy.now(),
            min_interval_minutes=self.settings.min_interval_minutes,
        )
        async with self.store.lock:
            moment = self.policy.now()
            self.control.expire_interests_locked(self.store.state, moment)
            self.control.cleanup_locked(self.store.state, moment)
            self.policy.expire_policies(self.store.state, moment)
            self._recover_running_locked()
            self._recover_reservations_locked()
            self._seed_checkpoints_locked()
            self._flush_locked()
        self._started = True
        await self.scheduler.start()
        logger.info(
            "Autonomous attention core started: passive=%s active=%s dry_run=%s.",
            self.settings.passive_mode,
            self.settings.active_enabled,
            self.settings.dry_run,
        )

    async def stop(self) -> None:
        """Stop all background work and persist the final state."""

        if not self._started:
            return
        self._started = False
        await self.scheduler.stop()
        async with self.store.lock:
            self._flush_locked()
        logger.info("Autonomous attention core stopped.")

    def update_settings(self, settings: Settings) -> None:
        """Apply a new configuration snapshot without restarting the plugin."""

        if settings != self.settings:
            self._settings_revision += 1
            self._passive_done.clear()
            self.store.discard_shadow()
        self.settings = settings
        self.policy.update_settings(settings)
        self.control.update_settings(settings)
        self.passive_runner.update_settings(settings)
        self.active_runner.update_settings(settings)
        self._passive_semaphore = asyncio.Semaphore(settings.max_concurrent_decisions)

    def _flush_locked(self) -> None:
        self.control.cleanup_locked(self.store.state, self.policy.now())
        self.store.flush()

    def _recover_running_locked(self) -> None:
        moment = self.policy.now()
        for raw in (self.store.state.get("checkpoints") or {}).values():
            if not isinstance(raw, dict) or raw.get("status") != "running":
                continue
            raw["status"] = "active"
            raw["run_id"] = ""
            raw["last_result"] = "interrupted"
            raw["last_error"] = "plugin restarted while the checkpoint was running"
            raw["revision"] = int(raw.get("revision") or 1) + 1
            raw["updated_at"] = to_iso(moment)
            raw["scheduled_at"] = to_iso(
                moment + timedelta(minutes=self.settings.min_interval_minutes)
            )
            logger.warning(
                "Recovered interrupted checkpoint %s; retry after the minimum interval.",
                raw.get("checkpoint_id"),
            )

    def _recover_reservations_locked(self) -> None:
        moment = self.policy.now()
        changed = 0
        for raw in (self.store.state.get("opportunities") or {}).values():
            if not isinstance(raw, dict) or raw.get("status") != "reserved":
                continue
            raw["status"] = "uncertain"
            raw["outcome"] = "restart_uncertain"
            raw["updated_at"] = to_iso(moment)
            raw["revision"] = int(raw.get("revision") or 1) + 1
            changed += 1
        for session in (self.store.state.get("sessions") or {}).values():
            if not isinstance(session, dict):
                continue
            if session.get("pending_reply_reservation"):
                session["pending_reply_reservation"] = ""
                session["revision"] = int(session.get("revision") or 1) + 1
        if changed:
            logger.warning(
                "Marked %d interrupted participation reservations as uncertain.",
                changed,
            )

    def _seed_checkpoints_locked(self) -> None:
        seeded = self.store.state.setdefault("seeded", [])
        if not isinstance(seeded, list):
            seeded = []
            self.store.state["seeded"] = seeded
        for item in self.settings.seed_checkpoints:
            signature = "|".join(
                str(item.get(key) or "")
                for key in ("session", "umo", "platform_id", "group_id", "at", "reason")
            )
            if signature in seeded:
                continue
            try:
                session = str(item.get("session") or item.get("umo") or "").strip()
                platform_id = str(item.get("platform_id") or "").strip()
                group_id = str(item.get("group_id") or "").strip()
                umo, platform_id, group_id = self._resolve_target(
                    session,
                    platform_id,
                    group_id,
                )
                if not group_id:
                    raise ValueError("seed checkpoint must target a group")
                scheduled = parse_iso(
                    str(item.get("at") or item.get("scheduled_at") or ""),
                    self.settings.tz,
                )
                reason = clip_text(
                    item.get("reason") or "configuration seed checkpoint",
                    200,
                )
                actor = Actor(
                    source=SOURCE_ADMIN,
                    session=umo,
                    group_id=group_id,
                    platform_id=platform_id,
                    is_admin=True,
                )
                result = self.control.create_checkpoint_locked(
                    self.store.state,
                    actor,
                    umo=umo,
                    platform_id=platform_id,
                    group_id=group_id,
                    kind="session_scan",
                    scheduled_at=scheduled,
                    reason=reason,
                    source=SOURCE_ADMIN,
                )
                if not result.ok:
                    raise ValueError(result.message)
                seeded.append(signature)
                logger.info("Seeded attention checkpoint for %s (%s).", umo, reason)
            except Exception as exc:
                logger.warning("Ignoring invalid seed checkpoint %s: %s", item, exc)

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _session_locked(
        self, umo: str, moment: datetime | None = None
    ) -> dict[str, Any]:
        return self.policy.ensure_session(
            self.store.state,
            umo,
            moment or self.policy.now(),
        )

    @staticmethod
    def _ids_from_messages(
        messages: list[RecentMessage],
    ) -> tuple[set[str], set[str]]:
        valid: set[str] = set()
        bot: set[str] = set()
        for message in messages:
            if not message.message_id:
                continue
            valid.add(message.message_id)
            if message.is_bot:
                bot.add(message.message_id)
        return valid, bot

    def _window_locked(self, umo: str) -> list[RecentMessage]:
        return self.store.recent_window(umo)

    def _window_ids_locked(self, umo: str) -> tuple[set[str], set[str]]:
        valid: set[str] = set()
        bot: set[str] = set()
        for message in self._window_locked(umo):
            if not message.message_id:
                continue
            valid.add(message.message_id)
            if message.is_bot:
                bot.add(message.message_id)
        return valid, bot

    def _ensure_message_locked(
        self,
        *,
        umo: str,
        sender_id: str,
        sender_name: str,
        text: str,
        message_id: str,
        created_at: str,
        created_ts: float,
        is_bot: bool = False,
    ) -> int:
        """Insert a message once and return its stable per-session sequence."""

        message_id = str(message_id or "").strip()
        for message in self._window_locked(umo):
            if message_id and message_id in [message.message_id, *message.aliases]:
                return message.seq
            if (
                not message_id
                and not message.is_bot
                and message.sender_id == sender_id
                and message.text == text
                and abs(message.created_ts - created_ts) < 1.0
            ):
                return message.seq
        session = self._session_locked(umo)
        known = session.setdefault("message_ids", {})
        if message_id and message_id in known:
            return int(known[message_id])
        if is_bot:
            seq = 0
        else:
            session["external_message_seq"] = (
                int(session.get("external_message_seq") or 0) + 1
            )
            session["revision"] = int(session.get("revision") or 1) + 1
            seq = int(session["external_message_seq"])
        if not message_id and not is_bot:
            digest = hashlib.sha1(umo.encode("utf-8")).hexdigest()[:8]
            message_id = f"local_{digest}_{seq}"
        if message_id:
            known[message_id] = seq
            while len(known) > 500:
                known.pop(next(iter(known)))
        if not is_bot:
            session["received_messages"] = (
                int(session.get("received_messages") or 0) + 1
            )
            shapes = session.setdefault("history_shapes", [])
            shapes.append(
                {
                    "sender_id": sender_id,
                    "text_digest": hashlib.sha256(
                        clip_text(text, 1000).encode()
                    ).hexdigest(),
                    "ts": created_ts,
                    "id": message_id,
                    "seq": seq,
                }
            )
            del shapes[:-500]
        self.store.remember_recent(
            umo,
            RecentMessage(
                role="bot" if is_bot else "user",
                sender_id=sender_id,
                sender_name=sender_name,
                text=clip_text(text, 1000),
                created_at=created_at,
                created_ts=created_ts,
                message_id=message_id,
                seq=seq,
                umo=umo,
                is_bot=is_bot,
            ),
            max_messages=max(20, self.settings.context_window_messages * 2),
            max_sessions=self.settings.max_context_sessions,
        )
        return seq

    def _evaluation_stamp_locked(self, umo: str) -> str:
        """Fingerprint controls that must invalidate asynchronous model results."""
        state = self.store.state
        controls = {
            "settings": self._settings_revision,
            "policies": state.get("policies"),
            "directives": {
                k: v
                for k, v in (state.get("directives") or {}).items()
                if isinstance(v, dict) and v.get("created_by_session") == umo
            },
            "interests": {
                k: v
                for k, v in (state.get("interests") or {}).items()
                if isinstance(v, dict) and v.get("umo") == umo
            },
        }
        return hashlib.sha256(json.dumps(controls, sort_keys=True).encode()).hexdigest()

    def _revalidate_locked(
        self,
        umo: str,
        evaluated_seq: int,
        stamp: str,
        *,
        checkpoint: dict[str, Any] | None = None,
        approval: ParticipationApproval | None = None,
    ) -> str:
        """Check live controls, snapshot, lease and reservation before committing.

        Args:
            umo: Trusted owner session.
            evaluated_seq: Last external sequence actually shown to the model.
            stamp: Control fingerprint captured before the asynchronous call.
            checkpoint: Expected checkpoint lease, for an active run.
            approval: Expected opportunity reservation, before generation/send.

        Returns:
            A stable rejection code, or an empty string when valid.
        """
        moment = self.policy.now()
        state = self.store.state
        if not self._started or not self.settings.enabled:
            return "policy_changed"
        if self._evaluation_stamp_locked(umo) != stamp:
            return "policy_changed"
        if self.policy.is_quiet(moment):
            return "quiet_hours"
        session = self._session_locked(umo, moment)
        if int(session.get("external_message_seq") or 0) > evaluated_seq:
            return "stale_snapshot"
        if self.control.stop_directive(state, umo=umo) is not None:
            if checkpoint is None or checkpoint.get("source") != SOURCE_ADMIN:
                return "stop_directive"
        if checkpoint is not None:
            raw = (state.get("checkpoints") or {}).get(checkpoint.get("checkpoint_id"))
            if (
                not isinstance(raw, dict)
                or raw.get("status") != "running"
                or raw.get("run_id") != checkpoint.get("run_id")
            ):
                return "lease_invalid"
            expires = parse_iso(raw.get("expires_at"), self.settings.tz)
            if expires is None or expires <= moment:
                return "checkpoint_expired"
            if not self.settings.active_enabled or self.policy.is_quiet(moment):
                return "policy_changed"
            if not self.policy.group_allowed(
                str(raw.get("group_id") or ""), umo, str(raw.get("platform_id") or "")
            ):
                return "policy_changed"
        if approval is not None:
            opportunity = (state.get("opportunities") or {}).get(
                (approval.opportunity or {}).get("opportunity_id")
            )
            if (
                not isinstance(opportunity, dict)
                or opportunity.get("umo") != umo
                or opportunity.get("status") != "reserved"
                or opportunity.get("reservation_id") != approval.reservation_id
                or session.get("pending_reply_reservation") != approval.reservation_id
            ):
                return "reservation_invalid"
            expires = parse_iso(opportunity.get("expires_at"), self.settings.tz)
            if expires is None or expires <= moment:
                return "opportunity_expired"
            interest = (state.get("interests") or {}).get(approval.interest_id)
            if not isinstance(interest, dict) or interest.get("umo") != umo:
                return "interest_not_found"
            expires = parse_iso(interest.get("expires_at"), self.settings.tz)
            if (
                interest.get("status") != "watching"
                or expires is None
                or expires <= moment
            ):
                return "interest_expired"
            anchor = next(
                (
                    m
                    for m in self._window_locked(umo)
                    if m.message_id == opportunity.get("anchor_message_id")
                ),
                None,
            )
            if anchor is None or self.policy.is_user_blocked(
                state,
                group_id=str(
                    session.get("group_id") or (checkpoint or {}).get("group_id") or ""
                ),
                user_id=anchor.sender_id,
                umo=umo,
                platform_id=str(session.get("platform_id") or ""),
                is_admin=False,
            ):
                return "hard_policy"
        return ""

    # ------------------------------------------------------------------
    # Observation / hard policies
    # ------------------------------------------------------------------

    async def observe_message(
        self,
        *,
        umo: str,
        platform_id: str,
        group_id: str,
        sender_id: str,
        sender_name: str,
        text: str,
        message_id: str,
        created_at: str,
        created_ts: float,
        is_bot: bool = False,
    ) -> None:
        """Record one group message and advance the session external version."""

        if not self._started or not self.settings.enabled:
            return
        if not group_id:
            return
        if not self.policy.group_allowed(group_id, umo, platform_id):
            return
        if not text.strip():
            return
        async with self.store.lock:
            if self.control._owner_mismatch(umo, platform_id, group_id):
                raise ValueError("scope_mismatch")
            session = self._session_locked(umo)
            session.update(platform_id=platform_id, group_id=group_id)
            self._ensure_message_locked(
                umo=umo,
                sender_id=sender_id,
                sender_name=sender_name,
                text=text,
                message_id=message_id,
                created_at=created_at,
                created_ts=created_ts,
                is_bot=is_bot,
            )

    async def check_hard_policy(
        self,
        *,
        group_id: str,
        user_id: str,
        umo: str,
        platform_id: str,
        is_admin: bool,
    ) -> dict[str, Any] | None:
        """Return the hard ignore policy for a sender, if any."""

        if not self._started or not self.settings.enabled:
            return None
        async with self.store.lock:
            moment = self.policy.now()
            self.policy.expire_policies(self.store.state, moment)
            policy = self.policy.is_user_blocked(
                self.store.state,
                group_id=group_id,
                user_id=user_id,
                umo=umo,
                platform_id=platform_id,
                is_admin=is_admin,
            )
            return dict(policy) if policy is not None else None

    # ------------------------------------------------------------------
    # Passive attention
    # ------------------------------------------------------------------

    @_simulated_operation
    async def handle_passive(self, request: PassiveRequest) -> PassiveOutcome:
        """Run the passive path for one woken message.

        Explicit requests pass directly after hard policy checks. Autonomous
        suggestions need code approval through the control layer; failures
        never default to passing the main model.
        """

        if not self._started or not self.settings.enabled:
            return PassiveOutcome(
                decision="pass", block=False, reason="plugin disabled"
            )
        if not self.settings.passive_full:
            return PassiveOutcome(
                decision="pass",
                block=False,
                reason=f"passive mode is {self.settings.passive_mode}",
            )
        if request.is_command:
            return PassiveOutcome(decision="pass", block=False, reason="command event")
        moment = self.policy.now()
        if not request.timestamp:
            request.timestamp = to_iso(moment)
        if request.is_group and not self.policy.group_allowed(
            request.group_id,
            request.umo,
            request.platform_id,
        ):
            return PassiveOutcome(
                decision="pass", block=False, reason="group not allowed"
            )

        with self.store.simulation(False):
            async with self.store.lock:
                if self.control._owner_mismatch(
                    request.umo, request.platform_id, request.group_id
                ):
                    return PassiveOutcome(
                        code="scope_mismatch", reason="session route mismatch"
                    )
                live_session = self._session_locked(request.umo, moment)
                live_session.update(
                    platform_id=request.platform_id, group_id=request.group_id
                )
                self._ensure_message_locked(
                    umo=request.umo,
                    sender_id=request.user_id,
                    sender_name=request.sender_name,
                    text=request.message_text,
                    message_id=request.message_id,
                    created_at=request.timestamp,
                    created_ts=_timestamp_from_iso(request.timestamp, self.settings.tz),
                )
        key = self._passive_key(request)
        async with self.store.lock:
            self.policy.expire_policies(self.store.state, moment)
            self.control.expire_interests_locked(self.store.state, moment)
            done = self._passive_done.get(key)
            if done is not None:
                self._record_decision_locked(
                    kind="passive",
                    umo=request.umo,
                    trigger=request.trigger,
                    code_decision="observe",
                    code="duplicate_event",
                    reason="event already handled",
                    wake_source=request.wake_source,
                    cached=True,
                )
                self._flush_locked()
                return PassiveOutcome(
                    decision="observe",
                    block=True,
                    code="duplicate_event",
                    reason="event already handled",
                )
            if key in self._passive_inflight:
                return PassiveOutcome(
                    decision="observe",
                    block=True,
                    reason="duplicate passive decision is already running",
                )
            if self.control._owner_mismatch(
                request.umo, request.platform_id, request.group_id
            ):
                return PassiveOutcome(
                    code="scope_mismatch", reason="session route mismatch"
                )
            session = self._session_locked(request.umo, moment)
            session.update(platform_id=request.platform_id, group_id=request.group_id)
            message_seq = self._ensure_message_locked(
                umo=request.umo,
                sender_id=request.user_id,
                sender_name=request.sender_name,
                text=request.message_text,
                message_id=request.message_id,
                created_at=request.timestamp,
                created_ts=_timestamp_from_iso(request.timestamp, self.settings.tz),
            )
            request.message_seq = message_seq
            if not request.message_id:
                request.message_id = next(
                    (
                        m.message_id
                        for m in self._window_locked(request.umo)
                        if m.seq == message_seq
                    ),
                    "",
                )
            blocked = self.policy.is_user_blocked(
                self.store.state,
                group_id=request.group_id,
                user_id=request.user_id,
                umo=request.umo,
                platform_id=request.platform_id,
                is_admin=request.is_admin,
            )
            if blocked is not None:
                self._remember_passive(
                    key,
                    PassiveOutcome(
                        decision="ignore",
                        block=True,
                        reason=str(blocked.get("reason") or "hard policy"),
                        code="hard_policy",
                    ),
                )
                self._record_decision_locked(
                    kind="passive",
                    umo=request.umo,
                    trigger=request.trigger,
                    wake_source=request.wake_source,
                    code_decision="ignore",
                    reason=str(blocked.get("reason") or "hard policy"),
                    source=(
                        SOURCE_ADMIN
                        if blocked.get("source") == SOURCE_ADMIN
                        else SOURCE_MAIN_AGENT
                        if blocked.get("source") == SOURCE_MAIN_AGENT
                        else SOURCE_ATTENTION_AGENT
                    ),
                    related_policy_id=str(blocked.get("policy_id") or ""),
                    message_seq=message_seq,
                )
                self._flush_locked()
                return PassiveOutcome(
                    decision="ignore",
                    block=True,
                    reason="hard user policy",
                    code="hard_policy",
                )

            if request.wake_source in EXPLICIT_WAKE_SOURCES:
                approval = self.control.approve_participation_locked(
                    self.store.state,
                    umo=request.umo,
                    wake_source=request.wake_source,
                    suggested_action="reply",
                    opportunity_payload=None,
                    moment=moment,
                    anchor_message_id=request.message_id,
                )
                if not approval.approved:
                    return PassiveOutcome(code=approval.code, reason=approval.message)
                self._record_decision_locked(
                    kind="passive",
                    umo=request.umo,
                    trigger=request.trigger,
                    wake_source=request.wake_source,
                    code_decision="explicit_pass",
                    suggested_action="reply",
                    reason="explicit request",
                    source=SOURCE_MAIN_AGENT,
                    anchor_message_id=request.message_id,
                    reservation_id=approval.reservation_id,
                    message_seq=message_seq,
                )
                self._remember_passive(
                    key,
                    PassiveOutcome(
                        decision="pass",
                        block=False,
                        reason="explicit request",
                        code="explicit_pass",
                        opportunity_id=str(
                            (approval.opportunity or {}).get("opportunity_id") or ""
                        ),
                        reservation_id=approval.reservation_id,
                    ),
                )
                self._flush_locked()
                return self._passive_done[key]

            stop = self.control.stop_directive(self.store.state, umo=request.umo)
            if stop is not None:
                self._record_decision_locked(
                    kind="passive",
                    umo=request.umo,
                    trigger=request.trigger,
                    wake_source=request.wake_source,
                    code_decision="observe",
                    code="stop_directive",
                    reason="main agent stop_focus directive",
                    message_seq=message_seq,
                )
                self._remember_passive(
                    key,
                    PassiveOutcome(
                        decision="observe",
                        block=True,
                        reason="main agent stop_focus directive",
                        code="stop_directive",
                    ),
                )
                self._flush_locked()
                return self._passive_done[key]

            if int(session.get("external_message_seq") or 0) <= int(
                session.get("last_decision_seq") or 0
            ):
                self._record_decision_locked(
                    kind="passive",
                    umo=request.umo,
                    trigger=request.trigger,
                    wake_source=request.wake_source,
                    code_decision="observe",
                    code="no_new_messages",
                    reason="no new external message since the last decision",
                    message_seq=message_seq,
                )
                self._remember_passive(
                    key,
                    PassiveOutcome(
                        decision="observe",
                        block=True,
                        reason="no new external message",
                        code="no_new_messages",
                    ),
                )
                self._flush_locked()
                return self._passive_done[key]

            next_evaluation = parse_iso(
                session.get("next_evaluation_at"), self.settings.tz
            )
            last_decision = parse_iso(session.get("last_decision_at"), self.settings.tz)
            throttled = (next_evaluation is not None and moment < next_evaluation) or (
                last_decision is not None
                and (moment - last_decision).total_seconds()
                < self.settings.min_evaluation_interval_seconds
            )
            if throttled:
                self._record_decision_locked(
                    kind="passive",
                    umo=request.umo,
                    trigger=request.trigger,
                    wake_source=request.wake_source,
                    code_decision="observe",
                    code="evaluation_throttled",
                    reason="minimum evaluation interval has not elapsed",
                    message_seq=message_seq,
                )
                self._remember_passive(
                    key,
                    PassiveOutcome(
                        decision="observe",
                        block=True,
                        reason="evaluation throttled",
                        code="evaluation_throttled",
                    ),
                )
                self._flush_locked()
                return self._passive_done[key]

            if self.policy.is_quiet(moment):
                return PassiveOutcome(
                    code="quiet_hours", reason="autonomous participation is quiet"
                )
            if request.umo in self._evaluating_sessions or any(
                isinstance(cp, dict)
                and cp.get("umo") == request.umo
                and cp.get("status") == "running"
                for cp in (self.store.state.get("checkpoints") or {}).values()
            ):
                return PassiveOutcome(
                    code="session_busy", reason="session evaluation in progress"
                )
            interests = self.control.session_interests_locked(
                self.store.state, request.umo
            )
            checkpoints = self.control.session_checkpoints_locked(
                self.store.state, request.umo
            )
            opportunities = self._recent_opportunities_locked(request.umo)
            policy_summary = self.control.prompt_summary_locked(
                self.store.state,
                umo=request.umo,
                user_id=request.user_id,
                group_id=request.group_id,
                platform_id=request.platform_id,
                message_text=request.message_text,
                message_id=request.message_id,
            )
            window = self._window_locked(request.umo)
            resources = self._resources_locked(request.umo)
            allowed, quota_reason = self.policy.can_reserve_autonomous(
                self.store.state, request.umo, moment
            )
            if not allowed:
                self._record_decision_locked(
                    kind="passive",
                    umo=request.umo,
                    trigger=request.trigger,
                    wake_source=request.wake_source,
                    code_decision="observe",
                    code=quota_reason,
                    reason="autonomous participation quota unavailable",
                    message_seq=message_seq,
                )
                self._remember_passive(
                    key,
                    PassiveOutcome(
                        decision="observe",
                        block=True,
                        reason="autonomous participation quota unavailable",
                        code=quota_reason,
                    ),
                )
                self._flush_locked()
                return self._passive_done[key]
            if not self.policy.consume_model_call(
                self.store.live_state, moment, "decision"
            ):
                self._record_decision_locked(
                    kind="passive",
                    umo=request.umo,
                    trigger=request.trigger,
                    wake_source=request.wake_source,
                    code_decision="observe",
                    code="budget_exhausted",
                    reason="daily model-call budget exhausted",
                    message_seq=message_seq,
                )
                self._remember_passive(
                    key,
                    PassiveOutcome(
                        decision="observe",
                        block=True,
                        reason="daily model-call budget exhausted",
                        code="budget_exhausted",
                    ),
                )
                self._flush_locked()
                return self._passive_done[key]
            evaluated_seq = int(session.get("external_message_seq") or 0)
            evaluation_stamp = self._evaluation_stamp_locked(request.umo)
            valid_ids, bot_ids = self._ids_from_messages(window)
            self._passive_inflight.add(key)
            self._evaluating_sessions.add(request.umo)
            session["next_evaluation_at"] = to_iso(
                moment
                + timedelta(seconds=self.settings.min_evaluation_interval_seconds)
            )
            try:
                self._flush_locked()
            except Exception:
                self._passive_inflight.discard(key)
                self._evaluating_sessions.discard(request.umo)
                raise

        try:
            async with self._passive_semaphore:
                run = await self.passive_runner.decide(
                    request,
                    interests=interests,
                    checkpoints=checkpoints,
                    policy_summary=policy_summary,
                    opportunities=opportunities,
                    resources=resources,
                    window=window,
                )
        except asyncio.CancelledError:
            self._passive_inflight.discard(key)
            self._evaluating_sessions.discard(request.umo)
            raise
        except Exception as exc:
            logger.exception("Passive attention runner failed.")
            run = PassiveRunResult(kind="error", error=str(exc))

        async with self.store.lock:
            self._passive_inflight.discard(key)
            self._evaluating_sessions.discard(request.umo)
            now_after = self.policy.now()
            fallback_code = ""
            if run.kind == "ok" and run.decision is not None:
                decision = run.decision
            else:
                fallback_code = (
                    "invalid_output" if run.kind == "invalid" else "model_error"
                )
                decision = PassiveDecision(
                    suggested_action="observe",
                    interest_level="none" if run.kind != "invalid" else "low",
                    reason=(
                        "invalid structured output; conservative fallback"
                        if run.kind == "invalid"
                        else "decision model unavailable; conservative fallback"
                    ),
                )
                logger.warning(
                    "Passive decision model failed for %s: %s. Observing instead "
                    "of passing through.",
                    request.umo,
                    run.error,
                )
            invalidated = self._revalidate_locked(
                request.umo, evaluated_seq, evaluation_stamp
            )
            if invalidated:
                outcome = PassiveOutcome(
                    code=invalidated,
                    reason="controls or messages changed during evaluation",
                )
                approval = None
            else:
                if decision.suggested_action not in {"end_focus", "policy_update"}:
                    self._apply_interest_update_locked(
                        {
                            "umo": request.umo,
                            "platform_id": request.platform_id,
                            "group_id": request.group_id,
                        },
                        decision,
                        now_after,
                        valid_ids=valid_ids,
                    )
                outcome, approval = self._apply_passive_decision_locked(
                    request, decision, now_after, valid_ids=valid_ids, bot_ids=bot_ids
                )
            outcome.model_error = run.error if run.kind != "ok" else ""
            if fallback_code and not invalidated:
                outcome.code = fallback_code
            if not invalidated and run.kind == "ok":
                self._consume_passive_interest_check_locked(
                    request.umo, decision, now_after
                )
            self._remember_passive(key, outcome)
            self._record_decision_locked(
                kind="passive",
                umo=request.umo,
                trigger=request.trigger,
                wake_source=request.wake_source,
                code_decision=outcome.decision,
                suggested_action=decision.suggested_action if run.kind == "ok" else "",
                code=outcome.code or fallback_code,
                reason=decision.reason,
                source=SOURCE_ATTENTION_AGENT,
                anchor_message_id=str(
                    (decision.opportunity or {}).get("anchor_message_id") or ""
                ),
                reservation_id=outcome.reservation_id,
                message_seq=evaluated_seq,
                interest_id=str(
                    (decision.opportunity or {}).get("interest_id")
                    or (decision.interest_update or {}).get("interest_id")
                    or ""
                ),
                model_suggestion=decision.suggested_action if run.kind == "ok" else "",
                semantic_evaluations=int(run.kind == "ok"),
                decision_model_calls=1,
            )
            if not invalidated and run.kind == "ok":
                session = self._session_locked(request.umo, now_after)
                session["last_decision_seq"] = max(
                    int(session.get("last_decision_seq") or 0),
                    evaluated_seq,
                )
                session["last_decision_at"] = to_iso(now_after)
                session["next_evaluation_at"] = to_iso(
                    now_after
                    + timedelta(seconds=self.settings.min_evaluation_interval_seconds)
                )
                session["revision"] = int(session.get("revision") or 1) + 1
            self._flush_locked()

        logger.info(
            "Passive attention: umo=%s user=%s decision=%s code=%s reason=%s",
            request.umo,
            request.user_id or "-",
            outcome.decision,
            outcome.code or "-",
            clip_text(outcome.reason, 120),
        )
        return outcome

    def _apply_passive_decision_locked(
        self,
        request: PassiveRequest,
        decision: PassiveDecision,
        moment: datetime,
        *,
        valid_ids: set[str],
        bot_ids: set[str],
    ) -> tuple[PassiveOutcome, ParticipationApproval | None]:
        action = decision.suggested_action
        if action == "reply":
            approval = self._approve_with_interest_locked(
                request,
                decision,
                moment,
                valid_ids=valid_ids,
                bot_ids=bot_ids,
            )
            if approval.approved:
                return (
                    PassiveOutcome(
                        decision="pass",
                        block=False,
                        reason=decision.reason or "approved participation",
                        code="autonomous_approved",
                        opportunity_id=str(
                            (approval.opportunity or {}).get("opportunity_id") or ""
                        ),
                        reservation_id=approval.reservation_id,
                    ),
                    approval,
                )
            return (
                PassiveOutcome(
                    decision="observe",
                    block=True,
                    reason=approval.message or "participation rejected",
                    code=approval.code or "not_approved",
                ),
                approval,
            )
        if action == "ignore":
            self._note_passive_no_progress_locked(request.umo, decision, moment)
            return (
                PassiveOutcome(
                    decision="ignore",
                    block=True,
                    reason=decision.reason or "ignored by interest model",
                    code="model_ignore",
                ),
                None,
            )
        if action == "policy_update":
            if decision.user_action:
                self._apply_auto_policy_locked(request, decision.user_action, moment)
            return (
                PassiveOutcome(
                    decision="ignore",
                    block=True,
                    reason=decision.reason or "policy updated",
                    code="policy_update",
                ),
                None,
            )
        if action == "retry_later":
            self._schedule_retry_locked(request, decision, moment)
            return (
                PassiveOutcome(
                    decision="observe",
                    block=True,
                    reason=decision.reason or "retry later",
                    code="retry_later",
                ),
                None,
            )
        if action == "end_focus":
            self._close_interest_from_update_locked(request.umo, decision, moment)
            return (
                PassiveOutcome(
                    decision="observe",
                    block=True,
                    reason=decision.reason or "interest closed",
                    code="end_focus",
                ),
                None,
            )
        self._note_passive_no_progress_locked(request.umo, decision, moment)
        return (
            PassiveOutcome(
                decision="observe",
                block=True,
                reason=decision.reason or "observe",
                code="model_observe",
            ),
            None,
        )

    def _consume_passive_interest_check_locked(
        self,
        umo: str,
        decision: PassiveDecision,
        moment: datetime,
    ) -> None:
        """Consume one interest check for a completed passive semantic evaluation."""

        interest_id = str(
            (decision.opportunity or {}).get("interest_id")
            or (decision.interest_update or {}).get("interest_id")
            or ""
        )
        if not interest_id:
            return
        raw = (self.store.state.get("interests") or {}).get(interest_id)
        if not isinstance(raw, dict) or raw.get("umo") != umo:
            return
        if raw.get("status") == "closed":
            return
        raw["remaining_checks"] = max(0, int(raw.get("remaining_checks") or 0) - 1)
        raw["last_evaluated_seq"] = int(
            self._session_locked(umo, moment).get("external_message_seq") or 0
        )
        for cp in (self.store.state.get("checkpoints") or {}).values():
            if (
                isinstance(cp, dict)
                and cp.get("umo") == umo
                and cp.get("interest_id") == interest_id
            ):
                cp["remaining_checks"] = raw["remaining_checks"]
                cp["no_progress_count"] = raw.get("no_progress_count", 0)
        raw["updated_at"] = to_iso(moment)
        raw["revision"] = int(raw.get("revision") or 1) + 1
        if int(raw["remaining_checks"]) <= 0:
            self.control.close_interest_locked(
                self.store.state,
                Actor(source=SOURCE_SYSTEM, session=umo, is_admin=True),
                interest_id=interest_id,
                reason="checks_exhausted",
                moment=moment,
            )

    def _note_passive_no_progress_locked(
        self,
        umo: str,
        decision: PassiveDecision,
        moment: datetime,
    ) -> None:
        if (decision.interest_update or {}).get("_progress"):
            return
        interest_id = str(
            (decision.interest_update or {}).get("interest_id")
            or (decision.opportunity or {}).get("interest_id")
            or ""
        )
        if not interest_id:
            return
        raw = (self.store.state.get("interests") or {}).get(interest_id)
        if not isinstance(raw, dict) or raw.get("umo") != umo:
            return
        if raw.get("status") == "closed":
            return
        raw["no_progress_count"] = int(raw.get("no_progress_count") or 0) + 1
        raw["updated_at"] = to_iso(moment)
        raw["revision"] = int(raw.get("revision") or 1) + 1
        if int(raw["no_progress_count"]) >= self.settings.no_progress_limit:
            self.control.close_interest_locked(
                self.store.state,
                Actor(
                    source=SOURCE_SYSTEM,
                    session=umo,
                    is_admin=True,
                ),
                interest_id=interest_id,
                reason="no_progress_limit",
                moment=moment,
            )

    def _approve_with_interest_locked(
        self,
        request: PassiveRequest,
        decision: PassiveDecision,
        moment: datetime,
        *,
        valid_ids: set[str],
        bot_ids: set[str],
    ) -> ParticipationApproval:
        payload = dict(decision.opportunity or {})
        interest_id = str(payload.get("interest_id") or "")
        if not interest_id:
            interest_id = str((decision.interest_update or {}).get("interest_id") or "")
        if not interest_id:
            admitted = self._admit_interest_from_update_locked(
                request.umo,
                decision.interest_update,
                payload,
                moment,
                valid_ids=valid_ids,
            )
            if admitted is not None:
                interest_id = str(admitted.get("interest_id") or "")
        if not interest_id:
            existing = self.control.session_interests_locked(
                self.store.state, request.umo
            )
            if len(existing) == 1:
                interest_id = str(existing[0].get("interest_id") or "")
        if not interest_id:
            return ParticipationApproval(
                approved=False,
                code="interest_required",
                message="自主参与必须引用一个已准入兴趣。",
                wake_source=request.wake_source,
            )
        payload["interest_id"] = interest_id
        return self.control.approve_participation_locked(
            self.store.state,
            umo=request.umo,
            wake_source=request.wake_source,
            suggested_action="reply",
            opportunity_payload=payload,
            moment=moment,
            valid_ids=valid_ids,
            bot_ids=bot_ids,
            anchor_message_id=str(payload.get("anchor_message_id") or ""),
            interest_id=interest_id,
            interest_update=decision.interest_update,
            consume_on_approval=True,
        )

    def _admit_interest_from_update_locked(
        self,
        umo: str,
        update: dict[str, Any] | None,
        opportunity: dict[str, Any] | None,
        moment: datetime,
        *,
        valid_ids: set[str],
    ) -> dict[str, Any] | None:
        update = update or {}
        opportunity = opportunity or {}
        summary = str(update.get("summary") or "").strip()
        wait_for = str(update.get("wait_for") or "").strip()
        if not summary or not wait_for:
            return None
        evidence = [
            str(item)
            for item in [
                *(update.get("evidence_message_ids") or []),
                *(opportunity.get("evidence_message_ids") or []),
            ]
            if str(item) in valid_ids
        ]
        anchor = str(opportunity.get("anchor_message_id") or "")
        if anchor and anchor in valid_ids and anchor not in evidence:
            evidence.insert(0, anchor)
        if not evidence:
            return None
        binding = self._session_locked(umo, moment)
        actor = Actor(
            source=SOURCE_ATTENTION_AGENT,
            session=umo,
            platform_id=str(binding.get("platform_id") or ""),
            group_id=str(binding.get("group_id") or ""),
        )
        known_progress = int(binding.get("last_decision_seq") or 0)
        evidence_messages = [
            message
            for message in self._window_locked(umo)
            if message.message_id in evidence and not message.is_bot
        ]
        new_progress = any(
            message.seq > known_progress for message in evidence_messages
        )
        anchor_message = next(
            (message for message in evidence_messages if message.message_id == anchor),
            None,
        )
        new_opening = bool(
            anchor_message
            and anchor_message.seq > known_progress
            and str(opportunity.get("expected_contribution") or "").strip()
            and opportunity.get("kind")
            in {"unanswered_question", "new_information", "natural_opening"}
            and moment.timestamp() - anchor_message.created_ts
            < self.settings.opportunity_ttl_minutes * 60
            and not any(
                isinstance(item, dict)
                and item.get("umo") == umo
                and item.get("anchor_message_id") == anchor
                for item in (self.store.state.get("opportunities") or {}).values()
            )
        )
        result = self.control.admit_interest_locked(
            self.store.state,
            actor,
            summary=summary,
            wait_for=wait_for,
            evidence_message_ids=evidence,
            source=SOURCE_ATTENTION_AGENT,
            moment=moment,
            priority_level=2 if new_opening else int(new_progress),
        )
        if not result.ok:
            return None
        interest = result.data.get("interest")
        if isinstance(interest, dict) and actor.group_id:
            self.control.create_checkpoint_locked(
                self.store.state,
                actor,
                umo=umo,
                platform_id=actor.platform_id,
                group_id=actor.group_id,
                kind="interest",
                interest_id=str(interest.get("interest_id") or ""),
                reason="new external interest evidence",
                source=SOURCE_ATTENTION_AGENT,
                moment=moment,
            )
        return interest

    def _apply_auto_policy_locked(
        self,
        request: PassiveRequest,
        user_action: dict[str, Any],
        moment: datetime,
    ) -> None:
        scope = str(user_action.get("scope") or "group_user")
        if scope == "group_user" and not request.group_id:
            scope = "user"
        if scope not in {"group_user", "user"} or not request.user_id:
            return
        minutes = user_action.get("expires_in_minutes")
        try:
            minutes = int(minutes) if minutes is not None else 0
        except (TypeError, ValueError):
            minutes = 0
        if minutes <= 0:
            minutes = self.settings.max_auto_policy_hours * 60
        minutes = max(1, min(self.settings.max_auto_policy_hours * 60, minutes))
        actor = Actor(
            source=SOURCE_ATTENTION_AGENT,
            user_id=request.user_id,
            session=request.umo,
            group_id=request.group_id,
            platform_id=request.platform_id,
            is_admin=request.is_admin,
        )
        self.control.set_policy_locked(
            self.store.state,
            actor,
            scope=scope,
            group_id=request.group_id if scope == "group_user" else "",
            user_id=request.user_id,
            reason=clip_text(user_action.get("reason"), 120) or "agent policy",
            expires_at=add_minutes(moment, minutes),
            source=SOURCE_ATTENTION_AGENT,
            created_by_session=request.umo,
            allow_permanent=False,
        )

    def _schedule_retry_locked(
        self,
        request: PassiveRequest,
        decision: PassiveDecision,
        moment: datetime,
    ) -> None:
        if not request.is_group or not self.settings.active_enabled:
            return
        if not self.policy.group_allowed(
            request.group_id,
            request.umo,
            request.platform_id,
        ):
            return
        actor = Actor(
            source=SOURCE_ATTENTION_AGENT,
            user_id=request.user_id,
            session=request.umo,
            group_id=request.group_id,
            platform_id=request.platform_id,
            is_admin=request.is_admin,
        )
        self.control.create_checkpoint_locked(
            self.store.state,
            actor,
            umo=request.umo,
            platform_id=request.platform_id,
            group_id=request.group_id,
            kind="interest",
            interest_id=str(
                (decision.interest_update or {}).get("interest_id")
                or (decision.opportunity or {}).get("interest_id")
                or ""
            ),
            minutes=self.settings.passive_retry_delay_minutes,
            reason=f"retry_later: {decision.reason}"[:200],
            source=SOURCE_ATTENTION_AGENT,
        )

    def _close_interest_from_update_locked(
        self,
        umo: str,
        decision: PassiveDecision,
        moment: datetime,
    ) -> None:
        interest_id = str((decision.interest_update or {}).get("interest_id") or "")
        if not interest_id:
            return
        actor = Actor(source=SOURCE_ATTENTION_AGENT, session=umo)
        self.control.close_interest_locked(
            self.store.state,
            actor,
            interest_id=interest_id,
            reason=decision.reason or "end_focus",
            moment=moment,
        )

    def _recent_opportunities_locked(self, umo: str) -> list[dict[str, Any]]:
        result = [
            dict(raw)
            for raw in (self.store.state.get("opportunities") or {}).values()
            if isinstance(raw, dict) and raw.get("umo") == umo
        ]
        result.sort(key=lambda item: str(item.get("updated_at") or ""))
        return result[-5:]

    def _resources_locked(self, umo: str) -> dict[str, Any]:
        moment = self.policy.now()
        session = self._session_locked(umo, moment)
        budget = self.policy.ensure_budget(self.store.state, moment)
        return {
            "session_attempts": int(session.get("autonomous_attempts") or 0),
            "session_limit": self.settings.session_autonomous_reply_budget,
            "global_attempts": int(budget.get("autonomous_attempts") or 0),
            "global_limit": self.settings.daily_autonomous_reply_budget,
        }

    @staticmethod
    def _passive_key(request: PassiveRequest) -> str:
        if request.message_id:
            return f"{request.umo}:{request.message_id}"
        digest = hashlib.sha1(
            f"{request.umo}|{request.user_id}|{request.message_text}|"
            f"{request.timestamp}".encode()
        ).hexdigest()
        return f"{request.umo}:{digest}"

    def _remember_passive(self, key: str, outcome: PassiveOutcome) -> None:
        self._passive_inflight.discard(key)
        self._passive_done[key] = outcome
        self._passive_done.move_to_end(key)
        while len(self._passive_done) > 500:
            self._passive_done.popitem(last=False)

    async def complete_passive_generation(
        self,
        umo: str,
        reservation_id: str,
        text: str = "",
    ) -> None:
        """Mark a passive approval as generated downstream (not platform-sent)."""

        if not reservation_id:
            return
        async with self.store.lock:
            self._complete_participation_locked(
                umo,
                reservation_id,
                "passed_through",
                self.policy.now(),
            )
            self._record_decision_locked(
                kind="passive_result",
                umo=umo,
                trigger="on_llm_response",
                code_decision="generated" if text.strip() else "generation_failed",
                code="",
                reason="downstream generation result; platform delivery is unconfirmed",
                reservation_id=reservation_id,
                main_model="generated" if text.strip() else "failed",
            )
            self._flush_locked()
        logger.debug(
            "Passive generation completed: umo=%s chars=%s",
            umo,
            len(text or ""),
        )

    # ------------------------------------------------------------------
    # Checkpoint administration
    # ------------------------------------------------------------------

    async def create_checkpoint(
        self,
        *,
        umo: str = "",
        platform_id: str = "",
        group_id: str = "",
        minutes: int | None = None,
        scheduled_at: datetime | None = None,
        reason: str = "",
        created_by: str = "admin",
        source: str = SOURCE_ADMIN,
        actor: Actor | None = None,
        interest_id: str = "",
        kind: str = "",
    ) -> dict[str, Any]:
        """Create or merge a checkpoint from an admin action or a test."""

        async with self.store.lock:
            umo, platform_id, group_id = self._resolve_target(
                umo, platform_id, group_id
            )
            actor = actor or Actor(
                source=source,
                session=umo,
                group_id=group_id,
                platform_id=platform_id,
                is_admin=source == SOURCE_ADMIN,
            )
            if not actor.session:
                actor = Actor(
                    source=actor.source,
                    session=umo,
                    group_id=group_id,
                    platform_id=platform_id,
                    is_admin=actor.is_admin,
                )
            result = self.control.create_checkpoint_locked(
                self.store.state,
                actor,
                umo=umo,
                platform_id=platform_id,
                group_id=group_id,
                kind=kind or ("interest" if interest_id else "session_scan"),
                scheduled_at=scheduled_at,
                minutes=minutes,
                reason=reason,
                source=source,
                interest_id=interest_id,
            )
            if not result.ok:
                raise ValueError(result.message)
            self._flush_locked()
            return dict(result.data.get("checkpoint") or {})

    async def list_checkpoints(self, umo: str = "") -> list[dict[str, Any]]:
        """Return checkpoints for one session, or all sessions when empty."""

        async with self.store.lock:
            if umo:
                items = self.control.session_checkpoints_locked(
                    self.store.state, umo, include_finished=True
                )
            else:
                items = [
                    dict(raw)
                    for raw in (self.store.state.get("checkpoints") or {}).values()
                    if isinstance(raw, dict)
                ]
        items.sort(
            key=lambda item: (
                str(item.get("status") or ""),
                str(item.get("scheduled_at") or ""),
            )
        )
        return items

    async def list_interests(self, umo: str = "") -> list[dict[str, Any]]:
        """Return interests for one session, or all sessions when empty."""

        async with self.store.lock:
            if umo:
                return self.control.session_interests_locked(
                    self.store.state, umo, include_closed=False
                )
            return [
                dict(raw)
                for raw in (self.store.state.get("interests") or {}).values()
                if isinstance(raw, dict) and raw.get("status") != "closed"
            ]

    async def list_policies(self) -> list[dict[str, Any]]:
        async with self.store.lock:
            return self.policy.active_policies(self.store.state)

    async def set_checkpoint_status(
        self,
        checkpoint_id: str,
        status: str,
        *,
        umo: str = "",
    ) -> tuple[bool, str]:
        """Pause, resume or cancel one checkpoint."""

        async with self.store.lock:
            raw = (self.store.state.get("checkpoints") or {}).get(checkpoint_id)
            if not isinstance(raw, dict):
                return False, "checkpoint not found"
            if raw.get("status") == "running":
                return False, "checkpoint is currently running"
            moment = self.policy.now()
            scheduled_at: str | None = None
            if status == "active":
                scheduled = parse_iso(raw.get("scheduled_at"), self.settings.tz)
                if scheduled is None or scheduled <= moment:
                    scheduled_at = to_iso(
                        self._next_interval(
                            moment, self.settings.default_interval_minutes
                        )
                    )
            elif status not in {"paused", "cancelled"}:
                return False, "unsupported checkpoint status"
            actor = Actor(
                source=SOURCE_ADMIN,
                session=umo or str(raw.get("umo") or ""),
                is_admin=True,
            )
            result = self.control.set_checkpoint_status_locked(
                self.store.state,
                actor,
                checkpoint_id,
                status,
                moment=moment,
                scheduled_at=scheduled_at,
            )
            if not result.ok:
                return False, result.message
            self._flush_locked()
            return True, str(result.data.get("status") or status)

    def _resolve_target(
        self,
        umo: str,
        platform_id: str,
        group_id: str,
    ) -> tuple[str, str, str]:
        umo = str(umo or "").strip()
        platform_id = str(platform_id or "").strip()
        group_id = str(group_id or "").strip()
        if umo and ":" in umo:
            parts = umo.split(":", 2)
            if len(parts) != 3 or parts[1] != "GroupMessage":
                raise ValueError("checkpoint UMO must be a group session")
            if platform_id and platform_id != parts[0]:
                raise ValueError("platform_id contradicts the target UMO")
            platform_id = parts[0]
            binding = (self.store.state.get("sessions") or {}).get(umo, {})
            if binding.get("group_id"):
                if group_id and group_id != binding["group_id"]:
                    raise ValueError("group_id contradicts the session route")
                group_id = binding["group_id"]
            elif not group_id:
                # For an unobserved explicit target, only the standard route is
                # known. Isolated routes require an explicit real group_id.
                if "_" in parts[2] or "%" in parts[2]:
                    raise ValueError("isolated UMO requires an explicit group_id")
                group_id = parts[2]
        elif umo and not group_id:
            group_id = umo
            umo = ""
        if group_id and not umo:
            if not platform_id:
                raise ValueError("platform_id is required when umo is not given")
            umo = f"{platform_id}:GroupMessage:{group_id}"
        return umo, platform_id, group_id

    # ------------------------------------------------------------------
    # Claiming / scheduling
    # ------------------------------------------------------------------

    @_simulated_operation
    async def claim_checkpoint(
        self,
        checkpoint_id: str,
        *,
        force: bool = False,
        umo: str = "",
    ) -> tuple[dict[str, Any] | None, str]:
        """Atomically claim a checkpoint for one active session."""

        async with self.store.lock:
            raw = (self.store.state.get("checkpoints") or {}).get(checkpoint_id)
            if isinstance(raw, dict) and umo and raw.get("umo") != umo:
                return None, "scope_mismatch"
            if not self.settings.enabled or not self.settings.active_enabled:
                return None, "active attention is disabled"
            if isinstance(raw, dict) and raw.get("umo") in self._evaluating_sessions:
                return None, "session_busy"
            if not self.policy.can_run_active_check(
                self.store.live_state, self.policy.now()
            ):
                return None, "daily active-check budget exhausted"
            checkpoint, reason = self.control.claim_checkpoint_locked(
                self.store.state,
                checkpoint_id,
                moment=self.policy.now(),
                force=force,
            )
            if checkpoint is None:
                return None, reason
            self._flush_locked()
            return checkpoint, ""

    @_simulated_operation
    async def claim_next_due(self) -> dict[str, Any] | None:
        """Claim the earliest due checkpoint using per-session fairness."""

        async with self.store.lock:
            if not self.settings.enabled or not self.settings.active_enabled:
                return None
            moment = self.policy.now()
            if not self.policy.can_run_active_check(self.store.live_state, moment):
                return None
            due: dict[str, list[dict[str, Any]]] = {}
            for raw in (self.store.state.get("checkpoints") or {}).values():
                if not isinstance(raw, dict) or raw.get("status") != "active":
                    continue
                if raw.get("umo") in self._evaluating_sessions:
                    continue
                if not self.policy.group_allowed(
                    str(raw.get("group_id") or ""),
                    str(raw.get("umo") or ""),
                    str(raw.get("platform_id") or ""),
                ):
                    continue
                scheduled = parse_iso(raw.get("scheduled_at"), self.settings.tz)
                if scheduled is None or scheduled <= moment:
                    due.setdefault(str(raw.get("umo") or ""), []).append(dict(raw))
            if not due:
                return None
            sessions = sorted(due)
            index = self._schedule_cursor % len(sessions)
            chosen_umo = sessions[index]
            self._schedule_cursor = (index + 1) % len(sessions)
            items = due[chosen_umo]
            items.sort(key=lambda item: str(item.get("scheduled_at") or ""))
            chosen_id = str(items[0].get("checkpoint_id") or "")
            checkpoint, _reason = self.control.claim_checkpoint_locked(
                self.store.state,
                chosen_id,
                moment=moment,
            )
            if checkpoint is None:
                return None
            merged_checkpoint_ids: list[str] = []
            merged_interest_ids: list[str] = []
            for other in items[1:]:
                other_id = str(other.get("checkpoint_id") or "")
                raw_other = (self.store.state.get("checkpoints") or {}).get(other_id)
                if (
                    not isinstance(raw_other, dict)
                    or raw_other.get("status") != "active"
                ):
                    continue
                raw_other["scheduled_at"] = to_iso(
                    self._next_interval(moment, self.settings.default_interval_minutes)
                )
                raw_other["last_result"] = "merged_pending"
                raw_other["last_error"] = f"merged into {chosen_id}"
                raw_other["updated_at"] = to_iso(moment)
                raw_other["revision"] = int(raw_other.get("revision") or 1) + 1
                merged_checkpoint_ids.append(other_id)
                interest_id = str(raw_other.get("interest_id") or "")
                if interest_id and interest_id not in merged_interest_ids:
                    merged_interest_ids.append(interest_id)
            raw_chosen = (self.store.state.get("checkpoints") or {}).get(chosen_id)
            if isinstance(raw_chosen, dict):
                raw_chosen["merged_checkpoint_ids"] = merged_checkpoint_ids
                raw_chosen["merged_interest_ids"] = merged_interest_ids
            checkpoint["merged_checkpoint_ids"] = merged_checkpoint_ids
            checkpoint["merged_interest_ids"] = merged_interest_ids
            self._flush_locked()
            return checkpoint

    @_simulated_operation
    async def seconds_until_next_due(self) -> int:
        """Return a scheduler sleep hint in seconds."""

        async with self.store.lock:
            if not self.settings.enabled or not self.settings.active_enabled:
                return 30
            moment = self.policy.now()
            delays: list[float] = []
            for raw in (self.store.state.get("checkpoints") or {}).values():
                if not isinstance(raw, dict) or raw.get("status") != "active":
                    continue
                if not self.policy.group_allowed(
                    str(raw.get("group_id") or ""),
                    str(raw.get("umo") or ""),
                    str(raw.get("platform_id") or ""),
                ):
                    continue
                scheduled = parse_iso(raw.get("scheduled_at"), self.settings.tz)
                if scheduled is not None:
                    delays.append((scheduled - moment).total_seconds())
            if not delays:
                return 60
            return max(5, min(60, int(min(delays))))

    async def trigger_checkpoint(
        self,
        checkpoint_id: str,
        *,
        umo: str = "",
    ) -> tuple[bool, str]:
        """Claim and enqueue a checkpoint for immediate execution."""

        claimed, reason = await self.claim_checkpoint(
            checkpoint_id, force=True, umo=umo
        )
        if claimed is None:
            return False, reason
        await self.scheduler.enqueue_checkpoint(checkpoint_id)
        return True, "checkpoint execution queued"

    # ------------------------------------------------------------------
    # Active attention
    # ------------------------------------------------------------------

    @_simulated_operation
    async def run_claimed(self, checkpoint_id: str) -> None:
        """Execute one claimed checkpoint and never leave it stuck running."""

        try:
            await self._run_claimed_inner(checkpoint_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Active attention checkpoint %s crashed.",
                checkpoint_id,
            )
            async with self.store.lock:
                raw = (self.store.state.get("checkpoints") or {}).get(checkpoint_id)
                if isinstance(raw, dict) and raw.get("status") == "running":
                    self._finalize_locked(
                        raw,
                        status="active",
                        next_minutes=None,
                        last_result="observe",
                        last_error=str(exc),
                        evaluated=False,
                        no_progress=False,
                    )
                    self._record_decision_locked(
                        kind="active",
                        umo=str(raw.get("umo") or ""),
                        trigger="checkpoint",
                        code_decision="observe",
                        code="checkpoint_crashed",
                        reason=f"checkpoint crashed: {exc}",
                        checkpoint_id=checkpoint_id,
                    )
                    self._flush_locked()

    async def _run_claimed_inner(self, checkpoint_id: str) -> None:
        async with self.store.lock:
            raw = (self.store.state.get("checkpoints") or {}).get(checkpoint_id)
            if not isinstance(raw, dict) or raw.get("status") != "running":
                return
            checkpoint = dict(raw)
            run_id = str(checkpoint.get("run_id") or "")
            moment = self.policy.now()
            umo = str(checkpoint.get("umo") or "")
            platform_id = str(checkpoint.get("platform_id") or "")
            group_id = str(checkpoint.get("group_id") or "")
            session = self._session_locked(umo, moment)
            interest = None
            if checkpoint.get("kind") == "interest":
                interest = (self.store.state.get("interests") or {}).get(
                    checkpoint.get("interest_id")
                )
                if (
                    not isinstance(interest, dict)
                    or interest.get("status") == "closed"
                    or interest.get("umo") != umo
                ):
                    self._finish_locked(
                        checkpoint,
                        status="completed",
                        last_result="interest_closed",
                        error="interest_closed",
                        evaluated=False,
                        no_progress=False,
                    )
                    return
                expires_at = parse_iso(interest.get("expires_at"), self.settings.tz)
                if expires_at is not None and expires_at <= moment:
                    self._finish_locked(
                        checkpoint,
                        status="cancelled",
                        last_result="interest_expired",
                        error="interest_expired",
                        evaluated=False,
                        no_progress=False,
                    )
                    return
            if not self.settings.enabled or not self.settings.active_enabled:
                self._finish_locked(
                    checkpoint,
                    status="active",
                    last_result="observe",
                    error="active attention disabled",
                    evaluated=False,
                    no_progress=False,
                )
                return
            if not self.policy.group_allowed(group_id, umo, platform_id):
                self._finish_locked(
                    checkpoint,
                    status="cancelled",
                    last_result="skip",
                    error="group is not allowed",
                    evaluated=False,
                    no_progress=False,
                )
                return
            if self.policy.is_quiet(moment):
                end = self.policy.quiet_end(moment)
                minutes = (
                    max(
                        self.settings.min_interval_minutes,
                        int((end - moment).total_seconds() // 60) + 1,
                    )
                    if end is not None
                    else self.settings.min_interval_minutes
                )
                self._finish_locked(
                    checkpoint,
                    status="active",
                    last_result="observe",
                    error="quiet hours; deferred",
                    next_minutes=minutes,
                    evaluated=False,
                    no_progress=False,
                )
                return
            stop = self.control.stop_directive(self.store.state, umo=umo)
            if stop is not None and checkpoint.get("source") != SOURCE_ADMIN:
                self._finish_locked(
                    checkpoint,
                    status="cancelled",
                    last_result="skip",
                    error="main agent stop_focus directive",
                    evaluated=False,
                    no_progress=False,
                )
                return
            evaluated_seq = int(session.get("external_message_seq") or 0)
            if checkpoint.get("kind") == "interest" and interest is not None:
                last_evaluated = int(interest.get("last_evaluated_seq") or 0)
            else:
                last_evaluated = int(session.get("last_decision_seq") or 0)
            forced = bool(checkpoint.get("forced"))
            if evaluated_seq <= last_evaluated and not forced and evaluated_seq > 0:
                self._finish_locked(
                    checkpoint,
                    status="active",
                    last_result="skip",
                    error="no_new_messages",
                    next_minutes=self._backoff_minutes(
                        int(checkpoint.get("no_progress_count") or 0) + 1
                    ),
                    evaluated=False,
                    no_progress=True,
                )
                return
            next_eval = parse_iso(session.get("next_evaluation_at"), self.settings.tz)
            if not forced and next_eval is not None and next_eval > moment:
                self._finish_locked(
                    checkpoint,
                    status="active",
                    last_result="skip",
                    error="evaluation_throttled",
                    next_minutes=max(
                        1, int((next_eval - moment).total_seconds() / 60) + 1
                    ),
                )
                return
            allowed, quota_reason = self.policy.can_reserve_autonomous(
                self.store.state, umo, moment
            )
            if not allowed:
                self._finish_locked(
                    checkpoint,
                    status="active",
                    last_result="skip",
                    error=quota_reason,
                    next_minutes=self.settings.default_interval_minutes,
                )
                return
            self._flush_locked()

        recent = []
        async with self.store.lock:
            recent = self.store.recent_window(umo)
        window = await self.context_provider.window(
            umo=umo,
            platform_id=platform_id,
            group_id=group_id,
            recent=recent,
            limit=self.settings.context_window_messages,
        )
        async with self.store.lock:
            if any(message.umo and message.umo != umo for message in window):
                self._finish_locked(
                    checkpoint,
                    status="cancelled",
                    last_result="skip",
                    error="scope_mismatch",
                )
                return
            # Import only dated history; absent timestamps never become fresh.
            for message in window:
                if message.source != "history":
                    continue
                created = parse_iso(message.created_at, self.settings.tz)
                if created is None or created > self.policy.now():
                    continue
                shape = next(
                    (
                        item
                        for item in self._session_locked(umo).get("history_shapes", [])
                        if item.get("sender_id") == message.sender_id
                        and item.get("text_digest")
                        == hashlib.sha256(
                            clip_text(message.text, 1000).encode()
                        ).hexdigest()
                        and abs(float(item.get("ts") or 0) - message.created_ts)
                        < self.settings.opportunity_ttl_minutes * 60
                    ),
                    None,
                )
                if shape is not None and shape.get("id") != message.message_id:
                    message.message_id = str(shape["id"])
                seq = self._ensure_message_locked(
                    umo=umo,
                    sender_id=message.sender_id,
                    sender_name=message.sender_name,
                    text=message.text,
                    message_id=message.message_id,
                    created_at=message.created_at,
                    created_ts=message.created_ts,
                    is_bot=message.is_bot,
                )
                message.seq = seq
            canonical = {m.message_id: m for m in self._window_locked(umo)}
            for message in window:
                message.seq = (
                    canonical[message.message_id].seq
                    if message.message_id in canonical
                    else 0
                )
                if message.seq:
                    known = self._session_locked(umo).setdefault("message_ids", {})
                    for alias in message.aliases:
                        known[alias] = message.seq
                    while len(known) > 500:
                        known.pop(next(iter(known)))
            evaluated_seq = max((m.seq for m in window if not m.is_bot), default=0)
            if evaluated_seq <= last_evaluated and not forced:
                self._finish_locked(
                    checkpoint,
                    status="active",
                    last_result="skip",
                    error="no_new_messages",
                    no_progress=True,
                    next_minutes=self._backoff_minutes(
                        int(checkpoint.get("no_progress_count") or 0) + 1
                    ),
                )
                return
        if not window:
            async with self.store.lock:
                self._finish_locked(
                    checkpoint,
                    status="active",
                    last_result="skip",
                    error="no readable group context",
                    next_minutes=self._backoff_minutes(
                        int(checkpoint.get("no_progress_count") or 0) + 1
                    ),
                    evaluated=False,
                    no_progress=True,
                )
            return
        persona = await self.context_provider.persona(umo)
        async with self.store.lock:
            interests = self.control.session_interests_locked(self.store.state, umo)
            policy_summary = self.control.prompt_summary_locked(
                self.store.state,
                umo=umo,
                group_id=group_id,
                platform_id=platform_id,
                message_text=window[-1].text if window else "",
                message_id=window[-1].message_id if window else "",
            )
            current_interest = None
            if checkpoint.get("kind") == "interest":
                current_interest = (self.store.state.get("interests") or {}).get(
                    checkpoint.get("interest_id")
                )
            evaluation_stamp = self._evaluation_stamp_locked(umo)
        ctx = ActiveContext(
            checkpoint=checkpoint,
            interest=dict(current_interest) if current_interest else None,
            interests=interests,
            window=window,
            persona=persona,
            policy_summary=policy_summary,
            evaluated_seq=evaluated_seq,
            forced=forced,
        )

        check_charged = False

        async def consume_call() -> bool:
            nonlocal check_charged
            async with self.store.lock:
                if not self.policy.can_call_model(
                    self.store.live_state, self.policy.now()
                ):
                    return False
                if not check_charged:
                    if not self.policy.consume_active_check(
                        self.store.live_state, self.policy.now()
                    ):
                        return False
                    check_charged = True
                allowed = self.policy.consume_model_call(
                    self.store.live_state,
                    self.policy.now(),
                    "decision",
                )
                if allowed:
                    self._flush_locked()
                return allowed

        async def window_loader(limit: int) -> list[RecentMessage]:
            async with self.store.lock:
                latest = self.store.recent_window(umo)
            refreshed = await self.context_provider.window(
                umo=umo,
                platform_id=platform_id,
                group_id=group_id,
                recent=latest,
                limit=limit,
            )
            async with self.store.lock:
                if any(message.umo and message.umo != umo for message in refreshed):
                    return []
                canonical = {m.message_id: m for m in self._window_locked(umo)}
                for message in refreshed:
                    message.seq = (
                        canonical[message.message_id].seq
                        if message.message_id in canonical
                        else 0
                    )
            return refreshed

        result = await self.active_runner.run(ctx, consume_call, window_loader)
        evaluated_seq = max(evaluated_seq, int(ctx.evaluated_seq or 0))

        async with self.store.lock:
            raw = (self.store.state.get("checkpoints") or {}).get(checkpoint_id)
            if (
                not isinstance(raw, dict)
                or raw.get("status") != "running"
                or str(raw.get("run_id") or "") != run_id
            ):
                return
            checkpoint = dict(raw)
            now_after = self.policy.now()
            invalidated = self._revalidate_locked(
                umo, evaluated_seq, evaluation_stamp, checkpoint=checkpoint
            )
            if invalidated:
                self._finish_locked(
                    checkpoint,
                    status="active",
                    last_result="observe",
                    error=invalidated,
                    evaluated=False,
                    model_suggestion=result.decision if not result.interrupted else "",
                    decision_model_calls=result.model_calls,
                    evaluated_seq=evaluated_seq,
                )
                return
            interest = None
            if checkpoint.get("kind") == "interest":
                interest = (self.store.state.get("interests") or {}).get(
                    checkpoint.get("interest_id")
                )
            self._apply_active_effects_locked(checkpoint, result, now_after)
            valid_ids, bot_ids = self._ids_from_messages(ctx.window)
            self._apply_interest_update_locked(
                checkpoint, result, now_after, valid_ids=valid_ids
            )
            if not result.interrupted:
                session = self._session_locked(umo, now_after)
                session["last_decision_seq"] = max(
                    int(session.get("last_decision_seq") or 0),
                    evaluated_seq,
                )
                session["last_decision_at"] = to_iso(now_after)
                session["next_evaluation_at"] = to_iso(
                    now_after
                    + timedelta(seconds=self.settings.min_evaluation_interval_seconds)
                )
                session["revision"] = int(session.get("revision") or 1) + 1
            no_progress = (
                not result.interrupted
                and result.decision in {"observe", "skip"}
                and not (result.interest_update or {}).get("_progress")
            )
            merged_interest_ids = [
                str(item)
                for item in checkpoint.get("merged_interest_ids") or []
                if str(item) in result.evaluated_interest_ids
                and str(item) != checkpoint.get("interest_id")
            ]
            if merged_interest_ids and not result.interrupted:
                self._advance_merged_interests_locked(
                    umo,
                    merged_interest_ids,
                    evaluated_seq,
                    now_after,
                    no_progress,
                )
            next_minutes = result.next_check_minutes
            status = "active"
            if result.decision == "end_focus":
                status = "completed"
                self._close_interest_locked(checkpoint, now_after, "end_focus")
            if interest is not None and interest.get("status") == "closed":
                status = "completed"
            if status == "active" and next_minutes is None:
                next_minutes = self._backoff_minutes(
                    int((interest or {}).get("no_progress_count") or 0)
                    if (interest or {}).get("no_progress_count")
                    else 0
                )
            if result.decision != "participate":
                self._finalize_locked(
                    checkpoint,
                    status=status,
                    next_minutes=next_minutes,
                    last_result=result.decision,
                    last_error=result.reason if result.interrupted else "",
                    evaluated=not result.interrupted,
                    no_progress=no_progress,
                )
                self._record_decision_locked(
                    kind="active",
                    umo=umo,
                    trigger="checkpoint",
                    code_decision=result.decision,
                    suggested_action=result.decision if not result.interrupted else "",
                    semantic_evaluations=int(not result.interrupted),
                    decision_model_calls=result.model_calls,
                    tool_call_count=result.tool_calls,
                    message_seq=evaluated_seq,
                    reason=result.reason,
                    checkpoint_id=checkpoint_id,
                    interest_id=str(checkpoint.get("interest_id") or ""),
                    code="",
                )
                self._flush_locked()
                logger.info(
                    "Active attention: checkpoint=%s umo=%s decision=%s reason=%s",
                    checkpoint_id,
                    umo,
                    result.decision,
                    clip_text(result.reason, 120),
                )
                return

            approval = self._approve_active_locked(
                checkpoint,
                result,
                now_after,
                valid_ids=valid_ids,
                bot_ids=bot_ids,
            )
            if not approval.approved:
                self._finalize_locked(
                    checkpoint,
                    status="active",
                    next_minutes=self.settings.min_interval_minutes,
                    last_result="observe",
                    last_error=approval.message,
                    evaluated=True,
                    no_progress=True,
                )
                self._record_decision_locked(
                    kind="active",
                    umo=umo,
                    trigger="checkpoint",
                    code_decision="observe",
                    suggested_action="participate",
                    code=approval.code or "not_approved",
                    semantic_evaluations=int(not result.interrupted),
                    decision_model_calls=result.model_calls,
                    tool_call_count=result.tool_calls,
                    message_seq=evaluated_seq,
                    reason=approval.message or "participation rejected",
                    checkpoint_id=checkpoint_id,
                    interest_id=str(checkpoint.get("interest_id") or ""),
                    anchor_message_id=str(
                        (result.opportunity or {}).get("anchor_message_id") or ""
                    ),
                )
                self._flush_locked()
                return
            if self.settings.dry_run:
                self._complete_participation_locked(
                    umo, approval.reservation_id, "simulated", now_after
                )
                self._finalize_locked(
                    checkpoint,
                    status="active",
                    next_minutes=None,
                    last_result="participate",
                    last_error="",
                    evaluated=True,
                    no_progress=False,
                )
                self._record_decision_locked(
                    kind="active",
                    umo=umo,
                    trigger="checkpoint",
                    code_decision="participate",
                    suggested_action="participate",
                    code="dry_run",
                    reason=result.reason,
                    checkpoint_id=checkpoint_id,
                    reservation_id=approval.reservation_id,
                    message_seq=evaluated_seq,
                )
                self._flush_locked()
                return
            self._record_decision_locked(
                kind="active",
                umo=umo,
                trigger="checkpoint",
                code_decision="participate",
                suggested_action="participate",
                code="autonomous_approved",
                reason=result.reason,
                checkpoint_id=checkpoint_id,
                interest_id=approval.interest_id,
                anchor_message_id=str(
                    (result.opportunity or {}).get("anchor_message_id") or ""
                ),
                reservation_id=approval.reservation_id,
                message_seq=evaluated_seq,
                decision_model_calls=result.model_calls,
                tool_call_count=result.tool_calls,
            )
            self._flush_locked()

        await self._handle_participation(
            checkpoint,
            result,
            approval,
            evaluated_seq,
            persona,
            ctx.window,
        )

    def _approve_active_locked(
        self,
        checkpoint: dict[str, Any],
        result: Any,
        moment: datetime,
        *,
        valid_ids: set[str],
        bot_ids: set[str],
    ) -> ParticipationApproval:
        umo = str(checkpoint.get("umo") or "")
        payload = dict(result.opportunity or {})
        interest_id = str(
            payload.get("interest_id")
            or checkpoint.get("interest_id")
            or (result.interest_update or {}).get("interest_id")
            or ""
        )
        if not interest_id:
            admitted = self._admit_interest_from_update_locked(
                umo,
                result.interest_update,
                payload,
                moment,
                valid_ids=valid_ids,
            )
            if admitted is not None:
                interest_id = str(admitted.get("interest_id") or "")
        if not interest_id:
            return ParticipationApproval(
                approved=False,
                code="interest_required",
                message="主动参与必须引用一个已准入兴趣。",
            )
        payload["interest_id"] = interest_id
        return self.control.approve_participation_locked(
            self.store.state,
            umo=umo,
            wake_source="ambient",
            suggested_action="participate",
            opportunity_payload=payload,
            moment=moment,
            valid_ids=valid_ids,
            bot_ids=bot_ids,
            anchor_message_id=str(payload.get("anchor_message_id") or ""),
            interest_id=interest_id,
            interest_update=result.interest_update,
        )

    async def _handle_participation(
        self,
        checkpoint: dict[str, Any],
        result: Any,
        approval: ParticipationApproval,
        evaluated_seq: int,
        persona: str,
        window: list[RecentMessage],
    ) -> None:
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        umo = str(checkpoint.get("umo") or "")
        group_id = str(checkpoint.get("group_id") or "")
        async with self.store.lock:
            generation_stamp = self._evaluation_stamp_locked(umo)
            moment = self.policy.now()
            invalidated = self._revalidate_locked(
                umo,
                evaluated_seq,
                generation_stamp,
                checkpoint=checkpoint,
                approval=approval,
            )
            if invalidated:
                self._complete_participation_locked(
                    umo, approval.reservation_id, invalidated, moment
                )
                self._finish_locked(
                    checkpoint,
                    status="active",
                    last_result="observe",
                    error=invalidated,
                    evaluated=True,
                    reservation_id=approval.reservation_id,
                    evaluated_seq=evaluated_seq,
                )
                return
            if (
                self.settings.max_main_model_calls < 1
                or not self.policy.consume_model_call(
                    self.store.live_state, moment, "main"
                )
            ):
                self._complete_participation_locked(
                    umo, approval.reservation_id, "quota_exhausted", moment
                )
                self._finish_locked(
                    checkpoint,
                    status="active",
                    last_result="observe",
                    error="quota_exhausted",
                    evaluated=True,
                    reservation_id=approval.reservation_id,
                    evaluated_seq=evaluated_seq,
                )
                return
            current_interest = (self.store.state.get("interests") or {}).get(
                approval.interest_id
            )
            ctx = ActiveContext(
                checkpoint=checkpoint,
                interest=dict(current_interest) if current_interest else None,
                interests=self.control.session_interests_locked(self.store.state, umo),
                window=list(window),
                persona=persona,
                policy_summary="",
                evaluated_seq=evaluated_seq,
            )
            self._flush_locked()

        reply_system = build_active_reply_system_prompt(ctx, self.settings)
        reply_prompt = build_active_reply_prompt(
            ctx,
            self.settings,
            result.reply_guidance,
        )
        reply_contexts = build_active_reply_contexts(ctx, self.settings)
        try:
            reply = await self.gateway.generate(
                purpose="main",
                umo=umo,
                system_prompt=reply_system,
                prompt=reply_prompt,
                contexts=reply_contexts,
                timeout=max(30, self.settings.decision_timeout_seconds),
            )
        except asyncio.CancelledError:
            async with self.store.lock:
                self._complete_participation_locked(
                    umo,
                    approval.reservation_id,
                    "generation_cancelled",
                    self.policy.now(),
                )
                self._flush_locked()
            raise
        except Exception as exc:
            reply = ModelResult(error=str(exc))
        message = (
            sanitize_reply(reply.text, self.settings.max_reply_chars)
            if reply.ok
            else None
        )

        async with self.store.lock:
            now_after = self.policy.now()
            invalidated = self._revalidate_locked(
                umo,
                evaluated_seq,
                generation_stamp,
                checkpoint=checkpoint,
                approval=approval,
            )
            if invalidated or not message:
                code = invalidated or "generation_failed"
                self._complete_participation_locked(
                    umo, approval.reservation_id, code, now_after
                )
                self._finish_locked(
                    checkpoint,
                    status="active",
                    last_result="observe",
                    error=code,
                    evaluated=True,
                    reservation_id=approval.reservation_id,
                    evaluated_seq=evaluated_seq,
                    main_model="generated" if message else "failed",
                )
                return
            opportunity = self.store.state["opportunities"][
                (approval.opportunity or {})["opportunity_id"]
            ]
            opportunity["outcome"] = "sending"
            self._flush_locked()

        try:
            sent, error = await self.sender.send_text(umo, message)
            outcome = (
                "sent"
                if sent
                else (
                    "uncertain_delivery"
                    if error.startswith("uncertain_delivery:")
                    else "send_failed"
                )
            )
        except asyncio.CancelledError:
            async with self.store.lock:
                self._complete_participation_locked(
                    umo,
                    approval.reservation_id,
                    "uncertain_delivery",
                    self.policy.now(),
                )
                self._flush_locked()
            raise
        except Exception as exc:
            sent, error, outcome = False, str(exc), "uncertain_delivery"
        async with self.store.lock:
            now_after = self.policy.now()
            self._complete_participation_locked(
                umo, approval.reservation_id, outcome, now_after
            )
            self.policy.record_send(
                self.store.state,
                umo=umo,
                group_id=group_id,
                moment=now_after,
                text=message,
                result=outcome,
            )
            if sent:
                self._ensure_message_locked(
                    umo=umo,
                    sender_id="bot",
                    sender_name="bot",
                    text=message,
                    message_id="",
                    created_at=to_iso(now_after),
                    created_ts=timestamp(now_after),
                    is_bot=True,
                )
            self._finalize_locked(
                checkpoint,
                status="active",
                next_minutes=None,
                last_result="participate" if sent else "observe",
                last_error="" if sent else error,
                evaluated=True,
                no_progress=not sent,
            )
            self._record_decision_locked(
                kind="active",
                umo=umo,
                trigger="checkpoint",
                code_decision="participate" if sent else "observe",
                code=outcome,
                reason=result.reason,
                checkpoint_id=checkpoint_id,
                reservation_id=approval.reservation_id,
            )
            self._flush_locked()
        logger.info(
            "Active reply: checkpoint=%s umo=%s sent=%s chars=%s",
            checkpoint_id,
            umo,
            sent,
            len(message),
        )

    def _apply_active_effects_locked(
        self,
        checkpoint: dict[str, Any],
        result: Any,
        moment: datetime,
    ) -> None:
        umo = str(checkpoint.get("umo") or "")
        actor = Actor(
            source=SOURCE_ATTENTION_AGENT,
            session=umo,
            group_id=str(checkpoint.get("group_id") or ""),
            platform_id=str(checkpoint.get("platform_id") or ""),
        )
        for effect in result.effects:
            if not isinstance(effect, dict):
                continue
            effect_type = effect.get("type")
            if effect_type == "create_checkpoint":
                interest_id = str(effect.get("interest_id") or "")
                if not interest_id:
                    continue
                self.control.create_checkpoint_locked(
                    self.store.state,
                    actor,
                    umo=umo,
                    platform_id=str(checkpoint.get("platform_id") or ""),
                    group_id=str(checkpoint.get("group_id") or ""),
                    kind="interest",
                    minutes=int(effect.get("minutes") or 60),
                    reason=str(effect.get("reason") or "agent checkpoint"),
                    source=SOURCE_ATTENTION_AGENT,
                    interest_id=interest_id,
                )
            elif effect_type == "user_policy":
                minutes = effect.get("expires_in_minutes")
                try:
                    minutes = int(minutes) if minutes is not None else 0
                except (TypeError, ValueError):
                    minutes = 0
                if minutes <= 0:
                    minutes = self.settings.max_auto_policy_hours * 60
                minutes = max(
                    1,
                    min(self.settings.max_auto_policy_hours * 60, minutes),
                )
                user_id = str(effect.get("user_id") or "")
                if not user_id:
                    continue
                self.control.set_policy_locked(
                    self.store.state,
                    actor,
                    scope=str(effect.get("scope") or "group_user"),
                    group_id=str(effect.get("group_id") or ""),
                    user_id=user_id,
                    reason=clip_text(effect.get("reason"), 120),
                    expires_at=add_minutes(moment, minutes),
                    source=SOURCE_ATTENTION_AGENT,
                    created_by_session=umo,
                    allow_permanent=False,
                )
            elif effect_type == "end_focus":
                interest_id = str(effect.get("interest_id") or "")
                if interest_id:
                    self.control.close_interest_locked(
                        self.store.state,
                        actor,
                        interest_id=interest_id,
                        reason="end_focus",
                        moment=moment,
                    )

    def _advance_merged_interests_locked(
        self,
        umo: str,
        interest_ids: list[str],
        evaluated_seq: int,
        moment: datetime,
        no_progress: bool,
    ) -> None:
        """Advance checkpoints merged into this session-level evaluation."""

        for interest_id in interest_ids:
            raw = (self.store.state.get("interests") or {}).get(interest_id)
            if not isinstance(raw, dict) or raw.get("umo") != umo:
                continue
            if raw.get("status") == "closed":
                continue
            raw["last_evaluated_seq"] = max(
                int(raw.get("last_evaluated_seq") or 0), evaluated_seq
            )
            for cp in list((self.store.state.get("checkpoints") or {}).values()):
                if (
                    isinstance(cp, dict)
                    and cp.get("umo") == umo
                    and cp.get("interest_id") == interest_id
                    and cp.get("status") == "active"
                ):
                    self.control.finalize_checkpoint_locked(
                        self.store.state,
                        str(cp["checkpoint_id"]),
                        status="active",
                        scheduled_at=to_iso(
                            self._next_interval(
                                moment,
                                self._backoff_minutes(
                                    int(raw.get("no_progress_count") or 0)
                                    + int(no_progress)
                                ),
                            )
                        ),
                        last_result="merged_evaluated",
                        last_error="",
                        moment=moment,
                        evaluated=True,
                        no_progress=no_progress,
                    )

    def _apply_interest_update_locked(
        self,
        checkpoint: dict[str, Any],
        result: Any,
        moment: datetime,
        *,
        valid_ids: set[str] | None = None,
    ) -> None:
        """Apply evidence-bound proposals without overwriting protected metadata."""
        update = result.interest_update or {}
        if not isinstance(update, dict) or not update:
            return
        umo = str(checkpoint.get("umo") or "")
        interest_id = str(
            update.get("interest_id") or checkpoint.get("interest_id") or ""
        )
        if valid_ids is None:
            valid_ids, _ = self._window_ids_locked(umo)
        evidence = [str(item) for item in update.get("evidence_message_ids") or []]
        if any(item not in valid_ids for item in evidence):
            return
        if not interest_id:
            admitted = self._admit_interest_from_update_locked(
                umo,
                update,
                getattr(result, "opportunity", {}),
                moment,
                valid_ids=valid_ids,
            )
            if admitted is not None:
                update["interest_id"] = str(admitted.get("interest_id") or "")
                update["_progress"] = True
                if checkpoint.get("kind") == "session_scan":
                    current = (self.store.state.get("checkpoints") or {}).get(
                        checkpoint.get("checkpoint_id")
                    )
                    if isinstance(current, dict):
                        current["converted_to_interest"] = update["interest_id"]
            return
        raw = (self.store.state.get("interests") or {}).get(interest_id)
        if not isinstance(raw, dict) or raw.get("umo") != umo or raw.get("quarantine"):
            return
        update["interest_id"] = interest_id
        actor = Actor(source=SOURCE_ATTENTION_AGENT, session=umo)
        operation = str(update.get("operation") or "keep")
        if operation == "close":
            self.control.close_interest_locked(
                self.store.state,
                actor,
                interest_id=interest_id,
                reason="interest_update close",
                moment=moment,
            )
            return
        if operation != "update" or raw.get("status") != "watching":
            return
        progressed = self.control.touch_interest_evidence_locked(
            self.store.state, interest_id, evidence, moment=moment
        )
        update["_progress"] = progressed
        if raw.get("source") == SOURCE_ATTENTION_AGENT:
            summary = str(update.get("summary") or "").strip()
            wait_for = str(update.get("wait_for") or "").strip()
            if summary:
                raw["summary"] = clip_text(summary, 300)
            if wait_for:
                raw["wait_for"] = clip_text(wait_for, 300)
            raw["updated_at"] = to_iso(moment)
            raw["revision"] = int(raw.get("revision") or 1) + 1
        if progressed:
            for cp in (self.store.state.get("checkpoints") or {}).values():
                if (
                    isinstance(cp, dict)
                    and cp.get("umo") == umo
                    and cp.get("interest_id") == interest_id
                    and cp.get("status") in {"active", "running", "paused"}
                ):
                    cp["expires_at"] = raw["expires_at"]
                    cp["remaining_checks"] = raw["remaining_checks"]
                    cp["no_progress_count"] = 0

    def _close_interest_locked(
        self,
        checkpoint: dict[str, Any],
        moment: datetime,
        reason: str,
        *,
        system: bool = False,
    ) -> None:
        interest_id = str(checkpoint.get("interest_id") or "")
        if not interest_id:
            return
        actor = Actor(
            source=SOURCE_SYSTEM if system else SOURCE_ATTENTION_AGENT,
            session=str(checkpoint.get("umo") or ""),
            is_admin=system,
        )
        self.control.close_interest_locked(
            self.store.state,
            actor,
            interest_id=interest_id,
            reason=reason,
            moment=moment,
        )

    def _complete_participation_locked(
        self,
        umo: str,
        reservation_id: str,
        outcome: str,
        moment: datetime,
    ) -> None:
        self.control.complete_participation_locked(
            self.store.state,
            umo=umo,
            reservation_id=reservation_id,
            outcome=outcome,
            moment=moment,
        )

    def _finish_locked(
        self,
        checkpoint: dict[str, Any],
        *,
        status: str,
        last_result: str,
        error: str,
        next_minutes: int | None = None,
        evaluated: bool = False,
        no_progress: bool = False,
        model_suggestion: str = "",
        decision_model_calls: int = 0,
        evaluated_seq: int = 0,
        reservation_id: str = "",
        main_model: str = "none",
    ) -> None:
        """Finalize a code-skipped checkpoint. Caller holds the store lock."""

        self._finalize_locked(
            checkpoint,
            status=status,
            next_minutes=next_minutes,
            last_result=last_result,
            last_error=error,
            evaluated=evaluated,
            no_progress=no_progress,
        )
        interest_id = str(checkpoint.get("interest_id") or "")
        interest = (self.store.state.get("interests") or {}).get(interest_id)
        if (
            isinstance(interest, dict)
            and interest.get("status") != "closed"
            and int(interest.get("no_progress_count") or 0)
            >= self.settings.no_progress_limit
        ):
            self.control.close_interest_locked(
                self.store.state,
                Actor(
                    source=SOURCE_SYSTEM,
                    session=str(checkpoint.get("umo") or ""),
                    is_admin=True,
                ),
                interest_id=interest_id,
                reason="no_progress_limit",
                moment=self.policy.now(),
            )
            raw = (self.store.state.get("checkpoints") or {}).get(
                str(checkpoint.get("checkpoint_id") or "")
            )
            if isinstance(raw, dict):
                raw["status"] = "completed"
                raw["scheduled_at"] = ""
                raw["last_result"] = "no_progress_limit"
                raw["last_error"] = error
                raw["updated_at"] = to_iso(self.policy.now())
                raw["revision"] = int(raw.get("revision") or 1) + 1
        self._record_decision_locked(
            kind="active",
            umo=str(checkpoint.get("umo") or ""),
            trigger="checkpoint",
            code_decision=last_result,
            code=error,
            reason=error,
            suggested_action=model_suggestion,
            model_suggestion=model_suggestion,
            decision_model_calls=decision_model_calls,
            message_seq=evaluated_seq,
            reservation_id=reservation_id,
            main_model=main_model,
            checkpoint_id=str(checkpoint.get("checkpoint_id") or ""),
            interest_id=interest_id,
        )
        self._flush_locked()
        logger.info(
            "Active attention skipped: checkpoint=%s code=%s",
            checkpoint.get("checkpoint_id"),
            error,
        )

    def _finalize_locked(
        self,
        checkpoint: dict[str, Any],
        *,
        status: str,
        next_minutes: int | None,
        last_result: str,
        last_error: str,
        evaluated: bool,
        no_progress: bool,
    ) -> None:
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        raw = (self.store.state.get("checkpoints") or {}).get(checkpoint_id)
        if (
            not isinstance(raw, dict)
            or raw.get("status") != "running"
            or raw.get("run_id") != checkpoint.get("run_id")
        ):
            return
        moment = self.policy.now()
        if (
            status == "active"
            and raw.get("kind") == "session_scan"
            and raw.get("converted_to_interest")
        ):
            status = "completed"
        scheduled_at = ""
        if status == "active":
            minutes = (
                next_minutes
                if next_minutes is not None
                else self.settings.default_interval_minutes
            )
            desired = self._next_interval(moment, minutes)
            expires = parse_iso(raw.get("expires_at"), self.settings.tz)
            scheduled_at = to_iso(
                min(desired, expires) if expires is not None else desired
            )
        self.control.finalize_checkpoint_locked(
            self.store.state,
            checkpoint_id,
            status=status,
            scheduled_at=scheduled_at,
            last_result=last_result,
            last_error=last_error,
            moment=moment,
            evaluated=evaluated,
            no_progress=no_progress,
        )

    def _next_interval(self, moment: datetime, minutes: int) -> datetime:
        minutes = max(
            self.settings.min_interval_minutes,
            min(
                self.settings.max_interval_minutes,
                self.settings.max_defer_minutes,
                int(minutes),
            ),
        )
        desired = moment + timedelta(minutes=minutes)
        if in_quiet_hours(desired, self.settings.quiet_hours):
            end = quiet_hours_end(desired, self.settings.quiet_hours)
            if end is not None:
                desired = end + timedelta(minutes=1)
        return desired

    def _backoff_minutes(self, no_progress: int) -> int:
        base = self.settings.min_interval_minutes
        factor = 2 ** min(max(0, int(no_progress)), 6)
        return min(
            self.settings.max_interval_minutes,
            self.settings.max_defer_minutes,
            base * factor,
        )

    # ------------------------------------------------------------------
    # Introspection and administrative policies
    # ------------------------------------------------------------------

    def _record_decision_locked(
        self,
        *,
        kind: str,
        umo: str,
        trigger: str,
        code_decision: str,
        reason: str,
        suggested_action: str = "",
        code: str = "",
        source: str = "",
        interest_id: str = "",
        checkpoint_id: str = "",
        anchor_message_id: str = "",
        related_policy_id: str = "",
        related_directive_id: str = "",
        reservation_id: str = "",
        wake_source: str = "unknown",
        model_suggestion: str = "",
        message_seq: int = 0,
        semantic_evaluations: int | None = None,
        decision_model_calls: int = 0,
        tool_call_count: int = 0,
        cached: bool = False,
        main_model: str = "none",
        send: str = "none",
    ) -> None:
        explicit = code_decision == "explicit_pass"
        suggestion = "" if explicit else (model_suggestion or suggested_action)
        if reservation_id:
            opportunity = next(
                (
                    item
                    for item in (self.store.state.get("opportunities") or {}).values()
                    if isinstance(item, dict)
                    and item.get("umo") == umo
                    and item.get("reservation_id") == reservation_id
                ),
                {},
            )
            interest_id = interest_id or str(opportunity.get("interest_id") or "")
            anchor_message_id = anchor_message_id or str(
                opportunity.get("anchor_message_id") or ""
            )
        if not message_seq:
            message_seq = int(self._session_locked(umo).get("last_decision_seq") or 0)
        record = {
            "timestamp": to_iso(self.policy.now()),
            "kind": kind,
            "umo": umo,
            "trigger": trigger,
            "wake_source": wake_source,
            "decision": code_decision,
            "code_decision": code_decision,
            "suggested_action": "" if explicit else suggested_action,
            "model_suggestion": clip_text(suggestion, 500),
            "semantic_evaluations": int(bool(suggestion))
            if semantic_evaluations is None
            else semantic_evaluations,
            "decision_model_calls": decision_model_calls,
            "tool_call_count": tool_call_count,
            "rejection_code": code,
            "interest": "none",
            "reason": clip_text(reason, self.settings.max_reason_chars),
            "source": source
            or ("passive_agent" if kind == "passive" else "active_agent"),
            "interest_id": interest_id,
            "checkpoint_id": checkpoint_id,
            "anchor_message_id": anchor_message_id,
            "related_policy_id": related_policy_id,
            "related_directive_id": related_directive_id,
            "reservation_id": reservation_id,
            "message_seq": int(message_seq or 0),
            "cached": cached,
            "tool_calls": [],
            "checkpoint_changes": [],
            "main_model": main_model
            if main_model != "none"
            else (
                "failed"
                if code == "generation_failed"
                else "generated"
                if code in {"sent", "send_failed", "uncertain_delivery"}
                else "none"
            ),
            "send": code
            if code in {"sent", "send_failed", "uncertain_delivery"}
            else send,
            "code_skipped": not bool(suggestion),
            "resource_consumption": {
                "decision_calls": decision_model_calls,
                "autonomous_attempt": bool(reservation_id)
                and code == "autonomous_approved",
            },
        }
        self.store.record_decision(record)

    async def status_snapshot(self, umo: str = "") -> dict[str, Any]:
        """Return a global or session-scoped status snapshot."""

        async with self.store.lock:
            state = self.store.state
            moment = self.policy.now()
            self.control.expire_interests_locked(state, moment)
            self.control.cleanup_locked(state, moment)
            checkpoints = state.get("checkpoints", {})
            counts: dict[str, int] = {}
            for raw in checkpoints.values():
                if not isinstance(raw, dict):
                    continue
                if umo and raw.get("umo") != umo:
                    continue
                status = str(raw.get("status") or "unknown")
                counts[status] = counts.get(status, 0) + 1
            if umo:
                decisions = [
                    item
                    for item in state.get("decisions", [])
                    if isinstance(item, dict) and item.get("umo") == umo
                ][-5:]
                interests = self.control.session_interests_locked(state, umo)
                opportunities = [
                    item
                    for item in (state.get("opportunities") or {}).values()
                    if isinstance(item, dict)
                    and item.get("umo") == umo
                    and item.get("status") in {"pending", "reserved"}
                ]
            else:
                decisions = [
                    item
                    for item in state.get("decisions", [])
                    if isinstance(item, dict)
                ][-5:]
                interests = [
                    item
                    for item in (state.get("interests") or {}).values()
                    if isinstance(item, dict) and item.get("status") != "closed"
                ]
                opportunities = [
                    item
                    for item in (state.get("opportunities") or {}).values()
                    if isinstance(item, dict)
                    and item.get("status") in {"pending", "reserved"}
                ]
            counters = [
                value
                for key, value in (state.get("metrics") or {}).items()
                if isinstance(value, dict) and (not umo or key == umo)
            ]
            metrics = {
                key: sum(int(item.get(key) or 0) for item in counters)
                for key in (
                    "semantic_decisions",
                    "suggestions",
                    "approved_reservations",
                    "passive_passes",
                    "explicit_passes",
                    "active_sends",
                )
            }
            metrics["received_messages"] = sum(
                int(item.get("received_messages") or 0)
                for key, item in (state.get("sessions") or {}).items()
                if isinstance(item, dict) and (not umo or key == umo)
            )
            rejections: dict[str, int] = {}
            for item in counters:
                for code, count in (item.get("rejection_counts") or {}).items():
                    rejections[code] = rejections.get(code, 0) + int(count)
            metrics["rejection_counts"] = rejections
            return {
                "started": self._started,
                "scope": umo or "all",
                "passive_mode": self.settings.passive_mode,
                "active_enabled": self.settings.active_enabled,
                "dry_run": self.settings.dry_run,
                "checkpoint_counts": counts,
                "interest_count": len(interests),
                "opportunity_count": len(opportunities),
                "policy_count": len(state.get("policies") or {}),
                "budget": self.policy.describe_budget(state, umo),
                "scheduler": self.scheduler.status(),
                "metrics": metrics,
                "recent_decisions": decisions,
                "store_path": str(self.store.path),
                "migrated": self.store.migrated,
                "migration": state.get("migration", {}),
            }

    async def set_policy(
        self,
        *,
        scope: str,
        group_id: str,
        user_id: str,
        reason: str,
        expires_minutes: int | None,
        created_by: str = "admin",
    ) -> tuple[bool, str]:
        """Create or replace an ignore policy from an admin command."""

        if scope not in {"group_user", "user", "group"}:
            return False, "unsupported policy scope"
        if scope in {"group_user", "group"} and not group_id:
            return False, "group_id is required for this scope"
        if scope in {"group_user", "user"} and not user_id:
            return False, "user_id is required for this scope"
        async with self.store.lock:
            moment = self.policy.now()
            expires_at = (
                add_minutes(moment, expires_minutes)
                if expires_minutes is not None
                else None
            )
            result = self.control.set_policy_locked(
                self.store.state,
                Actor(
                    source=SOURCE_ADMIN,
                    session="",
                    group_id=group_id,
                    is_admin=True,
                ),
                scope=scope,
                group_id=group_id,
                user_id=user_id,
                reason=clip_text(reason, 120) or "admin policy",
                expires_at=expires_at,
                source=SOURCE_ADMIN,
                created_by_session="",
                allow_permanent=True,
            )
            if result.ok:
                self._flush_locked()
                return True, str(result.data.get("policy_id", ""))
            return False, result.message

    async def remove_policy(
        self,
        *,
        scope: str,
        group_id: str,
        user_id: str,
    ) -> tuple[bool, str]:
        """Remove one explicit policy by its natural key."""

        async with self.store.lock:
            policy_id = self.policy.policy_id(scope, group_id, user_id)
            policy = (self.store.state.get("policies") or {}).get(policy_id)
            if not isinstance(policy, dict):
                return False, "policy not found"
            result = self.control.remove_policy_locked(
                self.store.state,
                Actor(source=SOURCE_ADMIN, is_admin=True),
                policy,
            )
            if result.ok:
                self._flush_locked()
                return True, policy_id
            return False, result.message
