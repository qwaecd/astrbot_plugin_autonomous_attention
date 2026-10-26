"""Attention engine: coordinates passive and active attention sessions."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any

from .active import ActiveAttentionRunner
from .control import Actor, AttentionControlService, ControlResult
from .models import (
    SOURCE_ADMIN,
    SOURCE_ATTENTION_AGENT,
    SOURCE_MAIN_AGENT,
    ActiveContext,
    Checkpoint,
    PassiveDecision,
    PassiveOutcome,
    PassiveRequest,
    RecentMessage,
    TopicState,
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


class AttentionEngine:
    """Own state, policies, budget accounting and both attention paths."""

    def __init__(
        self,
        *,
        settings: Settings,
        store: StateStore,
        gateway: ModelGateway,
        context_provider: ContextProvider,
        sender: MessageSender,
        logger: logging.Logger,
    ) -> None:
        self.settings = settings
        self.store = store
        self.gateway = gateway
        self.context_provider = context_provider
        self.sender = sender
        self.logger = logger
        self.policy = PolicyManager(settings, logger)
        self.control = AttentionControlService(
            store,
            settings,
            self.policy,
            logger,
        )
        self.passive_runner = PassiveAttentionRunner(settings, gateway, logger)
        self.active_runner = ActiveAttentionRunner(settings, gateway, logger)
        self.scheduler = AttentionScheduler(self, logger)
        self._passive_semaphore = asyncio.Semaphore(settings.max_concurrent_decisions)
        self._passive_inflight: set[str] = set()
        self._passive_done: OrderedDict[str, PassiveOutcome] = OrderedDict()
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load persisted state, recover interrupted checkpoints and schedule."""

        await self.store.load()
        async with self.store.lock:
            self.policy.expire_policies(self.store.state, self.policy.now())
            self._recover_running_locked()
            self._seed_checkpoints_locked()
            self._flush_locked()
        self._started = True
        await self.scheduler.start()
        self.logger.info(
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
        self.logger.info("Autonomous attention core stopped.")

    def update_settings(self, settings: Settings) -> None:
        """Apply a new configuration snapshot without restarting the plugin."""

        self.settings = settings
        self.policy.update_settings(settings)
        self.control.update_settings(settings)
        self.passive_runner.update_settings(settings)
        self.active_runner.update_settings(settings)
        self._passive_semaphore = asyncio.Semaphore(settings.max_concurrent_decisions)

    def _flush_locked(self) -> None:
        try:
            self.store.flush()
        except Exception:
            self.logger.exception("Failed to persist attention state.")

    def _recover_running_locked(self) -> None:
        moment = self.policy.now()
        for raw in self.store.state.get("checkpoints", {}).values():
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
            self.logger.warning(
                "Recovered interrupted checkpoint %s; it will retry after the "
                "minimum interval.",
                raw.get("checkpoint_id"),
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
                self._create_checkpoint_locked(
                    umo=umo,
                    platform_id=platform_id,
                    group_id=group_id,
                    scheduled_at=scheduled,
                    reason=reason,
                    created_by="config",
                    dedupe=False,
                    source=SOURCE_ADMIN,
                    actor=Actor(
                        source=SOURCE_ADMIN,
                        session=umo,
                        group_id=group_id,
                        platform_id=platform_id,
                        is_admin=True,
                    ),
                )
                seeded.append(signature)
                self.logger.info(
                    "Seeded attention checkpoint for %s (%s).",
                    group_id,
                    reason,
                )
            except Exception as exc:
                self.logger.warning(
                    "Ignoring invalid seed checkpoint %s: %s",
                    item,
                    exc,
                )

    # ------------------------------------------------------------------
    # Passive attention
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
        """Record one group message for active sliding-window context.

        Passive group messages are observed even when AstrBot's own group
        history is disabled. Only the in-memory window is updated here; it is
        persisted together with the next decision or checkpoint change.
        """

        if (
            not self._started
            or not self.settings.enabled
            or not self.settings.active_enabled
        ):
            return
        if not group_id:
            return
        if not self.policy.group_allowed(group_id, umo, platform_id):
            return
        message = RecentMessage(
            role="bot" if is_bot else "user",
            sender_id=sender_id,
            sender_name=sender_name,
            text=clip_text(text, 1000),
            created_at=created_at,
            created_ts=created_ts,
            message_id=message_id,
            is_bot=is_bot,
        )
        if not message.text:
            return
        async with self.store.lock:
            self.store.remember_recent(
                umo,
                message,
                max_messages=max(20, self.settings.context_window_messages * 2),
                max_sessions=self.settings.max_context_sessions,
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
        """Return the hard ignore policy for a sender, if any.

        This is used by the adapter before command handlers run, so ignored
        users cannot reach AstrBot commands or the main model.
        """

        if not self._started or not self.settings.enabled:
            return None
        async with self.store.lock:
            self.policy.expire_policies(self.store.state, self.policy.now())
            policy = self.policy.is_user_blocked(
                self.store.state,
                group_id=group_id,
                user_id=user_id,
                umo=umo,
                platform_id=platform_id,
                is_admin=is_admin,
            )
            return dict(policy) if policy is not None else None

    async def handle_passive(self, request: PassiveRequest) -> PassiveOutcome:
        """Run the passive attention path for one woken message.

        Args:
            request: Normalised woken message built by the adapter.

        Returns:
            An outcome telling the adapter whether to stop the event.
        """

        if not self._started or not self.settings.enabled:
            return PassiveOutcome(decision="pass", reason="plugin disabled")
        if not self.settings.passive_full:
            return PassiveOutcome(
                decision="pass",
                reason=f"passive mode is {self.settings.passive_mode}",
            )
        if request.is_command:
            return PassiveOutcome(decision="pass", reason="command event")

        moment = self.policy.now()
        if request.is_group and not self.policy.group_allowed(
            request.group_id,
            request.umo,
            request.platform_id,
        ):
            return PassiveOutcome(decision="pass", reason="group not allowed")

        key = self._passive_key(request)
        async with self.store.lock:
            self.policy.expire_policies(self.store.state, moment)
            done = self._passive_done.get(key)
            if done is not None:
                return done
            if key in self._passive_inflight:
                return PassiveOutcome(
                    decision="observe",
                    block=True,
                    reason="duplicate passive decision is already running",
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
                    ),
                )
                self._record_decision_locked(
                    kind="passive",
                    umo=request.umo,
                    trigger=request.trigger,
                    decision="ignore",
                    interest="none",
                    reason=str(blocked.get("reason") or "hard policy"),
                    source=(
                        SOURCE_ADMIN
                        if blocked.get("source") == SOURCE_ADMIN
                        else SOURCE_MAIN_AGENT
                        if blocked.get("source") == SOURCE_MAIN_AGENT
                        else "passive_agent"
                    ),
                    related_policy_id=str(blocked.get("policy_id") or ""),
                    extra={"policy_scope": blocked.get("scope", "")},
                )
                self._flush_locked()
                return PassiveOutcome(
                    decision="ignore",
                    block=True,
                    reason="hard user policy",
                )
            topic = self._copy_topic_locked(
                request.umo,
                request.group_id,
            )
            topic_id = str(topic.get("topic_id") or self._default_topic_id(request.umo))
            focus = self.control.focus_directive(
                self.store.state,
                group_id=request.group_id,
                umo=request.umo,
                topic_id=topic_id,
                message_text=request.message_text,
            )
            if focus is not None:
                outcome = PassiveOutcome(
                    decision="pass",
                    block=False,
                    reason=f"main agent focus directive: {focus.get('target')}",
                )
                self._remember_passive(key, outcome)
                self._record_decision_locked(
                    kind="passive",
                    umo=request.umo,
                    trigger=request.trigger,
                    decision="pass",
                    interest="high",
                    reason=outcome.reason,
                    source=SOURCE_MAIN_AGENT,
                    related_directive_id=str(focus.get("directive_id") or ""),
                )
                self._flush_locked()
                return outcome
            if not self.policy.consume_model_call(
                self.store.state,
                moment,
                "decision",
            ):
                self._record_decision_locked(
                    kind="passive",
                    umo=request.umo,
                    trigger=request.trigger,
                    decision="pass",
                    interest="none",
                    reason="daily model-call budget exhausted",
                    extra={"model_error": "budget"},
                )
                self._flush_locked()
                return PassiveOutcome(
                    decision="pass",
                    reason="daily model-call budget exhausted",
                )
            self._passive_inflight.add(key)
            policy_summary = self.control.prompt_summary_locked(
                self.store.state,
                group_id=request.group_id,
                user_id=request.user_id,
                umo=request.umo,
                platform_id=request.platform_id,
                topic_id=topic_id,
                message_text=request.message_text,
            )
            self._flush_locked()

        try:
            async with self._passive_semaphore:
                run = await self.passive_runner.decide(
                    request,
                    topic,
                    policy_summary,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.exception("Passive attention runner failed.")
            run = PassiveRunResult(kind="error", error=str(exc))

        if run.kind == "ok" and run.decision is not None:
            decision = run.decision
            model_error = ""
        elif run.kind == "invalid":
            decision = PassiveDecision(
                decision="observe",
                interest_level="low",
                reason="invalid structured output; safe fallback",
            )
            model_error = run.error
        else:
            decision = PassiveDecision(
                decision="pass",
                interest_level="none",
                reason="decision model unavailable; passed through",
            )
            model_error = run.error
            self.logger.warning(
                "Passive decision model failed for %s: %s. Passing through to "
                "AstrBot's main model.",
                request.umo,
                run.error,
            )

        async with self.store.lock:
            self._passive_inflight.discard(key)
            now_after = self.policy.now()
            focus_after = self.control.focus_directive(
                self.store.state,
                group_id=request.group_id,
                umo=request.umo,
                topic_id=topic_id,
                message_text=request.message_text,
            )
            if focus_after is not None and decision.decision != "pass":
                decision = PassiveDecision(
                    decision="pass",
                    interest_level="high",
                    reason=(
                        "main agent focus directive takes precedence: "
                        f"{focus_after.get('target')}"
                    ),
                )
            if self.settings.dry_run:
                outcome = PassiveOutcome(
                    decision=decision.decision,
                    block=False,
                    reason=decision.reason,
                    dry_run=True,
                    model_error=model_error,
                )
                self._remember_passive(key, outcome)
                self._record_decision_locked(
                    kind="passive",
                    umo=request.umo,
                    trigger=request.trigger,
                    decision=decision.decision,
                    interest=decision.interest_level,
                    reason=decision.reason,
                    extra={
                        "dry_run": True,
                        "model_error": model_error,
                        "provider": run.provider_id,
                        "usage": run.usage,
                    },
                )
                self._flush_locked()
                return outcome

            changes = self._apply_passive_decision_locked(
                request,
                decision,
                now_after,
            )
            self._update_topic_from_passive_locked(
                request,
                decision,
                now_after,
            )
            outcome = PassiveOutcome(
                decision=decision.decision,
                block=decision.decision != "pass",
                reason=decision.reason,
                model_error=model_error,
            )
            self._remember_passive(key, outcome)
            self._record_decision_locked(
                kind="passive",
                umo=request.umo,
                trigger=request.trigger,
                decision=decision.decision,
                interest=decision.interest_level,
                reason=decision.reason,
                extra={
                    "confidence": decision.confidence,
                    "checkpoint_changes": changes,
                    "provider": run.provider_id,
                    "usage": run.usage,
                    "main_model": "blocked" if outcome.block else "passed_through",
                    "model_error": model_error,
                },
            )
            self._flush_locked()

        self.logger.info(
            "Passive attention: umo=%s user=%s decision=%s interest=%s reason=%s",
            request.umo,
            request.user_id or "-",
            decision.decision,
            decision.interest_level,
            clip_text(decision.reason, 120),
        )
        return outcome

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

    def _apply_passive_decision_locked(
        self,
        request: PassiveRequest,
        decision: PassiveDecision,
        moment: datetime,
    ) -> list[str]:
        changes: list[str] = []
        if decision.user_action:
            policy = self._build_auto_policy_locked(
                decision.user_action,
                request,
                moment,
            )
            if policy is not None and policy.ok:
                changes.append(f"policy:{policy.data.get('policy_id', '')}")
                if policy.data.get("directive_id"):
                    changes.append(f"directive:{policy.data.get('directive_id', '')}")
        if decision.decision == "retry_later" and request.is_group:
            if self.settings.active_enabled and self.policy.group_allowed(
                request.group_id,
                request.umo,
                request.platform_id,
            ):
                try:
                    checkpoint = self._create_checkpoint_locked(
                        umo=request.umo,
                        platform_id=request.platform_id,
                        group_id=request.group_id,
                        scheduled_at=add_minutes(
                            moment,
                            self.settings.passive_retry_delay_minutes,
                        ),
                        reason=f"retry_later: {decision.reason}"[:200],
                        created_by="agent",
                        dedupe=True,
                        source=SOURCE_ATTENTION_AGENT,
                        actor=Actor(
                            source=SOURCE_ATTENTION_AGENT,
                            user_id=request.user_id,
                            session=request.umo,
                            group_id=request.group_id,
                            platform_id=request.platform_id,
                            is_admin=request.is_admin,
                        ),
                    )
                except Exception as exc:
                    self.logger.warning(
                        "retry_later checkpoint creation failed: %s",
                        exc,
                    )
                    checkpoint = None
                if checkpoint is not None:
                    changes.append(f"checkpoint:{checkpoint['checkpoint_id']}")
        return changes

    def _build_auto_policy_locked(
        self,
        user_action: dict[str, Any],
        request: PassiveRequest,
        moment: datetime,
    ) -> ControlResult | None:
        scope = str(user_action.get("scope") or "group_user")
        if scope == "group_user" and not request.group_id:
            scope = "user"
        if scope not in {"group_user", "user"}:
            return None
        if not request.user_id:
            return None
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
        return self.control.set_policy_locked(
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

    def _update_topic_from_passive_locked(
        self,
        request: PassiveRequest,
        decision: PassiveDecision,
        moment: datetime,
    ) -> None:
        topic = self._copy_topic_locked(request.umo, request.group_id)
        summary = decision.topic_summary
        if not summary and decision.topic_action == "update":
            summary = clip_text(request.message_text, 200)
        topic["summary"] = summary or topic.get("summary") or ""
        topic["interest_level"] = decision.interest_level
        topic["interest_source"] = "passive"
        topic["focus_status"] = (
            "closed" if decision.topic_action == "close" else "watching"
        )
        topic["last_observation"] = f"{decision.decision}: {decision.reason}"[:300]
        topic["has_opening"] = decision.decision in {"pass", "retry_later"}
        topic["next_step"] = (
            "等待下一次检查"
            if decision.decision == "retry_later"
            else "继续观察"
            if decision.decision in {"observe", "ignore", "policy_update"}
            else "可自然接话"
        )
        topic["last_activity_at"] = request.timestamp or to_iso(moment)
        topic["updated_at"] = to_iso(moment)
        topic["revision"] = int(topic.get("revision") or 0) + 1
        self.store.state.setdefault("topics", {})[topic["topic_id"]] = topic

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
        directive_id: str = "",
    ) -> dict[str, Any]:
        """Create a checkpoint from an explicit admin action."""

        async with self.store.lock:
            checkpoint = self._create_checkpoint_locked(
                umo=umo,
                platform_id=platform_id,
                group_id=group_id,
                scheduled_at=scheduled_at,
                minutes=minutes,
                reason=reason,
                created_by=created_by,
                dedupe=True,
                source=source,
                actor=actor,
                directive_id=directive_id,
            )
            self._flush_locked()
            return checkpoint

    def _create_checkpoint_locked(
        self,
        *,
        umo: str = "",
        platform_id: str = "",
        group_id: str = "",
        scheduled_at: datetime | None = None,
        minutes: int | None = None,
        reason: str = "",
        created_by: str = "agent",
        dedupe: bool = True,
        source: str = SOURCE_ATTENTION_AGENT,
        actor: Actor | None = None,
        directive_id: str = "",
    ) -> dict[str, Any] | None:
        umo, platform_id, group_id = self._resolve_target(
            umo,
            platform_id,
            group_id,
        )
        if not group_id:
            raise ValueError("checkpoint target must be a group session")
        if not self.policy.group_allowed(group_id, umo, platform_id):
            raise ValueError("group is not allowed by the active group policy")
        if self.policy.group_blocked(group_id, umo, platform_id):
            raise ValueError("group is blocked by policy")
        moment = self.policy.now()
        if scheduled_at is None:
            scheduled_at = add_minutes(
                moment,
                minutes
                if minutes is not None
                else self.settings.default_interval_minutes,
            )
        scheduled_at = self._clamp_schedule(moment, scheduled_at)
        reason = clip_text(reason, 200)
        if dedupe and created_by == "agent":
            for raw in self.store.state.get("checkpoints", {}).values():
                if not isinstance(raw, dict) or raw.get("status") != "active":
                    continue
                if raw.get("umo") != umo:
                    continue
                existing = parse_iso(raw.get("scheduled_at"), self.settings.tz)
                if (
                    existing is not None
                    and abs((existing - scheduled_at).total_seconds()) < 120
                ):
                    return dict(raw)
        checkpoint = Checkpoint(
            checkpoint_id=f"cp_{uuid.uuid4().hex[:12]}",
            umo=umo,
            platform_id=platform_id,
            group_id=group_id,
            scheduled_at=to_iso(scheduled_at),
            status="active",
            reason=reason,
            topic_id=self._default_topic_id(umo),
            created_by=source,
            revision=1,
            created_at=to_iso(moment),
            updated_at=to_iso(moment),
        ).to_dict()
        actor = actor or Actor(
            source=source,
            session=umo,
            group_id=group_id,
            platform_id=platform_id,
            is_admin=source == SOURCE_ADMIN,
        )
        result = self.control.register_checkpoint_locked(
            self.store.state,
            checkpoint,
            actor=actor,
            source=source,
            directive_id=directive_id,
        )
        if not result.ok:
            raise ValueError(result.message)
        return dict(result.data)

    @staticmethod
    def _resolve_target(
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
            platform_id = parts[0]
            if not group_id:
                group_id = parts[2]
        elif umo and not group_id:
            group_id = umo
            umo = ""
        if group_id and not umo:
            if not platform_id:
                raise ValueError("platform_id is required when umo is not given")
            umo = f"{platform_id}:GroupMessage:{group_id}"
        return umo, platform_id, group_id

    def _default_topic_id(self, umo: str) -> str:
        digest = hashlib.sha1(umo.encode("utf-8")).hexdigest()[:12]
        return f"topic_{digest}"

    def _copy_topic_locked(self, umo: str, group_id: str) -> dict[str, Any]:
        topic_id = self._default_topic_id(umo)
        raw = (self.store.state.get("topics") or {}).get(topic_id)
        if isinstance(raw, dict):
            return dict(raw)
        moment = self.policy.now()
        topic = TopicState(
            topic_id=topic_id,
            umo=umo,
            group_id=group_id,
            interest_level="none",
            focus_status="idle",
            revision=1,
            updated_at=to_iso(moment),
        )
        return topic.to_dict()

    def _clamp_schedule(
        self,
        moment: datetime,
        desired: datetime,
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
        return self._shift_out_of_quiet(moment, desired)

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

    def _next_schedule(
        self,
        moment: datetime,
        minutes: int | None,
    ) -> datetime:
        if minutes is None:
            minutes = self.settings.default_interval_minutes
        minutes = max(
            self.settings.min_interval_minutes,
            min(
                min(
                    self.settings.max_interval_minutes,
                    self.settings.max_defer_minutes,
                ),
                int(minutes),
            ),
        )
        return self._clamp_schedule(moment, add_minutes(moment, minutes))

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        """Return checkpoints sorted by next scheduled time."""

        async with self.store.lock:
            items = [
                dict(raw)
                for raw in self.store.state.get("checkpoints", {}).values()
                if isinstance(raw, dict)
            ]
        items.sort(
            key=lambda item: (
                str(item.get("status") or ""),
                str(item.get("scheduled_at") or ""),
            )
        )
        return items

    async def list_policies(self) -> list[dict[str, Any]]:
        async with self.store.lock:
            return self.policy.active_policies(self.store.state)

    async def set_checkpoint_status(
        self,
        checkpoint_id: str,
        status: str,
    ) -> tuple[bool, str]:
        """Pause, resume or cancel one checkpoint."""

        async with self.store.lock:
            raw = (self.store.state.get("checkpoints") or {}).get(checkpoint_id)
            if not isinstance(raw, dict):
                return False, "checkpoint not found"
            current = str(raw.get("status") or "")
            if current == "running":
                return False, "checkpoint is currently running"
            moment = self.policy.now()
            scheduled_at: str | None = None
            if status == "active":
                scheduled = parse_iso(raw.get("scheduled_at"), self.settings.tz)
                if scheduled is None or scheduled <= moment:
                    scheduled_at = to_iso(self._next_schedule(moment, None))
            elif status not in {"paused", "cancelled"}:
                return False, "unsupported checkpoint status"
            result = self.control.set_checkpoint_status_locked(
                self.store.state,
                Actor(source=SOURCE_ADMIN, is_admin=True),
                checkpoint_id,
                status,
                moment=moment,
                scheduled_at=scheduled_at,
            )
            if not result.ok:
                return False, result.message
            self._flush_locked()
            return True, str(result.data.get("status") or status)

    async def claim_checkpoint(
        self,
        checkpoint_id: str,
        *,
        force: bool = False,
    ) -> tuple[dict[str, Any] | None, str]:
        """Atomically claim a checkpoint for one active session.

        Args:
            checkpoint_id: Checkpoint to claim.
            force: Ignore the scheduled time (admin ``check`` action).

        Returns:
            ``(checkpoint, reason)``. A ``None`` checkpoint means the claim was
            rejected and ``reason`` explains why.
        """

        async with self.store.lock:
            raw = (self.store.state.get("checkpoints") or {}).get(checkpoint_id)
            if not isinstance(raw, dict):
                return None, "checkpoint not found"
            if not self.settings.enabled or not self.settings.active_enabled:
                return None, "active attention is disabled"
            if raw.get("status") not in {"active", "paused"}:
                return None, f"checkpoint status is {raw.get('status')}"
            if not self.policy.group_allowed(
                str(raw.get("group_id") or ""),
                str(raw.get("umo") or ""),
                str(raw.get("platform_id") or ""),
            ):
                return None, "group is not allowed by the group allowlist"
            moment = self.policy.now()
            scheduled = parse_iso(raw.get("scheduled_at"), self.settings.tz)
            if not force and scheduled is not None and scheduled > moment:
                return None, "checkpoint is not due"
            if not self.policy.can_run_active_check(self.store.state, moment):
                return None, "daily active-check budget exhausted"
            if not self.policy.consume_active_check(self.store.state, moment):
                return None, "daily active-check budget exhausted"
            raw["status"] = "running"
            raw["run_id"] = uuid.uuid4().hex[:12]
            raw["last_run_at"] = to_iso(moment)
            raw["attempts"] = int(raw.get("attempts") or 0) + 1
            raw["revision"] = int(raw.get("revision") or 1) + 1
            raw["updated_at"] = to_iso(moment)
            self._flush_locked()
            return dict(raw), ""

    async def claim_next_due(self) -> dict[str, Any] | None:
        """Claim the earliest due checkpoint for the scheduler loop."""

        async with self.store.lock:
            if not self.settings.enabled or not self.settings.active_enabled:
                return None
            moment = self.policy.now()
            if not self.policy.can_run_active_check(self.store.state, moment):
                return None
            due: list[dict[str, Any]] = []
            for raw in self.store.state.get("checkpoints", {}).values():
                if not isinstance(raw, dict) or raw.get("status") != "active":
                    continue
                if not self.policy.group_allowed(
                    str(raw.get("group_id") or ""),
                    str(raw.get("umo") or ""),
                    str(raw.get("platform_id") or ""),
                ):
                    continue
                scheduled = parse_iso(raw.get("scheduled_at"), self.settings.tz)
                if scheduled is None or scheduled <= moment:
                    due.append(dict(raw))
            if not due:
                return None
            due.sort(key=lambda item: str(item.get("scheduled_at") or ""))
            chosen = due[0]
            if not self.policy.consume_active_check(self.store.state, moment):
                return None
            raw = self.store.state["checkpoints"][chosen["checkpoint_id"]]
            raw["status"] = "running"
            raw["run_id"] = uuid.uuid4().hex[:12]
            raw["last_run_at"] = to_iso(moment)
            raw["attempts"] = int(raw.get("attempts") or 0) + 1
            raw["revision"] = int(raw.get("revision") or 1) + 1
            raw["updated_at"] = to_iso(moment)
            self._flush_locked()
            return dict(raw)

    async def seconds_until_next_due(self) -> int:
        """Return a scheduler sleep hint in seconds."""

        async with self.store.lock:
            if not self.settings.enabled or not self.settings.active_enabled:
                return 30
            moment = self.policy.now()
            delays: list[float] = []
            for raw in self.store.state.get("checkpoints", {}).values():
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
    ) -> tuple[bool, str]:
        """Claim and enqueue a checkpoint for immediate execution."""

        claimed, reason = await self.claim_checkpoint(checkpoint_id, force=True)
        if claimed is None:
            return False, reason
        await self.scheduler.enqueue_checkpoint(checkpoint_id)
        return True, "checkpoint execution queued"

    # ------------------------------------------------------------------
    # Active attention
    # ------------------------------------------------------------------

    async def run_claimed(self, checkpoint_id: str) -> None:
        """Execute one claimed checkpoint and never leave it stuck running."""

        try:
            await self._run_claimed_inner(checkpoint_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.exception(
                "Active attention checkpoint %s crashed.",
                checkpoint_id,
            )
            async with self.store.lock:
                raw = (self.store.state.get("checkpoints") or {}).get(checkpoint_id)
                if isinstance(raw, dict) and raw.get("status") == "running":
                    self._finalize_locked(
                        raw,
                        decision="observe",
                        next_minutes=None,
                        status="active",
                        moment=self.policy.now(),
                        error=str(exc),
                    )
                    self._record_decision_locked(
                        kind="active",
                        umo=str(raw.get("umo") or ""),
                        trigger="checkpoint",
                        decision="observe",
                        interest="none",
                        reason=f"checkpoint crashed: {exc}",
                        extra={"checkpoint_id": checkpoint_id},
                    )
                    self._flush_locked()

    async def _run_claimed_inner(self, checkpoint_id: str) -> None:
        """Execute one checkpoint previously claimed by the scheduler or admin."""

        async with self.store.lock:
            raw = (self.store.state.get("checkpoints") or {}).get(checkpoint_id)
            if not isinstance(raw, dict) or raw.get("status") != "running":
                return
            checkpoint = dict(raw)
            recent = self.store.recent_window(str(checkpoint.get("umo") or ""))
            topic = self._copy_topic_locked(
                str(checkpoint.get("umo") or ""),
                str(checkpoint.get("group_id") or ""),
            )
        moment = self.policy.now()
        umo = str(checkpoint.get("umo") or "")
        platform_id = str(checkpoint.get("platform_id") or "")
        group_id = str(checkpoint.get("group_id") or "")

        if not self.settings.enabled or not self.settings.active_enabled:
            await self._finish_without_reply(
                checkpoint,
                decision="observe",
                reason="active attention disabled",
                next_minutes=None,
                extra={"last_error": "active attention disabled"},
            )
            return
        if not self.policy.group_allowed(group_id, umo, platform_id):
            await self._finish_without_reply(
                checkpoint,
                decision="skip",
                reason="group is not allowed",
                next_minutes=None,
                status="cancelled",
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
            await self._finish_without_reply(
                checkpoint,
                decision="observe",
                reason="quiet hours; deferred",
                next_minutes=minutes,
            )
            return

        async with self.store.lock:
            stop_directive = self.control.stop_directive(
                self.store.state,
                group_id=group_id,
                umo=umo,
                topic_id=str(topic.get("topic_id") or ""),
            )
        if (
            stop_directive is not None
            and str(checkpoint.get("source") or "") != SOURCE_ADMIN
        ):
            await self._finish_without_reply(
                checkpoint,
                decision="skip",
                reason="main agent stop_focus directive",
                next_minutes=None,
                status="cancelled",
                extra={
                    "source": SOURCE_MAIN_AGENT,
                    "related_directive_id": str(
                        stop_directive.get("directive_id") or ""
                    ),
                },
            )
            return

        window = await self.context_provider.window(
            umo=umo,
            platform_id=platform_id,
            group_id=group_id,
            recent=recent,
            limit=self.settings.context_window_messages,
        )
        if not window:
            await self._finish_without_reply(
                checkpoint,
                decision="skip",
                reason="no readable group context",
                next_minutes=None,
            )
            return

        persona = await self.context_provider.persona(umo)
        async with self.store.lock:
            policy_summary = self.control.prompt_summary_locked(
                self.store.state,
                group_id=group_id,
                umo=umo,
                platform_id=platform_id,
                topic_id=str(topic.get("topic_id") or ""),
                message_text=window[-1].text if window else "",
            )
        ctx = ActiveContext(
            checkpoint=checkpoint,
            topic=topic,
            window=window,
            persona=persona,
            policy_summary=policy_summary,
        )

        async def consume_call() -> bool:
            async with self.store.lock:
                allowed = self.policy.consume_model_call(
                    self.store.state,
                    self.policy.now(),
                    "decision",
                )
                if allowed:
                    self._flush_locked()
                return allowed

        async def window_loader(limit: int) -> list[RecentMessage]:
            async with self.store.lock:
                latest = self.store.recent_window(umo)
            return await self.context_provider.window(
                umo=umo,
                platform_id=platform_id,
                group_id=group_id,
                recent=latest,
                limit=limit,
            )

        result = await self.active_runner.run(ctx, consume_call, window_loader)

        if result.decision != "participate":
            await self._finish_active_result(ctx, result)
            return

        await self._handle_participation(ctx, result, checkpoint)

    async def _finish_active_result(
        self,
        ctx: ActiveContext,
        result: Any,
    ) -> None:
        """Apply effects and finalize a non-speaking active result."""

        checkpoint = ctx.checkpoint
        moment = self.policy.now()
        async with self.store.lock:
            changes = self._apply_active_effects_locked(ctx, result, moment)
            self._update_topic_from_active_locked(ctx, result, moment)
            stop_directive = self.control.stop_directive(
                self.store.state,
                group_id=str(checkpoint.get("group_id") or ""),
                umo=str(checkpoint.get("umo") or ""),
                topic_id=str(ctx.topic.get("topic_id") or ""),
            )
            status = "completed" if result.decision == "end_focus" else "active"
            if (
                stop_directive is not None
                and str(checkpoint.get("source") or "") != SOURCE_ADMIN
            ):
                status = "cancelled"
            self._finalize_locked(
                checkpoint,
                decision=result.decision,
                next_minutes=result.next_check_minutes,
                status=status,
                moment=moment,
                error=result.reason if result.interrupted else "",
            )
            self._record_decision_locked(
                kind="active",
                umo=str(checkpoint.get("umo") or ""),
                trigger="checkpoint",
                decision=result.decision,
                interest=result.interest_level,
                reason=result.reason,
                extra={
                    "checkpoint_id": checkpoint.get("checkpoint_id"),
                    "tool_calls": result.tool_calls,
                    "model_calls": result.model_calls,
                    "checkpoint_changes": changes,
                    "main_model": "not_called",
                    "send": "none",
                },
            )
            self._flush_locked()
        self.logger.info(
            "Active attention: checkpoint=%s group=%s decision=%s interest=%s reason=%s",
            checkpoint.get("checkpoint_id"),
            checkpoint.get("group_id"),
            result.decision,
            result.interest_level or "-",
            clip_text(result.reason, 120),
        )

    async def _handle_participation(
        self,
        ctx: ActiveContext,
        result: Any,
        checkpoint: dict[str, Any],
    ) -> None:
        """Generate and send one proactive message after a participate decision."""

        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        umo = str(checkpoint.get("umo") or "")
        group_id = str(checkpoint.get("group_id") or "")
        platform_id = str(checkpoint.get("platform_id") or "")
        moment = self.policy.now()

        async with self.store.lock:
            changes = self._apply_active_effects_locked(ctx, result, moment)
            self._update_topic_from_active_locked(ctx, result, moment)
            stop_directive = self.control.stop_directive(
                self.store.state,
                group_id=group_id,
                umo=umo,
                topic_id=str(ctx.topic.get("topic_id") or ""),
            )
            if (
                stop_directive is not None
                and str(checkpoint.get("source") or "") != SOURCE_ADMIN
            ):
                self._finalize_locked(
                    checkpoint,
                    decision="observe",
                    next_minutes=result.next_check_minutes,
                    status="cancelled",
                    moment=moment,
                    error="main agent stop_focus directive",
                )
                self._record_decision_locked(
                    kind="active",
                    umo=umo,
                    trigger="checkpoint",
                    decision="observe",
                    interest=result.interest_level,
                    reason="main agent stop_focus directive",
                    source=SOURCE_MAIN_AGENT,
                    related_directive_id=str(stop_directive.get("directive_id") or ""),
                    extra={
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_changes": changes,
                        "main_model": "skipped",
                        "send": "none",
                    },
                )
                self._flush_locked()
                return
            if self.settings.max_main_model_calls < 1:
                self._finalize_locked(
                    checkpoint,
                    decision="observe",
                    next_minutes=result.next_check_minutes,
                    status="active",
                    moment=moment,
                    error="main model calls disabled by configuration",
                )
                self._record_decision_locked(
                    kind="active",
                    umo=umo,
                    trigger="checkpoint",
                    decision="observe",
                    interest=result.interest_level,
                    reason="main model calls disabled",
                    extra={
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_changes": changes,
                        "tool_calls": result.tool_calls,
                        "model_calls": result.model_calls,
                        "main_model": "skipped",
                        "send": "none",
                    },
                )
                self._flush_locked()
                return
            can_send, send_reason = self.policy.can_send(
                self.store.state,
                umo=umo,
                group_id=group_id,
                platform_id=platform_id,
                moment=moment,
            )
            if not can_send:
                self._finalize_locked(
                    checkpoint,
                    decision="observe",
                    next_minutes=result.next_check_minutes,
                    status="active",
                    moment=moment,
                    error=send_reason,
                )
                self._record_decision_locked(
                    kind="active",
                    umo=umo,
                    trigger="checkpoint",
                    decision="observe",
                    interest=result.interest_level,
                    reason=f"send blocked: {send_reason}",
                    extra={
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_changes": changes,
                        "tool_calls": result.tool_calls,
                        "model_calls": result.model_calls,
                        "main_model": "skipped",
                        "send": send_reason,
                    },
                )
                self._flush_locked()
                return
            if not self.policy.consume_model_call(
                self.store.state,
                moment,
                "main",
            ):
                self._finalize_locked(
                    checkpoint,
                    decision="observe",
                    next_minutes=result.next_check_minutes,
                    status="active",
                    moment=moment,
                    error="daily model-call budget exhausted",
                )
                self._record_decision_locked(
                    kind="active",
                    umo=umo,
                    trigger="checkpoint",
                    decision="observe",
                    interest=result.interest_level,
                    reason="daily model-call budget exhausted",
                    extra={
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_changes": changes,
                        "tool_calls": result.tool_calls,
                        "model_calls": result.model_calls,
                        "main_model": "skipped",
                        "send": "none",
                    },
                )
                self._flush_locked()
                return
            self._flush_locked()

        persona = await self.context_provider.persona(umo)
        ctx.persona = persona
        reply_system = build_active_reply_system_prompt(ctx, self.settings)
        reply_prompt = build_active_reply_prompt(
            ctx,
            self.settings,
            result.reply_guidance,
        )
        reply_contexts = build_active_reply_contexts(ctx, self.settings)
        reply = await self.gateway.generate(
            purpose="main",
            umo=umo,
            system_prompt=reply_system,
            prompt=reply_prompt,
            contexts=reply_contexts,
            timeout=max(30, self.settings.decision_timeout_seconds),
        )
        message = (
            sanitize_reply(reply.text, self.settings.max_reply_chars)
            if reply.ok
            else None
        )
        if not message:
            now = self.policy.now()
            async with self.store.lock:
                self._finalize_locked(
                    checkpoint,
                    decision="observe",
                    next_minutes=result.next_check_minutes,
                    status="active",
                    moment=now,
                    error=reply.error or "main model returned no usable reply",
                )
                self._record_decision_locked(
                    kind="active",
                    umo=umo,
                    trigger="checkpoint",
                    decision="observe",
                    interest=result.interest_level,
                    reason="main model returned no usable reply",
                    extra={
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_changes": changes,
                        "tool_calls": result.tool_calls,
                        "model_calls": result.model_calls,
                        "main_model": "error",
                        "send": "none",
                    },
                )
                self._flush_locked()
            return

        if self.settings.dry_run:
            now = self.policy.now()
            async with self.store.lock:
                self._finalize_locked(
                    checkpoint,
                    decision="observe",
                    next_minutes=result.next_check_minutes,
                    status="active",
                    moment=now,
                    error="dry-run: message not sent",
                )
                self._record_decision_locked(
                    kind="active",
                    umo=umo,
                    trigger="checkpoint",
                    decision="participate",
                    interest=result.interest_level,
                    reason=result.reason,
                    extra={
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_changes": changes,
                        "tool_calls": result.tool_calls,
                        "model_calls": result.model_calls,
                        "main_model": "called",
                        "send": "dry_run",
                        "reply_length": len(message),
                    },
                )
                self._flush_locked()
            self.logger.info(
                "Dry-run active reply for %s suppressed (%s chars).",
                umo,
                len(message),
            )
            return

        now = self.policy.now()
        async with self.store.lock:
            stop_directive = self.control.stop_directive(
                self.store.state,
                group_id=group_id,
                umo=umo,
                topic_id=str(ctx.topic.get("topic_id") or ""),
            )
            if (
                stop_directive is not None
                and str(checkpoint.get("source") or "") != SOURCE_ADMIN
            ):
                self._finalize_locked(
                    checkpoint,
                    decision="observe",
                    next_minutes=result.next_check_minutes,
                    status="cancelled",
                    moment=now,
                    error="main agent stop_focus directive",
                )
                self._record_decision_locked(
                    kind="active",
                    umo=umo,
                    trigger="checkpoint",
                    decision="observe",
                    interest=result.interest_level,
                    reason="main agent stop_focus directive before send",
                    source=SOURCE_MAIN_AGENT,
                    related_directive_id=str(stop_directive.get("directive_id") or ""),
                    extra={
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_changes": changes,
                        "main_model": "called",
                        "send": "blocked_by_directive",
                    },
                )
                self._flush_locked()
                return
            can_send, send_reason = self.policy.can_send(
                self.store.state,
                umo=umo,
                group_id=group_id,
                platform_id=platform_id,
                moment=now,
            )
            if not can_send:
                self._finalize_locked(
                    checkpoint,
                    decision="observe",
                    next_minutes=result.next_check_minutes,
                    status="active",
                    moment=now,
                    error=send_reason,
                )
                self._record_decision_locked(
                    kind="active",
                    umo=umo,
                    trigger="checkpoint",
                    decision="observe",
                    interest=result.interest_level,
                    reason=f"send blocked after generation: {send_reason}",
                    extra={
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_changes": changes,
                        "tool_calls": result.tool_calls,
                        "model_calls": result.model_calls,
                        "main_model": "called",
                        "send": send_reason,
                    },
                )
                self._flush_locked()
                return

        sent, error = await self.sender.send_text(umo, message)
        now = self.policy.now()
        async with self.store.lock:
            self.policy.record_send(
                self.store.state,
                umo=umo,
                group_id=group_id,
                moment=now,
                text=message,
                result="sent" if sent else f"failed:{clip_text(error, 80)}",
            )
            topic = self._copy_topic_locked(umo, group_id)
            topic["last_activity_at"] = to_iso(now)
            topic["last_observation"] = f"participate: {clip_text(result.reason, 120)}"
            topic["focus_status"] = "watching"
            topic["interest_source"] = "active"
            if result.interest_level:
                topic["interest_level"] = result.interest_level
            if result.topic_summary:
                topic["summary"] = result.topic_summary
            topic["updated_at"] = to_iso(now)
            topic["revision"] = int(topic.get("revision") or 0) + 1
            self.store.state.setdefault("topics", {})[topic["topic_id"]] = topic
            self._finalize_locked(
                checkpoint,
                decision="participate" if sent else "observe",
                next_minutes=result.next_check_minutes,
                status="active",
                moment=now,
                error="" if sent else error,
            )
            self._record_decision_locked(
                kind="active",
                umo=umo,
                trigger="checkpoint",
                decision="participate" if sent else "observe",
                interest=result.interest_level,
                reason=result.reason,
                extra={
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_changes": changes,
                    "main_model": "called",
                    "send": "sent" if sent else "failed",
                    "reply_length": len(message),
                    "send_error": clip_text(error, 120),
                },
            )
            self._flush_locked()
        if sent:
            await self.observe_message(
                umo=umo,
                platform_id=platform_id,
                group_id=group_id,
                sender_id="bot",
                sender_name="bot",
                text=message,
                message_id="",
                created_at=to_iso(now),
                created_ts=timestamp(now),
                is_bot=True,
            )
        self.logger.info(
            "Active reply: checkpoint=%s group=%s sent=%s chars=%s",
            checkpoint_id,
            group_id,
            sent,
            len(message),
        )

    def _apply_active_effects_locked(
        self,
        ctx: ActiveContext,
        result: Any,
        moment: datetime,
    ) -> list[str]:
        changes: list[str] = []
        for effect in result.effects:
            if not isinstance(effect, dict):
                continue
            effect_type = effect.get("type")
            if effect_type == "create_checkpoint":
                try:
                    current_group = str(ctx.checkpoint.get("group_id") or "")
                    requested_group = str(effect.get("group_id") or "")
                    requested_umo = str(effect.get("umo") or "")
                    target_umo = requested_umo or str(ctx.checkpoint.get("umo") or "")
                    target_platform = str(ctx.checkpoint.get("platform_id") or "")
                    target_group = requested_group or current_group
                    if (
                        requested_group
                        and requested_group != current_group
                        and not requested_umo
                    ):
                        target_umo = ""
                    checkpoint = self._create_checkpoint_locked(
                        umo=target_umo,
                        platform_id=target_platform,
                        group_id=target_group,
                        minutes=int(effect.get("minutes") or 60),
                        reason=str(effect.get("reason") or "agent checkpoint"),
                        created_by="agent",
                        dedupe=True,
                        source=SOURCE_ATTENTION_AGENT,
                        actor=Actor(
                            source=SOURCE_ATTENTION_AGENT,
                            session=str(ctx.checkpoint.get("umo") or ""),
                            group_id=current_group,
                            platform_id=target_platform,
                        ),
                    )
                except Exception as exc:
                    self.logger.warning("Agent checkpoint creation rejected: %s", exc)
                    continue
                if checkpoint is not None:
                    changes.append(f"checkpoint:{checkpoint['checkpoint_id']}")
            elif effect_type == "user_policy":
                minutes = effect.get("expires_in_minutes")
                try:
                    minutes = int(minutes) if minutes is not None else 0
                except (TypeError, ValueError):
                    minutes = 0
                if minutes <= 0:
                    minutes = self.settings.max_auto_policy_hours * 60
                minutes = max(1, min(self.settings.max_auto_policy_hours * 60, minutes))
                user_id = str(effect.get("user_id") or "")
                if not user_id:
                    continue
                scope = str(effect.get("scope") or "group_user")
                if scope == "group_user" and not effect.get("group_id"):
                    scope = "user"
                policy_result = self.control.set_policy_locked(
                    self.store.state,
                    Actor(
                        source=SOURCE_ATTENTION_AGENT,
                        session=str(ctx.checkpoint.get("umo") or ""),
                        group_id=str(ctx.checkpoint.get("group_id") or ""),
                        platform_id=str(ctx.checkpoint.get("platform_id") or ""),
                    ),
                    scope=scope,
                    group_id=str(effect.get("group_id") or ""),
                    user_id=user_id,
                    reason=clip_text(effect.get("reason"), 120),
                    expires_at=add_minutes(moment, minutes),
                    source=SOURCE_ATTENTION_AGENT,
                    created_by_session=str(ctx.checkpoint.get("umo") or ""),
                    allow_permanent=False,
                )
                if policy_result.ok:
                    changes.append(f"policy:{policy_result.data.get('policy_id', '')}")
                else:
                    self.logger.info(
                        "Active user-policy update rejected: %s",
                        policy_result.message,
                    )
        return changes

    def _update_topic_from_active_locked(
        self,
        ctx: ActiveContext,
        result: Any,
        moment: datetime,
    ) -> None:
        topic = self._copy_topic_locked(
            str(ctx.checkpoint.get("umo") or ""),
            str(ctx.checkpoint.get("group_id") or ""),
        )
        if result.topic_summary:
            topic["summary"] = result.topic_summary
        if result.interest_level:
            topic["interest_level"] = result.interest_level
        if result.topic_status == "closed" or result.decision == "end_focus":
            topic["focus_status"] = "closed"
        else:
            topic["focus_status"] = "watching"
        topic["interest_source"] = "active"
        topic["last_observation"] = f"{result.decision}: {result.reason}"[:300]
        topic["has_opening"] = result.decision == "participate"
        topic["next_step"] = (
            "已参与，继续观察"
            if result.decision == "participate"
            else "结束关注"
            if result.decision == "end_focus"
            else "等待下一次检查"
        )
        topic["last_activity_at"] = to_iso(moment)
        topic["updated_at"] = to_iso(moment)
        topic["revision"] = int(topic.get("revision") or 0) + 1
        self.store.state.setdefault("topics", {})[topic["topic_id"]] = topic

    async def _finish_without_reply(
        self,
        checkpoint: dict[str, Any],
        *,
        decision: str,
        reason: str,
        next_minutes: int | None,
        status: str = "active",
        extra: dict[str, Any] | None = None,
    ) -> None:
        moment = self.policy.now()
        async with self.store.lock:
            self._finalize_locked(
                checkpoint,
                decision=decision,
                next_minutes=next_minutes,
                status=status,
                moment=moment,
                error=str((extra or {}).get("last_error") or ""),
            )
            self._record_decision_locked(
                kind="active",
                umo=str(checkpoint.get("umo") or ""),
                trigger="checkpoint",
                decision=decision,
                interest="none",
                reason=reason,
                extra={
                    "checkpoint_id": checkpoint.get("checkpoint_id"),
                    **(extra or {}),
                },
            )
            self._flush_locked()
        self.logger.info(
            "Active attention skipped: checkpoint=%s decision=%s reason=%s",
            checkpoint.get("checkpoint_id"),
            decision,
            clip_text(reason, 120),
        )

    def _finalize_locked(
        self,
        checkpoint: dict[str, Any],
        *,
        decision: str,
        next_minutes: int | None,
        status: str,
        moment: datetime,
        error: str = "",
    ) -> None:
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        raw = (self.store.state.get("checkpoints") or {}).get(checkpoint_id)
        if not isinstance(raw, dict):
            return
        scheduled_at = (
            to_iso(self._next_schedule(moment, next_minutes))
            if status == "active"
            else ""
        )
        self.control.finalize_checkpoint_locked(
            self.store.state,
            checkpoint_id,
            status=status,
            scheduled_at=scheduled_at,
            last_result=decision,
            last_error=error,
            moment=moment,
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
        decision: str,
        interest: str,
        reason: str,
        source: str = "",
        related_policy_id: str = "",
        related_directive_id: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": to_iso(self.policy.now()),
            "kind": kind,
            "umo": umo,
            "trigger": trigger,
            "decision": decision,
            "interest": interest,
            "reason": clip_text(reason, self.settings.max_reason_chars),
            "source": source
            or ("passive_agent" if kind == "passive" else "active_agent"),
            "related_policy_id": related_policy_id,
            "related_directive_id": related_directive_id,
            "tool_calls": [],
            "checkpoint_changes": [],
            "main_model": "none",
            "send": "none",
        }
        if extra:
            record.update(extra)
        self.store.record_decision(record)

    async def status_snapshot(self) -> dict[str, Any]:
        """Return a compact status snapshot for the admin command."""

        async with self.store.lock:
            checkpoints = self.store.state.get("checkpoints", {})
            counts: dict[str, int] = {}
            for raw in checkpoints.values():
                if isinstance(raw, dict):
                    status = str(raw.get("status") or "unknown")
                    counts[status] = counts.get(status, 0) + 1
            decisions = [
                item
                for item in self.store.state.get("decisions", [])
                if isinstance(item, dict)
            ][-5:]
            return {
                "started": self._started,
                "passive_mode": self.settings.passive_mode,
                "active_enabled": self.settings.active_enabled,
                "dry_run": self.settings.dry_run,
                "checkpoint_counts": counts,
                "policy_count": len(self.store.state.get("policies") or {}),
                "budget": self.policy.describe_budget(self.store.state),
                "scheduler": self.scheduler.status(),
                "recent_decisions": decisions,
                "store_path": str(self.store.path),
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
            return False, "group_id is required for this policy scope"
        if scope in {"group_user", "user"} and not user_id:
            return False, "user_id is required for this policy scope"
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
