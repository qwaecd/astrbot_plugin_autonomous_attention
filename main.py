"""Autonomous Attention plugin entry point and AstrBot adapter wiring."""

# ruff: noqa: I001

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from typing import Any

from astrbot.api import AstrBotConfig
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools

from .astrbot_adapter import (
    AstrBotContextProvider,
    AstrBotMessageSender,
    AstrBotModelGateway,
    build_passive_request,
    classify_wake_source,
)
from .attention import AttentionEngine, Settings, StateStore
from .attention.control import Actor
from .attention.models import DIRECT_PASS_WAKE_SOURCES, SOURCE_MAIN_AGENT
from .attention.timeutil import to_iso
from .attention.validation import clip_text

PLUGIN_NAME = "astrbot_plugin_autonomous_attention"
COMMAND_FILTER_CLASS_NAMES = {"CommandFilter", "CommandGroupFilter"}
DURATION_RE = re.compile(r"^(\d+)\s*(m|min|分钟|h|小时|d|天)?$", re.IGNORECASE)


class AdminOnlyFilter(filter.CustomFilter):
    """Require an AstrBot administrator for the command-group tree."""

    def filter(self, event: AstrMessageEvent, cfg: Any) -> bool:
        return event.is_admin()


HELP_TEXT = """Autonomous Attention 管理指令：
/attention status [UMO|--all] — 查看当前会话/全局状态、兴趣和最近决策
/attention list [UMO|--all] — 列出当前会话/全局检查点
/attention interests [UMO|--all] — 列出当前会话/全局兴趣
/attention add [分钟] [原因] — 为当前群创建扫描检查点
/attention add <UMO> [分钟] [原因] — 为指定群创建检查点
/attention check <检查点ID> — 立即执行一次检查
/attention pause|resume|remove <检查点ID> [UMO] — 暂停/恢复/取消
/attention ignore <用户ID或@成员> [时长] [原因] — 设置忽略策略
/attention unignore <用户ID或@成员> — 解除当前群忽略策略
/attention policies — 列出忽略策略
/attention budget [UMO|--all] — 查看当天预算
/attention dryrun on|off — 切换 dry-run
/attention help — 显示本帮助
时长示例：30m、2h、1d、永久；省略时默认 60 分钟。"""


class AutonomousAttentionPlugin(Star):
    """Wire the platform-independent attention core into AstrBot."""

    _wake_prefixes: tuple[str, ...] = ()
    _friend_needs_wake_prefix = False

    def __init__(
        self,
        context: Context,
        config: AstrBotConfig | None = None,
    ) -> None:
        super().__init__(context)
        self.config: Any = config if config is not None else {}
        self.engine: AttentionEngine | None = None
        self.gateway: AstrBotModelGateway | None = None
        self.context_provider: AstrBotContextProvider | None = None
        self.store: StateStore | None = None
        self._ready = False
        self._fallback_timezone = str(
            self.context.get_config().get("timezone") or "",
        )
        self._wake_prefixes: tuple[str, ...] = ()
        self._friend_needs_wake_prefix = False
        self._refresh_wake_metadata()
        self.settings = Settings.from_mapping(
            self.config,
            self._fallback_timezone,
        )
        self._apply_log_level()

    def _refresh_wake_metadata(self) -> None:
        """Cache the AstrBot wake settings used to classify request sources."""

        try:
            config = self.context.get_config()
        except Exception:
            config = {}
        prefixes = config.get("wake_prefix") or []
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        self._wake_prefixes = tuple(str(item) for item in prefixes if str(item).strip())
        platform_settings = config.get("platform_settings") or {}
        self._friend_needs_wake_prefix = bool(
            platform_settings.get("friend_message_needs_wake_prefix", False)
            if isinstance(platform_settings, dict)
            else False
        )

    def _apply_log_level(self) -> None:
        try:
            logger.setLevel(self.settings.log_level)
        except Exception:
            pass

    async def initialize(self) -> None:
        """Create the core and start the checkpoint scheduler."""

        try:
            data_dir = StarTools.get_data_dir(PLUGIN_NAME)
            self.store = StateStore(
                data_dir / "attention_state.json",
                max_decisions=self.settings.max_decision_log,
            )
            self.gateway = AstrBotModelGateway(
                self.context,
                self.settings,
            )
            self.context_provider = AstrBotContextProvider(
                self.context,
                self.settings,
            )
            self.engine = AttentionEngine(
                settings=self.settings,
                store=self.store,
                gateway=self.gateway,
                context_provider=self.context_provider,
                sender=AstrBotMessageSender(self.context),
            )
            await self.engine.start()
            self._ready = True
            logger.info(
                "Autonomous Attention initialized (passive=%s, active=%s).",
                self.settings.passive_mode,
                self.settings.active_enabled,
            )
        except Exception:
            self._ready = False
            logger.exception("Autonomous Attention failed to initialize.")
            if self.engine is not None:
                try:
                    await self.engine.stop()
                except Exception:
                    logger.exception("Attention cleanup after failure failed.")

    async def terminate(self) -> None:
        """Stop every background task and flush state."""

        self._ready = False
        if self.engine is not None:
            try:
                await self.engine.stop()
            except Exception:
                logger.exception("Autonomous Attention shutdown failed.")
        logger.info("Autonomous Attention terminated.")

    def _reload_settings(self) -> None:
        self._refresh_wake_metadata()
        self.settings = Settings.from_mapping(
            self.config,
            self._fallback_timezone,
        )
        self._apply_log_level()
        if self.store is not None:
            self.store.max_decisions = self.settings.max_decision_log
        if self.gateway is not None:
            self.gateway.update_settings(self.settings)
        if self.context_provider is not None:
            self.context_provider.update_settings(self.settings)
        if self.engine is not None:
            self.engine.update_settings(self.settings)

    def _set_config_value(self, section: str, key: str, value: Any) -> None:
        """Write one grouped config value and persist it when possible."""

        if not isinstance(self.config, dict):
            return
        section_data = self.config.get(section)
        if not isinstance(section_data, dict):
            section_data = {}
            self.config[section] = section_data
        section_data[key] = value
        save_config = getattr(self.config, "save_config", None)
        if callable(save_config):
            save_config()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    @filter.on_waiting_llm_request(priority=-20000)
    async def mark_non_user_llm_request(self, event: AstrMessageEvent) -> None:
        """Mark requests prepared by a plugin before the main-agent build.

        AstrBot stores a normal user's request in ``provider_request`` while
        building the main agent, after this waiting hook has already run. A
        plugin-provided request is stored before the hook, so this is the one
        lifecycle point where the extra still distinguishes the two sources.
        """

        if event.get_extra("provider_request") is not None:
            event.set_extra("_aa_skip_passive", True)

    @filter.event_message_type(filter.EventMessageType.ALL, priority=20000)
    async def observe_and_guard(self, event: AstrMessageEvent) -> None:
        """Record group context and enforce hard ignore policies.

        The handler is registered for all message types. Group observation is
        what makes an active checkpoint able to read a recent window even when
        AstrBot's own group-history feature is disabled. Hard policies are
        applied before command handlers and before any main-model request.
        """

        if not self._ready or self.engine is None:
            return
        group_id = event.get_group_id()
        platform_id = event.get_platform_id()
        policy = await self.engine.check_hard_policy(
            group_id=group_id,
            user_id=event.get_sender_id(),
            umo=event.unified_msg_origin,
            platform_id=platform_id,
            is_admin=event.is_admin(),
        )
        if policy is not None:
            if event.is_at_or_wake_command:
                event.stop_event()
                logger.info(
                    "Hard ignore policy blocked woken message: umo=%s user=%s "
                    "scope=%s reason=%s",
                    event.unified_msg_origin,
                    event.get_sender_id(),
                    policy.get("scope"),
                    clip_text(policy.get("reason"), 80),
                )
            return

        if not group_id:
            return
        if self._is_command_event(event):
            return
        text = event.message_str or event.get_message_outline()
        if not text:
            return
        created = datetime.fromtimestamp(
            event.created_at or time.time(),
            tz=self.settings.tz,
        )
        await self.engine.observe_message(
            umo=event.unified_msg_origin,
            platform_id=platform_id,
            group_id=group_id,
            sender_id=event.get_sender_id(),
            sender_name=event.get_sender_name(),
            text=text,
            message_id=str(
                getattr(event.message_obj, "message_id", "") or "",
            ),
            created_at=created.isoformat(timespec="seconds"),
            created_ts=created.timestamp(),
        )

    @filter.on_llm_request(priority=-20000)
    async def passive_gate(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest,
    ) -> None:
        """Decide whether a woken message may reach the main model.

        This hook runs last so the request already contains the same persona,
        history and system prompt that the main model would receive.
        """

        umo = str(getattr(event, "unified_msg_origin", "") or "")
        if not self._ready or self.engine is None:
            logger.info(
                "Passive attention gate: umo=%s verdict=skip code=not_ready", umo
            )
            return
        wake_source = classify_wake_source(event, self._wake_prefixes)
        if (
            event.get_platform_name() == "cron"
            or event.get_extra("cron_payload") is not None
        ):
            logger.info(
                "Passive attention gate: umo=%s wake_source=%s verdict=skip "
                "code=cron_event",
                umo,
                wake_source,
            )
            return
        if event.get_extra("_aa_skip_passive"):
            logger.info(
                "Passive attention gate: umo=%s wake_source=%s verdict=skip "
                "code=plugin_request",
                umo,
                wake_source,
            )
            return
        if not event.is_at_or_wake_command:
            logger.info(
                "Passive attention gate: umo=%s wake_source=%s verdict=skip "
                "code=not_woken",
                umo,
                wake_source,
            )
            return
        if event.get_extra("_aa_passive_handled"):
            logger.info(
                "Passive attention gate: umo=%s wake_source=%s verdict=skip "
                "code=already_handled",
                umo,
                wake_source,
            )
            return
        event.set_extra("_aa_passive_handled", True)
        try:
            request_obj = build_passive_request(
                event,
                request,
                is_command=self._is_command_event(event),
                timestamp=to_iso(self.engine.policy.now()),
                wake_prefixes=self._wake_prefixes,
                friend_needs_wake_prefix=self._friend_needs_wake_prefix,
            )
            outcome = await self.engine.handle_passive(request_obj)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Passive attention gate failed.")
            if wake_source not in DIRECT_PASS_WAKE_SOURCES:
                # Judged requests (mentions, replies, autonomous) must not
                # fail open when the gate itself raises.
                try:
                    event.stop_event()
                    event.set_extra("_aa_blocked", True)
                except Exception:
                    pass
            return
        if outcome.block:
            event.stop_event()
            event.set_extra("_aa_blocked", True)
            logger.info(
                "Passive attention blocked main model: umo=%s user=%s "
                "decision=%s code=%s reason=%s",
                event.unified_msg_origin,
                event.get_sender_id(),
                outcome.decision,
                outcome.code or "-",
                clip_text(outcome.reason, 100),
            )
        elif outcome.reservation_id:
            event.set_extra("_aa_passive_reservation", outcome.reservation_id)
            event.set_extra("_aa_passive_umo", request_obj.umo)

    @filter.on_llm_response()
    async def remember_bot_reply(
        self,
        event: AstrMessageEvent,
        response: Any,
    ) -> None:
        """Add the bot's own reply to the active sliding window."""

        if not self._ready or self.engine is None:
            return
        reservation_id = str(event.get_extra("_aa_passive_reservation") or "")
        if reservation_id:
            try:
                await self.engine.complete_passive_generation(
                    str(event.get_extra("_aa_passive_umo") or event.unified_msg_origin),
                    reservation_id,
                    str(getattr(response, "completion_text", "") or ""),
                )
            except Exception:
                logger.exception("Failed to complete passive participation.")
        if not event.get_group_id():
            return
        if not self.settings.active_enabled:
            return
        text = str(getattr(response, "completion_text", "") or "").strip()
        if not text:
            chain = getattr(response, "result_chain", None)
            if chain is not None:
                try:
                    text = str(chain.get_plain_text() or "").strip()
                except Exception:
                    text = ""
        if not text:
            return
        moment = datetime.now(self.settings.tz)
        await self.engine.observe_message(
            umo=event.unified_msg_origin,
            platform_id=event.get_platform_id(),
            group_id=event.get_group_id(),
            sender_id=event.get_self_id() or "bot",
            sender_name="bot",
            text=text,
            message_id="",
            created_at=moment.isoformat(timespec="seconds"),
            created_ts=moment.timestamp(),
            is_bot=True,
        )

    # ------------------------------------------------------------------
    # Main-agent attention tools
    # ------------------------------------------------------------------

    def _tool_actor(self, event: AstrMessageEvent) -> Actor:
        return Actor(
            source=SOURCE_MAIN_AGENT,
            user_id=str(event.get_sender_id() or ""),
            session=str(event.unified_msg_origin),
            group_id=str(event.get_group_id() or ""),
            platform_id=str(event.get_platform_id() or ""),
            is_admin=bool(event.is_admin()),
        )

    def _tools_ready(self, event: AstrMessageEvent) -> str | None:
        if not self._ready or self.engine is None:
            return json.dumps(
                {"ok": False, "code": "not_ready", "message": "插件尚未初始化。"},
                ensure_ascii=False,
            )
        if not self.settings.main_agent_tools_enabled:
            return json.dumps(
                {
                    "ok": False,
                    "code": "tools_disabled",
                    "message": "主模型注意力工具已关闭。",
                },
                ensure_ascii=False,
            )
        return None

    @staticmethod
    def _tool_failure(code: str, message: str) -> str:
        return json.dumps(
            {"ok": False, "code": code, "message": str(message)},
            ensure_ascii=False,
        )

    def _group_tool_blocked(self, event: AstrMessageEvent) -> str | None:
        if self.engine is None:
            return None
        group_id = str(event.get_group_id() or "")
        if group_id and not self.engine.policy.group_allowed(
            group_id,
            str(event.unified_msg_origin),
            str(event.get_platform_id() or ""),
        ):
            return json.dumps(
                {
                    "ok": False,
                    "code": "group_not_allowed",
                    "message": "当前群不在 group_allowlist 中，不能进行注意力控制。",
                },
                ensure_ascii=False,
            )
        return None

    @filter.llm_tool(name="attention_query")
    async def attention_query(
        self,
        event: AstrMessageEvent,
        scope: str = "session",
    ) -> str:
        """查询当前会话范围内的注意力状态摘要。

        后台注意力 Agent 是执行单元。仅在用户提及相关状态、需要解释 Bot 行为或需要基于注意力状态作决定时调用。只返回当前会话的紧凑摘要，`all` 表示当前会话内全部类别，不返回其他会话、聊天历史或完整状态文件。

        Args:
            scope(string): session、interests、checkpoints、directives、recent 或 all。
        """
        error = self._tools_ready(event)
        if error:
            return error
        try:
            result = await self.engine.control.query(
                self._tool_actor(event),
                scope=scope,
            )
        except Exception as exc:
            return self._tool_failure("query_failed", str(exc))
        return result.to_json()

    @filter.llm_tool(name="attention_set_intent")
    async def attention_set_intent(
        self,
        event: AstrMessageEvent,
        intent_type: str = "",
        target: str = "",
        reason: str = "",
        expires_in_minutes: int = 0,
        minutes: int = 0,
        interest_id: str = "",
    ) -> str:
        """表达主 Agent 的关注或停止关注意图。

        关注意图只会提高本会话兴趣的优先级，不能绕过证据和参与许可。注意力 Agent 在后台执行，但不能覆盖管理员的硬性策略。不要假设意图已经存在，必须以返回结果为准。

        Args:
            intent_type(string): focus_group、focus_topic、ignore_user、stop_focus 或 schedule_check。
            target(string): 话题关键字或用户 ID；为空时使用当前会话。
            reason(string): 简短原因。
            expires_in_minutes(number): 意图有效分钟数；0 表示使用默认时限。
            minutes(number): schedule_check 的延迟分钟数。
            interest_id(string): schedule_check 可选关联的本会话兴趣 ID。
        """
        error = self._tools_ready(event) or self._group_tool_blocked(event)
        if error:
            return error
        try:
            result = await self.engine.control.set_intent(
                self._tool_actor(event),
                intent_type=intent_type,
                target=target,
                reason=reason,
                expires_in_minutes=expires_in_minutes or None,
                minutes=minutes or None,
                interest_id=interest_id,
            )
        except Exception as exc:
            return self._tool_failure("set_intent_failed", str(exc))
        return result.to_json()

    @filter.llm_tool(name="attention_ignore_user")
    async def attention_ignore_user(
        self,
        event: AstrMessageEvent,
        target_user: str = "",
        reason: str = "",
        expires_in_minutes: int = 0,
        scope: str = "group_user",
    ) -> str:
        """创建用户忽略策略。默认有时限；普通成员不能创建永久策略。

        结果以返回 JSON 为准。不要仅因为发起了工具调用就假设策略已生效。

        Args:
            target_user(string): 要忽略的用户 ID。
            reason(string): 简短原因。
            expires_in_minutes(number): 有效分钟数；0 表示使用默认时限。
            scope(string): group_user（默认，当前群）或 user（全局，仅管理员）。
        """
        error = self._tools_ready(event) or self._group_tool_blocked(event)
        if error:
            return error
        try:
            result = await self.engine.control.ignore_user(
                self._tool_actor(event),
                target_user=target_user,
                reason=reason,
                expires_in_minutes=expires_in_minutes or None,
                scope=scope,
            )
        except Exception as exc:
            return self._tool_failure("ignore_user_failed", str(exc))
        return result.to_json()

    @filter.llm_tool(name="attention_clear_user_policy")
    async def attention_clear_user_policy(
        self,
        event: AstrMessageEvent,
        target_user: str = "",
        policy_id: str = "",
    ) -> str:
        """解除用户忽略策略。不能解除管理员策略。

        结果以返回 JSON 为准。

        Args:
            target_user(string): 目标用户 ID；与 policy_id 二选一。
            policy_id(string): 可选策略 ID。
        """
        error = self._tools_ready(event) or self._group_tool_blocked(event)
        if error:
            return error
        try:
            result = await self.engine.control.clear_user_policy(
                self._tool_actor(event),
                target_user=target_user,
                policy_id=policy_id,
            )
        except Exception as exc:
            return self._tool_failure("clear_policy_failed", str(exc))
        return result.to_json()

    @filter.llm_tool(name="attention_focus_topic")
    async def attention_focus_topic(
        self,
        event: AstrMessageEvent,
        topic: str = "",
        reason: str = "",
        expires_in_minutes: int = 0,
    ) -> str:
        """设置主 Agent 的话题关注意图。

        注意力 Agent 会读取该意图并在检查点中执行，但不能修改它。

        Args:
            topic(string): 话题关键字或话题 ID。
            reason(string): 简短原因。
            expires_in_minutes(number): 有效分钟数；0 表示使用默认时限。
        """
        error = self._tools_ready(event) or self._group_tool_blocked(event)
        if error:
            return error
        try:
            result = await self.engine.control.focus_topic(
                self._tool_actor(event),
                topic=topic,
                reason=reason,
                expires_in_minutes=expires_in_minutes or None,
            )
        except Exception as exc:
            return self._tool_failure("focus_topic_failed", str(exc))
        return result.to_json()

    @filter.llm_tool(name="attention_stop_focus")
    async def attention_stop_focus(
        self,
        event: AstrMessageEvent,
        target: str = "",
    ) -> str:
        """停止某个关注目标，并撤销主 Agent 相应的关注/检查点意图。

        普通成员只能停止当前会话中由自己发起的意图；管理员可以停止更多目标，但仍不能覆盖管理员硬策略。

        Args:
            target(string): 群号、UMO、话题或 all；为空时使用当前会话。
        """
        error = self._tools_ready(event) or self._group_tool_blocked(event)
        if error:
            return error
        try:
            result = await self.engine.control.stop_focus(
                self._tool_actor(event),
                target=target,
            )
        except Exception as exc:
            return self._tool_failure("stop_focus_failed", str(exc))
        return result.to_json()

    # ------------------------------------------------------------------
    # Admin commands
    # ------------------------------------------------------------------

    @filter.command_group("attention", alias={"aa", "注意力"})
    @filter.custom_filter(AdminOnlyFilter)
    def attention_group(self) -> None:
        """Autonomous Attention 管理指令组。"""
        pass

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("help", alias={"帮助"})
    async def command_help(self, event: AstrMessageEvent):
        """显示帮助。"""
        yield event.plain_result(HELP_TEXT)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("status", alias={"状态"})
    async def command_status(self, event: AstrMessageEvent):
        """查看当前会话或全局的运行状态。"""
        if not self._ready or self.engine is None:
            yield event.plain_result(
                "Autonomous Attention 尚未初始化成功，请检查日志。"
            )
            return
        args = self._command_args(event)
        global_scope = any(item == "--all" for item in args)
        target_umo = ""
        if not global_scope:
            target_umo = next(
                (item for item in args if ":" in item),
                event.unified_msg_origin,
            )
        snapshot = await self.engine.status_snapshot(target_umo)
        counts = snapshot.get("checkpoint_counts") or {}
        count_text = "、".join(f"{key}={value}" for key, value in counts.items())
        scheduler = snapshot.get("scheduler") or {}
        lines = [
            "Autonomous Attention 状态",
            f"范围：{snapshot.get('scope')}",
            f"被动模式：{snapshot.get('passive_mode')}",
            f"主动检查：{'开启' if snapshot.get('active_enabled') else '关闭'}",
            f"dry-run：{'开启' if snapshot.get('dry_run') else '关闭'}",
            f"检查点：{count_text or '无'}",
            f"兴趣：{snapshot.get('interest_count', 0)} 个；"
            f"待用参与机会：{snapshot.get('opportunity_count', 0)} 个",
            f"忽略策略：{snapshot.get('policy_count', 0)} 条",
        ]
        directives = snapshot.get("directives") or []
        lines.append(f"活跃指令：{len(directives)} 条")
        for item in directives[:5]:
            lines.append(
                f"- {item.get('directive_id') or '-'} | {item.get('type') or '-'} | "
                f"目标 {item.get('target') or '-'} | "
                f"来源 {item.get('source') or '-'} | "
                f"到期 {self._short_time(item.get('expires_at')) or '永久'} | "
                f"{clip_text(item.get('reason'), 40)}"
            )
        lines.extend(
            [
                f"预算：{snapshot.get('budget')}",
                f"调度器：{'运行中' if scheduler.get('running') else '已停止'}，"
                f"活动任务 {scheduler.get('active_tasks', 0)} 个",
            ]
        )
        if snapshot.get("migrated"):
            lines.append("状态已从 v1 迁移到 v2。")
        metrics = snapshot.get("metrics") or {}
        if metrics:
            lines.append(
                "指标：收到 {received} / 语义判断 {semantic} / 建议 {suggestions} / "
                "获批 {approved} / 被动放行 {passive} / 主动发送 {sent} / "
                "显式放行 {explicit}".format(
                    received=metrics.get("received_messages", 0),
                    semantic=metrics.get("semantic_decisions", 0),
                    suggestions=metrics.get("suggestions", 0),
                    approved=metrics.get("approved_reservations", 0),
                    passive=metrics.get("passive_passes", 0),
                    sent=metrics.get("active_sends", 0),
                    explicit=metrics.get("explicit_passes", 0),
                )
            )
            rejection = metrics.get("rejection_counts") or {}
            if rejection:
                top = sorted(rejection.items(), key=lambda item: -item[1])[:3]
                lines.append(
                    "拒绝原因：" + "、".join(f"{key}={value}" for key, value in top)
                )
        recent = snapshot.get("recent_decisions") or []
        if recent:
            lines.append("最近决策：")
            for item in recent[-3:]:
                lines.append(
                    f"- {item.get('timestamp', '')} [{item.get('kind')}] "
                    f"{item.get('decision')} / {item.get('code') or '-'} "
                    f"{clip_text(item.get('reason'), 60)}"
                )
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("list", alias={"列表"})
    async def command_list(self, event: AstrMessageEvent):
        """列出当前会话或全局的检查点。"""
        if not self._ready or self.engine is None:
            yield event.plain_result("插件尚未初始化。")
            return
        args = self._command_args(event)
        global_scope = any(item == "--all" for item in args)
        target_umo = ""
        if not global_scope:
            target_umo = next(
                (item for item in args if ":" in item),
                event.unified_msg_origin,
            )
        items = await self.engine.list_checkpoints(target_umo)
        if not items:
            yield event.plain_result(
                "当前没有检查点。使用 /attention add 创建，"
                "或让主 Agent 调用 schedule_check。"
            )
            return
        lines = [f"检查点列表（{target_umo or '全部会话'}）："]
        for item in items[:30]:
            lines.append(
                f"- {item.get('checkpoint_id')} | {item.get('status')} | "
                f"{self._short_time(item.get('scheduled_at'))} | "
                f"{item.get('kind') or 'session_scan'} | "
                f"兴趣 {item.get('interest_id') or '-'} | "
                f"{clip_text(item.get('reason'), 50)}"
            )
            if item.get("last_result"):
                lines.append(
                    f"  上次：{item.get('last_result')} "
                    f"{clip_text(item.get('last_error'), 50)}"
                )
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("interests", alias={"兴趣"})
    async def command_interests(self, event: AstrMessageEvent):
        """列出当前会话或全局的兴趣。"""
        if not self._ready or self.engine is None:
            yield event.plain_result("插件尚未初始化。")
            return
        args = self._command_args(event)
        global_scope = any(item == "--all" for item in args)
        target_umo = ""
        if not global_scope:
            target_umo = next(
                (item for item in args if ":" in item),
                event.unified_msg_origin,
            )
        items = await self.engine.list_interests(target_umo)
        if not items:
            yield event.plain_result("当前没有兴趣。")
            return
        lines = [f"兴趣列表（{target_umo or '全部会话'}）："]
        for item in items[:30]:
            lines.append(
                f"- {item.get('interest_id')} | {item.get('status')} | "
                f"{clip_text(item.get('summary'), 60)} | 等待 "
                f"{clip_text(item.get('wait_for'), 40) or '-'} | 剩余检查 "
                f"{item.get('remaining_checks')} | 无进展 "
                f"{item.get('no_progress_count')} | 到期 "
                f"{self._short_time(item.get('expires_at')) or '-'}"
            )
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("add", alias={"添加", "create"})
    async def command_add(self, event: AstrMessageEvent):
        """为当前群或指定 UMO 创建检查点。"""
        if not self._ready or self.engine is None:
            yield event.plain_result("插件尚未初始化。")
            return
        args = self._command_args(event)
        umo = ""
        minutes = self.settings.default_interval_minutes
        if args and not args[0].isdigit() and args[0].count(":") >= 2:
            umo = args.pop(0)
        if args and args[0].isdigit():
            minutes = int(args.pop(0))
        elif not umo and not event.get_group_id():
            yield event.plain_result(
                "私聊中请提供目标群 UMO，例如：/attention add "
                "platform_id:GroupMessage:123456 60",
            )
            return
        reason = " ".join(args).strip() or "管理员手动创建"
        target_umo = umo or (event.unified_msg_origin if event.get_group_id() else "")
        try:
            checkpoint = await self.engine.create_checkpoint(
                umo=target_umo,
                platform_id="" if umo else event.get_platform_id(),
                group_id="" if umo else event.get_group_id(),
                minutes=minutes,
                reason=reason,
                created_by="admin",
            )
        except Exception as exc:
            yield event.plain_result(f"创建检查点失败：{exc}")
            return
        yield event.plain_result(
            f"已创建检查点 {checkpoint.get('checkpoint_id')}\n"
            f"目标：{checkpoint.get('umo')}\n"
            f"计划时间：{checkpoint.get('scheduled_at')}\n"
            f"原因：{checkpoint.get('reason') or '-'}",
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("check", alias={"检查", "run"})
    async def command_check(self, event: AstrMessageEvent):
        """立即执行一个检查点。"""
        if not self._ready or self.engine is None:
            yield event.plain_result("插件尚未初始化。")
            return
        args = self._command_args(event)
        if not args:
            yield event.plain_result("用法：/attention check <检查点ID> [UMO]")
            return
        target_umo = next(
            (item for item in args[1:] if ":" in item),
            event.unified_msg_origin,
        )
        ok, message = await self.engine.trigger_checkpoint(
            args[0],
            umo=target_umo,
        )
        yield event.plain_result(
            f"已触发检查：{args[0]}" if ok else f"无法触发：{message}"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("pause", alias={"暂停"})
    async def command_pause(self, event: AstrMessageEvent):
        """暂停一个检查点。"""
        async for result in self._set_status_command(event, "paused", "已暂停"):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("resume", alias={"恢复"})
    async def command_resume(self, event: AstrMessageEvent):
        """恢复一个检查点。"""
        async for result in self._set_status_command(event, "active", "已恢复"):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("remove", alias={"删除", "cancel"})
    async def command_remove(self, event: AstrMessageEvent):
        """取消一个检查点。"""
        async for result in self._set_status_command(
            event,
            "cancelled",
            "已取消",
        ):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("ignore", alias={"忽略"})
    async def command_ignore(self, event: AstrMessageEvent):
        """为用户添加忽略策略。"""
        if not self._ready or self.engine is None:
            yield event.plain_result("插件尚未初始化。")
            return
        args = self._command_args(event)
        user_id, rest = self._extract_user(event, args)
        if not user_id:
            yield event.plain_result(
                "请提供用户 ID 或 @ 成员：/attention ignore <用户> [时长] [原因]",
            )
            return
        minutes, reason = self._parse_duration(rest, default_minutes=60)
        scope = "group_user" if event.get_group_id() else "user"
        ok, detail = await self.engine.set_policy(
            scope=scope,
            group_id=event.get_group_id() if scope == "group_user" else "",
            user_id=user_id,
            reason=reason or "管理员手动忽略",
            expires_minutes=minutes,
            created_by="admin",
        )
        if not ok:
            yield event.plain_result(f"设置失败：{detail}")
            return
        duration = "永久" if minutes is None else f"{minutes} 分钟"
        yield event.plain_result(
            f"已忽略用户 {user_id}（{duration}）。策略：{detail}",
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("unignore", alias={"解除忽略"})
    async def command_unignore(self, event: AstrMessageEvent):
        """解除当前群或全局用户忽略策略。"""
        if not self._ready or self.engine is None:
            yield event.plain_result("插件尚未初始化。")
            return
        args = self._command_args(event)
        user_id, _ = self._extract_user(event, args)
        if not user_id:
            yield event.plain_result("请提供用户 ID 或 @ 成员。")
            return
        removed = []
        if event.get_group_id():
            ok, detail = await self.engine.remove_policy(
                scope="group_user",
                group_id=event.get_group_id(),
                user_id=user_id,
            )
            if ok:
                removed.append(detail)
        ok, detail = await self.engine.remove_policy(
            scope="user",
            group_id="",
            user_id=user_id,
        )
        if ok:
            removed.append(detail)
        if not removed:
            yield event.plain_result(f"没有找到用户 {user_id} 的忽略策略。")
            return
        yield event.plain_result(
            f"已解除用户 {user_id} 的忽略策略：{', '.join(removed)}",
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("policies", alias={"策略"})
    async def command_policies(self, event: AstrMessageEvent):
        """列出忽略策略。"""
        if not self._ready or self.engine is None:
            yield event.plain_result("插件尚未初始化。")
            return
        policies = await self.engine.list_policies()
        if not policies:
            yield event.plain_result("当前没有忽略策略。")
            return
        lines = ["忽略策略："]
        for policy in policies[:30]:
            target = policy.get("user_id") or policy.get("group_id") or "全局"
            lines.append(
                f"- {policy.get('scope')} {target} | 来源 "
                f"{policy.get('source') or '-'} | 到期 "
                f"{self._short_time(policy.get('expires_at')) or '永久'} | "
                f"{clip_text(policy.get('reason'), 50)}"
            )
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("budget", alias={"预算"})
    async def command_budget(self, event: AstrMessageEvent):
        """查看当天预算。"""
        if not self._ready or self.engine is None:
            yield event.plain_result("插件尚未初始化。")
            return
        args = self._command_args(event)
        global_scope = any(item == "--all" for item in args)
        target_umo = ""
        if not global_scope:
            target_umo = next(
                (item for item in args if ":" in item),
                event.unified_msg_origin,
            )
        snapshot = await self.engine.status_snapshot(target_umo)
        yield event.plain_result(str(snapshot.get("budget") or "无预算信息"))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @attention_group.command("dryrun", alias={"模拟"})
    async def command_dryrun(self, event: AstrMessageEvent):
        """切换 dry-run 模式。"""
        args = self._command_args(event)
        if not args or args[0].lower() not in {
            "on",
            "off",
            "true",
            "false",
            "开",
            "关",
        }:
            yield event.plain_result("用法：/attention dryrun on|off")
            return
        enabled = args[0].lower() in {"on", "true", "开"}
        try:
            self._set_config_value("general", "dry_run", enabled)
        except Exception:
            logger.exception("Failed to persist dry_run setting.")
        self._reload_settings()
        yield event.plain_result(
            f"dry-run 已{'开启' if enabled else '关闭'}。"
            "开启时不会拦截主模型，也不会真正发送主动消息。"
        )

    async def _set_status_command(
        self,
        event: AstrMessageEvent,
        status: str,
        label: str,
    ):
        if not self._ready or self.engine is None:
            yield event.plain_result("插件尚未初始化。")
            return
        args = self._command_args(event)
        if not args:
            yield event.plain_result("请提供检查点 ID。")
            return
        target_umo = next(
            (item for item in args[1:] if ":" in item),
            event.unified_msg_origin,
        )
        ok, detail = await self.engine.set_checkpoint_status(
            args[0],
            status,
            umo=target_umo,
        )
        if not ok:
            yield event.plain_result(f"操作失败：{detail}")
            return
        yield event.plain_result(
            f"{label}检查点 {args[0]}（UMO：{target_umo}，状态：{detail}）。"
        )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _command_args(event: AstrMessageEvent) -> list[str]:
        raw = re.sub(r"\s+", " ", event.message_str.strip())
        parts = raw.split(" ")
        return parts[2:] if len(parts) > 2 else []

    @staticmethod
    def _is_command_event(event: AstrMessageEvent) -> bool:
        handlers = event.get_extra("activated_handlers", []) or []
        for handler in handlers:
            for event_filter in getattr(handler, "event_filters", []) or []:
                if type(event_filter).__name__ in COMMAND_FILTER_CLASS_NAMES:
                    return True
        return False

    @staticmethod
    def _extract_user(
        event: AstrMessageEvent,
        args: list[str],
    ) -> tuple[str, list[str]]:
        remaining = list(args)
        for component in event.get_messages():
            if (
                isinstance(component, At)
                and str(component.qq) != str(event.get_self_id())
                and str(component.qq).lower() != "all"
            ):
                return str(component.qq), remaining
        if remaining and re.fullmatch(r"\d+", remaining[0]):
            return remaining.pop(0), remaining
        return "", remaining

    @staticmethod
    def _parse_duration(
        args: list[str],
        default_minutes: int = 60,
    ) -> tuple[int | None, str]:
        if not args:
            return default_minutes, ""
        first = args[0].strip()
        if first in {"永久", "permanent", "forever"}:
            return None, " ".join(args[1:]).strip()
        match = DURATION_RE.fullmatch(first)
        if not match:
            return default_minutes, " ".join(args).strip()
        amount = int(match.group(1))
        unit = (match.group(2) or "m").lower()
        if amount <= 0:
            return None, " ".join(args[1:]).strip()
        if unit in {"h", "小时"}:
            amount *= 60
        elif unit in {"d", "天"}:
            amount *= 1440
        return amount, " ".join(args[1:]).strip()

    @staticmethod
    def _short_time(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return text.replace("T", " ")[:19]
