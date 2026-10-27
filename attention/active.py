"""Bounded active attention loop with validated tool actions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .models import ActiveAction, ActiveContext, ActiveResult
from .ports import ModelGateway
from .prompts import (
    build_active_reply_prompt,
    build_active_system_prompt,
    build_active_tool_result_prompt,
    build_active_user_prompt,
    clip_text,
    format_window,
)
from .settings import Settings
from .validation import (
    coerce_int,
    extract_json_object,
    is_known_active_action,
    validate_active_action,
)


class ActiveAttentionRunner:
    """Run one bounded active attention session.

    The runner only produces a result and a list of policy/checkpoint effects.
    The engine validates and applies those effects while holding the store lock,
    which keeps the runner free of persistence concerns.
    """

    def __init__(
        self,
        settings: Settings,
        gateway: ModelGateway,
    ) -> None:
        self.settings = settings
        self.gateway = gateway

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    async def run(
        self,
        ctx: ActiveContext,
        consume_call: Callable[[], Awaitable[bool]],
        window_loader: Callable[[int], Awaitable[list]],
    ) -> ActiveResult:
        """Run the loop until a final action, a quota, or an error.

        Args:
            ctx: Checkpoint, topic, window, persona and policy snapshot.
            consume_call: Async budget check/reservation callback.
            window_loader: Async loader used by the ``read_messages`` tool.

        Returns:
            A validated result. The default result is ``observe`` so a failure
            can never turn into a proactive message.
        """

        result = ActiveResult()
        window = list(ctx.window)
        ctx.window = window
        system_prompt = build_active_system_prompt(ctx, self.settings)
        history: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": build_active_user_prompt(ctx, self.settings),
            }
        ]
        invalid_outputs = 0
        mutations = 0

        for _ in range(self.settings.max_agent_rounds):
            if not await consume_call():
                result.reason = "daily model-call budget exhausted"
                result.interrupted = True
                return result
            response = await self.gateway.generate(
                purpose="decision",
                umo=str(ctx.checkpoint.get("umo") or ""),
                system_prompt=system_prompt,
                contexts=history,
                timeout=self.settings.decision_timeout_seconds,
            )
            result.model_calls += 1
            if not response.ok:
                result.reason = f"decision model error: {response.error}"[:200]
                result.interrupted = True
                return result

            history.append({"role": "assistant", "content": response.text})
            action = validate_active_action(
                extract_json_object(response.text),
                self.settings.max_reason_chars,
            )
            if action is None or not is_known_active_action(action.action):
                invalid_outputs += 1
                if invalid_outputs >= 2:
                    result.reason = "repeated invalid structured output"
                    result.interrupted = True
                    return result
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "上一轮输出不是合法 JSON 或 action 不受支持。"
                            "请只输出一个符合 schema 的 JSON 对象。"
                        ),
                    }
                )
                continue
            invalid_outputs = 0

            result.reason = action.reason or result.reason
            if action.interest_level:
                result.interest_level = action.interest_level
            if action.topic_summary:
                result.topic_summary = action.topic_summary
            if action.topic_status:
                result.topic_status = action.topic_status

            if action.action in {"participate", "request_main_reply"}:
                result.decision = "participate"
                result.reply_guidance = action.reply_guidance or clip_text(
                    action.arguments.get("guidance"), 500
                )
                return result

            if action.action in {"observe", "skip", "end_focus"}:
                result.decision = action.action
                minutes = coerce_int(action.arguments.get("minutes"))
                if minutes is not None:
                    result.next_check_minutes = minutes
                if action.action == "end_focus":
                    result.topic_status = "closed"
                return result

            if result.tool_calls >= self.settings.max_tool_calls:
                result.reason = "tool-call budget exhausted"
                result.decision = "observe"
                result.interrupted = True
                return result

            is_mutation = action.action in {
                "schedule_next_check",
                "defer_check",
                "create_checkpoint",
                "update_user_policy",
            }
            if is_mutation and mutations >= self.settings.max_checkpoint_mutations:
                result.reason = "checkpoint mutation budget exhausted"
                result.decision = "observe"
                return result
            tool_output = await self._execute_tool(
                action,
                ctx,
                result,
                window_loader,
            )
            result.tool_calls += 1
            if tool_output is None:
                # The action could not be executed with the supplied arguments.
                result.reason = "invalid tool arguments"
                result.decision = "observe"
                return result
            if is_mutation:
                mutations += 1
            if isinstance(tool_output, tuple):
                text, refreshed_window = tool_output
                ctx.window = refreshed_window
                window = refreshed_window
                history.append(
                    {
                        "role": "user",
                        "content": build_active_tool_result_prompt(
                            action.action,
                            text,
                        ),
                    }
                )
            else:
                history.append(
                    {
                        "role": "user",
                        "content": build_active_tool_result_prompt(
                            action.action,
                            str(tool_output),
                        ),
                    }
                )
            self._trim_history(history)

        result.decision = "observe"
        result.reason = result.reason or "agent loop reached its round limit"
        result.interrupted = True
        return result

    def _trim_history(self, history: list[dict[str, Any]]) -> None:
        """Keep the oldest prompt and newest tool turns under the budget.

        The initial prompt contains the checkpoint and window and is retained.
        Older assistant/tool-result pairs are dropped when the accumulated
        history grows beyond twice ``context_max_chars``.
        """

        budget = max(2000, self.settings.context_max_chars * 2)

        def total_chars() -> int:
            return sum(len(str(item.get("content") or "")) for item in history)

        while len(history) > 3 and total_chars() > budget:
            history.pop(1)
            if len(history) > 1:
                history.pop(1)

    async def _execute_tool(
        self,
        action: ActiveAction,
        ctx: ActiveContext,
        result: ActiveResult,
        window_loader: Callable[[int], Awaitable[list]],
    ) -> str | tuple[str, list] | None:
        """Execute one read-only or deferred tool action.

        Args:
            action: Validated model action.
            ctx: Active session context.
            result: Mutable result collecting effects and counters.
            window_loader: Loader for refreshed message windows.

        Returns:
            Tool output text, ``(text, refreshed_window)`` for message reads, or
            ``None`` when a mutation quota is exceeded.
        """

        arguments = action.arguments

        if action.action == "read_messages":
            limit = coerce_int(arguments.get("limit"), 20) or 20
            limit = max(1, min(self.settings.context_window_messages, limit))
            window = await window_loader(limit)
            if not window:
                return "没有读到新的消息。"
            return (
                format_window(
                    window,
                    self.settings.context_max_chars,
                    max_messages=limit,
                ),
                window,
            )

        if action.action == "get_topic":
            summary = clip_text(str(ctx.topic.get("summary") or ""), 500)
            return (
                f"话题摘要：{summary or '（无）'}\n"
                f"兴趣：{ctx.topic.get('interest_level') or 'none'}\n"
                f"关注状态：{ctx.topic.get('focus_status') or 'idle'}"
            )

        if action.action in {"schedule_next_check", "defer_check"}:
            minutes = coerce_int(arguments.get("minutes"))
            if minutes is None:
                return "缺少整数参数 minutes。"
            result.next_check_minutes = minutes
            return f"已记录下次检查间隔：{minutes} 分钟（受上下限约束）。"

        if action.action == "create_checkpoint":
            minutes = coerce_int(arguments.get("minutes"), 60) or 60
            reason = clip_text(
                arguments.get("reason") or action.reason or "关注该话题",
                200,
            )
            effect = {
                "type": "create_checkpoint",
                "minutes": minutes,
                "reason": reason,
                "group_id": clip_text(arguments.get("group_id"), 80),
                "umo": clip_text(arguments.get("umo"), 200),
            }
            result.effects.append(effect)
            return "已创建新的检查点请求，将按策略校验后生效。"

        if action.action == "update_user_policy":
            user_id = clip_text(
                arguments.get("user_id") or arguments.get("target"),
                80,
            )
            if not user_id:
                return "缺少 user_id。"
            expires = coerce_int(arguments.get("expires_in_minutes"))
            if expires is not None:
                expires = max(1, min(self.settings.max_auto_policy_hours * 60, expires))
            effect = {
                "type": "user_policy",
                "scope": "user" if not ctx.checkpoint.get("group_id") else "group_user",
                "group_id": str(ctx.checkpoint.get("group_id") or ""),
                "user_id": user_id,
                "expires_in_minutes": expires,
                "reason": clip_text(
                    arguments.get("reason") or action.reason or "策略调整",
                    120,
                ),
            }
            result.effects.append(effect)
            return "已记录用户策略变更请求，将按策略校验后生效。"

        if action.action == "record_interest":
            level = str(
                arguments.get("interest_level") or action.interest_level or ""
            ).lower()
            if level in {"none", "low", "medium", "high"}:
                result.interest_level = level
                ctx.topic["interest_level"] = level
            summary = clip_text(
                arguments.get("summary") or action.topic_summary,
                500,
            )
            if summary:
                result.topic_summary = summary
                ctx.topic["summary"] = summary
            return "已记录兴趣和话题摘要。"

        return None

    def build_reply_prompt(self, ctx: ActiveContext, guidance: str) -> str:
        """Build the final main-model prompt for a proactive reply."""

        return build_active_reply_prompt(ctx, self.settings, guidance)
