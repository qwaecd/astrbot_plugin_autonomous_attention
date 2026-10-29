"""Bounded active attention loop with validated tool actions.

The runner only produces a suggestion plus policy/checkpoint effects. The engine
validates and applies them under the store lock, and the control layer decides
whether participation is approved.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .models import ActiveAction, ActiveContext, ActiveResult
from .ports import ModelGateway
from .prompts import (
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
    """Run one bounded active attention session."""

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
            ctx: Checkpoint, session interests, window, persona and policies.
            consume_call: Async global gateway budget callback.
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
            assessed = action.arguments.get("evaluated_interest_ids")
            if isinstance(assessed, list):
                visible = {item.get("interest_id") for item in ctx.interests}
                result.evaluated_interest_ids = [
                    str(item)
                    for item in assessed
                    if isinstance(item, str) and item in visible
                ]

            result.reason = action.reason or result.reason
            if action.interest_level:
                result.interest_level = action.interest_level
            if action.interest_update:
                result.interest_update = action.interest_update

            if action.action in {"participate", "request_main_reply"}:
                result.decision = "participate"
                result.opportunity = dict(action.opportunity or {})
                result.opportunity.setdefault(
                    "interest_id",
                    str(
                        result.opportunity.get("interest_id")
                        or action.arguments.get("interest_id")
                        or (ctx.interest or {}).get("interest_id")
                        or ""
                    ),
                )
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
                    target = action.arguments.get("interest_id") or (
                        ctx.interest or {}
                    ).get("interest_id")
                    result.effects.append(
                        {"type": "end_focus", "interest_id": str(target or "")}
                    )
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
                "record_interest",
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
        """Keep the oldest prompt and newest tool turns under the budget."""

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
        """Execute one read-only or deferred tool action."""

        arguments = action.arguments

        if action.action == "read_messages":
            limit = coerce_int(arguments.get("limit"), 20) or 20
            limit = max(1, min(self.settings.context_window_messages, limit))
            window = await window_loader(limit)
            if not window:
                return "没有读到新的消息。"
            for message in window:
                if int(message.seq or 0) > ctx.evaluated_seq:
                    ctx.evaluated_seq = int(message.seq)
            return (
                format_window(
                    window,
                    self.settings.context_max_chars,
                    max_messages=limit,
                ),
                window,
            )

        if action.action in {"get_interest", "get_topic"}:
            interest_id = str(
                arguments.get("interest_id")
                or (ctx.interest or {}).get("interest_id")
                or ""
            )
            selected = None
            for item in ctx.interests:
                if item.get("interest_id") == interest_id:
                    selected = item
                    break
            if selected is None:
                return "本会话没有该兴趣，或它已经关闭。"
            return (
                f"兴趣：{selected.get('interest_id')}\n"
                f"摘要：{clip_text(selected.get('summary'), 300) or '（无）'}\n"
                f"等待：{clip_text(selected.get('wait_for'), 200) or '（无）'}\n"
                f"状态：{selected.get('status')}；剩余检查："
                f"{selected.get('remaining_checks', 0)}；到期："
                f"{selected.get('expires_at') or '-'}"
            )

        if action.action in {"schedule_next_check", "defer_check"}:
            minutes = coerce_int(arguments.get("minutes"))
            if minutes is None:
                return "缺少整数参数 minutes。"
            result.next_check_minutes = minutes
            return f"已记录下次检查间隔：{minutes} 分钟（受上下限约束）。"

        if action.action == "create_checkpoint":
            interest_id = str(
                arguments.get("interest_id")
                or (ctx.interest or {}).get("interest_id")
                or ""
            )
            minutes = coerce_int(arguments.get("minutes"), 60) or 60
            reason = clip_text(
                arguments.get("reason") or action.reason or "关注该话题",
                200,
            )
            if not interest_id:
                return "缺少 interest_id；不能创建无主扫描器。"
            result.effects.append(
                {
                    "type": "create_checkpoint",
                    "interest_id": interest_id,
                    "minutes": minutes,
                    "reason": reason,
                }
            )
            return "已记录检查点请求，将按兴趣期限和去重规则校验。"

        if action.action == "update_user_policy":
            user_id = clip_text(
                arguments.get("user_id") or arguments.get("target"),
                80,
            )
            if not user_id:
                return "缺少 user_id。"
            expires = coerce_int(arguments.get("expires_in_minutes"))
            if expires is not None:
                expires = max(
                    1,
                    min(self.settings.max_auto_policy_hours * 60, expires),
                )
            result.effects.append(
                {
                    "type": "user_policy",
                    "scope": "user"
                    if not ctx.checkpoint.get("group_id")
                    else "group_user",
                    "group_id": str(ctx.checkpoint.get("group_id") or ""),
                    "user_id": user_id,
                    "expires_in_minutes": expires,
                    "reason": clip_text(
                        arguments.get("reason") or action.reason or "策略调整",
                        120,
                    ),
                }
            )
            return "已记录用户策略变更请求，将按权限和时限校验。"

        if action.action == "record_interest":
            level = str(
                arguments.get("interest_level") or action.interest_level or ""
            ).lower()
            if level in {"none", "low", "medium", "high"}:
                result.interest_level = level
            summary = clip_text(
                arguments.get("summary")
                or (action.interest_update or {}).get("summary"),
                300,
            )
            wait_for = clip_text(arguments.get("wait_for"), 300)
            result.interest_update = {
                "operation": "update",
                "summary": summary,
                "wait_for": wait_for,
                "interest_id": str(
                    arguments.get("interest_id")
                    or (ctx.interest or {}).get("interest_id")
                    or ""
                ),
                "evidence_message_ids": list(
                    arguments.get("evidence_message_ids") or []
                ),
            }
            return "已记录兴趣更新请求，仍需代码校验。"

        return None
