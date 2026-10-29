"""AstrBot adapter implementations for the attention ports.

This is the only module that depends on AstrBot. The core sees small typed
interfaces instead of ``AstrMessageEvent``, message chains, or provider objects.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from astrbot.api import logger
from astrbot.api.message_components import At, Reply
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.utils.error_redaction import safe_error

from .attention.context import merge_window
from .attention.models import ModelResult, RecentMessage
from .attention.settings import Settings


class AstrBotModelGateway:
    """Call AstrBot providers for decisions and proactive replies."""

    def __init__(
        self,
        context: Any,
        settings: Settings,
    ) -> None:
        self.context = context
        self.settings = settings

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    async def _provider_id(self, purpose: str, umo: str) -> str:
        configured = (
            self.settings.decision_provider_id
            if purpose == "decision"
            else self.settings.main_provider_id
        )
        if configured:
            return configured
        return await self.context.get_current_chat_provider_id(umo=umo)

    async def generate(
        self,
        *,
        purpose: str,
        umo: str,
        system_prompt: str,
        prompt: str | None = None,
        contexts: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> ModelResult:
        """Generate one completion through AstrBot's provider API."""

        try:
            provider_id = await self._provider_id(purpose, umo)
        except Exception as exc:
            return ModelResult(error=safe_error("provider resolution failed: ", exc))

        clean_contexts: list[dict[str, Any]] = []
        for item in contexts or []:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role and content:
                clean_contexts.append({"role": str(role), "content": content})

        timeout_seconds = float(timeout or self.settings.decision_timeout_seconds)
        if purpose == "decision":
            logger.info(
                "Attention decision model call: umo=%s provider=%s timeout=%ss",
                umo,
                provider_id or "-",
                int(timeout_seconds),
            )
        try:
            response = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    contexts=clean_contexts or None,
                    system_prompt=system_prompt,
                    request_max_retries=1,
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            return ModelResult(
                provider_id=provider_id,
                error=f"model call timed out after {timeout_seconds:.0f}s",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ModelResult(
                provider_id=provider_id,
                error=safe_error(f"{type(exc).__name__}: ", exc),
            )

        usage: dict[str, int] = {}
        if getattr(response, "usage", None) is not None:
            try:
                usage = {
                    "input": int(response.usage.input),
                    "output": int(response.usage.output),
                    "total": int(response.usage.total),
                }
            except Exception:
                usage = {}
        text = str(getattr(response, "completion_text", "") or "")
        logger.debug(
            "Attention model call: purpose=%s provider=%s chars=%s",
            purpose,
            provider_id,
            len(text),
        )
        return ModelResult(
            text=text,
            provider_id=provider_id,
            usage=usage,
        )


class AstrBotContextProvider:
    """Read AstrBot's group history and resolve the active persona."""

    def __init__(
        self,
        context: Any,
        settings: Settings,
    ) -> None:
        self.context = context
        self.settings = settings

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    async def window(
        self,
        *,
        umo: str,
        platform_id: str,
        group_id: str,
        recent: list[RecentMessage],
        limit: int,
    ) -> list[RecentMessage]:
        history = await self._read_history(umo, platform_id, limit)
        return merge_window(history, recent, limit)

    async def _read_history(
        self,
        umo: str,
        platform_id: str,
        limit: int,
    ) -> list[RecentMessage]:
        manager = getattr(self.context, "message_history_manager", None)
        if manager is None or not platform_id:
            return []
        try:
            records = await manager.get(
                platform_id=platform_id,
                user_id=umo,
                page_size=max(limit * 2, 20),
            )
        except Exception:
            logger.debug(
                "Reading AstrBot group history failed for %s.",
                umo,
                exc_info=True,
            )
            return []
        result: list[RecentMessage] = []
        for record in records or []:
            try:
                message = self._record_to_message(record)
            except Exception:
                continue
            if message is not None and message.text.strip():
                message.umo = umo
                result.append(message)
        return result

    def _record_to_message(self, record: Any) -> RecentMessage | None:
        content = getattr(record, "content", None)
        if not isinstance(content, dict):
            return None
        parts = content.get("message")
        if not isinstance(parts, list):
            return None
        pieces: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "plain":
                pieces.append(str(part.get("text") or ""))
            elif part_type == "at":
                name = part.get("name") or part.get("user_id") or ""
                pieces.append(f"@{name}")
            elif part_type == "reply":
                pieces.append(
                    f"[引用 {part.get('sender_name') or ''}: {part.get('text') or ''}]"
                )
        text = " ".join(piece for piece in pieces if piece).strip()
        if not text:
            return None
        role = "bot" if content.get("type") == "bot" else "user"
        created = getattr(record, "created_at", None)
        if not isinstance(created, datetime):
            return None
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        local_created = created.astimezone(self.settings.tz)
        return RecentMessage(
            role=role,
            sender_id=str(getattr(record, "sender_id", "") or ""),
            sender_name=str(getattr(record, "sender_name", "") or ""),
            text=text,
            created_at=local_created.isoformat(timespec="seconds"),
            created_ts=created.timestamp(),
            message_id=str(
                content.get("message_id") or f"history:{getattr(record, 'id', '')}"
            ),
            source="history",
            is_bot=role == "bot",
        )

    async def persona(self, umo: str) -> str:
        if self.settings.persona_source == "custom":
            return self.settings.custom_persona
        manager = getattr(self.context, "persona_manager", None)
        if manager is None:
            return ""
        try:
            persona = await manager.get_default_persona_v3(umo)
        except Exception:
            logger.debug("Persona lookup failed for %s.", umo, exc_info=True)
            return ""
        if isinstance(persona, dict):
            return str(persona.get("prompt") or "")
        return str(getattr(persona, "get", lambda *_: "")("prompt") or "")


class AstrBotMessageSender:
    """Send proactive messages through AstrBot's platform manager."""

    def __init__(
        self,
        context: Any,
    ) -> None:
        self.context = context

    async def send_text(self, umo: str, text: str) -> tuple[bool, str]:
        try:
            chain = MessageChain().message(text)
            sent = await self.context.send_message(umo, chain)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Sending proactive message to %s failed.", umo)
            return False, "uncertain_delivery:" + safe_error(
                f"{type(exc).__name__}: ", exc
            )
        if not sent:
            return False, "platform not found for session"
        return True, ""


def _raw_message_text(event: AstrMessageEvent) -> str:
    """Return the platform message text before AstrBot strips the wake prefix."""

    message_obj = getattr(event, "message_obj", None)
    raw = getattr(message_obj, "raw_message", None)
    message = raw.get("message") if isinstance(raw, dict) else None
    if isinstance(message, list):
        parts = [
            str((segment.get("data") or {}).get("text") or "")
            for segment in message
            if isinstance(segment, dict) and segment.get("type") == "text"
        ]
        text = "".join(parts)
        if text:
            return text
    parts = []
    for component in getattr(event, "get_messages", lambda: [])():
        text = getattr(component, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def classify_wake_source(
    event: AstrMessageEvent,
    wake_prefixes: tuple[str, ...],
) -> str:
    """Classify why this event reached the main model.

    AstrBot only reports the combined ``is_at_or_wake_command`` flag and strips
    the wake prefix before hooks run. This function therefore inspects the
    trusted message components and the raw platform payload, never model text.

    Args:
        event: Incoming AstrBot event.
        wake_prefixes: Configured non-empty wake prefixes.

    Returns:
        One of ``private``, ``mention``, ``reply_to_bot``, ``wake_prefix``,
        ``ambient`` or ``unknown``.
    """

    if not getattr(event, "get_group_id", lambda: "")():
        return "private"
    self_id = str(getattr(event, "get_self_id", lambda: "")() or "")
    messages = getattr(event, "get_messages", lambda: [])()
    for component in messages:
        if isinstance(component, At) and str(component.qq) == self_id:
            return "mention"
    for component in messages:
        if isinstance(component, Reply) and str(component.sender_id) == self_id:
            return "reply_to_bot"
    raw_text = _raw_message_text(event)
    for prefix in wake_prefixes:
        if prefix and raw_text.startswith(prefix):
            return "wake_prefix"
    if getattr(event, "is_at_or_wake_command", False):
        return "unknown"
    return "ambient"


def build_passive_request(
    event: AstrMessageEvent,
    request: ProviderRequest,
    *,
    is_command: bool,
    timestamp: str,
    wake_prefixes: tuple[str, ...] = (),
    friend_needs_wake_prefix: bool = False,
) -> Any:
    """Convert an AstrBot LLM request into a core passive request.

    Args:
        event: Trusted platform event.
        request: Main-model request assembled by AstrBot.
        is_command: Whether the event is a command.
        timestamp: Session-local ISO timestamp.
        wake_prefixes: Configured wake prefixes, used to classify the source.
        friend_needs_wake_prefix: Whether private chats also require a prefix.

    Returns:
        A normalised :class:`PassiveRequest`.
    """

    from .attention.models import PassiveRequest

    _ = friend_needs_wake_prefix
    contexts: list[dict[str, Any]] = []
    for item in getattr(request, "contexts", None) or []:
        if isinstance(item, dict):
            dumped = item
        elif hasattr(item, "model_dump"):
            try:
                dumped = item.model_dump()
            except Exception:
                continue
        else:
            continue
        if not isinstance(dumped, dict):
            continue

        role = str(dumped.get("role") or "").lower()
        if role not in {"user", "assistant"}:
            continue
        if dumped.get("tool_call_id") or dumped.get("tool_calls"):
            continue

        content = dumped.get("content")
        content_parts = content if isinstance(content, list) else [content]
        if any(
            isinstance(part, dict)
            and str(part.get("type") or "").lower() in {"tool_use", "tool_result"}
            for part in content_parts
        ):
            continue
        if not content:
            continue
        contexts.append({"role": role, "content": content})
    return PassiveRequest(
        umo=str(event.unified_msg_origin),
        platform_id=str(event.get_platform_id() or ""),
        group_id=str(event.get_group_id() or ""),
        user_id=str(event.get_sender_id() or ""),
        sender_name=str(event.get_sender_name() or ""),
        message_id=str(
            getattr(getattr(event, "message_obj", None), "message_id", "") or ""
        ),
        message_text=str(event.message_str or ""),
        wake_source=classify_wake_source(event, wake_prefixes),
        is_group=bool(event.get_group_id()),
        is_admin=bool(event.is_admin()),
        is_command=is_command,
        system_prompt=str(getattr(request, "system_prompt", "") or ""),
        contexts=contexts,
        prompt=str(getattr(request, "prompt", None) or event.message_str or ""),
        timestamp=timestamp,
    )
