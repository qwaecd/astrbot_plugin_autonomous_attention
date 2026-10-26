"""Adapter smoke tests (skipped when AstrBot is not installed)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("astrbot")

PLUGIN_PARENT = Path(__file__).resolve().parent.parent.parent
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))

from astrbot_plugin_autonomous_attention.astrbot_adapter import (  # noqa: E402
    AstrBotContextProvider,
    AstrBotMessageSender,
    AstrBotModelGateway,
    build_passive_request,
)
from astrbot_plugin_autonomous_attention.attention.models import (  # noqa: E402
    RecentMessage,
)
from astrbot_plugin_autonomous_attention.attention.settings import (  # noqa: E402
    Settings,
)
from astrbot_plugin_autonomous_attention.main import (  # noqa: E402
    AutonomousAttentionPlugin,
)


class FakeProviderResponse:
    def __init__(self, text: str = "ok") -> None:
        self.completion_text = text
        self.usage = SimpleNamespace(input=3, output=4, total=7)


class FakeContext:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.sent: list[tuple[str, object]] = []
        self.message_history_manager = FakeHistory()
        self.persona_manager = FakePersona()

    async def get_current_chat_provider_id(self, umo: str) -> str:
        return "provider-1"

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        return FakeProviderResponse("模型回复")

    async def send_message(self, umo: str, chain) -> bool:
        self.sent.append((umo, chain))
        return True


class FakeHistory:
    def __init__(self) -> None:
        self.records = [
            SimpleNamespace(
                id=1,
                sender_id="u1",
                sender_name="用户",
                created_at=datetime(2026, 1, 1, 4, 0, tzinfo=timezone.utc),
                content={
                    "type": "user",
                    "message": [{"type": "plain", "text": "历史消息"}],
                },
            )
        ]

    async def get(self, **kwargs):
        return list(self.records)


class FakePersona:
    async def get_default_persona_v3(self, umo: str):
        return {"prompt": "测试人格"}


@pytest.mark.asyncio
async def test_model_gateway_uses_llm_generate() -> None:
    context = FakeContext()
    settings = Settings.from_mapping({})
    gateway = AstrBotModelGateway(context, settings, logger=_logger())
    result = await gateway.generate(
        purpose="decision",
        umo="p:GroupMessage:g",
        system_prompt="sys",
        prompt="hello",
        timeout=5,
    )
    assert result.text == "模型回复"
    assert result.usage == {"input": 3, "output": 4, "total": 7}
    assert context.calls[0]["chat_provider_id"] == "provider-1"
    assert context.calls[0]["system_prompt"] == "sys"


@pytest.mark.asyncio
async def test_context_provider_reads_history_and_persona() -> None:
    context = FakeContext()
    settings = Settings.from_mapping({"timezone": "Asia/Shanghai"})
    provider = AstrBotContextProvider(context, settings, logger=_logger())
    window = await provider.window(
        umo="p:GroupMessage:g",
        platform_id="p",
        group_id="g",
        recent=[],
        limit=10,
    )
    assert window and window[0].text == "历史消息"
    assert window[0].is_bot is False
    assert await provider.persona("p:GroupMessage:g") == "测试人格"


@pytest.mark.asyncio
async def test_message_sender_calls_context() -> None:
    context = FakeContext()
    sender = AstrBotMessageSender(context, logger=_logger())
    sent, error = await sender.send_text("p:GroupMessage:g", "hello")
    assert sent is True
    assert error == ""
    assert context.sent[0][0] == "p:GroupMessage:g"


def test_build_passive_request_from_event() -> None:
    event = SimpleNamespace(
        unified_msg_origin="p:GroupMessage:g",
        message_obj=SimpleNamespace(message_id="m1"),
        message_str="你好",
        get_platform_id=lambda: "p",
        get_group_id=lambda: "g",
        get_sender_id=lambda: "u1",
        get_sender_name=lambda: "用户",
        is_admin=lambda: False,
    )
    request = SimpleNamespace(
        system_prompt="sys",
        contexts=[{"role": "user", "content": "hi"}],
        prompt="你好",
    )
    passive = build_passive_request(
        event,
        request,
        is_command=False,
        timestamp="2026-01-01T00:00:00+08:00",
    )
    assert passive.umo == "p:GroupMessage:g"
    assert passive.group_id == "g"
    assert passive.contexts[0]["content"] == "hi"
    assert passive.timestamp.startswith("2026-01-01")


def _logger():
    import logging

    return logging.getLogger("adapter-tests")


def test_recent_message_roundtrip() -> None:
    message = RecentMessage(
        sender_id="u1",
        text="hello",
        created_ts=1.0,
    )
    assert RecentMessage.from_dict(message.to_dict()).text == "hello"


@pytest.mark.asyncio
async def test_plugin_initialize_and_terminate(tmp_path, monkeypatch) -> None:
    from astrbot.core.star.star_tools import StarTools

    monkeypatch.setattr(
        StarTools, "get_data_dir", classmethod(lambda cls, name: tmp_path)
    )
    context = FakeContext()
    context.get_config = lambda: {"timezone": "Asia/Shanghai"}
    plugin = AutonomousAttentionPlugin(context, {"target": "test"})
    await plugin.initialize()
    assert plugin._ready is True
    assert plugin.engine is not None
    assert plugin.engine.scheduler.status()["running"] is True
    await plugin.terminate()
    assert plugin._ready is False
    assert plugin.engine.scheduler.status()["running"] is False


def _tool_event(is_admin: bool = False):
    return SimpleNamespace(
        unified_msg_origin="p:GroupMessage:g",
        get_group_id=lambda: "g",
        get_platform_id=lambda: "p",
        get_sender_id=lambda: "u1",
        get_sender_name=lambda: "用户",
        is_admin=lambda: is_admin,
    )


def test_main_agent_attention_tools_registered() -> None:
    from astrbot.core.provider.register import llm_tools

    names = {tool.name for tool in llm_tools.func_list}
    for name in (
        "attention_query",
        "attention_set_intent",
        "attention_ignore_user",
        "attention_clear_user_policy",
        "attention_focus_topic",
        "attention_stop_focus",
    ):
        assert name in names


@pytest.mark.asyncio
async def test_main_agent_tool_ignore_and_query(tmp_path, monkeypatch) -> None:
    from astrbot.core.star.star_tools import StarTools

    monkeypatch.setattr(
        StarTools, "get_data_dir", classmethod(lambda cls, name: tmp_path)
    )
    context = FakeContext()
    context.get_config = lambda: {"timezone": "Asia/Shanghai"}
    plugin = AutonomousAttentionPlugin(
        context,
        {
            "group_allowlist": ["g"],
            "passive_mode": "block_only",
            "main_agent_tools_enabled": True,
        },
    )
    await plugin.initialize()
    try:
        ignored = json.loads(
            await plugin.attention_ignore_user(
                _tool_event(),
                target_user="u2",
                reason="test",
                expires_in_minutes=30,
            )
        )
        assert ignored["ok"] is True
        blocked = await plugin.engine.check_hard_policy(
            group_id="g",
            user_id="u2",
            umo="p:GroupMessage:g",
            platform_id="p",
            is_admin=False,
        )
        assert blocked is not None
        assert blocked["source"] == "main_agent"
        queried = json.loads(
            await plugin.attention_query(_tool_event(), scope="session")
        )
        assert queried["ok"] is True
        failed = json.loads(
            await plugin.attention_ignore_user(
                _tool_event(),
                target_user="",
                reason="missing",
            )
        )
        assert failed["ok"] is False
    finally:
        await plugin.terminate()
