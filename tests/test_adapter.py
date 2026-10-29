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
    gateway = AstrBotModelGateway(context, settings)
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
    provider = AstrBotContextProvider(context, settings)
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
    sender = AstrBotMessageSender(context)
    sent, error = await sender.send_text("p:GroupMessage:g", "hello")
    assert sent is True
    assert error == ""
    assert context.sent[0][0] == "p:GroupMessage:g"


def _event(
    *,
    umo: str = "p:GroupMessage:g",
    group_id: str = "g",
    message_str: str = "你好",
    message_id: str = "m1",
    messages: list | None = None,
    self_id: str = "bot",
    raw_message: dict | None = None,
    is_admin: bool = False,
):
    return SimpleNamespace(
        unified_msg_origin=umo,
        message_obj=SimpleNamespace(
            message_id=message_id,
            raw_message=raw_message,
        ),
        message_str=message_str,
        get_platform_id=lambda: "p",
        get_group_id=lambda: group_id,
        get_sender_id=lambda: "u1",
        get_sender_name=lambda: "用户",
        get_self_id=lambda: self_id,
        get_messages=lambda: list(messages or []),
        is_at_or_wake_command=True,
        is_admin=lambda: is_admin,
    )


def test_build_passive_request_from_event() -> None:
    event = _event()
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
    assert passive.wake_source == "unknown"


def test_build_passive_request_classifies_wake_sources() -> None:
    from astrbot.api.message_components import At, Plain, Reply

    reply = Reply(
        id="x",
        chain=[],
        sender_id="123",
        sender_nickname="bot",
        time=0,
        message_str="旧消息",
        text="旧消息",
        qq=123,
    )
    at = At(qq=123)
    cases = [
        (_event(group_id="", umo="p:FriendMessage:u1"), (), "private"),
        (
            _event(messages=[at], self_id="123"),
            (),
            "mention",
        ),
        (
            _event(messages=[reply], self_id="123"),
            (),
            "reply_to_bot",
        ),
        (
            _event(
                message_str="dev 你好",
                raw_message={
                    "message": [{"type": "text", "data": {"text": "/dev 你好"}}]
                },
            ),
            ("/dev",),
            "wake_prefix",
        ),
        (_event(messages=[Plain("你好")]), (), "unknown"),
    ]
    for event, prefixes, expected in cases:
        request = SimpleNamespace(system_prompt="", contexts=[], prompt="你好")
        passive = build_passive_request(
            event,
            request,
            is_command=False,
            timestamp="2026-01-01T00:00:00+08:00",
            wake_prefixes=prefixes,
        )
        assert passive.wake_source == expected, (passive.wake_source, expected)


def test_build_passive_request_filters_tool_contexts() -> None:
    event = _event()
    request = SimpleNamespace(
        system_prompt="sys",
        contexts=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "用户消息"},
            {"role": "assistant", "content": "普通助手回复"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1"}],
            },
            {
                "role": "tool",
                "content": "工具结果",
                "tool_call_id": "call-1",
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "text": "工具结果"}],
            },
        ],
        prompt="你好",
    )

    passive = build_passive_request(
        event,
        request,
        is_command=False,
        timestamp="2026-01-01T00:00:00+08:00",
    )

    assert passive.contexts == [
        {"role": "user", "content": "用户消息"},
        {"role": "assistant", "content": "普通助手回复"},
    ]


@pytest.mark.asyncio
async def test_waiting_hook_marks_plugin_request_for_passive_gate():
    plugin = object.__new__(AutonomousAttentionPlugin)
    extras = {"provider_request": object()}
    event = SimpleNamespace(
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=lambda key, value: extras.__setitem__(key, value),
    )

    await plugin.mark_non_user_llm_request(event)

    assert extras["_aa_skip_passive"] is True


@pytest.mark.asyncio
async def test_passive_gate_does_not_call_attention_model_for_user_requests():
    plugin = object.__new__(AutonomousAttentionPlugin)
    plugin._ready = True
    plugin.settings = Settings.from_mapping({})
    calls: list[object] = []

    async def handle_passive(request):
        calls.append(request)
        return SimpleNamespace(block=False, reservation_id="", code="", reason="")

    plugin.engine = SimpleNamespace(
        policy=SimpleNamespace(now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)),
        handle_passive=handle_passive,
    )
    extras = {"provider_request": object()}
    event = SimpleNamespace(
        is_at_or_wake_command=True,
        message_str="你好",
        unified_msg_origin="p:GroupMessage:g",
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=lambda key, value: extras.__setitem__(key, value),
        get_platform_name=lambda: "aiocqhttp",
        get_group_id=lambda: "g",
        get_sender_id=lambda: "u1",
        get_self_id=lambda: "bot",
        get_messages=lambda: [],
    )

    await plugin.passive_gate(event, SimpleNamespace(system_prompt=""))

    assert calls == []


@pytest.mark.asyncio
async def test_passive_gate_skips_plugin_and_cron_requests():
    plugin = object.__new__(AutonomousAttentionPlugin)
    plugin._ready = True
    calls: list[object] = []

    async def handle_passive(request):
        calls.append(request)
        return SimpleNamespace(block=False)

    plugin.engine = SimpleNamespace(
        policy=SimpleNamespace(now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)),
        handle_passive=handle_passive,
    )

    for platform_name, extras in (
        ("aiocqhttp", {"_aa_skip_passive": True}),
        ("aiocqhttp", {"cron_payload": {"job_id": "job-1"}}),
        ("cron", {}),
    ):
        event = SimpleNamespace(
            is_at_or_wake_command=True,
            message_str="你好",
            get_extra=lambda key, default=None, values=extras: values.get(key, default),
            set_extra=lambda key, value, values=extras: values.__setitem__(key, value),
            get_platform_name=lambda name=platform_name: name,
        )
        await plugin.passive_gate(event, object())

    assert calls == []


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


@pytest.mark.asyncio
async def test_plugin_private_passive_gate_is_direct(tmp_path, monkeypatch) -> None:
    from astrbot.core.star.star_tools import StarTools

    monkeypatch.setattr(
        StarTools, "get_data_dir", classmethod(lambda cls, name: tmp_path)
    )
    context = FakeContext()
    context.get_config = lambda: {
        "timezone": "Asia/Shanghai",
        "wake_prefix": ["/"],
        "platform_settings": {"friend_message_needs_wake_prefix": True},
    }
    plugin = AutonomousAttentionPlugin(
        context,
        {"passive_mode": "full", "group_allowlist": []},
    )
    await plugin.initialize()
    extras: dict = {}
    event = SimpleNamespace(
        unified_msg_origin="p:FriendMessage:u1",
        message_str="你好",
        message_obj=SimpleNamespace(message_id="m1", raw_message=None),
        is_at_or_wake_command=True,
        get_platform_name=lambda: "aiocqhttp",
        get_platform_id=lambda: "p",
        get_group_id=lambda: "",
        get_sender_id=lambda: "u1",
        get_sender_name=lambda: "用户",
        get_self_id=lambda: "bot",
        get_messages=lambda: [],
        is_admin=lambda: False,
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=lambda key, value: extras.__setitem__(key, value),
        stop_event=lambda: extras.__setitem__("_stopped", True),
    )
    request = SimpleNamespace(system_prompt="", contexts=[], prompt="你好")
    try:
        await plugin.passive_gate(event, request)
        assert extras.get("_stopped") is None
        assert "Autonomous Attention Protocol" in request.system_prompt
        assert plugin.engine.store.state["decisions"] == []
        assert context.calls == []
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_plugin_group_passive_gate_allows_main_model_without_attention_call(
    tmp_path, monkeypatch
) -> None:
    from astrbot.api.message_components import Plain
    from astrbot.core.star.star_tools import StarTools

    monkeypatch.setattr(
        StarTools, "get_data_dir", classmethod(lambda cls, name: tmp_path)
    )
    context = FakeContext()
    context.get_config = lambda: {
        "timezone": "Asia/Shanghai",
        "wake_prefix": ["/"],
        "platform_settings": {"friend_message_needs_wake_prefix": True},
    }
    plugin = AutonomousAttentionPlugin(
        context,
        {"passive_mode": "full", "group_allowlist": ["g"]},
    )
    await plugin.initialize()
    extras: dict = {}
    event = SimpleNamespace(
        unified_msg_origin="p:GroupMessage:g",
        message_str="插件怎么配置",
        message_obj=SimpleNamespace(message_id="m1", raw_message=None),
        is_at_or_wake_command=True,
        get_platform_name=lambda: "aiocqhttp",
        get_platform_id=lambda: "p",
        get_group_id=lambda: "g",
        get_sender_id=lambda: "u1",
        get_sender_name=lambda: "用户",
        get_self_id=lambda: "bot",
        get_messages=lambda: [Plain("插件怎么配置")],
        is_admin=lambda: False,
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=lambda key, value: extras.__setitem__(key, value),
        stop_event=lambda: extras.__setitem__("_stopped", True),
    )
    request = SimpleNamespace(system_prompt="", contexts=[], prompt="插件怎么配置")
    try:
        await plugin.passive_gate(event, request)
        assert extras.get("_stopped") is None
        assert plugin.engine.store.state["decisions"] == []
        assert context.calls == []
    finally:
        await plugin.terminate()


def _tool_event(is_admin: bool = False):
    extras: dict[str, object] = {}
    return SimpleNamespace(
        unified_msg_origin="p:GroupMessage:g",
        get_group_id=lambda: "g",
        get_platform_id=lambda: "p",
        get_sender_id=lambda: "u1",
        get_sender_name=lambda: "用户",
        is_admin=lambda: is_admin,
        set_extra=lambda key, value: extras.__setitem__(key, value),
        stop_event=lambda: extras.__setitem__("_stopped", True),
        get_extra=lambda key, default=None: extras.get(key, default),
    )


def test_metadata_declares_qq_platforms() -> None:
    import yaml

    metadata = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "metadata.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["support_platforms"] == [
        "aiocqhttp",
        "qq_official",
        "qq_official_webhook",
    ]


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


def test_config_schema_is_grouped_and_loads_nested_defaults(tmp_path) -> None:
    from astrbot.core.config.astrbot_config import AstrBotConfig

    schema_path = Path(__file__).resolve().parent.parent / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    visible_schema = {
        key: node for key, node in schema.items() if not node.get("invisible")
    }
    assert set(visible_schema) == {
        "general",
        "models",
        "groups",
        "context",
        "scheduling",
        "agent_limits",
        "attention_limits",
        "budgets",
        "safety",
    }
    for node in visible_schema.values():
        assert node["type"] == "object"
        assert isinstance(node.get("items"), dict) and node["items"]

    config = AstrBotConfig(
        config_path=str(tmp_path / "plugin_config.json"),
        schema=schema,
    )
    assert config["general"]["passive_mode"] == "block_only"
    assert config["agent_limits"]["max_agent_rounds"] == 4
    settings = Settings.from_mapping(dict(config))
    assert settings.passive_mode == "block_only"
    assert settings.max_agent_rounds == 4
    assert settings.group_allowlist == ()
    assert settings.main_agent_attention_prompt_enabled is True
    assert "attention_ignore_user" in settings.main_agent_attention_prompt


def test_main_agent_attention_prompt_can_be_disabled_or_customized() -> None:
    disabled = AutonomousAttentionPlugin.__new__(AutonomousAttentionPlugin)
    disabled.settings = Settings.from_mapping(
        {"general": {"main_agent_attention_prompt_enabled": False}}
    )
    request = SimpleNamespace(system_prompt="persona")
    assert disabled._inject_attention_protocol(request) is False
    assert request.system_prompt == "persona"

    customized = AutonomousAttentionPlugin.__new__(AutonomousAttentionPlugin)
    customized.settings = Settings.from_mapping(
        {"general": {"main_agent_attention_prompt": "CUSTOM ATTENTION RULE"}}
    )
    request = SimpleNamespace(system_prompt="persona")
    assert customized._inject_attention_protocol(request) is True
    assert request.system_prompt == (
        "persona\n\n[Autonomous Attention Protocol]\nCUSTOM ATTENTION RULE\n"
    )


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
        before_budget = dict(
            plugin.engine.policy.ensure_budget(
                plugin.engine.store.state,
                plugin.engine.policy.now(),
            )
        )
        tool_event = _tool_event()
        ignored = json.loads(
            await plugin.attention_ignore_user(
                tool_event,
                target_user="u1",
                reason="test",
                expires_in_minutes=30,
            )
        )
        assert ignored["ok"] is True
        assert tool_event.get_extra("_stopped") is None
        assert tool_event.get_extra("_aa_suppress_current_response") is None
        assert (
            plugin.engine.store.state["decisions"][-1]["code_decision"]
            == "policy_saved"
        )
        after_budget = plugin.engine.policy.ensure_budget(
            plugin.engine.store.state,
            plugin.engine.policy.now(),
        )
        assert after_budget["decision_calls"] == before_budget["decision_calls"]
        assert after_budget["main_calls"] == before_budget["main_calls"]
        assert (
            after_budget["autonomous_attempts"] == before_budget["autonomous_attempts"]
        )
        blocked = await plugin.engine.check_hard_policy(
            group_id="g",
            user_id="u1",
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
