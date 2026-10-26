"""Core tests that do not import AstrBot."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from attention.control import Actor
from attention.engine import AttentionEngine
from attention.models import (
    SOURCE_ADMIN,
    SOURCE_ATTENTION_AGENT,
    SOURCE_MAIN_AGENT,
    ModelResult,
    PassiveRequest,
    RecentMessage,
)
from attention.policy import PolicyManager
from attention.prompts import build_active_reply_system_prompt
from attention.settings import Settings
from attention.store import StateStore, normalize_state
from attention.validation import (
    extract_json_object,
    sanitize_reply,
    validate_passive_decision,
)

LOGGER = logging.getLogger("attention-tests")
LOGGER.addHandler(logging.NullHandler())


def make_settings(**overrides: Any) -> Settings:
    data = {
        "passive_mode": "full",
        "active_enabled": True,
        "group_allowlist": ["g", "g1"],
        "min_interval_minutes": 1,
        "max_interval_minutes": 120,
        "default_interval_minutes": 5,
        "max_defer_minutes": 120,
        "group_send_cooldown_minutes": 0,
        "global_send_cooldown_minutes": 0,
        "daily_model_call_budget": 100,
        "daily_active_message_budget": 10,
        "daily_active_check_budget": 100,
        "max_agent_rounds": 4,
        "max_tool_calls": 5,
        "max_checkpoint_mutations": 2,
        "decision_timeout_seconds": 5,
        "timezone": "Asia/Shanghai",
    }
    data.update(overrides)
    return Settings.from_mapping(data)


class FakeGateway:
    def __init__(self, decision_responses: list[ModelResult] | None = None) -> None:
        self.decision_responses = list(decision_responses or [])
        self.calls: list[dict[str, Any]] = []
        self.main_response = ModelResult(text="主动回复内容")

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
        self.calls.append(
            {
                "purpose": purpose,
                "umo": umo,
                "prompt": prompt,
                "contexts": contexts,
            }
        )
        if purpose == "main":
            return self.main_response
        if not self.decision_responses:
            return ModelResult(text='{"decision":"pass"}')
        return self.decision_responses.pop(0)


class FakeContextProvider:
    def __init__(self, messages: list[RecentMessage] | None = None) -> None:
        self.messages = messages or []

    async def window(
        self,
        *,
        umo: str,
        platform_id: str,
        group_id: str,
        recent: list[RecentMessage],
        limit: int,
    ) -> list[RecentMessage]:
        return list(self.messages or recent)[-limit:]

    async def persona(self, umo: str) -> str:
        return "你是一个友善的群成员。"


class FakeSender:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.sent: list[tuple[str, str]] = []

    async def send_text(self, umo: str, text: str) -> tuple[bool, str]:
        self.sent.append((umo, text))
        if self.ok:
            return True, ""
        return False, "fake send error"


async def make_engine(
    tmp_path: Path,
    *,
    settings: Settings | None = None,
    gateway: FakeGateway | None = None,
    context_provider: FakeContextProvider | None = None,
    sender: FakeSender | None = None,
) -> tuple[AttentionEngine, StateStore, FakeGateway]:
    settings = settings or make_settings()
    gateway = gateway or FakeGateway()
    context_provider = context_provider or FakeContextProvider()
    sender = sender or FakeSender()
    store = StateStore(tmp_path / "state.json", LOGGER, 50)
    engine = AttentionEngine(
        settings=settings,
        store=store,
        gateway=gateway,
        context_provider=context_provider,
        sender=sender,
        logger=LOGGER,
    )
    await engine.start()
    return engine, store, gateway


def passive_request(
    *,
    umo: str = "p:GroupMessage:g",
    group_id: str = "g",
    user_id: str = "u1",
    message_id: str = "m1",
    text: str = "你好",
) -> PassiveRequest:
    return PassiveRequest(
        umo=umo,
        platform_id="p",
        group_id=group_id,
        user_id=user_id,
        sender_name="用户",
        message_id=message_id,
        message_text=text,
        is_group=bool(group_id),
        prompt=text,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def test_validation_extracts_fenced_json_and_rejects_bad_decisions() -> None:
    raw = extract_json_object(
        '说明如下：\n```json\n{"decision": "ignore", "reason": "不适配"}\n```'
    )
    assert raw is not None
    decision = validate_passive_decision(raw)
    assert decision is not None
    assert decision.decision == "ignore"
    assert validate_passive_decision({"decision": "reply"}) is None
    assert validate_passive_decision({"decision": "policy_update"}) is None


def test_sanitize_reply_strips_fences_and_quotes() -> None:
    assert sanitize_reply('```\n"你好呀"\n```') == "你好呀"
    assert sanitize_reply("   ") is None
    long_text = "一" * 30
    assert sanitize_reply(long_text, 10) == "一" * 10


def test_settings_clamp_and_defaults() -> None:
    settings = Settings.from_mapping(
        {
            "passive_mode": "unknown",
            "min_interval_minutes": 30,
            "max_interval_minutes": 10,
            "default_interval_minutes": 9999,
            "max_agent_rounds": 0,
        }
    )
    assert settings.passive_mode == "block_only"
    assert settings.min_interval_minutes == 30
    assert settings.max_interval_minutes == 30
    assert settings.default_interval_minutes == 30
    assert settings.max_agent_rounds == 1


@pytest.mark.asyncio
async def test_store_roundtrip_and_recovery(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    store = StateStore(path, LOGGER, 50)
    async with store.lock:
        store.state["policies"]["x"] = {"scope": "user", "user_id": "u"}
        store.flush()
    loaded = StateStore(path, LOGGER, 50)
    await loaded.load()
    assert loaded.state["policies"]["x"]["user_id"] == "u"

    path.write_text("{broken", encoding="utf-8")
    corrupt = StateStore(path, LOGGER, 50)
    await corrupt.load()
    assert corrupt.state["policies"] == {}
    assert list(tmp_path.glob("state.corrupt-*.json"))


def test_state_migration_adds_source_and_priority() -> None:
    state = normalize_state(
        {
            "policies": {
                "group_user:g:u": {
                    "policy_id": "group_user:g:u",
                    "scope": "group_user",
                    "group_id": "g",
                    "user_id": "u",
                    "created_by": "agent",
                }
            },
            "checkpoints": {
                "cp1": {
                    "checkpoint_id": "cp1",
                    "created_by": "config",
                    "status": "active",
                }
            },
            "decisions": [{"kind": "passive", "decision": "pass"}],
        }
    )
    assert state["policies"]["group_user:g:u"]["source"] == SOURCE_ATTENTION_AGENT
    assert state["policies"]["group_user:g:u"]["priority"] == 10
    assert state["checkpoints"]["cp1"]["source"] == SOURCE_ADMIN
    assert state["checkpoints"]["cp1"]["priority"] == 100
    assert state["decisions"][0]["source"] == "passive_agent"
    assert state["directives"] == {}


@pytest.mark.asyncio
async def test_scheduler_loop_awaits_due_hint(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.ERROR, logger="attention-tests")
    engine, _, _ = await make_engine(tmp_path)
    try:
        await asyncio.sleep(0.1)
        hint = await engine.seconds_until_next_due()
    finally:
        await engine.stop()
    assert isinstance(hint, int)
    assert not any(
        "scheduler loop iteration failed" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_passive_pass_then_dedup(tmp_path: Path) -> None:
    engine, _, gateway = await make_engine(
        tmp_path,
        gateway=FakeGateway(
            [
                ModelResult(
                    text=json.dumps(
                        {
                            "decision": "pass",
                            "interest_level": "high",
                            "reason": "相关问题",
                        }
                    )
                )
            ]
        ),
    )
    try:
        request = passive_request()
        first = await engine.handle_passive(request)
        second = await engine.handle_passive(request)
    finally:
        await engine.stop()
    assert first.block is False
    assert second.block is False
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_passive_ignore_blocks_and_records(tmp_path: Path) -> None:
    engine, store, _ = await make_engine(
        tmp_path,
        gateway=FakeGateway(
            [
                ModelResult(
                    text=json.dumps(
                        {
                            "decision": "ignore",
                            "interest_level": "none",
                            "reason": "没有接话空间",
                        }
                    )
                )
            ]
        ),
    )
    try:
        outcome = await engine.handle_passive(passive_request())
        decisions = list(store.state["decisions"])
    finally:
        await engine.stop()
    assert outcome.block is True
    assert outcome.decision == "ignore"
    assert decisions[-1]["kind"] == "passive"
    assert decisions[-1]["decision"] == "ignore"


@pytest.mark.asyncio
async def test_passive_invalid_output_falls_back_to_observe(tmp_path: Path) -> None:
    engine, _, _ = await make_engine(
        tmp_path,
        gateway=FakeGateway([ModelResult(text="这不是 JSON")]),
    )
    try:
        outcome = await engine.handle_passive(passive_request())
    finally:
        await engine.stop()
    assert outcome.block is True
    assert outcome.decision == "observe"


@pytest.mark.asyncio
async def test_passive_model_error_passes_through(tmp_path: Path) -> None:
    engine, _, _ = await make_engine(
        tmp_path,
        gateway=FakeGateway([ModelResult(error="timeout")]),
    )
    try:
        outcome = await engine.handle_passive(passive_request())
    finally:
        await engine.stop()
    assert outcome.block is False
    assert outcome.decision == "pass"


@pytest.mark.asyncio
async def test_passive_retry_later_creates_checkpoint(tmp_path: Path) -> None:
    engine, store, _ = await make_engine(
        tmp_path,
        gateway=FakeGateway(
            [
                ModelResult(
                    text=json.dumps(
                        {
                            "decision": "retry_later",
                            "reason": "稍后再看",
                        }
                    )
                )
            ]
        ),
    )
    try:
        outcome = await engine.handle_passive(passive_request())
        checkpoints = list(store.state["checkpoints"].values())
    finally:
        await engine.stop()
    assert outcome.block is True
    assert len(checkpoints) == 1
    assert checkpoints[0]["scheduled_at"]


@pytest.mark.asyncio
async def test_passive_policy_update_applies_policy(tmp_path: Path) -> None:
    engine, store, _ = await make_engine(
        tmp_path,
        gateway=FakeGateway(
            [
                ModelResult(
                    text=json.dumps(
                        {
                            "decision": "policy_update",
                            "reason": "持续越界",
                            "user_action": {
                                "scope": "group_user",
                                "mode": "ignore",
                                "expires_in_minutes": 30,
                                "reason": "越界",
                            },
                        }
                    )
                )
            ]
        ),
    )
    try:
        outcome = await engine.handle_passive(passive_request())
        policies = list(store.state["policies"].values())
    finally:
        await engine.stop()
    assert outcome.block is True
    assert policies and policies[0]["scope"] == "group_user"
    assert policies[0]["user_id"] == "u1"


@pytest.mark.asyncio
async def test_passive_daily_budget_is_consumed(tmp_path: Path) -> None:
    settings = make_settings(daily_model_call_budget=1)
    engine, store, gateway = await make_engine(
        tmp_path,
        settings=settings,
        gateway=FakeGateway(
            [
                ModelResult(text=json.dumps({"decision": "pass", "reason": "ok"})),
                ModelResult(text=json.dumps({"decision": "pass", "reason": "ok"})),
            ]
        ),
    )
    try:
        first = await engine.handle_passive(passive_request(message_id="m1"))
        second = await engine.handle_passive(passive_request(message_id="m2"))
        decision_calls = store.state["budget"]["decision_calls"]
    finally:
        await engine.stop()
    assert first.block is False
    assert second.block is False
    assert decision_calls == 1
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_hard_policy_blocks_without_model_call(tmp_path: Path) -> None:
    engine, _, gateway = await make_engine(tmp_path)
    try:
        await engine.set_policy(
            scope="group_user",
            group_id="g",
            user_id="u1",
            reason="test",
            expires_minutes=60,
        )
        outcome = await engine.handle_passive(passive_request())
    finally:
        await engine.stop()
    assert outcome.block is True
    assert outcome.decision == "ignore"
    assert gateway.calls == []


@pytest.mark.asyncio
async def test_dry_run_does_not_block_or_send(tmp_path: Path) -> None:
    settings = make_settings(dry_run=True)
    engine, store, _ = await make_engine(
        tmp_path,
        settings=settings,
        gateway=FakeGateway(
            [
                ModelResult(
                    text=json.dumps(
                        {
                            "decision": "ignore",
                            "reason": "dry run",
                        }
                    )
                )
            ]
        ),
    )
    try:
        outcome = await engine.handle_passive(passive_request())
        decisions = list(store.state["decisions"])
    finally:
        await engine.stop()
    assert outcome.block is False
    assert decisions[-1]["dry_run"] is True


@pytest.mark.asyncio
async def test_checkpoint_keeps_isolated_session_and_group(tmp_path: Path) -> None:
    engine, _, _ = await make_engine(tmp_path)
    try:
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:user123_g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="isolated session",
            created_by="admin",
        )
    finally:
        await engine.stop()
    assert checkpoint["umo"] == "p:GroupMessage:user123_g"
    assert checkpoint["group_id"] == "g"


@pytest.mark.asyncio
async def test_active_participate_sends_and_reschedules(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    messages = [
        RecentMessage(
            role="user",
            sender_id="u1",
            sender_name="用户",
            text="这个功能怎么用？",
            created_at=now.isoformat(timespec="seconds"),
            created_ts=now.timestamp(),
            message_id="1",
        )
    ]
    gateway = FakeGateway(
        [
            ModelResult(
                text=json.dumps(
                    {
                        "action": "read_messages",
                        "arguments": {"limit": 5},
                        "reason": "需要更多上下文",
                    }
                )
            ),
            ModelResult(
                text=json.dumps(
                    {
                        "action": "participate",
                        "arguments": {},
                        "reply_guidance": "简短回答",
                        "reason": "有明确问题",
                    }
                )
            ),
        ]
    )
    sender = FakeSender()
    engine, store, _ = await make_engine(
        tmp_path,
        gateway=gateway,
        context_provider=FakeContextProvider(messages),
        sender=sender,
    )
    try:
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="test",
            created_by="admin",
        )
        claimed, reason = await engine.claim_checkpoint(
            checkpoint["checkpoint_id"],
            force=True,
        )
        assert claimed is not None, reason
        await engine.run_claimed(checkpoint["checkpoint_id"])
        raw = store.state["checkpoints"][checkpoint["checkpoint_id"]]
    finally:
        await engine.stop()
    assert sender.sent == [("p:GroupMessage:g", "主动回复内容")]
    assert raw["status"] == "active"
    assert raw["scheduled_at"]
    assert raw["last_result"] == "participate"
    assert any(call["purpose"] == "main" for call in gateway.calls)


@pytest.mark.asyncio
async def test_empty_group_allowlist_blocks_checkpoint_creation(
    tmp_path: Path,
) -> None:
    settings = make_settings(group_allowlist=[])
    engine, _, _ = await make_engine(tmp_path, settings=settings)
    try:
        with pytest.raises(ValueError):
            await engine.create_checkpoint(
                umo="p:GroupMessage:g",
                platform_id="p",
                group_id="g",
                minutes=5,
                reason="test",
                created_by="admin",
            )
        assert not engine.policy.group_allowed("g", "p:GroupMessage:g")
        allowed, reason = engine.policy.can_send(
            engine.store.state,
            umo="p:GroupMessage:g",
            group_id="g",
            platform_id="p",
            moment=engine.policy.now(),
        )
    finally:
        await engine.stop()
    assert allowed is False
    assert reason == "group_not_allowed"


@pytest.mark.asyncio
async def test_group_allowlist_blocks_unlisted_checkpoint(tmp_path: Path) -> None:
    settings = make_settings(group_allowlist=["g1"])
    engine, _, _ = await make_engine(tmp_path, settings=settings)
    try:
        with pytest.raises(ValueError):
            await engine.create_checkpoint(
                umo="p:GroupMessage:g",
                platform_id="p",
                group_id="g",
                minutes=5,
                reason="not listed",
                created_by="admin",
            )
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g1",
            platform_id="p",
            group_id="g1",
            minutes=5,
            reason="listed",
            created_by="admin",
        )
    finally:
        await engine.stop()
    assert checkpoint["group_id"] == "g1"
    assert checkpoint["status"] == "active"


@pytest.mark.asyncio
async def test_active_end_focus_completes_checkpoint(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    messages = [
        RecentMessage(
            role="user",
            sender_id="u1",
            sender_name="用户",
            text="随便聊聊",
            created_at=now.isoformat(timespec="seconds"),
            created_ts=now.timestamp(),
            message_id="1",
        )
    ]
    sender = FakeSender()
    engine, store, _ = await make_engine(
        tmp_path,
        gateway=FakeGateway(
            [
                ModelResult(
                    text=json.dumps(
                        {
                            "action": "end_focus",
                            "reason": "话题已经结束",
                        }
                    )
                )
            ]
        ),
        context_provider=FakeContextProvider(messages),
        sender=sender,
    )
    try:
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="test",
            created_by="admin",
        )
        await engine.claim_checkpoint(checkpoint["checkpoint_id"], force=True)
        await engine.run_claimed(checkpoint["checkpoint_id"])
        raw = store.state["checkpoints"][checkpoint["checkpoint_id"]]
    finally:
        await engine.stop()
    assert sender.sent == []
    assert raw["status"] == "completed"
    assert raw["last_result"] == "end_focus"


@pytest.mark.asyncio
async def test_active_main_model_failure_does_not_send(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    messages = [
        RecentMessage(
            role="user",
            sender_id="u1",
            sender_name="用户",
            text="随便聊聊",
            created_at=now.isoformat(timespec="seconds"),
            created_ts=now.timestamp(),
            message_id="1",
        )
    ]
    gateway = FakeGateway(
        [
            ModelResult(
                text=json.dumps(
                    {
                        "action": "participate",
                        "reason": "有切入点",
                    }
                )
            )
        ]
    )
    gateway.main_response = ModelResult(error="main down")
    sender = FakeSender()
    engine, store, _ = await make_engine(
        tmp_path,
        gateway=gateway,
        context_provider=FakeContextProvider(messages),
        sender=sender,
    )
    try:
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="test",
            created_by="admin",
        )
        await engine.claim_checkpoint(checkpoint["checkpoint_id"], force=True)
        await engine.run_claimed(checkpoint["checkpoint_id"])
        raw = store.state["checkpoints"][checkpoint["checkpoint_id"]]
    finally:
        await engine.stop()
    assert sender.sent == []
    assert raw["last_result"] == "observe"
    assert raw["status"] == "active"


def test_policy_quiet_hours_and_group_matching() -> None:
    settings = make_settings(quiet_hours="12:00-13:00", group_allowlist=["g1"])
    policy = PolicyManager(settings, LOGGER)
    tz = settings.tz
    assert policy.is_quiet(datetime(2026, 1, 1, 12, 30, tzinfo=tz))
    assert not policy.is_quiet(datetime(2026, 1, 1, 14, 30, tzinfo=tz))
    assert policy.group_allowed("g1", "p:GroupMessage:g1")
    assert not policy.group_allowed("g2", "p:GroupMessage:g2")
    state = {"budget": {}, "policies": {}, "send_stats": {}}
    assert policy.can_send(
        state,
        umo="p:GroupMessage:g1",
        group_id="g1",
        platform_id="p",
        moment=datetime(2026, 1, 1, 14, 30, tzinfo=tz),
    )[0]


def test_group_allowlist_is_the_single_boundary() -> None:
    moment = datetime(2026, 1, 1, 14, 30, tzinfo=make_settings().tz)
    empty = make_settings(group_allowlist=[])
    empty_policy = PolicyManager(empty, LOGGER)
    assert not empty_policy.group_allowed("g1", "p:GroupMessage:g1")
    state = {"budget": {}, "policies": {}, "send_stats": {}}
    allowed, reason = empty_policy.can_send(
        state,
        umo="p:GroupMessage:g1",
        group_id="g1",
        platform_id="p",
        moment=moment,
    )
    assert allowed is False
    assert reason == "group_not_allowed"

    explicit = make_settings(group_allowlist=["g1"])
    explicit_policy = PolicyManager(explicit, LOGGER)
    assert explicit_policy.group_allowed("g1", "p:GroupMessage:g1")
    assert not explicit_policy.group_allowed("g2", "p:GroupMessage:g2")
    assert explicit_policy.can_send(
        state,
        umo="p:GroupMessage:g1",
        group_id="g1",
        platform_id="p",
        moment=moment,
    )[0]


@pytest.mark.asyncio
async def test_main_agent_ignore_policy_blocks_even_when_passive_off(
    tmp_path: Path,
) -> None:
    engine, _, _ = await make_engine(tmp_path)
    actor = Actor(
        source=SOURCE_MAIN_AGENT,
        user_id="u1",
        session="p:GroupMessage:g",
        group_id="g",
        platform_id="p",
    )
    try:
        result = await engine.control.ignore_user(
            actor,
            target_user="u2",
            reason="test ignore",
            expires_in_minutes=30,
        )
        assert result.ok
        blocked = await engine.check_hard_policy(
            group_id="g",
            user_id="u2",
            umo="p:GroupMessage:g",
            platform_id="p",
            is_admin=False,
        )
        assert blocked is not None
        assert blocked["source"] == SOURCE_MAIN_AGENT

        engine.update_settings(
            make_settings(passive_mode="off", group_allowlist=["g", "g1"])
        )
        blocked_again = await engine.check_hard_policy(
            group_id="g",
            user_id="u2",
            umo="p:GroupMessage:g",
            platform_id="p",
            is_admin=False,
        )
    finally:
        await engine.stop()
    assert blocked_again is not None
    assert blocked_again["source"] == SOURCE_MAIN_AGENT


@pytest.mark.asyncio
async def test_policy_priority_admin_main_attention(tmp_path: Path) -> None:
    engine, store, _ = await make_engine(tmp_path)
    att = Actor(
        source=SOURCE_ATTENTION_AGENT,
        session="p:GroupMessage:g",
        group_id="g",
        platform_id="p",
    )
    main = Actor(
        source=SOURCE_MAIN_AGENT,
        user_id="u1",
        session="p:GroupMessage:g",
        group_id="g",
        platform_id="p",
    )
    try:
        async with engine.store.lock:
            att_result = engine.control.set_policy_locked(
                engine.store.state,
                att,
                scope="group_user",
                group_id="g",
                user_id="u2",
                reason="attention policy",
                expires_at=engine.policy.now() + timedelta(minutes=30),
                source=SOURCE_ATTENTION_AGENT,
                allow_permanent=False,
            )
        assert att_result.ok
        main_result = await engine.control.ignore_user(
            main,
            target_user="u2",
            reason="main policy",
            expires_in_minutes=30,
        )
        assert main_result.ok
        policy = store.state["policies"][
            engine.policy.policy_id("group_user", "g", "u2")
        ]
        assert policy["source"] == SOURCE_MAIN_AGENT

        async with engine.store.lock:
            att_overwrite = engine.control.set_policy_locked(
                engine.store.state,
                att,
                scope="group_user",
                group_id="g",
                user_id="u2",
                reason="attention cannot override main",
                expires_at=engine.policy.now() + timedelta(minutes=30),
                source=SOURCE_ATTENTION_AGENT,
                allow_permanent=False,
            )
        assert att_overwrite.ok is False
        att_intent = await engine.control.set_intent(
            att,
            intent_type="stop_focus",
            target="",
        )
        assert att_intent.ok is False

        admin_result = await engine.set_policy(
            scope="group_user",
            group_id="g",
            user_id="u2",
            reason="admin policy",
            expires_minutes=None,
        )
        assert admin_result[0] is True
        policy = store.state["policies"][
            engine.policy.policy_id("group_user", "g", "u2")
        ]
        assert policy["source"] == SOURCE_ADMIN

        overwritten = await engine.control.ignore_user(
            main,
            target_user="u2",
            reason="main cannot override admin",
            expires_in_minutes=30,
        )
        assert overwritten.ok is False
        cleared = await engine.control.clear_user_policy(
            main,
            target_user="u2",
        )
        assert cleared.ok is False
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_control_query_respects_scope(tmp_path: Path) -> None:
    engine, _, _ = await make_engine(tmp_path)
    main = Actor(
        source=SOURCE_MAIN_AGENT,
        user_id="u1",
        session="p:GroupMessage:g",
        group_id="g",
        platform_id="p",
    )
    admin = Actor(
        source=SOURCE_MAIN_AGENT,
        user_id="admin",
        session="p:GroupMessage:g",
        group_id="g",
        platform_id="p",
        is_admin=True,
    )
    try:
        await engine.control.ignore_user(
            main,
            target_user="u2",
            reason="query test",
            expires_in_minutes=30,
        )
        session_result = await engine.control.query(main, scope="session")
        denied = await engine.control.query(
            main,
            scope="user",
            target_user="u2",
        )
        allowed = await engine.control.query(
            admin,
            scope="user",
            target_user="u2",
        )
    finally:
        await engine.stop()
    assert session_result.ok
    assert session_result.data["user_policy"]["target_user"] == "u1"
    assert denied.data["user_policy"]["allowed"] is False
    assert allowed.data["user_policy"]["ignored"] is True
    assert allowed.data["user_policy"]["source"] == SOURCE_MAIN_AGENT


@pytest.mark.asyncio
async def test_main_focus_forces_pass_and_stop_revokes_it(tmp_path: Path) -> None:
    gateway = FakeGateway(
        [
            ModelResult(
                text=json.dumps(
                    {
                        "decision": "ignore",
                        "reason": "would normally ignore",
                    }
                )
            )
        ]
    )
    engine, store, _ = await make_engine(tmp_path, gateway=gateway)
    main = Actor(
        source=SOURCE_MAIN_AGENT,
        user_id="u1",
        session="p:GroupMessage:g",
        group_id="g",
        platform_id="p",
        is_admin=True,
    )
    try:
        intent = await engine.control.set_intent(
            main,
            intent_type="focus_group",
            reason="main focus",
        )
        assert intent.ok
        focused = await engine.handle_passive(passive_request(message_id="focus-1"))
        assert focused.block is False
        assert gateway.calls == []

        stopped = await engine.control.stop_focus(main)
        assert stopped.ok
        after_stop = await engine.handle_passive(passive_request(message_id="focus-2"))
        directives = list(store.state["directives"].values())
    finally:
        await engine.stop()
    assert after_stop.block is True
    assert any(item["type"] == "stop_focus" for item in directives)


@pytest.mark.asyncio
async def test_main_stop_cancels_attention_checkpoint(tmp_path: Path) -> None:
    engine, store, _ = await make_engine(tmp_path)
    att = Actor(
        source=SOURCE_ATTENTION_AGENT,
        session="p:GroupMessage:g",
        group_id="g",
        platform_id="p",
    )
    main = Actor(
        source=SOURCE_MAIN_AGENT,
        user_id="u1",
        session="p:GroupMessage:g",
        group_id="g",
        platform_id="p",
        is_admin=True,
    )
    try:
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="attention checkpoint",
            created_by="agent",
            source=SOURCE_ATTENTION_AGENT,
            actor=att,
        )
        stopped = await engine.control.stop_focus(main)
        status = store.state["checkpoints"][checkpoint["checkpoint_id"]]["status"]
    finally:
        await engine.stop()
    assert stopped.ok
    assert status == "cancelled"


@pytest.mark.asyncio
async def test_non_admin_cannot_focus_group_or_stop_attention_checkpoint(
    tmp_path: Path,
) -> None:
    engine, store, _ = await make_engine(tmp_path)
    att = Actor(
        source=SOURCE_ATTENTION_AGENT,
        session="p:GroupMessage:g",
        group_id="g",
        platform_id="p",
    )
    member = Actor(
        source=SOURCE_MAIN_AGENT,
        user_id="u1",
        session="p:GroupMessage:g",
        group_id="g",
        platform_id="p",
    )
    try:
        intent = await engine.control.set_intent(
            member,
            intent_type="focus_group",
            reason="member focus",
        )
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="attention checkpoint",
            created_by="agent",
            source=SOURCE_ATTENTION_AGENT,
            actor=att,
        )
        stopped = await engine.control.stop_focus(member)
        status = store.state["checkpoints"][checkpoint["checkpoint_id"]]["status"]
    finally:
        await engine.stop()
    assert intent.ok is False
    assert stopped.ok is True
    assert status == "active"


def test_active_reply_prompt_has_no_dynamic_state() -> None:
    from attention.models import ActiveContext

    ctx = ActiveContext(
        checkpoint={
            "group_id": "g",
            "umo": "p:GroupMessage:g",
            "reason": "SECRET_CHECKPOINT_REASON",
        },
        topic={"summary": "SECRET_TOPIC_SUMMARY", "policy": "SECRET_POLICY"},
        window=[],
        persona="测试人格",
        policy_summary="SECRET_POLICY_SUMMARY",
    )
    prompt = build_active_reply_system_prompt(ctx, make_settings())
    assert "测试人格" in prompt
    assert "SECRET_" not in prompt
