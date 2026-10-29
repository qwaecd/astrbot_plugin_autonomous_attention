"""Core regression tests for the session-scoped attention model."""

from __future__ import annotations

import json
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
    ActiveContext,
    ModelResult,
    PassiveRequest,
    RecentMessage,
)
from attention.policy import PolicyManager
from attention.prompts import build_active_reply_system_prompt
from attention.settings import Settings
from attention.store import StateStore, StateVersionError, migrate_v1
from attention.validation import (
    extract_json_object,
    sanitize_reply,
    validate_passive_decision,
)


def make_settings(**overrides: Any) -> Settings:
    data = {
        "passive_mode": "full",
        "active_enabled": True,
        "group_allowlist": ["g", "g1", "g2"],
        "min_interval_minutes": 1,
        "max_interval_minutes": 120,
        "default_interval_minutes": 5,
        "max_defer_minutes": 120,
        "session_autonomous_cooldown_minutes": 0,
        "global_send_cooldown_minutes": 0,
        "daily_model_call_budget": 100,
        "daily_autonomous_reply_budget": 10,
        "daily_active_check_budget": 100,
        "session_autonomous_reply_budget": 6,
        "max_interests_per_session": 3,
        "interest_ttl_minutes": 120,
        "interest_check_limit": 3,
        "min_evaluation_interval_seconds": 0,
        "opportunity_ttl_minutes": 10,
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
        self.calls.append({"purpose": purpose, "umo": umo, "prompt": prompt})
        if purpose == "main":
            return self.main_response
        if not self.decision_responses:
            return ModelResult(
                text='{"suggested_action":"observe","reason":"empty queue"}'
            )
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
    store = StateStore(tmp_path / "state.json", max_decisions=50)
    engine = AttentionEngine(
        settings=settings,
        store=store,
        gateway=gateway,
        context_provider=context_provider,
        sender=sender,
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
    wake_source: str = "unknown",
) -> PassiveRequest:
    return PassiveRequest(
        umo=umo,
        platform_id="p",
        group_id=group_id,
        user_id=user_id,
        sender_name="用户",
        message_id=message_id,
        message_text=text,
        wake_source=wake_source,
        is_group=bool(group_id),
        prompt=text,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


async def add_message(
    engine: AttentionEngine,
    *,
    umo: str = "p:GroupMessage:g",
    group_id: str = "g",
    message_id: str = "m1",
    text: str = "消息",
    sender_id: str = "u1",
    sender_name: str = "用户",
) -> None:
    now = datetime.now(engine.settings.tz)
    await engine.observe_message(
        umo=umo,
        platform_id="p",
        group_id=group_id,
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
        message_id=message_id,
        created_at=now.isoformat(timespec="seconds"),
        created_ts=now.timestamp(),
    )


async def admit_interest(
    engine: AttentionEngine,
    *,
    umo: str = "p:GroupMessage:g",
    summary: str = "插件配置问题",
    evidence: list[str] | None = None,
    wait_for: str = "等待用户提供配置",
    source: str = SOURCE_ATTENTION_AGENT,
    priority_level: int = 0,
    group_id: str = "g",
) -> dict[str, Any]:
    actor = Actor(
        source=source,
        session=umo,
        group_id=group_id,
        platform_id="p",
        is_admin=source == SOURCE_ADMIN,
    )
    async with engine.store.lock:
        result = engine.control.admit_interest_locked(
            engine.store.state,
            actor,
            summary=summary,
            wait_for=wait_for,
            evidence_message_ids=evidence or ["m1"],
            source=source,
            priority_level=priority_level,
            moment=engine.policy.now(),
        )
        engine.store.flush()
    assert result.ok, result.message
    return dict(result.data["interest"])


def reply_suggestion(
    *,
    anchor: str = "m1",
    evidence: list[str] | None = None,
    contribution: str = "补充配置入口",
    interest_id: str = "",
) -> str:
    payload: dict[str, Any] = {
        "suggested_action": "reply",
        "reason": "可以补充具体信息",
        "opportunity": {
            "kind": "unanswered_question",
            "anchor_message_id": anchor,
            "evidence_message_ids": evidence or [anchor],
            "expected_contribution": contribution,
        },
    }
    if interest_id:
        payload["interest_id"] = interest_id
        payload["opportunity"]["interest_id"] = interest_id
    return json.dumps(payload)


def active_participate(
    *,
    anchor: str = "m1",
    evidence: list[str] | None = None,
    contribution: str = "补充配置入口",
) -> str:
    return json.dumps(
        {
            "action": "participate",
            "arguments": {},
            "reply_guidance": "简短回答",
            "reason": "有明确问题",
            "opportunity": {
                "kind": "unanswered_question",
                "anchor_message_id": anchor,
                "evidence_message_ids": evidence or [anchor],
                "expected_contribution": contribution,
            },
        }
    )


def test_validation_extracts_fenced_json_and_requires_evidence() -> None:
    raw = extract_json_object(
        '说明如下：\n```json\n{"suggested_action": "ignore", "reason": "不适配"}\n```'
    )
    assert raw is not None
    decision = validate_passive_decision(raw)
    assert decision is not None
    assert decision.suggested_action == "ignore"
    assert validate_passive_decision({"suggested_action": "reply"}) is None
    legacy = validate_passive_decision({"decision": "pass"})
    assert legacy is None
    valid = validate_passive_decision(json.loads(reply_suggestion()))
    assert valid is not None and valid.suggested_action == "reply"


def test_sanitize_reply_strips_fences_and_quotes() -> None:
    assert sanitize_reply('```\n"你好呀"\n```') == "你好呀"
    assert sanitize_reply("   ") is None
    assert sanitize_reply("一" * 30, 10) == "一" * 10


def test_settings_clamp_and_new_limits() -> None:
    settings = Settings.from_mapping(
        {
            "passive_mode": "unknown",
            "min_interval_minutes": 30,
            "max_interval_minutes": 10,
            "default_interval_minutes": 9999,
            "max_agent_rounds": 0,
            "attention_limits": {
                "max_interests_per_session": 5,
                "interest_ttl_minutes": 60,
                "min_evaluation_interval_seconds": 12,
            },
        }
    )
    assert settings.passive_mode == "block_only"
    assert settings.min_interval_minutes == 30
    assert settings.max_interval_minutes == 30
    assert settings.default_interval_minutes == 30
    assert settings.max_agent_rounds == 1
    assert settings.max_interests_per_session == 5
    assert settings.interest_ttl_minutes == 60
    assert settings.min_evaluation_interval_seconds == 12


def test_settings_reads_legacy_budget_and_cooldown_keys() -> None:
    settings = Settings.from_mapping(
        {
            "budgets": {"daily_active_message_budget": 4},
            "scheduling": {"group_send_cooldown_minutes": 7},
        }
    )
    assert settings.daily_autonomous_reply_budget == 4
    assert settings.session_autonomous_cooldown_minutes == 7


@pytest.mark.asyncio
async def test_store_migrates_v1_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "topics": {
                    "topic_x": {
                        "topic_id": "topic_x",
                        "umo": "p:GroupMessage:g",
                        "group_id": "g",
                        "summary": "旧话题",
                        "focus_status": "watching",
                        "updated_at": "2026-01-01T00:00:00+08:00",
                    }
                },
                "checkpoints": {
                    "cp1": {
                        "checkpoint_id": "cp1",
                        "umo": "p:GroupMessage:g",
                        "platform_id": "p",
                        "group_id": "g",
                        "topic_id": "topic_x",
                        "status": "active",
                        "created_by": "agent",
                        "scheduled_at": "2026-01-01T01:00:00+08:00",
                    }
                },
                "directives": {},
                "policies": {},
                "decisions": [],
                "budget": {"messages_sent": 2},
                "send_stats": {},
                "recent_messages": {},
                "seeded": [],
            }
        ),
        encoding="utf-8",
    )
    store = StateStore(path, max_decisions=50)
    await store.load()
    assert store.state["version"] == 2
    assert store.migrated is True
    assert store.state["interests"]["topic_x"]["umo"] == "p:GroupMessage:g"
    checkpoint = store.state["checkpoints"]["cp1"]
    assert checkpoint["interest_id"] == "topic_x"
    assert checkpoint["kind"] == "interest"
    assert store.state["budget"]["autonomous_attempts"] == 2
    assert list(tmp_path.glob("state.v1-*.json"))
    first = json.dumps(store.state, sort_keys=True)
    again = StateStore(path, max_decisions=50)
    await again.load()
    assert again.migrated is False
    assert json.dumps(again.state, sort_keys=True) == first


@pytest.mark.asyncio
async def test_store_rejects_newer_version(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 99}), encoding="utf-8")
    store = StateStore(path, max_decisions=50)
    with pytest.raises(StateVersionError):
        await store.load()


def test_migrate_v1_quarantines_topic_without_session() -> None:
    state = migrate_v1(
        {
            "topics": {
                "topic_orphan": {
                    "summary": "无会话",
                    "focus_status": "watching",
                }
            }
        }
    )
    interest = state["interests"]["topic_orphan"]
    assert interest["status"] == "paused"
    assert interest["quarantine"] is True


@pytest.mark.asyncio
async def test_engine_starts_from_v1_state(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "topics": {
                    "topic_old": {
                        "topic_id": "topic_old",
                        "umo": "p:GroupMessage:g",
                        "group_id": "g",
                        "summary": "旧兴趣",
                        "focus_status": "watching",
                    }
                },
                "checkpoints": {},
                "directives": {},
                "policies": {},
                "decisions": [],
                "budget": {},
                "send_stats": {},
                "recent_messages": {},
                "seeded": [],
            }
        ),
        encoding="utf-8",
    )
    store = StateStore(path, max_decisions=50)
    engine = AttentionEngine(
        settings=make_settings(),
        store=store,
        gateway=FakeGateway(),
        context_provider=FakeContextProvider(),
        sender=FakeSender(),
    )
    try:
        await engine.start()
        assert store.state["version"] == 2
        assert store.state["interests"]["topic_old"]["umo"] == "p:GroupMessage:g"
        assert store.migrated is True
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_second_autonomous_reply_hits_session_quota(tmp_path: Path) -> None:
    settings = make_settings(session_autonomous_reply_budget=1)
    engine, store, _ = await make_engine(
        tmp_path,
        settings=settings,
        gateway=FakeGateway(
            [
                ModelResult(text=reply_suggestion(anchor="m1")),
                ModelResult(text=reply_suggestion(anchor="m2")),
            ]
        ),
    )
    try:
        await add_message(engine, message_id="m1")
        await admit_interest(engine, evidence=["m1"])
        first = await engine.handle_passive(
            passive_request(message_id="m1", wake_source="unknown")
        )
        await add_message(engine, message_id="m2")
        async with store.lock:
            store.state["sessions"]["p:GroupMessage:g"]["last_decision_at"] = (
                datetime.now(engine.settings.tz) - timedelta(minutes=1)
            ).isoformat(timespec="seconds")
            store.state["sessions"]["p:GroupMessage:g"]["next_evaluation_at"] = ""
        second = await engine.handle_passive(
            passive_request(message_id="m2", wake_source="unknown")
        )
        session = store.state["sessions"]["p:GroupMessage:g"]
    finally:
        await engine.stop()
    assert first.block is False
    assert first.code == "autonomous_approved"
    assert second.block is True
    assert second.code == "session_quota_exhausted"
    assert session["autonomous_attempts"] == 1


@pytest.mark.asyncio
async def test_active_dry_run_restores_checkpoint_without_side_effects(
    tmp_path: Path,
) -> None:
    settings = make_settings(dry_run=True)
    now = datetime.now(timezone.utc)
    messages = [
        RecentMessage(
            role="user",
            sender_id="u1",
            sender_name="用户",
            text="这个插件怎么配置？",
            created_at=now.isoformat(timespec="seconds"),
            created_ts=now.timestamp(),
            message_id="m1",
        )
    ]
    gateway = FakeGateway([ModelResult(text=active_participate(anchor="m1"))])
    sender = FakeSender()
    engine, store, _ = await make_engine(
        tmp_path,
        settings=settings,
        gateway=gateway,
        context_provider=FakeContextProvider(messages),
        sender=sender,
    )
    try:
        await add_message(engine, message_id="m1")
        interest = await admit_interest(engine, evidence=["m1"])
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="test",
            interest_id=interest["interest_id"],
        )
        original_schedule = checkpoint["scheduled_at"]
        await engine.claim_checkpoint(checkpoint["checkpoint_id"], force=True)
        await engine.run_claimed(checkpoint["checkpoint_id"])
        raw = store.state["checkpoints"][checkpoint["checkpoint_id"]]
        interest_raw = store.state["interests"][interest["interest_id"]]
        opportunities = list(store.state["opportunities"].values())
        budget = store.state["budget"]
    finally:
        await engine.stop()
    assert sender.sent == []
    assert opportunities == []
    assert raw["status"] == "active"
    assert raw["scheduled_at"] == original_schedule
    assert raw["last_result"] == ""
    assert interest_raw["remaining_checks"] == 3
    assert budget["autonomous_attempts"] == 0


@pytest.mark.asyncio
async def test_due_checkpoints_in_one_session_are_merged(tmp_path: Path) -> None:
    engine, store, gateway = await make_engine(
        tmp_path,
        gateway=FakeGateway(
            [
                ModelResult(
                    text=json.dumps(
                        {
                            "action": "observe",
                            "arguments": {"minutes": 5},
                            "reason": "继续观察",
                        }
                    )
                )
            ]
        ),
    )
    try:
        await add_message(engine, message_id="m1")
        interest = await admit_interest(engine, evidence=["m1"])
        interest_checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="interest",
            interest_id=interest["interest_id"],
        )
        scan_checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="scan",
        )
        old_interest = (
            datetime.now(engine.settings.tz) - timedelta(minutes=10)
        ).isoformat(timespec="seconds")
        old_scan = (datetime.now(engine.settings.tz) - timedelta(minutes=11)).isoformat(
            timespec="seconds"
        )
        async with store.lock:
            store.state["checkpoints"][interest_checkpoint["checkpoint_id"]][
                "scheduled_at"
            ] = old_interest
            store.state["checkpoints"][scan_checkpoint["checkpoint_id"]][
                "scheduled_at"
            ] = old_scan
        claimed = await engine.claim_next_due()
        assert claimed is not None
        assert claimed["checkpoint_id"] == scan_checkpoint["checkpoint_id"]
        assert claimed["merged_interest_ids"] == [interest["interest_id"]]
        merged = store.state["checkpoints"][interest_checkpoint["checkpoint_id"]]
        assert merged["status"] == "active"
        assert merged["last_result"] == "merged_pending"
        await engine.run_claimed(scan_checkpoint["checkpoint_id"])
        interest_raw = store.state["interests"][interest["interest_id"]]
    finally:
        await engine.stop()
    assert len(gateway.calls) == 1
    assert interest_raw["remaining_checks"] == 3
    assert interest_raw["last_evaluated_seq"] == 0


@pytest.mark.asyncio
async def test_admin_checkpoint_status_requires_explicit_matching_session(
    tmp_path: Path,
) -> None:
    engine, _, _ = await make_engine(tmp_path)
    try:
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="scoped",
        )
        wrong, wrong_detail = await engine.set_checkpoint_status(
            checkpoint["checkpoint_id"],
            "paused",
            umo="p:GroupMessage:g1",
        )
        right, right_detail = await engine.set_checkpoint_status(
            checkpoint["checkpoint_id"],
            "paused",
            umo="p:GroupMessage:g",
        )
    finally:
        await engine.stop()
    assert wrong is False
    assert wrong_detail == "不能修改其他会话的检查点。"
    assert right is True
    assert right_detail == "paused"


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduled_at", ["", "2000-01-01T00:00:00+00:00"])
async def test_resume_paused_checkpoint_reschedules_missing_or_past_time(
    tmp_path: Path, scheduled_at: str
) -> None:
    engine, store, _ = await make_engine(tmp_path)
    await engine.scheduler.stop()
    try:
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
        )
        checkpoint_id = checkpoint["checkpoint_id"]
        paused, _ = await engine.set_checkpoint_status(
            checkpoint_id, "paused", umo=checkpoint["umo"]
        )
        assert paused
        store.state["checkpoints"][checkpoint_id]["scheduled_at"] = scheduled_at
        resumed, status = await engine.set_checkpoint_status(
            checkpoint_id, "active", umo=checkpoint["umo"]
        )
        assert resumed
        assert status == "active"
        scheduled = datetime.fromisoformat(
            store.state["checkpoints"][checkpoint_id]["scheduled_at"]
        )
        assert scheduled > engine.policy.now()
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_cleanup_removes_old_terminal_resources(tmp_path: Path) -> None:
    engine, store, _ = await make_engine(tmp_path)
    old = (datetime.now(engine.settings.tz) - timedelta(days=30)).isoformat(
        timespec="seconds"
    )
    try:
        async with store.lock:
            store.state["interests"]["i_old"] = {
                "interest_id": "i_old",
                "umo": "p:GroupMessage:g",
                "status": "closed",
                "updated_at": old,
            }
            store.state["checkpoints"]["cp_old"] = {
                "checkpoint_id": "cp_old",
                "umo": "p:GroupMessage:g",
                "status": "completed",
                "updated_at": old,
            }
            store.state["opportunities"]["op_old"] = {
                "opportunity_id": "op_old",
                "umo": "p:GroupMessage:g",
                "status": "consumed",
                "updated_at": old,
            }
            removed = engine.control.cleanup_locked(store.state, engine.policy.now())
    finally:
        await engine.stop()
    assert removed == {"interests": 1, "checkpoints": 1, "opportunities": 1}
    assert "i_old" not in store.state["interests"]
    assert "cp_old" not in store.state["checkpoints"]
    assert "op_old" not in store.state["opportunities"]


@pytest.mark.asyncio
async def test_checkpoint_listing_is_session_scoped(tmp_path: Path) -> None:
    engine, _, _ = await make_engine(tmp_path)
    try:
        first = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="first",
        )
        second = await engine.create_checkpoint(
            umo="p:GroupMessage:g1",
            platform_id="p",
            group_id="g1",
            minutes=5,
            reason="second",
        )
        session_items = await engine.list_checkpoints("p:GroupMessage:g")
        all_items = await engine.list_checkpoints("")
    finally:
        await engine.stop()
    assert [item["checkpoint_id"] for item in session_items] == [first["checkpoint_id"]]
    assert {item["checkpoint_id"] for item in all_items} == {
        first["checkpoint_id"],
        second["checkpoint_id"],
    }


@pytest.mark.asyncio
async def test_session_isolation_of_interests_and_queries(tmp_path: Path) -> None:
    engine, _, _ = await make_engine(tmp_path)
    try:
        await add_message(engine, umo="p:GroupMessage:g", message_id="m1")
        await add_message(
            engine,
            umo="p:GroupMessage:g2",
            group_id="g2",
            message_id="m2",
        )
        interest_a = await admit_interest(
            engine, umo="p:GroupMessage:g", evidence=["m1"], summary="A 兴趣"
        )
        await admit_interest(
            engine,
            umo="p:GroupMessage:g2",
            group_id="g2",
            evidence=["m2"],
            summary="B 兴趣",
        )
        actor_a = Actor(source=SOURCE_MAIN_AGENT, session="p:GroupMessage:g")
        actor_b = Actor(source=SOURCE_MAIN_AGENT, session="p:GroupMessage:g2")
        result_a = await engine.control.query(actor_a, scope="interests")
        result_b = await engine.control.query(actor_b, scope="interests")
    finally:
        await engine.stop()
    assert [item["interest_id"] for item in result_a.data["interests"]] == [
        interest_a["interest_id"]
    ]
    assert [item["summary"] for item in result_b.data["interests"]] == ["B 兴趣"]
    assert all(
        item["interest_id"] != interest_a["interest_id"]
        for item in result_b.data["interests"]
    )


@pytest.mark.asyncio
async def test_interest_capacity_and_replacement(tmp_path: Path) -> None:
    settings = make_settings(max_interests_per_session=2)
    engine, store, _ = await make_engine(tmp_path, settings=settings)
    try:
        await add_message(engine, message_id="m1")
        await admit_interest(engine, summary="兴趣 1", evidence=["m1"])
        await admit_interest(engine, summary="兴趣 2", evidence=["m1"])
        actor = Actor(source=SOURCE_ATTENTION_AGENT, session="p:GroupMessage:g")
        async with engine.store.lock:
            rejected = engine.control.admit_interest_locked(
                store.state,
                actor,
                summary="兴趣 3",
                wait_for="等待新证据",
                evidence_message_ids=["m1"],
            )
        assert rejected.ok is False
        assert rejected.code == "interest_capacity_exhausted"
        await admit_interest(
            engine,
            summary="主 Agent 兴趣",
            evidence=["m1"],
            source=SOURCE_MAIN_AGENT,
            priority_level=2,
        )
        active = [
            item
            for item in store.state["interests"].values()
            if item["status"] != "closed"
        ]
    finally:
        await engine.stop()
    assert len(active) == 2
    assert any(item["summary"] == "主 Agent 兴趣" for item in active)


@pytest.mark.asyncio
async def test_explicit_private_request_passes_without_model(tmp_path: Path) -> None:
    engine, store, gateway = await make_engine(tmp_path)
    try:
        outcome = await engine.handle_passive(
            passive_request(
                umo="p:FriendMessage:u1",
                group_id="",
                wake_source="private",
                text="帮我看看",
            )
        )
        decisions = list(store.state["decisions"])
    finally:
        await engine.stop()
    assert outcome.block is False
    assert outcome.code == "explicit_pass"
    assert gateway.calls == []
    assert decisions[-1]["code_decision"] == "explicit_pass"
    assert decisions[-1]["wake_source"] == "private"


@pytest.mark.asyncio
async def test_autonomous_reply_requires_valid_evidence(tmp_path: Path) -> None:
    engine, store, gateway = await make_engine(
        tmp_path,
        gateway=FakeGateway([ModelResult(text=reply_suggestion(anchor="missing"))]),
    )
    try:
        await add_message(engine, message_id="m1")
        await admit_interest(engine, evidence=["m1"])
        outcome = await engine.handle_passive(
            passive_request(wake_source="unknown", text="插件怎么配置")
        )
        opportunities = list(store.state["opportunities"].values())
    finally:
        await engine.stop()
    assert outcome.block is True
    assert outcome.code in {"invalid_anchor", "interest_required"}
    assert opportunities == []
    assert gateway.calls and gateway.calls[0]["purpose"] == "decision"


@pytest.mark.asyncio
async def test_autonomous_reply_with_opportunity_is_approved(tmp_path: Path) -> None:
    engine, store, gateway = await make_engine(
        tmp_path,
        gateway=FakeGateway([ModelResult(text=reply_suggestion(anchor="m1"))]),
    )
    try:
        await add_message(engine, message_id="m1")
        interest = await admit_interest(engine, evidence=["m1"])
        outcome = await engine.handle_passive(
            passive_request(
                wake_source="unknown",
                text="插件怎么配置",
            )
        )
        session = store.state["sessions"]["p:GroupMessage:g"]
        opportunity = next(iter(store.state["opportunities"].values()))
        approved_status = opportunity["status"]
        await engine.complete_passive_generation(
            "p:GroupMessage:g", outcome.reservation_id
        )
        consumed = store.state["opportunities"][opportunity["opportunity_id"]]
    finally:
        await engine.stop()
    assert outcome.block is False
    assert outcome.code == "autonomous_approved"
    assert interest["interest_id"] == opportunity["interest_id"]
    assert approved_status == "consumed"
    assert session["autonomous_attempts"] == 1
    assert session["pending_reply_reservation"] == ""
    assert consumed["status"] == "consumed"


@pytest.mark.asyncio
async def test_focus_does_not_force_pass(tmp_path: Path) -> None:
    engine, store, gateway = await make_engine(
        tmp_path,
        gateway=FakeGateway(
            [
                ModelResult(
                    text=json.dumps(
                        {
                            "suggested_action": "observe",
                            "reason": "没有新的具体贡献",
                        }
                    )
                )
            ]
        ),
    )
    main = Actor(
        source=SOURCE_MAIN_AGENT,
        session="p:GroupMessage:g",
        group_id="g",
        platform_id="p",
        is_admin=True,
    )
    try:
        await add_message(engine, message_id="m1", text="插件怎么配置")
        intent = await engine.control.set_intent(
            main,
            intent_type="focus_topic",
            target="插件",
            reason="关注插件问题",
        )
        outcome = await engine.handle_passive(
            passive_request(wake_source="unknown", text="插件怎么配置")
        )
        directives = list(store.state["directives"].values())
    finally:
        await engine.stop()
    assert intent.ok
    assert outcome.block is True
    assert outcome.decision == "observe"
    assert len(gateway.calls) == 1
    assert any(item["type"] == "focus_topic" for item in directives)


@pytest.mark.asyncio
async def test_active_participation_sends_and_consumes_opportunity(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    messages = [
        RecentMessage(
            role="user",
            sender_id="u1",
            sender_name="用户",
            text="这个插件怎么配置？",
            created_at=now.isoformat(timespec="seconds"),
            created_ts=now.timestamp(),
            message_id="m1",
        )
    ]
    gateway = FakeGateway([ModelResult(text=active_participate(anchor="m1"))])
    sender = FakeSender()
    engine, store, _ = await make_engine(
        tmp_path,
        gateway=gateway,
        context_provider=FakeContextProvider(messages),
        sender=sender,
    )
    try:
        await add_message(engine, message_id="m1", text="这个插件怎么配置？")
        interest = await admit_interest(engine, evidence=["m1"])
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="test",
            interest_id=interest["interest_id"],
        )
        claimed, reason = await engine.claim_checkpoint(
            checkpoint["checkpoint_id"], force=True
        )
        assert claimed is not None, reason
        await engine.run_claimed(checkpoint["checkpoint_id"])
        raw = store.state["checkpoints"][checkpoint["checkpoint_id"]]
        opportunity = next(iter(store.state["opportunities"].values()))
        session = store.state["sessions"]["p:GroupMessage:g"]
    finally:
        await engine.stop()
    assert sender.sent == [("p:GroupMessage:g", "主动回复内容")]
    assert raw["status"] == "active"
    assert raw["last_result"] == "participate"
    assert opportunity["status"] == "consumed"
    assert opportunity["outcome"] == "sent"
    assert session["autonomous_attempts"] == 1


@pytest.mark.asyncio
async def test_active_skips_model_without_new_messages(tmp_path: Path) -> None:
    engine, store, gateway = await make_engine(tmp_path)
    try:
        await add_message(engine, message_id="m1")
        interest = await admit_interest(engine, evidence=["m1"])
        async with store.lock:
            store.state["interests"][interest["interest_id"]]["last_evaluated_seq"] = 1
            store.state["sessions"]["p:GroupMessage:g"]["last_decision_seq"] = 1
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="test",
            interest_id=interest["interest_id"],
        )
        async with store.lock:
            store.state["checkpoints"][checkpoint["checkpoint_id"]]["scheduled_at"] = (
                datetime.now(engine.settings.tz) - timedelta(minutes=1)
            ).isoformat(timespec="seconds")
        claimed, reason = await engine.claim_checkpoint(
            checkpoint["checkpoint_id"], force=False
        )
        assert claimed is not None, reason
        await engine.run_claimed(checkpoint["checkpoint_id"])
        raw = store.state["checkpoints"][checkpoint["checkpoint_id"]]
    finally:
        await engine.stop()
    assert gateway.calls == []
    assert raw["last_result"] == "skip"
    assert raw["last_error"] == "no_new_messages"
    assert raw["no_progress_count"] == 1


@pytest.mark.asyncio
async def test_dry_run_has_no_resource_side_effects(tmp_path: Path) -> None:
    settings = make_settings(dry_run=True)
    engine, store, gateway = await make_engine(
        tmp_path,
        settings=settings,
        gateway=FakeGateway([ModelResult(text=reply_suggestion(anchor="m1"))]),
    )
    try:
        await add_message(engine, message_id="m1")
        await admit_interest(engine, evidence=["m1"])
        outcome = await engine.handle_passive(
            passive_request(wake_source="unknown", text="插件怎么配置")
        )
        opportunities = list(store.state["opportunities"].values())
        session = store.state["sessions"]["p:GroupMessage:g"]
    finally:
        await engine.stop()
    assert outcome.dry_run is True
    assert outcome.block is False
    assert gateway.calls[0]["purpose"] == "decision"
    assert opportunities == []
    assert session["autonomous_attempts"] == 0


@pytest.mark.asyncio
async def test_interest_expiry_closes_interest_and_checkpoint(
    tmp_path: Path,
) -> None:
    engine, store, _ = await make_engine(tmp_path)
    try:
        await add_message(engine, message_id="m1")
        interest = await admit_interest(engine, evidence=["m1"])
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="test",
            interest_id=interest["interest_id"],
        )
        async with store.lock:
            store.state["interests"][interest["interest_id"]]["expires_at"] = (
                datetime.now(engine.settings.tz) - timedelta(minutes=1)
            ).isoformat(timespec="seconds")
            engine.control.expire_interests_locked(store.state, engine.policy.now())
        interest_raw = store.state["interests"][interest["interest_id"]]
        checkpoint_raw = store.state["checkpoints"][checkpoint["checkpoint_id"]]
    finally:
        await engine.stop()
    assert interest_raw["status"] == "closed"
    assert checkpoint_raw["status"] == "cancelled"


@pytest.mark.asyncio
async def test_active_end_focus_completes_and_closes(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    messages = [
        RecentMessage(
            role="user",
            sender_id="u1",
            sender_name="用户",
            text="随便聊聊",
            created_at=now.isoformat(timespec="seconds"),
            created_ts=now.timestamp(),
            message_id="m1",
        )
    ]
    engine, store, _ = await make_engine(
        tmp_path,
        gateway=FakeGateway(
            [
                ModelResult(
                    text=json.dumps({"action": "end_focus", "reason": "话题已经结束"})
                )
            ]
        ),
        context_provider=FakeContextProvider(messages),
    )
    try:
        await add_message(engine, message_id="m1")
        interest = await admit_interest(engine, evidence=["m1"])
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="test",
            interest_id=interest["interest_id"],
        )
        await engine.claim_checkpoint(checkpoint["checkpoint_id"], force=True)
        await engine.run_claimed(checkpoint["checkpoint_id"])
        raw = store.state["checkpoints"][checkpoint["checkpoint_id"]]
        interest_raw = store.state["interests"][interest["interest_id"]]
    finally:
        await engine.stop()
    assert raw["status"] == "completed"
    assert raw["last_result"] == "end_focus"
    assert interest_raw["status"] == "closed"


@pytest.mark.asyncio
async def test_active_main_model_failure_does_not_send(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    messages = [
        RecentMessage(
            role="user",
            sender_id="u1",
            sender_name="用户",
            text="这个插件怎么配置？",
            created_at=now.isoformat(timespec="seconds"),
            created_ts=now.timestamp(),
            message_id="m1",
        )
    ]
    gateway = FakeGateway([ModelResult(text=active_participate(anchor="m1"))])
    gateway.main_response = ModelResult(error="main down")
    sender = FakeSender()
    engine, store, _ = await make_engine(
        tmp_path,
        gateway=gateway,
        context_provider=FakeContextProvider(messages),
        sender=sender,
    )
    try:
        await add_message(engine, message_id="m1")
        interest = await admit_interest(engine, evidence=["m1"])
        checkpoint = await engine.create_checkpoint(
            umo="p:GroupMessage:g",
            platform_id="p",
            group_id="g",
            minutes=5,
            reason="test",
            interest_id=interest["interest_id"],
        )
        await engine.claim_checkpoint(checkpoint["checkpoint_id"], force=True)
        await engine.run_claimed(checkpoint["checkpoint_id"])
        raw = store.state["checkpoints"][checkpoint["checkpoint_id"]]
        opportunity = next(iter(store.state["opportunities"].values()))
    finally:
        await engine.stop()
    assert sender.sent == []
    assert raw["last_result"] == "observe"
    assert opportunity["status"] == "consumed"


@pytest.mark.asyncio
async def test_quota_exhaustion_rejects_autonomous_reply(tmp_path: Path) -> None:
    settings = make_settings(session_autonomous_reply_budget=0)
    engine, store, _ = await make_engine(
        tmp_path,
        settings=settings,
        gateway=FakeGateway([ModelResult(text=reply_suggestion(anchor="m1"))]),
    )
    try:
        await add_message(engine, message_id="m1")
        await admit_interest(engine, evidence=["m1"])
        outcome = await engine.handle_passive(
            passive_request(wake_source="unknown", text="插件怎么配置")
        )
        opportunities = list(store.state["opportunities"].values())
    finally:
        await engine.stop()
    assert outcome.block is True
    assert outcome.code == "quota_disabled"
    assert all(item["status"] == "expired" for item in opportunities)


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
        outcome = await engine.handle_passive(passive_request(wake_source="unknown"))
    finally:
        await engine.stop()
    assert outcome.block is True
    assert outcome.decision == "ignore"
    assert gateway.calls == []


def test_policy_quiet_hours_and_group_matching() -> None:
    settings = make_settings(quiet_hours="12:00-13:00", group_allowlist=["g1"])
    policy = PolicyManager(settings)
    tz = settings.tz
    assert policy.is_quiet(datetime(2026, 1, 1, 12, 30, tzinfo=tz))
    assert not policy.is_quiet(datetime(2026, 1, 1, 14, 30, tzinfo=tz))
    assert policy.group_allowed("g1", "p:GroupMessage:g1")
    assert not policy.group_allowed("g2", "p:GroupMessage:g2")


def test_policy_accepts_injected_clock() -> None:
    fixed = datetime(2026, 2, 3, 4, 5, tzinfo=make_settings().tz)
    policy = PolicyManager(make_settings(), lambda: fixed)
    assert policy.now() == fixed


def test_group_allowlist_is_the_single_boundary() -> None:
    empty = make_settings(group_allowlist=[])
    assert not PolicyManager(empty).group_allowed("g1", "p:GroupMessage:g1")
    explicit = make_settings(group_allowlist=["g1"])
    assert PolicyManager(explicit).group_allowed("g1", "p:GroupMessage:g1")
    assert not PolicyManager(explicit).group_allowed("g2", "p:GroupMessage:g2")


@pytest.mark.asyncio
async def test_query_rejects_unknown_scope(tmp_path: Path) -> None:
    engine, _, _ = await make_engine(tmp_path)
    actor = Actor(source=SOURCE_MAIN_AGENT, session="p:GroupMessage:g")
    try:
        result = await engine.control.query(actor, scope="global")
    finally:
        await engine.stop()
    assert result.ok is False
    assert result.code == "invalid_scope"


def test_active_reply_prompt_has_no_dynamic_state() -> None:
    ctx = ActiveContext(
        checkpoint={
            "group_id": "g",
            "umo": "p:GroupMessage:g",
            "reason": "SECRET_CHECKPOINT_REASON",
        },
        interest={"summary": "SECRET_INTEREST_SUMMARY"},
        interests=[{"summary": "SECRET_INTEREST_SUMMARY"}],
        window=[],
        persona="测试人格",
        policy_summary="SECRET_POLICY_SUMMARY",
    )
    prompt = build_active_reply_system_prompt(ctx, make_settings())
    assert "测试人格" in prompt
    assert "SECRET_" not in prompt
