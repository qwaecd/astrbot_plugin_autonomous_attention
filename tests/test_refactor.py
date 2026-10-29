"""Behavioral regressions for approval, isolation and finite attention resources."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from attention.context import merge_window
from attention.control import Actor
from attention.models import SOURCE_MAIN_AGENT, ModelResult, RecentMessage
from attention.settings import Settings
from attention.store import StateStore, migrate_v1
from test_core import (
    FakeGateway,
    FakeSender,
    active_participate,
    add_message,
    admit_interest,
    make_engine,
    make_settings,
    passive_request,
    reply_suggestion,
)


class HookGateway(FakeGateway):
    """Change real inputs while an asynchronous model request is outstanding."""

    hook = None
    hook_purpose = "main"

    async def generate(self, **kwargs):
        if self.hook is not None and kwargs["purpose"] == self.hook_purpose:
            await self.hook()
        return await super().generate(**kwargs)


async def active_case(tmp_path, gateway=None, settings=None, sender=None):
    engine, store, gateway = await make_engine(
        tmp_path, gateway=gateway, settings=settings, sender=sender
    )
    await engine.scheduler.stop()
    await add_message(engine)
    interest = await admit_interest(engine)
    checkpoint = await engine.create_checkpoint(
        umo="p:GroupMessage:g",
        platform_id="p",
        group_id="g",
        minutes=5,
        reason="regression",
        interest_id=interest["interest_id"],
    )
    return engine, store, gateway, interest, checkpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["decision", "main"])
@pytest.mark.parametrize("mutation", ["message", "policy", "stop", "config", "expiry"])
async def test_async_active_results_are_revalidated(tmp_path, phase, mutation):
    gateway = HookGateway([ModelResult(text=active_participate())])
    sender = FakeSender()
    engine, store, _, interest, cp = await active_case(tmp_path, gateway, sender=sender)
    gateway.hook_purpose = phase

    async def mutate():
        if mutation == "message":
            await add_message(engine, message_id="m2")
        elif mutation == "policy":
            await engine.set_policy(
                scope="group_user",
                group_id="g",
                user_id="u1",
                reason="stop",
                expires_minutes=60,
            )
        elif mutation == "stop":
            await engine.control.stop_focus(
                Actor(source=SOURCE_MAIN_AGENT, session=cp["umo"]),
                target=interest["interest_id"],
            )
        elif mutation == "config":
            engine.update_settings(make_settings(group_allowlist=["g2"]))
        else:
            future = engine.policy.now() + timedelta(hours=3)
            engine.policy._clock = lambda: future

    gateway.hook = mutate
    try:
        assert (await engine.claim_checkpoint(cp["checkpoint_id"], force=True))[0]
        await engine.run_claimed(cp["checkpoint_id"])
        assert sender.sent == []
        if phase == "decision":
            assert not any(call["purpose"] == "main" for call in gateway.calls)
        assert not any(
            op["status"] == "reserved" for op in store.state["opportunities"].values()
        )
        assert not store.state["sessions"][cp["umo"]].get("pending_reply_reservation")
    finally:
        await engine.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["message", "policy", "stop", "config"])
async def test_passive_result_cannot_commit_over_changed_inputs(tmp_path, mutation):
    gateway = HookGateway()
    engine, store, _, interest, cp = await active_case(tmp_path, gateway)
    gateway.decision_responses = [
        ModelResult(text=reply_suggestion(interest_id=interest["interest_id"]))
    ]
    gateway.hook_purpose = "decision"

    async def mutate():
        if mutation == "message":
            await add_message(engine, message_id="m2")
        elif mutation == "policy":
            await engine.set_policy(
                scope="group_user",
                group_id="g",
                user_id="u1",
                reason="stop",
                expires_minutes=60,
            )
        elif mutation == "stop":
            await engine.control.stop_focus(
                Actor(source=SOURCE_MAIN_AGENT, session=cp["umo"]),
                target=interest["interest_id"],
            )
        else:
            engine.update_settings(make_settings(passive_mode="block_only"))

    gateway.hook = mutate
    try:
        outcome = await engine.handle_passive(passive_request())
        assert outcome.block
        assert outcome.code in {"stale_snapshot", "policy_changed"}
        assert store.state["budget"]["autonomous_attempts"] == 0
        assert store.state["sessions"][cp["umo"]]["last_decision_seq"] == 0
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_isolated_umo_and_targeted_stop_keep_other_interests(tmp_path):
    engine, store, _ = await make_engine(tmp_path)
    await engine.scheduler.stop()
    umo = "p:GroupMessage:u1_g"
    try:
        await add_message(engine, umo=umo)
        first = await admit_interest(
            engine, umo=umo, source=SOURCE_MAIN_AGENT, summary="first"
        )
        second = await admit_interest(
            engine, umo=umo, source=SOURCE_MAIN_AGENT, summary="second"
        )
        cp = await engine.create_checkpoint(
            umo=umo,
            platform_id="p",
            group_id="g",
            interest_id=first["interest_id"],
            minutes=5,
        )
        assert (await engine.claim_checkpoint(cp["checkpoint_id"], force=True))[0]
        result = await engine.control.stop_focus(
            Actor(source=SOURCE_MAIN_AGENT, session=umo, platform_id="p", group_id="g"),
            target=first["interest_id"],
        )
        assert result.ok
        assert store.state["interests"][first["interest_id"]]["status"] == "closed"
        assert store.state["interests"][second["interest_id"]]["status"] == "watching"
        assert store.state["checkpoints"][cp["checkpoint_id"]]["status"] == "cancelled"
        assert engine._resolve_target(umo, "p", "")[2] == "g"
        with pytest.raises(ValueError):
            engine._resolve_target(umo, "other", "g")
        with pytest.raises(ValueError):
            engine._resolve_target(umo, "p", "g2")
    finally:
        await engine.stop()


def test_history_merges_to_observed_identity_without_collapsing_live_messages():
    now = datetime.now(timezone.utc)
    live = RecentMessage(
        message_id="platform:1",
        sender_id="u",
        text="same",
        created_ts=now.timestamp(),
        seq=1,
    )
    history = RecentMessage(
        message_id="history:42",
        sender_id="u",
        text="same",
        created_ts=now.timestamp(),
        source="history",
    )
    second = RecentMessage(
        message_id="platform:2",
        sender_id="u",
        text="same",
        created_ts=now.timestamp(),
        seq=2,
    )
    merged = merge_window([history], [live, second], 10)
    assert [item.message_id for item in merged] == ["platform:1", "platform:2"]
    assert "history:42" in merged[0].aliases


@pytest.mark.asyncio
async def test_old_or_unknown_anchor_never_becomes_fresh(tmp_path):
    engine, store, _, interest, _ = await active_case(tmp_path)
    try:
        async with store.lock:
            for anchor in ("db42", "m1"):
                if anchor == "m1":
                    store.state["recent_messages"]["p:GroupMessage:g"][0][
                        "created_at"
                    ] = (engine.policy.now() - timedelta(hours=1)).isoformat()
                approval = engine.control.approve_participation_locked(
                    store.state,
                    umo="p:GroupMessage:g",
                    wake_source="ambient",
                    suggested_action="participate",
                    opportunity_payload=json.loads(active_participate(anchor=anchor))[
                        "opportunity"
                    ],
                    interest_id=interest["interest_id"],
                    valid_ids={anchor},
                    bot_ids=set(),
                    moment=engine.policy.now(),
                )
                assert not approval.approved
                assert approval.code in {
                    "invalid_anchor",
                    "opportunity_expired",
                    "invalid_evidence",
                }
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_dry_run_progress_quota_and_mode_switch_are_isolated(tmp_path):
    gateway = FakeGateway([ModelResult(text=active_participate())])
    engine, store, _, interest, cp = await active_case(
        tmp_path, gateway, make_settings(dry_run=True)
    )
    original = deepcopy(store.state["checkpoints"])
    try:
        assert (await engine.claim_checkpoint(cp["checkpoint_id"], force=True))[0]
        await engine.run_claimed(cp["checkpoint_id"])
        assert store.state["checkpoints"] == original
        assert store.state["opportunities"] == {}
        assert store.state["budget"]["decision_calls"] == 1
        assert store.state["budget"]["autonomous_attempts"] == 0
        assert await engine.claim_next_due() is None
        with store.simulation(True):
            assert store.state["sessions"][cp["umo"]]["last_decision_seq"] == 1
            assert store.state["sessions"][cp["umo"]]["autonomous_attempts"] == 1
            assert (
                store.state["interests"][interest["interest_id"]]["remaining_checks"]
                == 2
            )
        engine.update_settings(make_settings(dry_run=False))
        assert store._shadow is None
        assert store.state["sessions"][cp["umo"]]["autonomous_attempts"] == 0
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_dry_run_passive_uses_shadow_quota_before_model(tmp_path):
    gateway = FakeGateway([ModelResult(text=reply_suggestion())])
    engine, store, _, _, cp = await active_case(
        tmp_path,
        gateway,
        make_settings(dry_run=True, session_autonomous_reply_budget=1),
    )
    try:
        first = await engine.handle_passive(passive_request())
        assert first.dry_run and not first.block and first.code == "autonomous_approved"
        await add_message(engine, message_id="m2")
        with store.simulation(True):
            store.state["sessions"][cp["umo"]]["next_evaluation_at"] = ""
            store.state["sessions"][cp["umo"]]["last_decision_at"] = ""
        second = await engine.handle_passive(passive_request(message_id="m2"))
        assert not second.block and second.code == "session_quota_exhausted"
        assert len(gateway.calls) == 1
        assert store.state["opportunities"] == {}
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_observe_admits_and_only_new_evidence_renews_interest(tmp_path):
    update = {
        "operation": "update",
        "summary": "topic",
        "wait_for": "new results",
        "evidence_message_ids": ["m1"],
    }
    gateway = FakeGateway(
        [
            ModelResult(
                text=json.dumps(
                    {"suggested_action": "observe", "interest_update": update}
                )
            )
        ]
    )
    engine, store, _ = await make_engine(tmp_path, gateway=gateway)
    await engine.scheduler.stop()
    try:
        await engine.handle_passive(passive_request())
        interest = next(iter(store.state["interests"].values()))
        assert interest["remaining_checks"] == 2
        assert interest["group_id"] == "g"
        assert len(store.state["checkpoints"]) == 1
        original_expiry = interest["expires_at"]
        original_revision = interest["revision"]
        assert not engine.control.touch_interest_evidence_locked(
            store.state, interest["interest_id"], ["m1", "forged"]
        )
        assert interest["revision"] == original_revision
        await add_message(engine, message_id="m2")
        future = engine.policy.now() + timedelta(minutes=1)
        engine.policy._clock = lambda: future
        assert engine.control.touch_interest_evidence_locked(
            store.state, interest["interest_id"], ["m2"]
        )
        assert interest["remaining_checks"] == 3
        assert interest["expires_at"] > original_expiry
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_session_eval_and_reply_reservations_are_exclusive(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    gateway = HookGateway()
    engine, store, _, interest, cp = await active_case(tmp_path, gateway)
    gateway.hook_purpose = "decision"
    gateway.decision_responses = [
        ModelResult(text=reply_suggestion(interest_id=interest["interest_id"]))
    ]

    async def hold():
        entered.set()
        await release.wait()

    gateway.hook = hold
    try:
        task = asyncio.create_task(engine.handle_passive(passive_request()))
        await entered.wait()
        second = await engine.handle_passive(passive_request(message_id="m2"))
        assert second.code in {"session_busy", "evaluation_throttled"}
        assert (await engine.claim_checkpoint(cp["checkpoint_id"], force=True))[
            1
        ] == "session_busy"
        release.set()
        assert (await task).code == "stale_snapshot"
        assert len(gateway.calls) == 1
    finally:
        release.set()
        await engine.stop()


@pytest.mark.asyncio
async def test_unreadable_state_is_preserved(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("broken{", encoding="utf-8")
    with pytest.raises(RuntimeError):
        await StateStore(path).load()
    assert path.read_text(encoding="utf-8") == "broken{"


def test_migration_bounds_scans_and_quarantines_ambiguous_owners():
    state = migrate_v1(
        {
            "version": 1,
            "topics": {"topic": {"summary": "ambiguous"}},
            "checkpoints": {
                "a": {
                    "umo": "p:GroupMessage:g",
                    "platform_id": "p",
                    "group_id": "g",
                    "topic_id": "topic",
                },
                "b": {
                    "umo": "p:GroupMessage:g2",
                    "platform_id": "p",
                    "group_id": "g2",
                    "topic_id": "topic",
                },
                "scan": {
                    "umo": "p:GroupMessage:g",
                    "platform_id": "p",
                    "group_id": "g",
                    "topic_id": "missing",
                },
                "conflict": {
                    "umo": "q:GroupMessage:g",
                    "platform_id": "p",
                    "group_id": "g",
                },
            },
            "directives": {"d": {"type": "focus_group", "target": "p:GroupMessage:g"}},
        }
    )
    assert state["interests"]["topic"]["quarantine"]
    assert state["checkpoints"]["a"]["quarantine"]
    assert state["checkpoints"]["b"]["quarantine"]
    assert state["checkpoints"]["conflict"]["quarantine"]
    assert state["checkpoints"]["scan"]["remaining_checks"] == 3
    assert state["checkpoints"]["scan"]["expires_at"]
    assert state["directives"]["d"]["created_by_session"] == "p:GroupMessage:g"


@pytest.mark.parametrize("flat", [True, False])
def test_schema_defaults_preserve_legacy_configuration(tmp_path, flat):
    from astrbot.core.config.astrbot_config import AstrBotConfig

    path = tmp_path / "config.json"
    old_config = (
        {"daily_active_message_budget": 2, "group_send_cooldown_minutes": 90}
        if flat
        else {
            "budgets": {"daily_active_message_budget": 2},
            "scheduling": {"group_send_cooldown_minutes": 90},
        }
    )
    path.write_text(json.dumps(old_config), encoding="utf-8")
    schema = json.loads(
        (Path(__file__).resolve().parent.parent / "_conf_schema.json").read_text(
            encoding="utf-8"
        )
    )
    config = AstrBotConfig(config_path=str(path), schema=schema)
    settings = Settings.from_mapping(config)
    assert settings.daily_autonomous_reply_budget == 2
    assert settings.session_autonomous_cooldown_minutes == 90
    config["attention_limits"]["daily_autonomous_reply_budget"] = 10
    assert Settings.from_mapping(config).daily_autonomous_reply_budget == 10


@pytest.mark.asyncio
async def test_adapter_history_has_distinct_source_and_preserves_time():
    from test_adapter import AstrBotContextProvider

    provider = AstrBotContextProvider(SimpleNamespace(), make_settings())
    record = SimpleNamespace(
        id=42,
        sender_id="u",
        content={"message": [{"type": "plain", "text": "old"}]},
        created_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    message = provider._record_to_message(record)
    assert message.source == "history"
    assert message.message_id == "history:42"
    assert message.created_ts == record.created_at.timestamp()
    record.created_at = None
    assert provider._record_to_message(record) is None


@pytest.mark.asyncio
async def test_last_interest_check_can_send_then_closes(tmp_path):
    gateway = FakeGateway([ModelResult(text=active_participate())])
    sender = FakeSender()
    engine, store, _, interest, cp = await active_case(tmp_path, gateway, sender=sender)
    try:
        store.state["interests"][interest["interest_id"]]["remaining_checks"] = 1
        store.state["checkpoints"][cp["checkpoint_id"]]["remaining_checks"] = 1
        await engine.claim_checkpoint(cp["checkpoint_id"], force=True)
        await engine.run_claimed(cp["checkpoint_id"])
        assert len(sender.sent) == 1
        assert store.state["interests"][interest["interest_id"]]["status"] == "closed"
        assert store.state["checkpoints"][cp["checkpoint_id"]]["status"] in {
            "cancelled",
            "completed",
        }
        metrics = (await engine.status_snapshot(cp["umo"]))["metrics"]
        assert metrics["received_messages"] == 1
        assert metrics["semantic_decisions"] == 1
        assert metrics["suggestions"] == 1
        assert metrics["approved_reservations"] == 1
        assert metrics["active_sends"] == 1
        assert any(
            record["main_model"] == "generated" and record["send"] == "sent"
            for record in store.state["decisions"]
        )
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_send_exception_is_uncertain_and_cannot_retry(tmp_path):
    class UncertainSender(FakeSender):
        async def send_text(self, umo, text):
            raise TimeoutError("delivery unknown")

    gateway = FakeGateway([ModelResult(text=active_participate())])
    engine, store, _, _, cp = await active_case(
        tmp_path, gateway, sender=UncertainSender()
    )
    try:
        await engine.claim_checkpoint(cp["checkpoint_id"], force=True)
        await engine.run_claimed(cp["checkpoint_id"])
        opportunity = next(iter(store.state["opportunities"].values()))
        assert opportunity["status"] == "uncertain"
        assert opportunity["outcome"] == "uncertain_delivery"
        assert store.state["budget"]["autonomous_attempts"] == 1
        gateway.decision_responses.append(ModelResult(text=active_participate()))
        await engine.claim_checkpoint(cp["checkpoint_id"], force=True)
        await engine.run_claimed(cp["checkpoint_id"])
        assert store.state["budget"]["main_calls"] == 1
        assert store.state["budget"]["autonomous_attempts"] == 1
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_reservation_write_failure_prevents_generation(tmp_path, monkeypatch):
    gateway = FakeGateway([ModelResult(text=active_participate())])
    engine, store, _, _, cp = await active_case(tmp_path, gateway)
    await engine.claim_checkpoint(cp["checkpoint_id"], force=True)
    original_flush = store.flush

    def fail_reservation():
        if store.state["sessions"][cp["umo"]].get("pending_reply_reservation"):
            raise OSError("disk full")
        original_flush()

    monkeypatch.setattr(store, "flush", fail_reservation)
    try:
        with pytest.raises(OSError):
            await engine.run_claimed(cp["checkpoint_id"])
        assert not any(call["purpose"] == "main" for call in gateway.calls)
    finally:
        monkeypatch.setattr(store, "flush", original_flush)
        await engine.stop()


@pytest.mark.asyncio
async def test_session_scan_exits_after_three_empty_polls(tmp_path):
    engine, store, gateway = await make_engine(tmp_path)
    await engine.scheduler.stop()
    try:
        cp = await engine.create_checkpoint(
            umo="p:GroupMessage:g", platform_id="p", group_id="g", minutes=5
        )
        for _ in range(3):
            await engine.claim_checkpoint(cp["checkpoint_id"], force=True)
            await engine.run_claimed(cp["checkpoint_id"])
        assert store.state["checkpoints"][cp["checkpoint_id"]]["status"] == "completed"
        assert gateway.calls == []
        assert store.state["budget"]["active_checks"] == 0
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_active_quota_exhaustion_skips_decision_model(tmp_path):
    engine, store, gateway, _, cp = await active_case(
        tmp_path, settings=make_settings(session_autonomous_reply_budget=0)
    )
    try:
        await engine.claim_checkpoint(cp["checkpoint_id"], force=True)
        await engine.run_claimed(cp["checkpoint_id"])
        assert gateway.calls == []
        assert (
            store.state["checkpoints"][cp["checkpoint_id"]]["last_error"]
            == "quota_disabled"
        )
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_direct_pass_request_is_idempotent_and_not_a_semantic_decision(tmp_path):
    engine, store, gateway = await make_engine(tmp_path)
    await engine.scheduler.stop()
    try:
        request = passive_request(wake_source="wake_prefix")
        first = await engine.handle_passive(request)
        assert not first.block
        assert (await engine.handle_passive(request)).block
        assert gateway.calls == []
        metrics = (await engine.status_snapshot(request.umo))["metrics"]
        assert metrics["received_messages"] == 1
        assert metrics["explicit_passes"] == 1
        assert metrics["semantic_decisions"] == 0
        assert metrics["approved_reservations"] == 0
        await engine.complete_passive_generation(
            request.umo, first.reservation_id, "answer"
        )
        assert store.state["decisions"][-1]["main_model"] == "generated"
        assert store.state["decisions"][-1]["send"] == "none"
        await engine.stop()
        await engine.start()
        await engine.scheduler.stop()
        assert (await engine.handle_passive(request)).block
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_evicted_live_message_cannot_reappear_as_history_anchor(tmp_path):
    class HistoryProvider:
        async def window(self, *, recent, limit, **kwargs):
            return merge_window([RecentMessage(**history.to_dict())], recent, limit)

        async def persona(self, umo):
            return "persona"

    engine, store, gateway, interest, cp = await active_case(tmp_path)
    original = store.recent_window(cp["umo"])[0]
    history = RecentMessage.from_dict(original.to_dict())
    history.message_id, history.source, history.seq = "history:42", "history", 0
    engine.context_provider = HistoryProvider()
    try:
        await engine.handle_passive(passive_request(wake_source="wake_prefix"))
        for index in range(70):
            await add_message(
                engine, message_id=f"new{index}", text=f"different {index}"
            )
        assert "m1" not in {
            message.message_id for message in store.recent_window(cp["umo"])
        }
        gateway.decision_responses = [
            ModelResult(text=active_participate(anchor="history:42"))
        ]
        original_seq = store.state["sessions"][cp["umo"]]["external_message_seq"]
        await engine.claim_checkpoint(cp["checkpoint_id"], force=True)
        await engine.run_claimed(cp["checkpoint_id"])
        assert (
            store.state["sessions"][cp["umo"]]["external_message_seq"] == original_seq
        )
        assert not any(call["purpose"] == "main" for call in gateway.calls)
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_cleanup_keeps_anchor_watermark_after_detailed_record_removal(tmp_path):
    engine, store, _, interest, cp = await active_case(tmp_path)
    try:
        await engine.handle_passive(passive_request(wake_source="wake_prefix"))
        opportunity = next(iter(store.state["opportunities"].values()))
        opportunity["updated_at"] = (
            engine.policy.now() - timedelta(days=8)
        ).isoformat()
        engine.control.cleanup_locked(store.state)
        assert store.state["opportunities"] == {}
        assert store.state["sessions"][cp["umo"]]["handled_before_ts"] > 0
        payload = json.loads(active_participate())["opportunity"]
        approval = engine.control.approve_participation_locked(
            store.state,
            umo=cp["umo"],
            wake_source="ambient",
            suggested_action="participate",
            opportunity_payload=payload,
            interest_id=interest["interest_id"],
            valid_ids={"m1"},
            bot_ids=set(),
            moment=engine.policy.now(),
        )
        assert not approval.approved
        assert approval.code == "opportunity_consumed"
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_dry_run_ingests_private_message_into_real_window(tmp_path):
    engine, store, _ = await make_engine(tmp_path, settings=make_settings(dry_run=True))
    await engine.scheduler.stop()
    try:
        request = passive_request(
            umo="p:FriendMessage:u", group_id="", wake_source="private"
        )
        await engine.handle_passive(request)
        assert store.state["sessions"][request.umo]["external_message_seq"] == 1
        assert len(store.recent_window(request.umo)) == 1
        assert store.state["opportunities"] == {}
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_merged_assessed_interest_keeps_its_next_checkpoint(tmp_path):
    engine, store, gateway, interest, cp = await active_case(tmp_path)
    try:
        other = await admit_interest(engine, summary="second interest")
        other_cp = await engine.create_checkpoint(
            umo=cp["umo"],
            platform_id="p",
            group_id="g",
            interest_id=other["interest_id"],
            minutes=5,
        )
        past = (engine.policy.now() - timedelta(minutes=1)).isoformat()
        for item in store.state["checkpoints"].values():
            item["scheduled_at"] = past
        chosen = await engine.claim_next_due()
        merged = other if chosen["interest_id"] == interest["interest_id"] else interest
        gateway.decision_responses = [
            ModelResult(
                text=json.dumps(
                    {
                        "action": "observe",
                        "arguments": {
                            "evaluated_interest_ids": [merged["interest_id"]]
                        },
                    }
                )
            )
        ]
        await engine.run_claimed(chosen["checkpoint_id"])
        assert store.state["interests"][merged["interest_id"]]["remaining_checks"] == 2
        assert (
            store.state["interests"][merged["interest_id"]]["last_evaluated_seq"] == 1
        )
        assert store.state["checkpoints"][cp["checkpoint_id"]]["status"] == "active"
        assert (
            store.state["checkpoints"][other_cp["checkpoint_id"]]["status"] == "active"
        )
        assert await engine.claim_next_due() is None
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_competing_reservations_share_one_session_slot(tmp_path):
    engine, store, _, interest, cp = await active_case(tmp_path)
    try:
        await add_message(engine, message_id="m2")

        async def reserve(anchor):
            async with store.lock:
                return engine.control.approve_participation_locked(
                    store.state,
                    umo=cp["umo"],
                    wake_source="ambient",
                    suggested_action="participate",
                    opportunity_payload=json.loads(active_participate(anchor=anchor))[
                        "opportunity"
                    ],
                    interest_id=interest["interest_id"],
                    moment=engine.policy.now(),
                    valid_ids={"m1", "m2"},
                    bot_ids=set(),
                )

        approvals = await asyncio.gather(reserve("m1"), reserve("m2"))
        assert sum(approval.approved for approval in approvals) == 1
        assert (
            next(approval for approval in approvals if not approval.approved).code
            == "session_busy"
        )
        assert store.state["budget"]["autonomous_attempts"] == 1
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_migration_write_failure_keeps_original_and_backup(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    original = '{"version":1,"topics":{},"checkpoints":{}}'
    path.write_text(original, encoding="utf-8")
    store = StateStore(path)

    def fail_flush():
        raise OSError("disk unavailable")

    monkeypatch.setattr(store, "flush", fail_flush)
    with pytest.raises(RuntimeError):
        await store.load()
    assert path.read_text(encoding="utf-8") == original
    assert len(list(tmp_path.glob("state.v1-*.json"))) == 1
    assert (
        json.loads(next(tmp_path.glob("state.v1-*.json")).read_text(encoding="utf-8"))[
            "version"
        ]
        == 1
    )
