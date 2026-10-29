"""JSON-backed, restart-safe attention state store (schema version 2).

The store owns an ``asyncio.Lock``. Callers mutate ``store.state`` while holding
the lock and call :meth:`StateStore.flush` before releasing it. Writes use a
temporary file plus ``os.replace`` so a crash cannot leave truncated JSON.

Version 1 documents are migrated once, after a timestamped backup, into the
session-scoped model:

* ``interests`` replaces the single writable ``topics`` map.
* ``opportunities`` and ``sessions`` are introduced.
* checkpoints gain ``interest_id`` and ``kind``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .models import (
    SOURCE_ADMIN,
    SOURCE_ATTENTION_AGENT,
    SOURCE_MAIN_AGENT,
    SOURCE_PRIORITIES,
    RecentMessage,
)
from .timeutil import parse_iso, to_iso

STATE_VERSION = 2
SUPPORTED_VERSIONS = (1, 2)


class StateVersionError(RuntimeError):
    """Raised when the state file was written by a newer plugin version."""


def empty_state() -> dict[str, Any]:
    """Return a fresh version 2 state dictionary with every expected key."""

    return {
        "version": STATE_VERSION,
        "updated_at": "",
        "interests": {},
        "opportunities": {},
        "sessions": {},
        "checkpoints": {},
        "policies": {},
        "directives": {},
        "decisions": [],
        "send_records": [],
        "budget": {},
        "send_stats": {"last_send_at": "", "last_send_by_umo": {}},
        "recent_messages": {},
        "seeded": [],
        "migration": {},
        "metrics": {},
    }


def _legacy_source(created_by: Any) -> str:
    text = str(created_by or "").lower()
    if text in {"admin", "config", "owner"}:
        return SOURCE_ADMIN
    if text == SOURCE_MAIN_AGENT:
        return SOURCE_MAIN_AGENT
    return SOURCE_ATTENTION_AGENT


def _normalize_policy(raw: dict[str, Any]) -> None:
    source = str(raw.get("source") or _legacy_source(raw.get("created_by")))
    raw["source"] = source
    raw["priority"] = int(raw.get("priority") or SOURCE_PRIORITIES.get(source, 10))
    raw.setdefault("created_by_session", "")
    raw.setdefault("revision", 1)


def _normalize_directive(raw: dict[str, Any]) -> None:
    source = str(raw.get("source") or SOURCE_MAIN_AGENT)
    raw["source"] = source
    raw["priority"] = int(raw.get("priority") or SOURCE_PRIORITIES.get(source, 50))
    raw.setdefault("status", "active")
    raw.setdefault("created_by_session", "")
    raw.setdefault("revision", 1)


def _normalize_interest(raw: dict[str, Any]) -> dict[str, Any]:
    interest = {
        "interest_id": str(raw.get("interest_id") or raw.get("topic_id") or ""),
        "umo": str(raw.get("umo") or ""),
        "platform_id": str(raw.get("platform_id") or ""),
        "group_id": str(raw.get("group_id") or ""),
        "summary": str(raw.get("summary") or "")[:500],
        "status": str(raw.get("status") or "watching"),
        "source": str(raw.get("source") or SOURCE_ATTENTION_AGENT),
        "priority": int(raw.get("priority") or 10),
        "directive_id": str(raw.get("directive_id") or ""),
        "evidence_message_ids": [
            str(item)
            for item in (raw.get("evidence_message_ids") or [])
            if str(item).strip()
        ][-50:],
        "wait_for": str(raw.get("wait_for") or "")[:300],
        "last_external_evidence_at": str(raw.get("last_external_evidence_at") or ""),
        "expires_at": str(raw.get("expires_at") or ""),
        "remaining_checks": max(0, int(raw.get("remaining_checks") or 0)),
        "no_progress_count": max(0, int(raw.get("no_progress_count") or 0)),
        "last_evaluated_seq": max(0, int(raw.get("last_evaluated_seq") or 0)),
        "created_at": str(raw.get("created_at") or ""),
        "updated_at": str(raw.get("updated_at") or ""),
        "revision": int(raw.get("revision") or 1),
    }
    if interest["status"] not in {"watching", "paused", "closed"}:
        interest["status"] = "paused"
    if raw.get("quarantine"):
        interest["quarantine"] = True
        interest["migration_note"] = str(raw.get("migration_note") or "")
        interest["legacy_record"] = raw.get("legacy_record", {})
    return interest


def _normalize_opportunity(raw: dict[str, Any]) -> dict[str, Any]:
    opportunity = {
        "opportunity_id": str(raw.get("opportunity_id") or ""),
        "umo": str(raw.get("umo") or ""),
        "interest_id": str(raw.get("interest_id") or ""),
        "kind": str(raw.get("kind") or "natural_opening"),
        "anchor_message_id": str(raw.get("anchor_message_id") or ""),
        "anchor_created_at": str(
            raw.get("anchor_created_at") or raw.get("created_at") or ""
        ),
        "evidence_message_ids": [
            str(item)
            for item in (raw.get("evidence_message_ids") or [])
            if str(item).strip()
        ][-50:],
        "expected_contribution": str(raw.get("expected_contribution") or "")[:500],
        "status": str(raw.get("status") or "pending"),
        "expires_at": str(raw.get("expires_at") or ""),
        "reservation_id": str(raw.get("reservation_id") or ""),
        "reserved_at": str(raw.get("reserved_at") or ""),
        "consumed_at": str(raw.get("consumed_at") or ""),
        "outcome": str(raw.get("outcome") or ""),
        "created_at": str(raw.get("created_at") or ""),
        "updated_at": str(raw.get("updated_at") or ""),
        "revision": int(raw.get("revision") or 1),
    }
    if opportunity["kind"] not in {
        "direct_request",
        "unanswered_question",
        "new_information",
        "natural_opening",
    }:
        opportunity["kind"] = "natural_opening"
    if opportunity["status"] not in {
        "pending",
        "reserved",
        "consumed",
        "expired",
        "uncertain",
    }:
        opportunity["status"] = "uncertain"
    return opportunity


def _normalize_session(raw: dict[str, Any], umo: str) -> dict[str, Any]:
    return {
        "umo": str(raw.get("umo") or umo),
        "platform_id": str(raw.get("platform_id") or ""),
        "group_id": str(raw.get("group_id") or ""),
        "message_ids": dict(raw.get("message_ids") or {}),
        "history_shapes": list(raw.get("history_shapes") or [])[-500:],
        "handled_before_ts": float(raw.get("handled_before_ts") or 0),
        "received_messages": int(raw.get("received_messages") or 0),
        "external_message_seq": max(0, int(raw.get("external_message_seq") or 0)),
        "last_decision_seq": max(0, int(raw.get("last_decision_seq") or 0)),
        "last_decision_at": str(raw.get("last_decision_at") or ""),
        "next_evaluation_at": str(raw.get("next_evaluation_at") or ""),
        "last_autonomous_participation_at": str(
            raw.get("last_autonomous_participation_at") or ""
        ),
        "quota_date": str(raw.get("quota_date") or ""),
        "autonomous_attempts": max(0, int(raw.get("autonomous_attempts") or 0)),
        "pending_reply_reservation": str(raw.get("pending_reply_reservation") or ""),
        "revision": int(raw.get("revision") or 1),
    }


def _normalize_checkpoint(raw: dict[str, Any]) -> dict[str, Any]:
    source = str(raw.get("source") or _legacy_source(raw.get("created_by")))
    interest_id = str(raw.get("interest_id") or raw.get("topic_id") or "")
    kind = str(raw.get("kind") or ("interest" if interest_id else "session_scan"))
    checkpoint = {
        "checkpoint_id": str(raw.get("checkpoint_id") or ""),
        "umo": str(raw.get("umo") or ""),
        "platform_id": str(raw.get("platform_id") or ""),
        "group_id": str(raw.get("group_id") or ""),
        "interest_id": interest_id,
        "kind": kind if kind in {"interest", "session_scan"} else "session_scan",
        "scheduled_at": str(raw.get("scheduled_at") or ""),
        "status": str(raw.get("status") or "active"),
        "reason": str(raw.get("reason") or "")[:200],
        "last_result": str(raw.get("last_result") or ""),
        "source": source,
        "priority": int(raw.get("priority") or SOURCE_PRIORITIES.get(source, 10)),
        "directive_id": str(raw.get("directive_id") or ""),
        "created_by_session": str(raw.get("created_by_session") or ""),
        "expires_at": str(raw.get("expires_at") or ""),
        "remaining_checks": max(0, int(raw.get("remaining_checks") or 0)),
        "no_progress_count": max(0, int(raw.get("no_progress_count") or 0)),
        "revision": int(raw.get("revision") or 1),
        "created_at": str(raw.get("created_at") or ""),
        "updated_at": str(raw.get("updated_at") or ""),
        "last_run_at": str(raw.get("last_run_at") or ""),
        "last_error": str(raw.get("last_error") or ""),
        "run_id": str(raw.get("run_id") or ""),
        "attempts": int(raw.get("attempts") or 0),
    }
    if checkpoint["status"] not in {
        "active",
        "running",
        "paused",
        "completed",
        "cancelled",
    }:
        checkpoint["status"] = "paused"
    if raw.get("quarantine"):
        checkpoint.update(
            quarantine=True,
            migration_note=str(raw.get("migration_note") or ""),
            legacy_record=raw.get("legacy_record", {}),
        )
    return checkpoint


def _normalize_decision(raw: dict[str, Any]) -> dict[str, Any]:
    record = dict(raw)
    kind = str(record.get("kind") or "")
    record.setdefault("umo", "")
    record.setdefault(
        "source",
        "passive_agent" if kind == "passive" else "active_agent",
    )
    record.setdefault("wake_source", "unknown")
    record.setdefault("interest_id", "")
    record.setdefault("checkpoint_id", "")
    record.setdefault("anchor_message_id", "")
    record.setdefault("suggested_action", "")
    record.setdefault("model_suggestion", "")
    record.setdefault("code_decision", "")
    record.setdefault("rejection_code", "")
    record.setdefault("reservation_id", "")
    record.setdefault("related_policy_id", "")
    record.setdefault("related_directive_id", "")
    record.setdefault("message_seq", 0)
    record.setdefault("cached", False)
    return record


def _normalize_recent_windows(
    windows: dict[str, Any],
    sessions: dict[str, Any],
) -> dict[str, Any]:
    """Assign stable per-UMO sequence numbers to persisted windows."""

    result: dict[str, Any] = {}
    for umo, raw_window in windows.items():
        if not isinstance(raw_window, list):
            continue
        messages: list[dict[str, Any]] = []
        for item in raw_window:
            if not isinstance(item, dict):
                continue
            messages.append(RecentMessage.from_dict(item).to_dict())
        messages.sort(
            key=lambda item: (float(item.get("created_ts") or 0), item["message_id"])
        )
        next_seq = max(
            [
                int(sessions.get(umo, {}).get("external_message_seq") or 0),
                *[
                    int(item.get("seq") or 0)
                    for item in messages
                    if not item.get("is_bot")
                ],
            ]
        )
        assigned = 0
        for item in messages:
            if item.get("is_bot"):
                item["seq"] = 0
                if not item.get("message_id"):
                    item["message_id"] = (
                        f"bot_{umo}_{int(float(item.get('created_ts') or 0))}"
                    )
                continue
            if not item.get("seq"):
                assigned += 1
                item["seq"] = next_seq + assigned
            if not item.get("message_id"):
                item["message_id"] = f"legacy_{umo}_{item['seq']}"
        if assigned:
            sessions.setdefault(umo, _normalize_session({}, umo))[
                "external_message_seq"
            ] = next_seq + assigned
        result[umo] = messages
    return result


def normalize_state(raw: Any) -> dict[str, Any]:
    """Fill missing keys and coerce malformed containers back to defaults.

    Args:
        raw: Untrusted decoded JSON.

    Returns:
        A version 2 state dictionary that is safe for the rest of the core.
    """

    state = empty_state()
    if not isinstance(raw, dict):
        return state
    for key, value in raw.items():
        if key in state:
            state[key] = value
    if not isinstance(state.get("migration"), dict):
        state["migration"] = {}
    for key in (
        "interests",
        "opportunities",
        "sessions",
        "checkpoints",
        "policies",
        "directives",
        "recent_messages",
    ):
        if not isinstance(state[key], dict):
            state[key] = {}
    for key in ("decisions", "send_records", "seeded"):
        if not isinstance(state[key], list):
            state[key] = []
    for raw_policy in state["policies"].values():
        if isinstance(raw_policy, dict):
            _normalize_policy(raw_policy)
    for raw_directive in state["directives"].values():
        if isinstance(raw_directive, dict):
            _normalize_directive(raw_directive)
    state["interests"] = {
        str(key): _normalize_interest(value)
        for key, value in state["interests"].items()
        if isinstance(value, dict)
    }
    state["opportunities"] = {
        str(key): _normalize_opportunity(value)
        for key, value in state["opportunities"].items()
        if isinstance(value, dict)
    }
    state["sessions"] = {
        str(key): _normalize_session(value, str(key))
        for key, value in state["sessions"].items()
        if isinstance(value, dict)
    }
    state["checkpoints"] = {
        str(key): _normalize_checkpoint(value)
        for key, value in state["checkpoints"].items()
        if isinstance(value, dict)
    }
    state["decisions"] = [
        _normalize_decision(value)
        for value in state["decisions"]
        if isinstance(value, dict)
    ]
    if not isinstance(state["budget"], dict):
        state["budget"] = {}
    if not isinstance(state["send_stats"], dict):
        state["send_stats"] = {}
    send_stats = state["send_stats"]
    if not isinstance(send_stats.get("last_send_by_umo"), dict):
        send_stats["last_send_by_umo"] = {}
    state["recent_messages"] = _normalize_recent_windows(
        state["recent_messages"],
        state["sessions"],
    )
    for umo in state["recent_messages"]:
        state["sessions"].setdefault(umo, _normalize_session({}, umo))
    state["version"] = STATE_VERSION
    return state


def migrate_v1(
    raw: dict[str, Any],
    *,
    interest_ttl_minutes: int = 120,
    interest_check_limit: int = 3,
    moment: datetime | None = None,
    min_interval_minutes: int = 10,
) -> dict[str, Any]:
    """Convert a version 1 document into the version 2 session model.

    Args:
        raw: Decoded version 1 document.
        interest_ttl_minutes: Initial TTL for migrated interests.
        interest_check_limit: Initial check balance for migrated interests.

    Returns:
        A normalised version 2 state dictionary.
    """

    now = moment or datetime.now(timezone.utc)
    ttl = timedelta(minutes=max(1, int(interest_ttl_minutes)))
    state = empty_state()
    state["budget"] = dict(raw.get("budget") or {})
    state["send_stats"] = dict(raw.get("send_stats") or {})
    state["seeded"] = list(raw.get("seeded") or [])
    state["send_records"] = list(raw.get("send_records") or [])
    state["policies"] = dict(raw.get("policies") or {})
    state["directives"] = dict(raw.get("directives") or {})
    state["decisions"] = list(raw.get("decisions") or [])
    state["recent_messages"] = dict(raw.get("recent_messages") or {})

    raw_checkpoints = {
        str(key): value
        for key, value in (raw.get("checkpoints") or {}).items()
        if isinstance(value, dict)
    }
    topic_owners: dict[str, set[str]] = {}
    routes: dict[str, set[tuple[str, str]]] = {}
    for checkpoint in raw_checkpoints.values():
        topic_id = str(checkpoint.get("topic_id") or "")
        umo = str(checkpoint.get("umo") or "")
        if topic_id and umo:
            topic_owners.setdefault(topic_id, set()).add(umo)
        if umo:
            parts = umo.split(":", 2)
            platform = str(checkpoint.get("platform_id") or "")
            group = str(checkpoint.get("group_id") or "")
            if len(parts) == 3 and parts[1] == "GroupMessage":
                platform = platform or parts[0]
                if not group and "_" not in parts[2] and "%" not in parts[2]:
                    group = parts[2]
            routes.setdefault(umo, set()).add((platform, group))

    notes: list[str] = []
    interests: dict[str, dict[str, Any]] = {}
    for topic_id, raw_topic in (raw.get("topics") or {}).items():
        if not isinstance(raw_topic, dict):
            continue
        topic_id = str(topic_id or raw_topic.get("topic_id") or "")
        if not topic_id:
            continue
        owners = topic_owners.get(topic_id, set())
        own = str(raw_topic.get("umo") or "")
        if own:
            owners = owners | {own}
        umo = next(iter(owners)) if len(owners) == 1 else ""
        if umo and raw_topic.get("platform_id") and raw_topic.get("group_id"):
            routes.setdefault(umo, set()).add(
                (str(raw_topic["platform_id"]), str(raw_topic["group_id"]))
            )
        route = (
            next(iter(routes.get(umo, set())))
            if len(routes.get(umo, set())) == 1
            else ("", "")
        )
        summary = str(raw_topic.get("summary") or "")[:500]
        if not summary:
            notes.append(f"topic {topic_id} dropped: empty summary")
            continue
        status = "closed" if raw_topic.get("focus_status") == "closed" else "watching"
        quarantined = False
        migration_note = ""
        parts = umo.split(":", 2)
        if (
            not umo
            or len(parts) != 3
            or parts[1] != "GroupMessage"
            or len(routes.get(umo, set())) > 1
            or not all(route)
            or parts[0] != route[0]
        ):
            status = "paused"
            quarantined = True
            migration_note = "legacy topic has no resolvable session"
            notes.append(f"topic {topic_id} quarantined: no session")
        interests[topic_id] = {
            "interest_id": topic_id,
            "umo": umo,
            "platform_id": str(raw_topic.get("platform_id") or route[0]),
            "group_id": str(raw_topic.get("group_id") or route[1]),
            "summary": summary,
            "status": status,
            "source": SOURCE_ATTENTION_AGENT,
            "priority": 10,
            "directive_id": "",
            "evidence_message_ids": [],
            "wait_for": "",
            "last_external_evidence_at": str(
                raw_topic.get("last_activity_at") or raw_topic.get("updated_at") or ""
            ),
            "expires_at": to_iso(now + ttl),
            "remaining_checks": max(1, int(interest_check_limit)),
            "no_progress_count": 0,
            "last_evaluated_seq": 0,
            "created_at": str(raw_topic.get("updated_at") or to_iso(now)),
            "updated_at": to_iso(now),
            "revision": 1,
            "quarantine": quarantined,
            "migration_note": migration_note,
            "legacy_record": dict(raw_topic) if quarantined else {},
        }

    checkpoints: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for checkpoint_id, raw_checkpoint in raw_checkpoints.items():
        raw_checkpoint = dict(raw_checkpoint)
        raw_checkpoint.setdefault("checkpoint_id", checkpoint_id)
        checkpoint = _normalize_checkpoint(raw_checkpoint)
        interest_id = checkpoint["interest_id"]
        if interest_id and interest_id not in interests:
            checkpoint["kind"] = "session_scan"
            checkpoint["interest_id"] = ""
            checkpoint["last_error"] = "legacy checkpoint referenced a missing topic"
            notes.append(f"checkpoint {checkpoint_id} downgraded to session_scan")
        umo = checkpoint["umo"]
        route = (
            next(iter(routes.get(umo, set())))
            if len(routes.get(umo, set())) == 1
            else ("", "")
        )
        checkpoint["platform_id"], checkpoint["group_id"] = route
        owner_parts = umo.split(":", 2)
        interest = interests.get(checkpoint["interest_id"])
        invalid = (
            len(owner_parts) != 3
            or owner_parts[1] != "GroupMessage"
            or not all(route)
            or owner_parts[0] != route[0]
            or (
                interest is not None
                and (interest.get("quarantine") or interest.get("umo") != umo)
            )
        )
        checkpoint["expires_at"] = (interest or {}).get("expires_at") or to_iso(
            now + ttl
        )
        checkpoint["remaining_checks"] = int(
            (interest or {}).get("remaining_checks") or interest_check_limit
        )
        if invalid:
            checkpoint.update(
                status="paused",
                run_id="",
                quarantine=True,
                migration_note="unresolved or conflicting legacy ownership/route",
                legacy_record=raw_checkpoint,
            )
            checkpoints[checkpoint_id] = checkpoint
            notes.append(f"checkpoint {checkpoint_id} quarantined: ownership/route")
            continue
        key = (checkpoint["umo"], checkpoint["interest_id"])
        if checkpoint["status"] in {"active", "running", "paused"}:
            grouped.setdefault(key, []).append(checkpoint)
        else:
            checkpoints[checkpoint_id] = checkpoint
    for _key, items in grouped.items():
        items.sort(
            key=lambda item: (
                -int(item.get("priority") or 0),
                item.get("status") == "paused",
                str(item.get("scheduled_at") or ""),
                item["checkpoint_id"],
            )
        )
        keeper = items[0]
        scheduled = parse_iso(keeper.get("scheduled_at"), now.tzinfo)
        if scheduled is None or scheduled <= now:
            keeper["scheduled_at"] = to_iso(
                now + timedelta(minutes=max(1, min_interval_minutes))
            )
        for dropped in items[1:]:
            aliases[dropped["checkpoint_id"]] = keeper["checkpoint_id"]
            dropped = dict(dropped)
            dropped["status"] = "cancelled"
            dropped["last_result"] = "migrated_duplicate"
            dropped["last_error"] = (
                f"merged into {keeper['checkpoint_id']} during v1 migration"
            )
            checkpoints[dropped["checkpoint_id"]] = dropped
        checkpoints[keeper["checkpoint_id"]] = keeper
        interest = interests.get(keeper["interest_id"] or "")
        if interest and not keeper.get("expires_at"):
            keeper["expires_at"] = interest["expires_at"]
            keeper["remaining_checks"] = interest["remaining_checks"]

    directives: dict[str, dict[str, Any]] = {}
    for directive_id, raw_directive in (raw.get("directives") or {}).items():
        if not isinstance(raw_directive, dict):
            continue
        directive = dict(raw_directive)
        session = str(directive.get("created_by_session") or "")
        if directive.get("type") in {"focus_topic", "focus_group"} and not session:
            direct_target = str(directive.get("target") or "")
            if direct_target.count(":") >= 2:
                session = direct_target
            if not session:
                candidates = {
                    cp["umo"]
                    for cp in checkpoints.values()
                    if not cp.get("quarantine")
                    and (
                        cp.get("interest_id") == direct_target
                        or cp.get("group_id") == direct_target
                    )
                }
                if len(candidates) == 1:
                    session = next(iter(candidates))
            if not session:
                directive["status"] = "paused"
                directive["migration_note"] = (
                    "legacy directive has no unambiguous session; "
                    "recreate it from the target session"
                )
                notes.append(f"directive {directive_id} paused: ambiguous session")
        if session:
            directive["created_by_session"] = session
        directives[str(directive_id)] = directive
    state["directives"] = directives
    state["interests"] = interests
    state["checkpoints"] = checkpoints
    state["migration"] = {
        "from_version": 1,
        "at": to_iso(now),
        "notes": notes[:100],
        "checkpoint_aliases": aliases,
    }
    if "messages_sent" in state["budget"]:
        state["budget"]["autonomous_attempts"] = int(
            state["budget"].get("messages_sent") or 0
        )
    return normalize_state(state)


class StateStore:
    """Persist attention state in one JSON document."""

    def __init__(
        self,
        path: Path,
        max_decisions: int = 200,
    ) -> None:
        self.path = Path(path)
        self.max_decisions = max(10, int(max_decisions))
        self.lock = asyncio.Lock()
        self.live_state: dict[str, Any] = empty_state()
        self._simulation: ContextVar[bool] = ContextVar(
            "attention_simulation", default=False
        )
        self._shadow: dict[str, Any] | None = None
        self._shadow_baseline: dict[str, Any] = {}
        self.migrated = False

    @property
    def state(self) -> dict[str, Any]:
        """Return the live state or a task-local view of shared simulation state."""
        if not self._simulation.get():
            return self.live_state
        if self._shadow is None:
            self._shadow = deepcopy(self.live_state)
            self._shadow_baseline = deepcopy(self.live_state)
        # Import live messages and administrator writes without publishing any
        # simulated transition. Sync individual changed records/fields so a new
        # message cannot reset simulated quotas or evaluation progress.
        for container in (
            "interests",
            "checkpoints",
            "opportunities",
            "sessions",
            "policies",
            "directives",
            "recent_messages",
        ):
            current = self.live_state.get(container, {})
            baseline = self._shadow_baseline.get(container, {})
            target = self._shadow.setdefault(container, {})
            for key in set(current) | set(baseline):
                if current.get(key) == baseline.get(key):
                    continue
                if key not in current:
                    target.pop(key, None)
                elif container == "sessions" and key in target:
                    for field, value in current[key].items():
                        if value != baseline.get(key, {}).get(field):
                            target[key][field] = deepcopy(value)
                else:
                    target[key] = deepcopy(current[key])
            self._shadow_baseline[container] = deepcopy(current)
        return self._shadow

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self.live_state = value
        self.discard_shadow()

    @contextmanager
    def simulation(self, enabled: bool):
        """Select shadow business state for this task without affecting observers.

        Args:
            enabled: Whether the operation is a dry-run.

        Yields:
            None. Real gateway accounting and audit use live_state explicitly.
        """
        token = self._simulation.set(enabled)
        try:
            yield
        finally:
            self._simulation.reset(token)

    def discard_shadow(self) -> None:
        """Discard simulated resources when configuration mode changes."""
        self._shadow = None
        self._shadow_baseline = {}

    async def load(
        self,
        *,
        interest_ttl_minutes: int = 120,
        interest_check_limit: int = 3,
        moment: datetime | None = None,
        min_interval_minutes: int = 10,
    ) -> None:
        """Load state from disk, migrating version 1 once.

        Args:
            interest_ttl_minutes: Initial TTL applied to migrated interests.
            interest_check_limit: Initial check balance applied to migrated
                interests.

        Raises:
            StateVersionError: The document was written by a newer version.
            RuntimeError: Migration or the atomic version 2 write failed. The
                original document is left untouched in that case.
        """

        if not self.path.exists():
            self.state = empty_state()
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            backup = self.path.with_name(
                f"{self.path.stem}.corrupt-{int(datetime.now().timestamp())}.json"
            )
            try:
                shutil.copy2(self.path, backup)
                logger.error(
                    "Attention state was unreadable (%s); backed it up to %s and "
                    "startup was stopped.",
                    exc,
                    backup,
                )
            except OSError:
                logger.exception(
                    "Attention state was unreadable and could not be backed up.",
                )
            raise RuntimeError(
                f"unreadable attention state; original preserved at {self.path}"
            ) from exc

        version = int(raw.get("version") or 1) if isinstance(raw, dict) else 0
        if version > STATE_VERSION:
            raise StateVersionError(
                f"attention state version {version} is newer than supported "
                f"version {STATE_VERSION}"
            )
        if version < 1:
            raise StateVersionError(f"unsupported attention state version {version}")
        if version == 1:
            backup = self.path.with_name(
                f"{self.path.stem}.v1-{int(datetime.now().timestamp())}.json"
            )
            try:
                shutil.copy2(self.path, backup)
            except OSError as exc:
                raise RuntimeError(
                    f"could not back up version 1 attention state: {exc}"
                ) from exc
            try:
                self.state = migrate_v1(
                    raw,
                    interest_ttl_minutes=interest_ttl_minutes,
                    interest_check_limit=interest_check_limit,
                    moment=moment,
                    min_interval_minutes=min_interval_minutes,
                )
                self.flush()
            except Exception as exc:
                self.state = empty_state()
                raise RuntimeError(
                    f"attention state migration failed; original kept at "
                    f"{self.path}: {exc}"
                ) from exc
            self.migrated = True
            logger.info(
                "Attention state migrated from v1 to v2 (backup: %s).",
                backup,
            )
            return

        self.state = normalize_state(raw)

    def flush(self) -> None:
        """Write the in-memory state to disk atomically.

        The caller must hold :attr:`lock`.
        """

        self.live_state["version"] = STATE_VERSION
        self.live_state["updated_at"] = to_iso(datetime.now(timezone.utc))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f".{self.path.name}.tmp")
        payload = json.dumps(self.live_state, ensure_ascii=False, indent=2)
        temp_path.write_text(payload, encoding="utf-8")
        os.replace(temp_path, self.path)

    def remember_recent(
        self,
        umo: str,
        message: RecentMessage,
        *,
        max_messages: int,
        max_sessions: int,
    ) -> None:
        """Append a message to the in-memory sliding window.

        The caller must hold :attr:`lock` and is responsible for assigning
        ``message.seq`` when the message is new external content.
        """

        windows = self.state.setdefault("recent_messages", {})
        if not isinstance(windows, dict):
            windows = {}
            self.state["recent_messages"] = windows
        window = windows.get(umo)
        if not isinstance(window, list):
            window = []
            windows[umo] = window
        window.append(message.to_dict())
        if len(window) > max_messages:
            del window[: len(window) - max_messages]
        if len(windows) > max_sessions:
            ordered: list[tuple[str, float]] = []
            for key, value in windows.items():
                last = value[-1] if isinstance(value, list) and value else {}
                created = last.get("created_ts") if isinstance(last, dict) else 0
                ordered.append((key, float(created or 0)))
            ordered.sort(key=lambda item: item[1])
            for old_umo, _ in ordered[: len(windows) - max_sessions]:
                windows.pop(old_umo, None)

    def recent_window(self, umo: str) -> list[RecentMessage]:
        """Return a copy of the in-memory window for ``umo``."""

        raw = (self.state.get("recent_messages") or {}).get(umo)
        if not isinstance(raw, list):
            return []
        result: list[RecentMessage] = []
        for item in raw:
            if isinstance(item, dict):
                result.append(RecentMessage.from_dict(item))
        return result

    def record_decision(self, record: dict[str, Any]) -> None:
        """Append an audit record, keeping only the most recent entries.

        The caller must hold :attr:`lock`.
        """

        if self._simulation.get():
            record["dry_run"] = True
        decisions = self.live_state.setdefault("decisions", [])
        if not isinstance(decisions, list):
            decisions = []
            self.live_state["decisions"] = decisions
        decisions.append(record)
        del decisions[: max(0, len(decisions) - self.max_decisions)]
        if record.get("dry_run"):
            return
        metrics = self.live_state.setdefault("metrics", {}).setdefault(
            str(record.get("umo") or ""), {}
        )
        increments = {
            "semantic_decisions": int(record.get("semantic_evaluations") or 0),
            "suggestions": int(
                record.get("suggested_action") in {"reply", "participate"}
            ),
            "approved_reservations": int(
                bool(record.get("reservation_id"))
                and record.get("rejection_code") == "autonomous_approved"
            ),
            "passive_passes": int(
                record.get("kind") == "passive"
                and record.get("code_decision") == "pass"
            ),
            "explicit_passes": int(record.get("code_decision") == "explicit_pass"),
            "active_sends": int(record.get("send") == "sent"),
        }
        for key, count in increments.items():
            metrics[key] = int(metrics.get(key) or 0) + count
        code = str(record.get("rejection_code") or "")
        if code and record.get("code_decision") in {"observe", "ignore", "skip"}:
            rejections = metrics.setdefault("rejection_counts", {})
            rejections[code] = int(rejections.get(code) or 0) + 1
