"""JSON-backed, restart-safe attention state store.

The store owns an ``asyncio.Lock``. Callers mutate ``store.state`` while holding
the lock and call :meth:`StateStore.flush` before releasing it. Writes use a
temporary file plus ``os.replace`` so a crash cannot leave truncated JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    SOURCE_ADMIN,
    SOURCE_ATTENTION_AGENT,
    SOURCE_MAIN_AGENT,
    SOURCE_PRIORITIES,
    RecentMessage,
)
from .timeutil import to_iso

STATE_VERSION = 1


def empty_state() -> dict[str, Any]:
    """Return a fresh state dictionary with every expected key."""

    return {
        "version": STATE_VERSION,
        "updated_at": "",
        "checkpoints": {},
        "topics": {},
        "policies": {},
        "directives": {},
        "decisions": [],
        "send_records": [],
        "budget": {},
        "send_stats": {"last_send_at": "", "last_send_by_umo": {}},
        "recent_messages": {},
        "seeded": [],
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


def _normalize_checkpoint(raw: dict[str, Any]) -> None:
    source = str(raw.get("source") or _legacy_source(raw.get("created_by")))
    raw["source"] = source
    raw["priority"] = int(raw.get("priority") or SOURCE_PRIORITIES.get(source, 10))
    raw.setdefault("directive_id", "")
    raw.setdefault("created_by_session", "")
    raw.setdefault("revision", 1)


def _normalize_decision(raw: dict[str, Any]) -> None:
    kind = str(raw.get("kind") or "")
    raw.setdefault(
        "source",
        "passive_agent" if kind == "passive" else "active_agent",
    )
    raw.setdefault("related_policy_id", "")
    raw.setdefault("related_directive_id", "")


def normalize_state(raw: Any) -> dict[str, Any]:
    """Fill missing keys and coerce malformed containers back to defaults.

    Args:
        raw: Untrusted decoded JSON.

    Returns:
        A state dictionary that is safe for the rest of the core to use.
    """

    state = empty_state()
    if not isinstance(raw, dict):
        return state
    for key, value in raw.items():
        if key in state:
            state[key] = value
    for key in ("checkpoints", "topics", "policies", "directives", "recent_messages"):
        if not isinstance(state[key], dict):
            state[key] = {}
    for key in ("decisions", "send_records", "seeded"):
        if not isinstance(state[key], list):
            state[key] = []
    for raw_policy in state["policies"].values():
        if isinstance(raw_policy, dict):
            _normalize_policy(raw_policy)
    for raw_checkpoint in state["checkpoints"].values():
        if isinstance(raw_checkpoint, dict):
            _normalize_checkpoint(raw_checkpoint)
    for raw_directive in state["directives"].values():
        if isinstance(raw_directive, dict):
            source = str(raw_directive.get("source") or SOURCE_MAIN_AGENT)
            raw_directive["source"] = source
            raw_directive["priority"] = int(
                raw_directive.get("priority") or SOURCE_PRIORITIES.get(source, 50)
            )
            raw_directive.setdefault("status", "active")
            raw_directive.setdefault("revision", 1)
            raw_directive.setdefault("payload", {})
    for raw_decision in state["decisions"]:
        if isinstance(raw_decision, dict):
            _normalize_decision(raw_decision)
    if not isinstance(state["budget"], dict):
        state["budget"] = {}
    if not isinstance(state["send_stats"], dict):
        state["send_stats"] = {}
    send_stats = state["send_stats"]
    if not isinstance(send_stats.get("last_send_by_umo"), dict):
        send_stats["last_send_by_umo"] = {}
    state["version"] = STATE_VERSION
    return state


class StateStore:
    """Persist attention state in one JSON document."""

    def __init__(
        self,
        path: Path,
        logger: logging.Logger,
        max_decisions: int = 200,
    ) -> None:
        self.path = Path(path)
        self.logger = logger
        self.max_decisions = max(10, int(max_decisions))
        self.lock = asyncio.Lock()
        self.state: dict[str, Any] = empty_state()

    async def load(self) -> None:
        """Load state from disk, backing up an unreadable document."""

        if not self.path.exists():
            self.state = empty_state()
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.state = normalize_state(raw)
        except Exception as exc:
            backup = self.path.with_name(
                f"{self.path.stem}.corrupt-{int(datetime.now().timestamp())}.json"
            )
            try:
                shutil.copy2(self.path, backup)
                self.logger.error(
                    "Attention state was unreadable (%s); backed it up to %s and "
                    "started from an empty state.",
                    exc,
                    backup,
                )
            except OSError:
                self.logger.exception(
                    "Attention state was unreadable and could not be backed up.",
                )
            self.state = empty_state()

    def flush(self) -> None:
        """Write the in-memory state to disk atomically.

        The caller must hold :attr:`lock`.
        """

        self.state["version"] = STATE_VERSION
        self.state["updated_at"] = to_iso(datetime.now(timezone.utc))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f".{self.path.name}.tmp")
        payload = json.dumps(self.state, ensure_ascii=False, indent=2)
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

        The caller must hold :attr:`lock`.
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

        decisions = self.state.setdefault("decisions", [])
        if not isinstance(decisions, list):
            decisions = []
            self.state["decisions"] = decisions
        decisions.append(record)
        del decisions[: max(0, len(decisions) - self.max_decisions)]
