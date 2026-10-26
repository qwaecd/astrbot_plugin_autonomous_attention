"""Context window helpers shared by the passive and active paths."""

from __future__ import annotations

from .models import RecentMessage


def _dedupe_key(message: RecentMessage) -> tuple[str, int, str]:
    if message.message_id:
        return ("id", 0, message.message_id)
    return (
        "shape",
        int(message.created_ts),
        f"{message.sender_id}|{message.is_bot}|{message.text[:120]}",
    )


def merge_window(
    history: list[RecentMessage],
    recent: list[RecentMessage],
    limit: int,
) -> list[RecentMessage]:
    """Merge history with the in-memory window without duplicate messages.

    Args:
        history: Messages read from AstrBot's persisted group history.
        recent: Messages observed by this plugin.
        limit: Maximum number of messages to keep.

    Returns:
        Chronological messages, newest last.
    """

    if limit <= 0:
        return []
    merged: list[RecentMessage] = []
    seen: set[tuple[str, int, str]] = set()
    bot_signatures: dict[tuple[str, str], float] = {}
    for message in [*history, *recent]:
        key = _dedupe_key(message)
        if key in seen:
            continue
        if message.is_bot:
            signature = (message.sender_id, message.text[:160])
            previous = bot_signatures.get(signature)
            if previous is not None and abs(message.created_ts - previous) < 120:
                continue
            bot_signatures[signature] = message.created_ts
        seen.add(key)
        merged.append(message)
    merged.sort(key=lambda item: (item.created_ts, item.message_id))
    if len(merged) > limit:
        merged = merged[-limit:]
    return merged
