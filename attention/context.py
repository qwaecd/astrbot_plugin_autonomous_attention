"""Context window helpers shared by the passive and active paths."""

from __future__ import annotations

from .models import RecentMessage


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
    # Prefer observed platform IDs and sequences over database row IDs. Only
    # cross-source records may match by shape; equal live messages stay distinct.
    merged = list(recent)
    for message in history:
        match = next(
            (
                item
                for item in merged
                if (
                    message.message_id
                    and message.message_id in [item.message_id, *item.aliases]
                )
                or (
                    message.created_ts > 0
                    and item.created_ts > 0
                    and (message.source == "history" or item.source == "history")
                    and item.sender_id == message.sender_id
                    and item.is_bot == message.is_bot
                    and item.text == message.text
                    and abs(item.created_ts - message.created_ts) < 600
                )
            ),
            None,
        )
        if match is not None:
            if message.message_id != match.message_id:
                match.aliases = list(
                    dict.fromkeys([*match.aliases, message.message_id])
                )[-10:]
            continue
        merged.append(message)
    merged.sort(key=lambda item: (item.created_ts, item.message_id))
    return merged[-limit:]
