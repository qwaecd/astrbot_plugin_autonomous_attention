"""Structured output extraction and validation.

The interest model is treated as an untrusted source. No message is ever sent,
and no state is mutated, before an output has passed these validators.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models import (
    ACTIVE_DECISIONS,
    CONFIDENCE_LEVELS,
    INTEREST_LEVELS,
    PASSIVE_DECISIONS,
    POLICY_SCOPES,
    TOPIC_ACTIONS,
    TOPIC_STATUSES,
    ActiveAction,
    PassiveDecision,
)

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("「", "」"))


def clip_text(value: Any, limit: int) -> str:
    """Convert ``value`` to a trimmed string with a hard length limit."""

    text = str(value or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response.

    Handles plain JSON, fenced JSON, and responses with a short preamble. It
    never evaluates Python or executes arbitrary content.

    Args:
        text: Raw model output.

    Returns:
        The decoded object, or ``None`` when no valid object is present.
    """

    if not text:
        return None
    cleaned = str(text).strip().lstrip("\ufeff")
    if cleaned.startswith("```"):
        cleaned = _FENCE_RE.sub("", cleaned).strip()
    try:
        decoded = json.loads(cleaned)
        if isinstance(decoded, dict):
            return decoded
    except (TypeError, ValueError):
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            decoded, _ = decoder.raw_decode(cleaned[index:])
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, dict):
            return decoded
    return None


def coerce_int(value: Any, default: int | None = None) -> int | None:
    """Coerce a JSON scalar to an integer without accepting booleans."""

    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            try:
                return int(text)
            except ValueError:
                return default
    return default


def _choice(value: Any, allowed: frozenset[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def validate_passive_decision(
    raw: dict[str, Any] | None,
    max_reason_chars: int = 200,
) -> PassiveDecision | None:
    """Validate a passive decision object.

    Args:
        raw: Decoded model output.
        max_reason_chars: Maximum stored reason length.

    Returns:
        A validated decision, or ``None`` when the output is unusable and the
        caller must apply the safe fallback.
    """

    if not isinstance(raw, dict):
        return None
    decision = str(raw.get("decision") or "").strip().lower()
    if decision not in PASSIVE_DECISIONS:
        return None

    user_action: dict[str, Any] | None = None
    raw_action = raw.get("user_action")
    if isinstance(raw_action, dict):
        scope = str(raw_action.get("scope") or "group_user").strip()
        mode = str(raw_action.get("mode") or "ignore").strip().lower()
        if scope in POLICY_SCOPES and mode == "ignore":
            expires = coerce_int(raw_action.get("expires_in_minutes"))
            user_action = {
                "scope": scope,
                "mode": "ignore",
                "expires_in_minutes": expires,
                "reason": clip_text(raw_action.get("reason"), 120),
            }
    if decision == "policy_update" and not user_action:
        return None

    return PassiveDecision(
        decision=decision,
        interest_level=_choice(
            raw.get("interest_level"),
            INTEREST_LEVELS,
            "low",
        ),
        reason=clip_text(raw.get("reason"), max_reason_chars),
        user_action=user_action,
        topic_action=_choice(raw.get("topic_action"), TOPIC_ACTIONS, "keep"),
        confidence=_choice(raw.get("confidence"), CONFIDENCE_LEVELS, "low"),
        topic_summary=clip_text(raw.get("topic_summary"), 500),
    )


def validate_active_action(
    raw: dict[str, Any] | None,
    max_reason_chars: int = 200,
) -> ActiveAction | None:
    """Validate one active attention action.

    Args:
        raw: Decoded model output.
        max_reason_chars: Maximum stored reason length.

    Returns:
        A validated action, or ``None`` for an unknown or malformed action.
    """

    if not isinstance(raw, dict):
        return None
    action = str(raw.get("action") or "").strip().lower()
    if not action:
        return None
    arguments = raw.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    return ActiveAction(
        action=action,
        arguments=arguments,
        reason=clip_text(raw.get("reason"), max_reason_chars),
        interest_level=_choice(raw.get("interest_level"), INTEREST_LEVELS, ""),
        topic_summary=clip_text(raw.get("topic_summary"), 500),
        topic_status=_choice(raw.get("topic_status"), TOPIC_STATUSES, ""),
        reply_guidance=clip_text(
            raw.get("reply_guidance") or arguments.get("guidance"),
            500,
        ),
    )


def is_known_active_action(action: str) -> bool:
    """Return whether ``action`` is a final action or a supported tool."""

    if action in ACTIVE_DECISIONS:
        return True
    return action in {
        "read_messages",
        "get_topic",
        "schedule_next_check",
        "defer_check",
        "create_checkpoint",
        "update_user_policy",
        "record_interest",
        "request_main_reply",
    }


def sanitize_reply(text: str | None, max_chars: int = 500) -> str | None:
    """Clean a main-model reply before sending it.

    Args:
        text: Raw main-model output.
        max_chars: Maximum allowed message length.

    Returns:
        A non-empty cleaned message, or ``None`` when nothing safe remains.
    """

    if not text:
        return None
    cleaned = str(text).strip().lstrip("\ufeff")
    if cleaned.startswith("```"):
        cleaned = _FENCE_RE.sub("", cleaned).strip()
    for left, right in _QUOTE_PAIRS:
        if len(cleaned) >= 2 and cleaned.startswith(left) and cleaned.endswith(right):
            cleaned = cleaned[len(left) : -len(right)].strip()
            break
    for prefix in ("assistant:", "bot:", "回复：", "消息："):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    if not cleaned:
        return None
    if len(cleaned) > max_chars:
        head = cleaned[:max_chars]
        cut = max(head.rfind(mark) for mark in "。！？.!?；;\n")
        if cut >= max_chars // 2:
            head = head[: cut + 1]
        cleaned = head.rstrip()
    return cleaned or None
