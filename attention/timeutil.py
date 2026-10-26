"""Time helpers for the attention core.

All timestamps persisted by the plugin are ISO-8601 strings with an explicit
offset. This keeps state readable and avoids ambiguity when AstrBot and the
operating system use different time zones.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo

try:  # pragma: no cover - zoneinfo is available on supported Python versions.
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


def resolve_timezone(name: str | None) -> tzinfo:
    """Resolve an IANA time zone name, falling back to the host time zone.

    Args:
        name: IANA time zone name such as ``Asia/Shanghai``. Empty values and
            unknown names fall back to the local time zone.

    Returns:
        A ``tzinfo`` instance usable for aware datetimes.
    """

    if name and ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except Exception:
            pass
    local = datetime.now().astimezone().tzinfo
    return local or timezone.utc


def now(tz: tzinfo) -> datetime:
    """Return the current time in ``tz``."""

    return datetime.now(tz)


def to_iso(value: datetime) -> str:
    """Format an aware datetime as an ISO-8601 string with seconds precision."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat(timespec="seconds")


def parse_iso(value: str | None, tz: tzinfo) -> datetime | None:
    """Parse an ISO-8601 timestamp, attaching ``tz`` when no offset is present.

    Args:
        value: Timestamp text. Empty or invalid values return ``None``.
        tz: Time zone used for naive timestamps.

    Returns:
        An aware datetime, or ``None`` when the value cannot be parsed.
    """

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return parsed


def timestamp(value: datetime) -> float:
    """Return a POSIX timestamp for an aware datetime."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def add_minutes(value: datetime, minutes: int) -> datetime:
    """Return ``value`` advanced by ``minutes`` minutes."""

    return value + timedelta(minutes=int(minutes))


def parse_quiet_hours(spec: str | None) -> tuple[int, int] | None:
    """Parse a quiet-hours specification.

    Supported forms are ``HH:MM-HH:MM`` and ``HH:MM~HH:MM``. A window whose end
    is earlier than its start crosses midnight.

    Args:
        spec: Quiet-hours text. Empty or malformed text disables quiet hours.

    Returns:
        ``(start_minute, end_minute)`` where each value is minutes since
        midnight, or ``None`` when quiet hours are disabled.
    """

    if not spec:
        return None
    text = str(spec).strip().replace("～", "-").replace("~", "-")
    if "-" not in text:
        return None
    start_text, end_text = text.split("-", 1)

    def parse_clock(value: str) -> int | None:
        parts = value.strip().split(":")
        if len(parts) != 2:
            return None
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError:
            return None
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            return None
        return hour * 60 + minute

    start = parse_clock(start_text)
    end = parse_clock(end_text)
    if start is None or end is None or start == end:
        return None
    return start, end


def in_quiet_hours(value: datetime, spec: str | None) -> bool:
    """Return whether ``value`` lies inside the configured quiet window."""

    window = parse_quiet_hours(spec)
    if window is None:
        return False
    start, end = window
    current = value.hour * 60 + value.minute
    if start < end:
        return start <= current < end
    return current >= start or current < end


def quiet_hours_end(value: datetime, spec: str | None) -> datetime | None:
    """Return the next moment the quiet window ends.

    Args:
        value: Reference time.
        spec: Quiet-hours text.

    Returns:
        The first aware datetime outside the quiet window, or ``None`` when the
        reference time is not inside a quiet window.
    """

    window = parse_quiet_hours(spec)
    if window is None or not in_quiet_hours(value, spec):
        return None
    _, end = window
    end_time = value.replace(hour=end // 60, minute=end % 60, second=0, microsecond=0)
    if end_time <= value:
        end_time += timedelta(days=1)
    return end_time
