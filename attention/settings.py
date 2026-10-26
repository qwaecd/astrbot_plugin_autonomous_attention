"""Typed, read-only plugin settings.

AstrBot exposes plugin configuration as a mutable dictionary. The core consumes
this snapshot instead, which makes defaults and bounds explicit and keeps tests
free of AstrBot imports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import tzinfo
from typing import Any

from .prompts import DEFAULT_ACTIVE_CRITERIA, DEFAULT_PASSIVE_CRITERIA
from .timeutil import resolve_timezone

PASSIVE_MODES = frozenset({"off", "block_only", "full"})
LOG_LEVELS = frozenset({"INFO", "DEBUG", "WARNING", "ERROR"})


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return default


def _int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _str_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        return ()
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _dict_list(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(dict(item))
    return tuple(result)


@dataclass(frozen=True)
class Settings:
    """A validated snapshot of plugin configuration."""

    enabled: bool = True
    passive_mode: str = "block_only"
    active_enabled: bool = True
    decision_provider_id: str = ""
    main_provider_id: str = ""
    persona_source: str = "astrbot"
    custom_persona: str = ""
    passive_criteria: str = DEFAULT_PASSIVE_CRITERIA
    active_criteria: str = DEFAULT_ACTIVE_CRITERIA
    group_allowlist: tuple[str, ...] = ()
    blocked_group_ids: tuple[str, ...] = ()
    context_window_messages: int = 30
    context_max_chars: int = 8000
    quiet_hours: str = ""
    min_interval_minutes: int = 10
    max_interval_minutes: int = 1440
    default_interval_minutes: int = 60
    max_defer_minutes: int = 1440
    group_send_cooldown_minutes: int = 30
    global_send_cooldown_minutes: int = 10
    daily_model_call_budget: int = 200
    daily_active_message_budget: int = 10
    daily_active_check_budget: int = 50
    max_agent_rounds: int = 4
    max_tool_calls: int = 6
    max_checkpoint_mutations: int = 2
    max_main_model_calls: int = 1
    decision_timeout_seconds: int = 45
    max_reply_chars: int = 500
    max_reason_chars: int = 200
    dry_run: bool = False
    log_level: str = "INFO"
    max_decision_log: int = 200
    passive_retry_delay_minutes: int = 10
    owner_override: bool = True
    max_auto_policy_hours: int = 168
    main_agent_max_policy_hours: int = 24
    main_agent_max_intent_hours: int = 24
    main_agent_tools_enabled: bool = True
    timezone: str = ""
    tz: tzinfo = field(default_factory=lambda: resolve_timezone(""))
    seed_checkpoints: tuple[dict[str, Any], ...] = ()
    max_context_sessions: int = 50
    max_concurrent_decisions: int = 4

    @property
    def passive_full(self) -> bool:
        return self.passive_mode == "full"

    @property
    def persona_for_prompt(self) -> str:
        if self.persona_source == "custom":
            return self.custom_persona
        return ""

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any] | None,
        fallback_timezone: str = "",
    ) -> Settings:
        """Build a settings snapshot from an AstrBot plugin config mapping.

        Args:
            config: Plugin configuration mapping. Missing keys use defaults.
            fallback_timezone: AstrBot's global time zone, used when the plugin
                does not configure one.

        Returns:
            A validated settings object.
        """

        data = dict(config or {})
        passive_mode = _text(data.get("passive_mode"), "block_only").lower()
        if passive_mode not in PASSIVE_MODES:
            passive_mode = "block_only"
        log_level = _text(data.get("log_level"), "INFO").upper()
        if log_level not in LOG_LEVELS:
            log_level = "INFO"
        persona_source = _text(data.get("persona_source"), "astrbot").lower()
        if persona_source not in {"astrbot", "custom"}:
            persona_source = "astrbot"
        timezone_name = _text(data.get("timezone")) or _text(fallback_timezone)
        min_interval = _int(data.get("min_interval_minutes"), 10, 1, 10080)
        max_interval = _int(
            data.get("max_interval_minutes"),
            1440,
            min_interval,
            43200,
        )
        default_interval = _int(
            data.get("default_interval_minutes"),
            60,
            min_interval,
            max_interval,
        )
        max_defer = _int(
            data.get("max_defer_minutes"),
            1440,
            min_interval,
            43200,
        )
        return cls(
            enabled=_bool(data.get("enabled"), True),
            passive_mode=passive_mode,
            active_enabled=_bool(data.get("active_enabled"), True),
            decision_provider_id=_text(data.get("decision_provider_id")),
            main_provider_id=_text(data.get("main_provider_id")),
            persona_source=persona_source,
            custom_persona=_text(data.get("custom_persona")),
            passive_criteria=_text(
                data.get("passive_criteria"),
                DEFAULT_PASSIVE_CRITERIA,
            )
            or DEFAULT_PASSIVE_CRITERIA,
            active_criteria=_text(
                data.get("active_criteria"),
                DEFAULT_ACTIVE_CRITERIA,
            )
            or DEFAULT_ACTIVE_CRITERIA,
            group_allowlist=_str_list(data.get("group_allowlist")),
            blocked_group_ids=_str_list(data.get("blocked_group_ids")),
            context_window_messages=_int(
                data.get("context_window_messages"),
                30,
                5,
                200,
            ),
            context_max_chars=_int(data.get("context_max_chars"), 8000, 500, 60000),
            quiet_hours=_text(data.get("quiet_hours")),
            min_interval_minutes=min_interval,
            max_interval_minutes=max_interval,
            default_interval_minutes=default_interval,
            max_defer_minutes=max_defer,
            group_send_cooldown_minutes=_int(
                data.get("group_send_cooldown_minutes"),
                30,
                0,
                10080,
            ),
            global_send_cooldown_minutes=_int(
                data.get("global_send_cooldown_minutes"),
                10,
                0,
                10080,
            ),
            daily_model_call_budget=_int(
                data.get("daily_model_call_budget"),
                200,
                0,
                100000,
            ),
            daily_active_message_budget=_int(
                data.get("daily_active_message_budget"),
                10,
                0,
                1000,
            ),
            daily_active_check_budget=_int(
                data.get("daily_active_check_budget"),
                50,
                0,
                10000,
            ),
            max_agent_rounds=_int(data.get("max_agent_rounds"), 4, 1, 20),
            max_tool_calls=_int(data.get("max_tool_calls"), 6, 0, 50),
            max_checkpoint_mutations=_int(
                data.get("max_checkpoint_mutations"),
                2,
                0,
                20,
            ),
            max_main_model_calls=_int(
                data.get("max_main_model_calls"),
                1,
                0,
                3,
            ),
            decision_timeout_seconds=_int(
                data.get("decision_timeout_seconds"),
                45,
                5,
                600,
            ),
            max_reply_chars=_int(data.get("max_reply_chars"), 500, 20, 4000),
            max_reason_chars=_int(data.get("max_reason_chars"), 200, 20, 2000),
            dry_run=_bool(data.get("dry_run"), False),
            log_level=log_level,
            max_decision_log=_int(data.get("max_decision_log"), 200, 10, 5000),
            passive_retry_delay_minutes=_int(
                data.get("passive_retry_delay_minutes"),
                10,
                min_interval,
                max_interval,
            ),
            owner_override=_bool(data.get("owner_override"), True),
            max_auto_policy_hours=_int(
                data.get("max_auto_policy_hours"),
                168,
                1,
                8760,
            ),
            main_agent_max_policy_hours=_int(
                data.get("main_agent_max_policy_hours"),
                24,
                1,
                8760,
            ),
            main_agent_max_intent_hours=_int(
                data.get("main_agent_max_intent_hours"),
                24,
                1,
                8760,
            ),
            main_agent_tools_enabled=_bool(
                data.get("main_agent_tools_enabled"),
                True,
            ),
            timezone=timezone_name,
            tz=resolve_timezone(timezone_name),
            seed_checkpoints=_dict_list(data.get("seed_checkpoints")),
            max_context_sessions=_int(
                data.get("max_context_sessions"),
                50,
                1,
                1000,
            ),
            max_concurrent_decisions=_int(
                data.get("max_concurrent_decisions"),
                4,
                1,
                32,
            ),
        )
