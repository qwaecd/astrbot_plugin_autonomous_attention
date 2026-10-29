"""Typed, read-only plugin settings.

AstrBot exposes plugin configuration as a mutable dictionary. The core consumes
this snapshot instead, which makes defaults and bounds explicit.

The ``attention_limits`` group controls session resources. Legacy keys
(``daily_active_message_budget`` and ``scheduling.group_send_cooldown_minutes``)
are still read when the new keys are absent so an upgrade keeps its old values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import tzinfo
from typing import Any

from astrbot.api import logger

from .prompts import (
    DEFAULT_ACTIVE_CRITERIA,
    DEFAULT_MAIN_AGENT_ATTENTION_PROMPT,
    DEFAULT_PASSIVE_CRITERIA,
)
from .timeutil import resolve_timezone

PASSIVE_MODES = frozenset({"off", "block_only", "full"})
NO_PROGRESS_LIMIT = 3
_WARNED_COMPATIBILITY: set[str] = set()


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


def _get(data: Mapping[str, Any], section: str, key: str) -> Any:
    """Read a grouped config value, falling back to the legacy flat key."""

    section_data = data.get(section)
    if isinstance(section_data, Mapping) and key in section_data:
        return section_data[key]
    return data.get(key)


def _first(*values: Any) -> Any:
    """Return the first explicit value, skipping the legacy inheritance sentinel."""

    for value in values:
        if value is not None and str(value) != "-1":
            return value
    return None


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
    global_send_cooldown_minutes: int = 10
    session_autonomous_cooldown_minutes: int = 10
    daily_model_call_budget: int = 200
    daily_autonomous_reply_budget: int = 10
    daily_active_check_budget: int = 50
    max_interests_per_session: int = 3
    interest_ttl_minutes: int = 120
    interest_check_limit: int = 3
    min_evaluation_interval_seconds: int = 30
    opportunity_ttl_minutes: int = 10
    session_autonomous_reply_budget: int = 6
    max_agent_rounds: int = 4
    max_tool_calls: int = 6
    max_checkpoint_mutations: int = 2
    max_main_model_calls: int = 1
    decision_timeout_seconds: int = 45
    max_reply_chars: int = 500
    max_reason_chars: int = 200
    dry_run: bool = False
    max_decision_log: int = 200
    passive_retry_delay_minutes: int = 10
    owner_override: bool = True
    max_auto_policy_hours: int = 168
    main_agent_max_policy_hours: int = 24
    main_agent_max_intent_minutes: int = 30
    main_agent_tools_enabled: bool = True
    main_agent_attention_prompt_enabled: bool = True
    main_agent_attention_prompt: str = DEFAULT_MAIN_AGENT_ATTENTION_PROMPT
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

    @property
    def no_progress_limit(self) -> int:
        return NO_PROGRESS_LIMIT

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
        passive_mode = _text(
            _get(data, "general", "passive_mode"), "block_only"
        ).lower()
        if passive_mode not in PASSIVE_MODES:
            passive_mode = "block_only"
        persona_source = _text(
            _get(data, "models", "persona_source"), "astrbot"
        ).lower()
        if persona_source not in {"astrbot", "custom"}:
            persona_source = "astrbot"
        timezone_name = _text(_get(data, "general", "timezone")) or _text(
            fallback_timezone
        )
        min_interval = _int(
            _get(data, "scheduling", "min_interval_minutes"), 10, 1, 10080
        )
        max_interval = _int(
            _get(data, "scheduling", "max_interval_minutes"),
            1440,
            min_interval,
            43200,
        )
        default_interval = _int(
            _get(data, "scheduling", "default_interval_minutes"),
            60,
            min_interval,
            max_interval,
        )
        max_defer = _int(
            _get(data, "scheduling", "max_defer_minutes"),
            1440,
            min_interval,
            43200,
        )
        intent_minutes = _get(data, "budgets", "main_agent_max_intent_minutes")
        if intent_minutes is None or str(intent_minutes) == "-1":
            intent_minutes = None
            legacy_hours = _get(data, "budgets", "main_agent_max_intent_hours")
            if legacy_hours is not None and str(legacy_hours) != "-1":
                intent_minutes = _int(legacy_hours, 24, 0, 8760) * 60
        for new_key, old_section, old_key in (
            ("daily_autonomous_reply_budget", "budgets", "daily_active_message_budget"),
            (
                "session_autonomous_cooldown_minutes",
                "scheduling",
                "group_send_cooldown_minutes",
            ),
            (
                "main_agent_max_intent_minutes",
                "budgets",
                "main_agent_max_intent_hours",
            ),
        ):
            old_value = _get(data, old_section, old_key)
            if old_value is not None and new_key not in _WARNED_COMPATIBILITY:
                _WARNED_COMPATIBILITY.add(new_key)
                logger.warning(
                    "Attention config compatibility: %s inherits %s when absent or -1; "
                    "an explicit new value takes precedence.",
                    new_key,
                    old_key,
                )
        session_cooldown = _int(
            _first(
                _get(data, "attention_limits", "session_autonomous_cooldown_minutes"),
                _get(data, "scheduling", "group_send_cooldown_minutes"),
                data.get("group_send_cooldown_minutes"),
            ),
            10,
            0,
            1440,
        )
        daily_autonomous = _int(
            _first(
                _get(data, "attention_limits", "daily_autonomous_reply_budget"),
                _get(data, "budgets", "daily_active_message_budget"),
                data.get("daily_active_message_budget"),
            ),
            10,
            0,
            10000,
        )
        return cls(
            enabled=_bool(_get(data, "general", "enabled"), True),
            passive_mode=passive_mode,
            active_enabled=_bool(_get(data, "general", "active_enabled"), True),
            decision_provider_id=_text(_get(data, "models", "decision_provider_id")),
            main_provider_id=_text(_get(data, "models", "main_provider_id")),
            persona_source=persona_source,
            custom_persona=_text(_get(data, "models", "custom_persona")),
            passive_criteria=_text(
                _get(data, "models", "passive_criteria"),
                DEFAULT_PASSIVE_CRITERIA,
            )
            or DEFAULT_PASSIVE_CRITERIA,
            active_criteria=_text(
                _get(data, "models", "active_criteria"),
                DEFAULT_ACTIVE_CRITERIA,
            )
            or DEFAULT_ACTIVE_CRITERIA,
            group_allowlist=_str_list(_get(data, "groups", "group_allowlist")),
            blocked_group_ids=_str_list(_get(data, "groups", "blocked_group_ids")),
            context_window_messages=_int(
                _get(data, "context", "context_window_messages"),
                30,
                5,
                200,
            ),
            context_max_chars=_int(
                _get(data, "context", "context_max_chars"), 8000, 500, 60000
            ),
            quiet_hours=_text(_get(data, "scheduling", "quiet_hours")),
            min_interval_minutes=min_interval,
            max_interval_minutes=max_interval,
            default_interval_minutes=default_interval,
            max_defer_minutes=max_defer,
            global_send_cooldown_minutes=_int(
                _get(data, "scheduling", "global_send_cooldown_minutes"),
                10,
                0,
                10080,
            ),
            session_autonomous_cooldown_minutes=session_cooldown,
            daily_model_call_budget=_int(
                _get(data, "budgets", "daily_model_call_budget"),
                200,
                0,
                100000,
            ),
            daily_autonomous_reply_budget=daily_autonomous,
            daily_active_check_budget=_int(
                _get(data, "budgets", "daily_active_check_budget"),
                50,
                0,
                10000,
            ),
            max_interests_per_session=_int(
                _get(data, "attention_limits", "max_interests_per_session"),
                3,
                1,
                20,
            ),
            interest_ttl_minutes=_int(
                _get(data, "attention_limits", "interest_ttl_minutes"),
                120,
                10,
                10080,
            ),
            interest_check_limit=_int(
                _get(data, "attention_limits", "interest_check_limit"),
                3,
                1,
                20,
            ),
            min_evaluation_interval_seconds=_int(
                _get(data, "attention_limits", "min_evaluation_interval_seconds"),
                30,
                1,
                3600,
            ),
            opportunity_ttl_minutes=_int(
                _get(data, "attention_limits", "opportunity_ttl_minutes"),
                10,
                1,
                120,
            ),
            session_autonomous_reply_budget=_int(
                _get(data, "attention_limits", "session_autonomous_reply_budget"),
                6,
                0,
                1000,
            ),
            max_agent_rounds=_int(
                _get(data, "agent_limits", "max_agent_rounds"), 4, 1, 20
            ),
            max_tool_calls=_int(_get(data, "agent_limits", "max_tool_calls"), 6, 0, 50),
            max_checkpoint_mutations=_int(
                _get(data, "agent_limits", "max_checkpoint_mutations"),
                2,
                0,
                20,
            ),
            max_main_model_calls=_int(
                _get(data, "agent_limits", "max_main_model_calls"),
                1,
                0,
                3,
            ),
            decision_timeout_seconds=_int(
                _get(data, "agent_limits", "decision_timeout_seconds"),
                45,
                5,
                600,
            ),
            max_reply_chars=_int(
                _get(data, "agent_limits", "max_reply_chars"), 500, 20, 4000
            ),
            max_reason_chars=_int(
                _get(data, "agent_limits", "max_reason_chars"), 200, 20, 2000
            ),
            dry_run=_bool(_get(data, "general", "dry_run"), False),
            max_decision_log=_int(
                _get(data, "general", "max_decision_log"), 200, 10, 5000
            ),
            passive_retry_delay_minutes=_int(
                _get(data, "scheduling", "passive_retry_delay_minutes"),
                10,
                min_interval,
                max_interval,
            ),
            owner_override=_bool(_get(data, "safety", "owner_override"), True),
            max_auto_policy_hours=_int(
                _get(data, "budgets", "max_auto_policy_hours"),
                168,
                1,
                8760,
            ),
            main_agent_max_policy_hours=_int(
                _get(data, "budgets", "main_agent_max_policy_hours"),
                24,
                1,
                8760,
            ),
            main_agent_max_intent_minutes=_int(
                intent_minutes,
                30,
                1,
                525600,
            ),
            main_agent_tools_enabled=_bool(
                _get(data, "general", "main_agent_tools_enabled"),
                True,
            ),
            main_agent_attention_prompt_enabled=_bool(
                _get(data, "general", "main_agent_attention_prompt_enabled"),
                True,
            ),
            main_agent_attention_prompt=_text(
                _get(data, "general", "main_agent_attention_prompt"),
                DEFAULT_MAIN_AGENT_ATTENTION_PROMPT,
            )
            or DEFAULT_MAIN_AGENT_ATTENTION_PROMPT,
            timezone=timezone_name,
            tz=resolve_timezone(timezone_name),
            seed_checkpoints=_dict_list(_get(data, "scheduling", "seed_checkpoints")),
            max_context_sessions=_int(
                _get(data, "context", "max_context_sessions"),
                50,
                1,
                1000,
            ),
            max_concurrent_decisions=_int(
                _get(data, "general", "max_concurrent_decisions"),
                4,
                1,
                32,
            ),
        )
