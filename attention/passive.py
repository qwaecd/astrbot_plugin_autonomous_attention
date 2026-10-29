"""Passive attention runner: one bounded interest-model call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import PassiveDecision, PassiveRequest, RecentMessage
from .ports import ModelGateway
from .prompts import build_passive_system_prompt, build_passive_user_prompt
from .settings import Settings
from .validation import extract_json_object, validate_passive_decision


@dataclass
class PassiveRunResult:
    """Outcome of the single passive interest-model call."""

    kind: str = "error"
    decision: PassiveDecision | None = None
    raw_text: str = ""
    provider_id: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    error: str = ""


class PassiveAttentionRunner:
    """Decide whether one woken message may reach AstrBot's main model."""

    def __init__(
        self,
        settings: Settings,
        gateway: ModelGateway,
    ) -> None:
        self.settings = settings
        self.gateway = gateway

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    async def decide(
        self,
        request: PassiveRequest,
        *,
        interests: list[dict[str, Any]],
        checkpoints: list[dict[str, Any]],
        policy_summary: str,
        opportunities: list[dict[str, Any]],
        resources: dict[str, Any],
        window: list[RecentMessage],
    ) -> PassiveRunResult:
        """Run exactly one interest-model call with session-scoped data.

        Args:
            request: Normalised woken message.
            interests: Current session interest candidates.
            checkpoints: Current session active checkpoints.
            policy_summary: Human-readable hard-policy/interest summary.
            opportunities: Recently processed opportunities for the session.
            resources: Remaining session/global participation resources.
            window: Session message window with stable IDs.

        Returns:
            A result whose ``kind`` is ``ok``, ``invalid`` or ``error``.
        """

        system_prompt = build_passive_system_prompt(self.settings)
        user_prompt = build_passive_user_prompt(
            self.settings,
            request,
            interests,
            checkpoints,
            policy_summary,
            opportunities,
            resources,
            window,
        )
        result = await self.gateway.generate(
            purpose="decision",
            umo=request.umo,
            system_prompt=system_prompt,
            prompt=user_prompt,
            timeout=self.settings.decision_timeout_seconds,
        )
        if not result.ok:
            return PassiveRunResult(
                kind="error",
                provider_id=result.provider_id,
                usage=result.usage,
                error=result.error or "empty model response",
            )
        raw = extract_json_object(result.text)
        decision = validate_passive_decision(raw, self.settings.max_reason_chars)
        if decision is None:
            return PassiveRunResult(
                kind="invalid",
                raw_text=result.text,
                provider_id=result.provider_id,
                usage=result.usage,
                error="invalid structured output",
            )
        return PassiveRunResult(
            kind="ok",
            decision=decision,
            raw_text=result.text,
            provider_id=result.provider_id,
            usage=result.usage,
        )
