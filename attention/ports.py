"""Small ports implemented by the AstrBot adapter.

Keeping these as protocols (instead of importing AstrBot) is what allows the
core to run in unit tests and in a dry-run harness with fake gateways.
"""

from __future__ import annotations

from typing import Any, Protocol

from .models import ModelResult, RecentMessage


class ModelGateway(Protocol):
    """Generate text with either the decision model or the main model."""

    async def generate(
        self,
        *,
        purpose: str,
        umo: str,
        system_prompt: str,
        prompt: str | None = None,
        contexts: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> ModelResult:
        """Return one model completion."""
        ...


class ContextProvider(Protocol):
    """Provide group history and persona text."""

    async def window(
        self,
        *,
        umo: str,
        platform_id: str,
        group_id: str,
        recent: list[RecentMessage],
        limit: int,
    ) -> list[RecentMessage]:
        """Return a merged, chronological window for a group."""
        ...

    async def persona(self, umo: str) -> str:
        """Return the persona/system prompt used for a session."""
        ...


class MessageSender(Protocol):
    """Send one proactive message."""

    async def send_text(self, umo: str, text: str) -> tuple[bool, str]:
        """Send text and return ``(sent, error)``."""
        ...
