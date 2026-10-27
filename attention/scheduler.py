"""Checkpoint scheduler.

The scheduler only decides *when* to start an active session. It never decides
whether to speak; that belongs to the active attention runner and hard policy.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

if TYPE_CHECKING:
    from .engine import AttentionEngine


class AttentionScheduler:
    """Run due checkpoints in the background for one plugin instance."""

    def __init__(
        self,
        engine: AttentionEngine,
        poll_seconds: int = 20,
    ) -> None:
        self.engine = engine
        self.poll_seconds = max(5, int(poll_seconds))
        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._run_tasks: dict[str, asyncio.Task] = {}
        self._run_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the polling loop if it is not already running."""

        if self._loop_task is not None and not self._loop_task.done():
            return
        self._stop_event.clear()
        self._loop_task = asyncio.create_task(
            self._loop(),
            name="autonomous-attention-scheduler",
        )
        logger.info("Attention checkpoint scheduler started.")

    async def stop(self) -> None:
        """Stop the loop and cancel every task started by this scheduler."""

        self._stop_event.set()
        if self._loop_task is not None:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
            self._loop_task = None
        tasks = list(self._run_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._run_tasks.clear()
        logger.info("Attention checkpoint scheduler stopped.")

    async def enqueue_checkpoint(self, checkpoint_id: str) -> bool:
        """Run a claimed checkpoint as a tracked background task.

        Args:
            checkpoint_id: Checkpoint already marked as ``running`` by the
                engine's claim method.

        Returns:
            ``True`` when a task was created, ``False`` when it was already
            running.
        """

        existing = self._run_tasks.get(checkpoint_id)
        if existing is not None and not existing.done():
            return False
        task = asyncio.create_task(
            self._run_one(checkpoint_id),
            name=f"autonomous-attention-{checkpoint_id}",
        )
        self._run_tasks[checkpoint_id] = task
        task.add_done_callback(
            lambda _task, cp_id=checkpoint_id: self._run_tasks.pop(cp_id, None)
        )
        return True

    async def _run_one(self, checkpoint_id: str) -> None:
        """Execute one claimed checkpoint under the single-run lock."""

        async with self._run_lock:
            try:
                await self.engine.run_claimed(checkpoint_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Active attention checkpoint %s failed.",
                    checkpoint_id,
                )

    async def _loop(self) -> None:
        """Poll for due checkpoints until stopped."""

        while not self._stop_event.is_set():
            delay = self.poll_seconds
            try:
                settings = self.engine.settings
                if not settings.enabled or not settings.active_enabled:
                    delay = 30
                else:
                    claimed = await self.engine.claim_next_due()
                    if claimed is not None:
                        await self._run_one(str(claimed.get("checkpoint_id") or ""))
                        continue
                    delay = await self.engine.seconds_until_next_due()
                    delay = max(5, min(self.poll_seconds, int(delay)))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Attention scheduler loop iteration failed.")
                delay = 30

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
            except TimeoutError:
                continue

    def status(self) -> dict[str, Any]:
        """Return a small status snapshot for admin commands."""

        return {
            "running": bool(self._loop_task and not self._loop_task.done()),
            "active_tasks": sum(
                1 for task in self._run_tasks.values() if not task.done()
            ),
            "poll_seconds": self.poll_seconds,
        }
