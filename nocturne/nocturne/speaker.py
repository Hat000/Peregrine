"""The speaking pump.

A single background task drains a queue of ``SpeechItem``s and reads each aloud
through the active TTS backend. It fires callbacks the moment it starts talking
and when it goes quiet, so the orchestrator can duck the music under speech and
bring it back while thinking.

Barge-in is ``skip()``: it stops the current utterance immediately and drops the
rest of the queued response. The last fully-spoken response is retained for the
``/again`` (replay) and read-slower controls.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from .speech_filter import SpeechItem
from .tts.base import TTSBackend


class Speaker:
    def __init__(
        self,
        tts: TTSBackend,
        on_start: Callable[[], Awaitable[None] | None] | None = None,
        on_idle: Callable[[], Awaitable[None] | None] | None = None,
    ) -> None:
        self.tts = tts
        self._on_start = on_start
        self._on_idle = on_idle
        self._queue: asyncio.Queue[SpeechItem | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._speaking = False
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        # Replay memory.
        self._current: list[str] = []   # response being spoken now
        self._last: list[str] = []       # last completed response

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self._run())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        await self.tts.close()

    # ------------------------------------------------------------------ #
    def enqueue(self, item: SpeechItem) -> None:
        self._idle_event.clear()
        self._queue.put_nowait(item)

    def say_now(self, text: str, kind: str = "system") -> None:
        """Queue a one-off system utterance (warnings, confirmations)."""
        self.enqueue(SpeechItem(text=text, kind=kind))

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                continue
            self._speaking = True
            await _maybe_await(self._on_start)
            finished = await self.tts.speak(item.text)
            if item.kind in ("prose", "code_announce") and finished:
                self._current.append(item.text)
            self._speaking = False
            if self._queue.empty():
                self._on_response_end()
                await _maybe_await(self._on_idle)
                self._idle_event.set()

    def _on_response_end(self) -> None:
        if self._current:
            self._last = self._current
            self._current = []

    # ------------------------------------------------------------------ #
    async def skip(self) -> None:
        """Barge-in: cut current speech and drop the rest of the response."""
        await self.tts.stop()
        _drain(self._queue)
        self._on_response_end()
        self._speaking = False
        self._idle_event.set()

    async def drain(self) -> None:
        """Await until everything queued has been spoken."""
        await self._idle_event.wait()

    @property
    def is_speaking(self) -> bool:
        return self._speaking or not self._queue.empty()

    # ------------------------------------------------------------------ #
    def last_response_text(self) -> str:
        return " ".join(self._last)

    async def replay(self, rate_scale: float = 1.0) -> None:
        """Re-speak the last completed response (optionally slower)."""
        text = self.last_response_text()
        if not text:
            self.say_now("There's nothing to replay yet.")
            return
        if rate_scale != 1.0:
            old = self.tts.rate
            self.tts.set_rate(old * rate_scale)
            self.enqueue(SpeechItem(text=text, kind="system"))
            await self.drain()
            self.tts.set_rate(old)
        else:
            self.enqueue(SpeechItem(text=text, kind="system"))


def _drain(q: asyncio.Queue) -> None:
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break


async def _maybe_await(cb) -> None:
    if cb is None:
        return
    r = cb()
    if asyncio.iscoroutine(r):
        await r
