"""TTS backend abstraction.

Every backend implements a single cancellable ``speak()``. Barge-in works by
calling ``stop()`` from another task: it sets a stop flag the backend polls
between audio chunks and calls ``_interrupt()`` to cut any in-flight playback.

All blocking work (SAPI's COM calls, subprocess playback) runs on a single
dedicated worker thread per backend so Windows COM stays on one apartment.
"""

from __future__ import annotations

import abc
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from ..config import TTSConfig


class _SingleThread:
    """A one-thread executor so all backend calls share one OS thread."""

    def __init__(self, name: str) -> None:
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    async def run(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._ex, fn, *args)

    def shutdown(self) -> None:
        self._ex.shutdown(wait=False)


class TTSBackend(abc.ABC):
    """Base class for all speech backends."""

    name: str = "base"

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self.rate = cfg.rate
        self.volume = cfg.volume
        self.voice = cfg.voice
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self._lock = asyncio.Lock()
        self._worker = _SingleThread(self.name)

    # -- public API ------------------------------------------------------ #
    async def speak(self, text: str) -> bool:
        """Speak ``text``. Returns True if it finished, False if interrupted."""
        if not text or not text.strip():
            return True
        self._stop.clear()
        async with self._lock:
            self._speaking.set()
            try:
                return await self._worker.run(self._speak_blocking, text, self._stop)
            finally:
                self._speaking.clear()

    async def stop(self) -> None:
        """Interrupt the current utterance as fast as the backend allows."""
        self._stop.set()
        try:
            self._interrupt()
        except Exception:
            pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def set_rate(self, rate: float) -> None:
        self.rate = max(0.5, min(4.0, rate))

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))

    def set_voice(self, voice: str) -> None:
        self.voice = voice

    async def close(self) -> None:
        await self.stop()
        self._worker.shutdown()

    # -- backend hooks --------------------------------------------------- #
    @abc.abstractmethod
    def _speak_blocking(self, text: str, stop: threading.Event) -> bool:
        """Synchronously speak ``text``; poll ``stop`` and return False if it
        was set before finishing, True otherwise. Runs on the worker thread."""

    def _interrupt(self) -> None:
        """Optional immediate-cut hook, called off the worker thread."""

    # -- discovery ------------------------------------------------------- #
    @classmethod
    def is_available(cls, cfg: TTSConfig) -> bool:  # pragma: no cover - overridden
        return False
