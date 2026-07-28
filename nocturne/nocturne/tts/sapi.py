"""Windows SAPI / cross-platform backend via pyttsx3.

Uses the operating system's built-in speech engine (SAPI5 on Windows,
NSSpeechSynthesizer on macOS, espeak on Linux). Offline and low-latency.
Barge-in is handled by ``engine.stop()``.

CAVEAT (Windows): pyttsx3's SAPI5 driver reliably speaks only the FIRST
utterance per process — subsequent ``runAndWait()`` calls often silently
no-op. For a screenless app that streams many sentences that is fatal, so
``auto`` prefers the ``claude`` backend (System.Speech via PowerShell, a fresh
process per utterance) over this one. Pick ``sapi`` explicitly only if you know
your pyttsx3/driver combination keeps speaking across utterances.
"""

from __future__ import annotations

import threading

from ..config import TTSConfig
from .base import TTSBackend

_BASE_WPM = 200  # pyttsx3's default rate is ~200 words/min


class SapiTTS(TTSBackend):
    name = "sapi"

    def __init__(self, cfg: TTSConfig) -> None:
        super().__init__(cfg)
        self._engine = None
        self._engine_lock = threading.Lock()

    def _ensure_engine(self):
        # Must run on the worker thread (COM apartment affinity).
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._apply_voice(self._engine)
        return self._engine

    def _apply_voice(self, engine) -> None:
        if not self.voice:
            return
        try:
            for v in engine.getProperty("voices"):
                if self.voice.lower() in (v.id or "").lower() or \
                        self.voice.lower() in (getattr(v, "name", "") or "").lower():
                    engine.setProperty("voice", v.id)
                    return
        except Exception:
            pass

    def _speak_blocking(self, text: str, stop: threading.Event) -> bool:
        if stop.is_set():
            return False
        with self._engine_lock:
            engine = self._ensure_engine()
            engine.setProperty("rate", int(_BASE_WPM * self.rate))
            engine.setProperty("volume", float(self.volume))
            if self.voice:
                self._apply_voice(engine)
            engine.say(text)
            engine.runAndWait()
        return not stop.is_set()

    def _interrupt(self) -> None:
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    @classmethod
    def is_available(cls, cfg: TTSConfig) -> bool:
        try:
            import pyttsx3  # noqa: F401
            return True
        except Exception:
            return False
