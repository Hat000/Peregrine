"""Pluggable TTS backends and the factory that selects one.

Backends: ``sapi`` (pyttsx3), ``piper`` (local neural), ``elevenlabs`` (cloud),
``claude`` (built-in system voice, zero-dependency). ``backend = "auto"`` picks
the first that initialises, preferring quality then zero-friction.
"""

from __future__ import annotations

from ..config import TTSConfig
from .base import TTSBackend
from .claude_voice import ClaudeVoiceTTS
from .elevenlabs import ElevenLabsTTS
from .piper import PiperTTS
from .sapi import SapiTTS

_REGISTRY: dict[str, type[TTSBackend]] = {
    "sapi": SapiTTS,
    "piper": PiperTTS,
    "elevenlabs": ElevenLabsTTS,
    "claude": ClaudeVoiceTTS,
}

# Order tried when backend == "auto": quality first (piper/elevenlabs if set
# up), then the built-in system voice. ``claude`` (System.Speech / say / spd-say)
# is preferred over ``sapi`` (pyttsx3) because pyttsx3's SAPI5 driver only speaks
# the FIRST utterance per process — every later one silently no-ops — which is
# fatal for a stream of sentences. ``sapi`` stays available as an explicit pick.
_AUTO_ORDER = ["piper", "elevenlabs", "claude", "sapi"]


class NoTTSBackend(TTSBackend):
    """Last-resort no-op backend that prints instead of speaking, so the app
    still runs on a machine with no working speech engine at all."""

    name = "none"

    def _speak_blocking(self, text: str, stop) -> bool:  # noqa: D401
        print(f"[tts:none] {text}")
        return not stop.is_set()

    @classmethod
    def is_available(cls, cfg: TTSConfig) -> bool:
        return True


# "none" is selectable (silent / print mode) but never auto-picked.
_REGISTRY["none"] = NoTTSBackend


def available_backends(cfg: TTSConfig) -> list[str]:
    return [name for name, cls in _REGISTRY.items() if cls.is_available(cfg)]


def create_tts(cfg: TTSConfig) -> TTSBackend:
    """Instantiate the configured backend, or auto-select, with a safe
    fallback to the no-op backend."""
    choice = (cfg.backend or "auto").lower()
    if choice != "auto":
        cls = _REGISTRY.get(choice)
        if cls is None:
            raise ValueError(f"unknown TTS backend {choice!r}; "
                             f"choose from {sorted(_REGISTRY)} or 'auto'")
        if cls.is_available(cfg):
            return cls(cfg)
        # Configured backend not usable — fall through to auto with a warning.
        print(f"[nocturne] TTS backend {choice!r} unavailable; auto-selecting.")
    for name in _AUTO_ORDER:
        cls = _REGISTRY[name]
        if cls.is_available(cfg):
            return cls(cfg)
    return NoTTSBackend(cfg)


__all__ = [
    "TTSBackend", "SapiTTS", "PiperTTS", "ElevenLabsTTS", "ClaudeVoiceTTS",
    "NoTTSBackend", "create_tts", "available_backends",
]
