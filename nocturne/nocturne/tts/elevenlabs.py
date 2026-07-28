"""ElevenLabs backend — cloud neural TTS, best quality.

Streams raw PCM from the ElevenLabs streaming endpoint (no mp3 decoder needed)
and plays it chunk-by-chunk through sounddevice so barge-in stays responsive.
Needs an API key in the env var named by ``tts.elevenlabs_api_key_env``.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request

from ..config import TTSConfig
from .base import TTSBackend

_API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream?output_format=pcm_22050"
_SR = 22050
_CHUNK = 4096


class ElevenLabsTTS(TTSBackend):
    name = "elevenlabs"

    def __init__(self, cfg: TTSConfig) -> None:
        super().__init__(cfg)
        self._api_key = os.environ.get(cfg.elevenlabs_api_key_env, "")
        self._voice_id = cfg.elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM"  # "Rachel"
        self._model = cfg.elevenlabs_model

    def _speak_blocking(self, text: str, stop: threading.Event) -> bool:
        if stop.is_set():
            return False
        import numpy as np
        import sounddevice as sd

        voice = self.voice or self._voice_id
        url = _API.format(voice=voice)
        # ElevenLabs exposes speed via voice_settings on some models; otherwise
        # rate is applied at playback time by the OS. We keep 1.0 here and let
        # the caller pick a faster voice/model if needed.
        body = json.dumps({
            "text": text,
            "model_id": self._model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={
                "xi-api-key": self._api_key,
                "content-type": "application/json",
                "accept": "audio/pcm",
            },
        )
        finished = True
        stream = sd.RawOutputStream(samplerate=_SR, channels=1, dtype="int16")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                stream.start()
                vol = float(self.volume)
                while True:
                    if stop.is_set():
                        finished = False
                        break
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    if vol < 0.999:
                        s = np.frombuffer(chunk, dtype=np.int16)
                        chunk = (s * vol).astype(np.int16).tobytes()
                    stream.write(chunk)
        finally:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass
        return finished and not stop.is_set()

    @classmethod
    def is_available(cls, cfg: TTSConfig) -> bool:
        if not os.environ.get(cfg.elevenlabs_api_key_env):
            return False
        try:
            import numpy  # noqa: F401
            import sounddevice  # noqa: F401
        except Exception:
            return False
        return True
