"""Piper backend — local neural TTS, good for bedtime.

Runs the ``piper`` binary, reads raw 16-bit PCM off its stdout, and plays it
through sounddevice in small chunks so barge-in can cut it mid-word. Requires a
one-time voice-model (.onnx) download plus the piper binary on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

from ..config import TTSConfig
from .base import TTSBackend

_DEFAULT_SR = 22050
_CHUNK = 4096  # bytes per playback chunk (~93 ms of 16-bit mono @ 22 kHz)


class PiperTTS(TTSBackend):
    name = "piper"

    def __init__(self, cfg: TTSConfig) -> None:
        super().__init__(cfg)
        self._exe = cfg.piper_exe or "piper"
        self._model = cfg.piper_model
        self._sr = _read_sample_rate(self._model)
        self._proc: subprocess.Popen | None = None

    def _speak_blocking(self, text: str, stop: threading.Event) -> bool:
        if stop.is_set():
            return False
        import numpy as np
        import sounddevice as sd

        length_scale = 1.0 / max(0.5, self.rate)  # lower = faster
        cmd = [
            self._exe, "--model", self._model, "--output_raw",
            "--length_scale", f"{length_scale:.3f}",
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        proc = self._proc
        assert proc.stdin and proc.stdout
        finished = True
        stream = sd.RawOutputStream(samplerate=self._sr, channels=1, dtype="int16")
        try:
            proc.stdin.write(text.encode("utf-8") + b"\n")
            proc.stdin.close()
            stream.start()
            vol = float(self.volume)
            while True:
                if stop.is_set():
                    finished = False
                    break
                chunk = proc.stdout.read(_CHUNK)
                if not chunk:
                    break
                if vol < 0.999:
                    samples = np.frombuffer(chunk, dtype=np.int16)
                    chunk = (samples * vol).astype(np.int16).tobytes()
                stream.write(chunk)
        finally:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass
            self._terminate()
        return finished and not stop.is_set()

    def _terminate(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _interrupt(self) -> None:
        self._terminate()

    @classmethod
    def is_available(cls, cfg: TTSConfig) -> bool:
        exe = cfg.piper_exe or "piper"
        has_exe = bool(shutil.which(exe) or (os.path.isfile(exe) and os.access(exe, os.X_OK)))
        has_model = bool(cfg.piper_model and Path(cfg.piper_model).is_file())
        try:
            import numpy  # noqa: F401
            import sounddevice  # noqa: F401
        except Exception:
            return False
        return has_exe and has_model


def _read_sample_rate(model_path: str) -> int:
    """Piper ships a ``<model>.onnx.json`` with the audio sample rate."""
    if not model_path:
        return _DEFAULT_SR
    cfg_path = Path(model_path + ".json")
    if not cfg_path.is_file():
        cfg_path = Path(model_path).with_suffix(".onnx.json")
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        return int(data.get("audio", {}).get("sample_rate", _DEFAULT_SR))
    except Exception:
        return _DEFAULT_SR
