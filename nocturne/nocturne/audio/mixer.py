"""Audio engine: earcons + the local ambient fallback, with fades.

Earcons are short and must be low-latency and non-blocking, so on Windows we
use ``winsound`` (zero dependency, mixes at the OS level) and elsewhere fall
back to sounddevice. The ambient loop (a gentle pad, used only when Spotify is
unavailable) runs on a sounddevice callback stream whose gain eases toward a
target so start/stop/duck are smooth fades, never hard cuts.
"""

from __future__ import annotations

import math
import sys
import threading
import wave
from pathlib import Path

from ..config import AudioConfig
from .earcons import SAMPLE_RATE, ensure_default_set

try:  # optional; ambient + non-Windows earcons need these
    import numpy as np
    import sounddevice as sd
    _HAS_SD = True
except Exception:  # pragma: no cover - depends on the machine
    _HAS_SD = False

_IS_WIN = sys.platform == "win32"
if _IS_WIN:
    import winsound


class AudioEngine:
    def __init__(self, cfg: AudioConfig, assets_root: Path) -> None:
        self.cfg = cfg
        self._earcon_dir = Path(assets_root) / "earcons" / (cfg.earcon_set or "default")
        ensure_default_set(Path(assets_root) / "earcons" / "default")
        self._earcon_wavs: dict[str, Path] = {}
        self._earcon_pcm: dict[str, "np.ndarray"] = {}
        self._load_earcons()

        # ambient state
        self._amb_stream = None
        self._amb_buf: "np.ndarray | None" = None
        self._amb_pos = 0
        self._amb_gain = 0.0
        self._amb_target = 0.0
        self._amb_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Earcons
    # ------------------------------------------------------------------ #
    def _load_earcons(self) -> None:
        for wav in self._earcon_dir.glob("*.wav"):
            self._earcon_wavs[wav.stem] = wav
        if _HAS_SD:
            for name, path in self._earcon_wavs.items():
                try:
                    self._earcon_pcm[name] = _read_wav_f32(path) * self.cfg.earcon_volume
                except Exception:
                    pass

    def play(self, name: str) -> None:
        """Fire an earcon; never blocks, never raises."""
        path = self._earcon_wavs.get(name)
        if path is None:
            return
        try:
            if _HAS_SD and name in self._earcon_pcm:
                sd.play(self._earcon_pcm[name], SAMPLE_RATE)
            elif _IS_WIN:
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Ambient fallback (used when Spotify is down)
    # ------------------------------------------------------------------ #
    def ambient_available(self) -> bool:
        return _HAS_SD

    def _ensure_ambient_buffer(self) -> None:
        if self._amb_buf is not None:
            return
        if self.cfg.ambient_fallback and Path(self.cfg.ambient_fallback).is_file():
            self._amb_buf = _read_wav_f32(Path(self.cfg.ambient_fallback))
        else:
            self._amb_buf = _generate_pad()

    def _amb_callback(self, outdata, frames, time_info, status):  # noqa: ANN001
        buf = self._amb_buf
        if buf is None:
            outdata[:] = 0
            return
        # ease gain toward target (~40 ms time constant per block)
        with self._amb_lock:
            step = (self._amb_target - self._amb_gain)
            self._amb_gain += step * 0.15
            gain = self._amb_gain
            pos = self._amb_pos
        n = len(buf)
        idx = (np.arange(frames) + pos) % n
        outdata[:, 0] = buf[idx] * gain
        with self._amb_lock:
            self._amb_pos = (pos + frames) % n

    def ambient_start(self) -> None:
        if not _HAS_SD:
            return
        self._ensure_ambient_buffer()
        with self._amb_lock:
            self._amb_target = 1.0
        if self._amb_stream is None:
            self._amb_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                callback=self._amb_callback, blocksize=1024,
            )
            self._amb_stream.start()

    def ambient_duck(self, gain: float = 0.15) -> None:
        with self._amb_lock:
            self._amb_target = max(0.0, min(1.0, gain))

    def ambient_stop(self) -> None:
        with self._amb_lock:
            self._amb_target = 0.0

    def close(self) -> None:
        try:
            if self._amb_stream is not None:
                self._amb_stream.stop(); self._amb_stream.close()
        except Exception:
            pass
        try:
            if _HAS_SD:
                sd.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
def _read_wav_f32(path: Path) -> "np.ndarray":
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data


def _generate_pad(seconds: float = 8.0) -> "np.ndarray":
    """A soft, slow, loopable drone for the thinking-indicator fallback."""
    t = np.linspace(0, seconds, int(seconds * SAMPLE_RATE), endpoint=False)
    base = 110.0  # A2
    tone = (0.5 * np.sin(2 * math.pi * base * t)
            + 0.3 * np.sin(2 * math.pi * base * 1.5 * t)
            + 0.2 * np.sin(2 * math.pi * base * 2.0 * t))
    lfo = 0.5 + 0.5 * np.sin(2 * math.pi * 0.12 * t)  # slow swell
    env = tone * lfo * 0.12
    # equal-power crossfade at the loop seam
    xf = int(0.25 * SAMPLE_RATE)
    ramp = np.linspace(0, 1, xf)
    env[:xf] *= ramp
    env[-xf:] *= ramp[::-1]
    return env.astype(np.float32)
