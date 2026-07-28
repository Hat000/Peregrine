"""The earcon vocabulary.

Audio is the only output channel, so every state has a short, distinct,
learnable sound. These are synthesised procedurally with the standard library
(no numpy, no binary assets committed) into 16-bit mono WAVs the first time the
app runs. Swap in your own set by pointing ``audio.earcon_set`` at another
folder under ``assets/earcons``.

Vocabulary:
    your-turn      soft rising two-tone   -> control is back with you (IDLE)
    working        low single blip        -> work started (used only if music off)
    done           bright major arpeggio  -> turn finished
    error          low descending buzz    -> something went wrong
    needs-confirm  urgent triple beep     -> risky action awaits your yes/no
    context-low    two falling mid tones  -> context window filling up
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 44100
_AMP = 0.5  # base amplitude headroom before per-earcon volume

# name -> list of notes; a note is (freq_hz, dur_s, wave, gain)
# wave: "sine" | "tri" | "square"
_SPECS: dict[str, list[tuple[float, float, str, float]]] = {
    "your-turn": [(523.25, 0.11, "sine", 0.8), (659.25, 0.14, "sine", 0.8)],
    "working":   [(196.00, 0.10, "sine", 0.7)],
    "done":      [(523.25, 0.09, "sine", 0.9), (659.25, 0.09, "sine", 0.9),
                  (783.99, 0.16, "sine", 0.9)],
    "error":     [(220.00, 0.16, "tri", 0.9), (174.61, 0.22, "tri", 0.9)],
    "needs-confirm": [(587.33, 0.10, "square", 0.6), (0.0, 0.05, "sine", 0.0),
                      (587.33, 0.10, "square", 0.6), (0.0, 0.05, "sine", 0.0),
                      (587.33, 0.14, "square", 0.6)],
    "context-low": [(494.00, 0.16, "sine", 0.8), (392.00, 0.22, "sine", 0.8)],
}

EARCON_NAMES = tuple(_SPECS.keys())


def _wave_sample(kind: str, phase: float) -> float:
    if kind == "sine":
        return math.sin(phase)
    if kind == "tri":
        # triangle from phase in [0, 2pi)
        x = (phase / (2 * math.pi)) % 1.0
        return 4 * abs(x - 0.5) - 1
    if kind == "square":
        return 1.0 if math.sin(phase) >= 0 else -1.0
    return math.sin(phase)


def _render(spec: list[tuple[float, float, str, float]]) -> list[int]:
    samples: list[int] = []
    for freq, dur, kind, gain in spec:
        n = int(dur * SAMPLE_RATE)
        fade = min(int(0.008 * SAMPLE_RATE), n // 2)  # 8 ms click-guard
        for i in range(n):
            if freq <= 0:
                samples.append(0)
                continue
            phase = 2 * math.pi * freq * (i / SAMPLE_RATE)
            v = _wave_sample(kind, phase) * _AMP * gain
            # attack/release envelope
            if i < fade:
                v *= i / fade
            elif i > n - fade:
                v *= max(0.0, (n - i) / fade)
            samples.append(int(max(-1.0, min(1.0, v)) * 32767))
    return samples


def write_earcon(path: Path, name: str) -> None:
    spec = _SPECS[name]
    samples = _render(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"".join(struct.pack("<h", s) for s in samples))


def ensure_default_set(out_dir: Path) -> Path:
    """Generate any missing earcon WAVs in ``out_dir``. Returns the directory."""
    out_dir = Path(out_dir)
    for name in EARCON_NAMES:
        wav = out_dir / f"{name}.wav"
        if not wav.is_file():
            write_earcon(wav, name)
    return out_dir


if __name__ == "__main__":  # regenerate the committed-free default set on demand
    here = Path(__file__).resolve().parents[2] / "assets" / "earcons" / "default"
    ensure_default_set(here)
    print(f"earcons written to {here}")
