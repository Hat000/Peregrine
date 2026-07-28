"""Built-in Claude Code voice — the zero-dependency system-speech fallback.

No pip installs, no model downloads, no API key: it drives the operating
system's own speech engine through a shell command.

* **Windows** -> PowerShell ``System.Speech.Synthesis.SpeechSynthesizer``
* **macOS**   -> ``say``
* **Linux**   -> ``spd-say`` (falls back to ``espeak`` / ``espeak-ng``)

Text is fed on stdin (never interpolated into the command) so quotes and
newlines are safe. Barge-in kills the subprocess, which stops playback at once.
Override everything with ``tts.claude_voice_cmd`` — a template where ``{rate}``,
``{volume}`` and ``{voice}`` are substituted and text arrives on stdin.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading

from ..config import TTSConfig
from .base import TTSBackend

_PS_SCRIPT = (
    "Add-Type -AssemblyName System.Speech;"
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "$s.Rate = {rate}; $s.Volume = {volume};"
    "{voice_select}"
    "$t = [Console]::In.ReadToEnd(); $s.Speak($t);"
)


class ClaudeVoiceTTS(TTSBackend):
    name = "claude"

    def __init__(self, cfg: TTSConfig) -> None:
        super().__init__(cfg)
        self._proc: subprocess.Popen | None = None
        self._custom = cfg.claude_voice_cmd

    # -- command construction ------------------------------------------- #
    def _build_cmd(self) -> tuple[list[str], bytes]:
        """Return (argv, stdin_bytes) for the current platform."""
        if self._custom:
            cmd = self._custom.format(
                rate=self._os_rate(), volume=int(self.volume * 100), voice=self.voice,
            )
            return (["/bin/sh", "-c", cmd] if sys.platform != "win32"
                    else ["cmd", "/c", cmd]), b""

        if sys.platform == "win32":
            ps = _powershell_exe()
            voice_select = f'$s.SelectVoice("{self.voice}");' if self.voice else ""
            script = _PS_SCRIPT.format(
                rate=self._ps_rate(), volume=int(self.volume * 100),
                voice_select=voice_select,
            )
            return [ps, "-NoProfile", "-NonInteractive", "-Command", script], b"stdin"

        if sys.platform == "darwin":
            argv = ["say", "-r", str(int(175 * self.rate))]
            if self.voice:
                argv += ["-v", self.voice]
            return argv, b"stdin"

        # Linux
        if shutil.which("spd-say"):
            return ["spd-say", "-e", "-w", "-r", str(self._spd_rate())], b"stdin"
        espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        argv = [espeak or "espeak", "-s", str(int(175 * self.rate)), "--stdin"]
        return argv, b"stdin"

    def _os_rate(self) -> int:
        return int(175 * self.rate)

    def _ps_rate(self) -> int:
        # PowerShell SpeechSynthesizer.Rate is -10..10 (0 = normal).
        return max(-10, min(10, round((self.rate - 1.0) * 10)))

    def _spd_rate(self) -> int:
        # speech-dispatcher rate is -100..100.
        return max(-100, min(100, round((self.rate - 1.0) * 50)))

    # -- speak ----------------------------------------------------------- #
    def _speak_blocking(self, text: str, stop: threading.Event) -> bool:
        if stop.is_set():
            return False
        argv, _ = self._build_cmd()
        try:
            self._proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return False
        proc = self._proc
        assert proc.stdin is not None
        try:
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        # Poll so a stop() (which terminates the process) returns promptly.
        while proc.poll() is None:
            if stop.wait(0.05):
                self._terminate()
                return False
        self._proc = None
        return not stop.is_set()

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
        if cfg.claude_voice_cmd:
            return True
        if sys.platform == "win32":
            return bool(_powershell_exe())
        if sys.platform == "darwin":
            return bool(shutil.which("say"))
        return bool(shutil.which("spd-say") or shutil.which("espeak-ng")
                    or shutil.which("espeak"))


def _powershell_exe() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")
