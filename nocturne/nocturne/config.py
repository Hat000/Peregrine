"""Configuration: a single TOML file mapped onto dataclasses.

Everything Nocturne needs to know lives in one ``nocturne.toml``. If no file is
found we fall back to the dataclass defaults, so the app runs out of the box.
Resolution order for the config path:

1. ``--config PATH`` on the command line (handled by the caller).
2. ``$NOCTURNE_CONFIG``.
3. ``./nocturne.toml`` in the current directory.
4. ``~/.config/nocturne/nocturne.toml``.
5. The packaged ``nocturne.toml`` shipped next to the source tree.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints


# --------------------------------------------------------------------------- #
# Sub-sections
# --------------------------------------------------------------------------- #
@dataclass
class TTSConfig:
    # backend: "sapi" | "piper" | "elevenlabs" | "claude" | "auto"
    # "auto" picks the first backend that imports and initialises cleanly,
    # preferring, in order: piper, sapi, claude, elevenlabs.
    backend: str = "auto"
    voice: str = ""             # backend-specific voice id / name ("" = default)
    rate: float = 1.6          # speaking rate multiplier (1.0 = natural)
    volume: float = 0.9        # 0.0 - 1.0
    # Piper-specific
    piper_model: str = ""       # path to a .onnx voice model
    piper_exe: str = "piper"   # piper binary on PATH or an absolute path
    # ElevenLabs-specific
    elevenlabs_api_key_env: str = "ELEVENLABS_API_KEY"
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = "eleven_turbo_v2_5"
    # Claude Code built-in voice backend
    claude_voice_cmd: str = ""  # override command template; see tts/claude_voice.py


@dataclass
class SpeechConfig:
    # Verbosity for the speech filter.
    #   "quiet"   -> prose only, no tool breadcrumbs
    #   "normal"  -> prose + terse tool breadcrumbs + result summaries
    #   "verbose" -> normal + a little more detail on results
    verbosity: str = "normal"
    speak_tool_use: bool = True        # "editing config.py"
    speak_tool_results: bool = True    # "wrote 42 lines to motor.py"
    # Blocks that must never be read aloud verbatim (code/diffs/logs); they are
    # summarised instead. Matched as fenced ``` blocks and heuristics.
    suppress_code_blocks: bool = True
    max_result_chars: int = 2000       # tool results longer than this are always summarised


@dataclass
class SpotifyConfig:
    enabled: bool = True
    client_id_env: str = "SPOTIFY_CLIENT_ID"
    client_secret_env: str = "SPOTIFY_CLIENT_SECRET"
    redirect_uri: str = "http://127.0.0.1:8888/callback"
    # Context URI to play as the "thinking" music. Empty = just resume whatever
    # was last playing on the active device.
    thinking_context_uri: str = ""
    device_id: str = ""                # explicit device; "" = current active device
    restore_on_exit: bool = True       # snapshot playback on start, restore on exit
    fade_ms: int = 900                 # fade in/out duration for play/pause
    duck_volume: int = 20              # % volume while ducking under speech (if not pausing)
    duck_instead_of_pause: bool = False


@dataclass
class AudioConfig:
    # Local ambient fallback used when Spotify is unavailable.
    ambient_fallback: str = ""         # path to a loopable wav; "" = generated pad
    earcon_set: str = "default"        # name of a subfolder under assets/earcons
    earcon_volume: float = 0.5
    output_device: str = ""            # sounddevice device name/index; "" = default


@dataclass
class ContextConfig:
    # Model context window in tokens. 200k standard; 1M on newer models.
    window_tokens: int = 200_000
    # Fraction thresholds at which to speak a warning (once each per session).
    warn_thresholds: list[float] = field(default_factory=lambda: [0.75, 0.90])
    # Auto-compaction typically triggers near the top of the window; warn just
    # before it so the user can choose to wrap up vs. let it compact.
    warn_before_compaction_at: float = 0.92
    speak_rate_limit_warnings: bool = True


@dataclass
class BedtimeConfig:
    # mode: "bedtime" (confirm-all-writes / read-only-ish) or "focused" (more autonomous)
    mode: str = "focused"
    idle_sleep_minutes: float = 12.0   # after this much silence, sleep (stop TTS + pause music)
    idle_warn_seconds: float = 20.0    # soft "still there?" this long before sleeping
    wind_down_voice: str = ""          # softer/warmer voice id used in bedtime mode ("" = same)
    wind_down_volume: float = 0.6      # lower default volume at night
    confirm_dangerous: bool = True     # spoken confirmation gate before risky tools


@dataclass
class SessionConfig:
    log_dir: str = "~/.local/state/nocturne/logs"
    write_transcript: bool = True
    resume_last: bool = False          # continue the previous session_id on startup
    session_id: str = ""               # explicit session id to resume ("" = new)
    # Away notifications (optional): "none" | "ntfy" | "pushover"
    away_notifier: str = "none"
    ntfy_topic: str = ""
    ntfy_server: str = "https://ntfy.sh"
    away_after_seconds: float = 90.0   # ping if a confirm/prompt goes unanswered this long


@dataclass
class AgentConfig:
    # driver: how Nocturne talks to Claude.
    #   "cli" (default) -> spawn the Claude Code CLI in headless stream-json mode.
    #                      Runs on your Pro/Max SUBSCRIPTION login. No API key,
    #                      no API credits. This is the point of Nocturne.
    #   "sdk"           -> the Claude Agent SDK (needs ANTHROPIC_API_KEY, billed
    #                      as pay-per-token API credits). Opt in only if you want
    #                      API billing.
    driver: str = "cli"
    # Path to the claude executable. "" = auto-detect (PATH, then the Windows
    # install under %APPDATA%/Claude/claude-code/<ver>/claude.exe).
    claude_cli_path: str = ""

    model: str = ""                    # "" = CLI/SDK default
    system_prompt: str = ""            # extra system prompt appended; "" = none
    cwd: str = ""                      # working dir for the Claude Code session; "" = cwd
    # allowed_tools: [] = default set. Populate to restrict.
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    # permission_mode: "default" | "acceptEdits" | "plan" | "bypassPermissions"
    permission_mode: str = "default"
    include_partial_messages: bool = True
    # Which on-disk settings to honour (CLAUDE.md, hooks, permissions). Loading
    # "project"/"user"/"local" makes Nocturne behave like the normal CLI. Set to
    # [] for a hermetic session. (SDK driver only; the CLI honours them natively.)
    setting_sources: list[str] = field(default_factory=lambda: ["user", "project", "local"])


@dataclass
class Config:
    tts: TTSConfig = field(default_factory=TTSConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    spotify: SpotifyConfig = field(default_factory=SpotifyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    bedtime: BedtimeConfig = field(default_factory=BedtimeConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Where this config was loaded from (for diagnostics / earcon assets).
    source_path: str = ""

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, explicit_path: str | os.PathLike[str] | None = None) -> "Config":
        path = _resolve_path(explicit_path)
        if path is None:
            return cls()
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        cfg = _from_dict(cls, data)
        cfg.source_path = str(path)
        return cfg


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_path(explicit: str | os.PathLike[str] | None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("NOCTURNE_CONFIG")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.cwd() / "nocturne.toml")
    candidates.append(Path.home() / ".config" / "nocturne" / "nocturne.toml")
    candidates.append(Path(__file__).resolve().parent.parent / "nocturne.toml")
    for c in candidates:
        if c and c.is_file():
            return c
    return None


def _from_dict(dc_type: type, data: dict[str, Any]) -> Any:
    """Recursively build a (nested) dataclass from a plain dict, ignoring
    unknown keys and keeping defaults for missing ones."""
    kwargs: dict[str, Any] = {}
    # ``from __future__ import annotations`` stringises ``field.type``, so
    # resolve real types (needed to recognise nested dataclasses).
    type_hints = get_type_hints(dc_type)
    for f in fields(dc_type):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = type_hints.get(f.name)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(ftype, value)  # type: ignore[arg-type]
        else:
            kwargs[f.name] = value
    return dc_type(**kwargs)
