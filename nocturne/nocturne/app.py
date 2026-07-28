"""The orchestrator — the loop over interaction states.

Wires the pieces together and owns the state machine:

    IDLE     -> silence, your-turn earcon, wait for a typed line
    WORKING  -> music (thinking indicator) fades in, breadcrumbs
    SPEAKING -> music ducks, TTS reads the prose
    NEEDS_CONFIRM -> music pauses, urgent earcon, spoken yes/no gate
    ERROR    -> error earcon + one spoken sentence

Everything runs on asyncio with clean cancellation so barge-in can cut speech
mid-sentence. A new typed message during a turn interrupts the agent (draining
the stream per the SDK contract) and starts the next turn.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from . import safety
from .audio import AudioEngine
from .config import Config
from .context_tracker import ContextTracker
from .events import (
    ProseComplete, ProseDelta, RateLimitWarning, StreamError, ToolFinished,
    ToolStarted, TurnFinished,
)
from .input import InputController, InputLine
from .logsession import SessionLog
from .speaker import Speaker
from .speech_filter import (
    ProseFilter, describe_tool_use, summarize_tool_result,
)
from .states import State
from .tts import create_tts

_ASSETS = Path(__file__).resolve().parent.parent / "assets"

_HELP = (
    "Commands: again, faster, slower, skip, stop, context, recap, mode, "
    "voice, help, quit. Chords: control-K skip, control-R replay, "
    "control-N slower, control-T mode, control-G context."
)


class NocturneApp:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = State.IDLE
        self.running = True
        self.mode = cfg.bedtime.mode
        self.debug = bool(os.environ.get("NOCTURNE_DEBUG"))

        self.audio = AudioEngine(cfg.audio, _ASSETS)
        self.tts = create_tts(cfg.tts)
        self.context = ContextTracker(cfg.context)
        self.log = SessionLog(cfg.session.log_dir, cfg.session.write_transcript)

        from .notify import AwayNotifier
        self.notifier = AwayNotifier(cfg.session)

        from .spotify import SpotifyController
        self.spotify = SpotifyController(cfg.spotify, ambient=self.audio)

        self.speaker = Speaker(
            self.tts,
            on_start=self._on_speech_start,
            on_idle=self._on_speech_idle,
        )
        self.input = InputController(
            on_barge_in=self._request_barge_in,
            hotkeys={
                "skip": lambda: asyncio.ensure_future(self.speaker.skip()),
                "replay": lambda: asyncio.ensure_future(self.speaker.replay()),
                "slower": lambda: asyncio.ensure_future(self.speaker.replay(rate_scale=0.75)),
                "mode": lambda: asyncio.ensure_future(self._toggle_mode()),
                "context": lambda: asyncio.ensure_future(self._speak_context()),
            },
            prompt_text=self._prompt_text,
        )

        self.agent = self._make_agent(cfg)

        self._tool_names: dict[str, str] = {}   # tool_id -> name
        self._turn_active = False
        self._pending_confirm: asyncio.Future[bool] | None = None
        self._input_task: asyncio.Task | None = None

    def _make_agent(self, cfg: Config):
        """Select the agent source. Default 'cli' drives the Claude Code CLI on
        the user's subscription; 'sdk' uses the API-key Agent SDK."""
        if cfg.agent.driver == "sdk":
            from .agent_loop import AgentSession
            hooks = safety.build_confirmation_hook(
                self._confirm, enabled=cfg.bedtime.confirm_dangerous)
            return AgentSession(
                cfg.agent, hooks=hooks or None,
                resume=cfg.session.session_id or None,
                continue_conversation=cfg.session.resume_last,
            )
        from .claude_cli import ClaudeCLISession
        return ClaudeCLISession(
            cfg.agent,
            resume=cfg.session.session_id or None,
            continue_conversation=cfg.session.resume_last,
        )

    # ================================================================== #
    # Lifecycle
    # ================================================================== #
    async def run(self) -> None:
        self.speaker.start()
        try:
            await self.agent.start()
        except Exception as e:
            self.audio.play("error")
            self.speaker.say_now(
                "I couldn't start Claude Code. Check the terminal for details.")
            await self.speaker.drain()
            print(f"[nocturne] failed to start Claude Code:\n{e}", file=__import__("sys").stderr)
            await self.speaker.close()
            self.audio.close()
            return
        await self.spotify.connect()
        input_runner = asyncio.ensure_future(self.input.run())

        self.audio.play("your-turn")
        self.speaker.say_now(self._greeting())
        await self.speaker.drain()

        try:
            await self._main_loop()
        finally:
            self.input.stop()
            input_runner.cancel()
            await self._shutdown()

    def _greeting(self) -> str:
        source = ("your Claude subscription" if self.cfg.agent.driver == "cli"
                  else "the Claude API")
        return (f"Nocturne ready, on {source}. {self.mode} mode. "
                f"{self.tts.name} voice. {_HELP}")

    async def _shutdown(self) -> None:
        self.speaker.say_now("Good night.")
        try:
            await asyncio.wait_for(self.speaker.drain(), timeout=4)
        except Exception:
            pass
        await self.spotify.restore()
        await self.speaker.close()
        self.audio.close()
        await self.agent.close()
        if self.log.path:
            print(f"[nocturne] transcript: {self.log.path}")

    # ================================================================== #
    # Main loop
    # ================================================================== #
    async def _main_loop(self) -> None:
        while self.running:
            self._enter(State.IDLE)
            line = await self._await_idle_input()
            if line is None:
                continue
            if line.kind == "eof":
                self.running = False
                break
            text = line.text.strip()
            if not text:
                continue
            if text.startswith("/"):
                await self._handle_command(text[1:].strip())
                continue
            next_prompt: str | None = text
            while next_prompt is not None and self.running:
                next_prompt = await self._run_turn(next_prompt)

    async def _await_idle_input(self) -> InputLine | None:
        """Wait for input in IDLE, with idle-sleep warn/quiet behaviour."""
        warn_after = self.cfg.bedtime.idle_sleep_minutes * 60
        try:
            return await asyncio.wait_for(self.input.get(), timeout=warn_after)
        except asyncio.TimeoutError:
            self.audio.play("your-turn")
            self.speaker.say_now("Still there?")
            try:
                return await asyncio.wait_for(
                    self.input.get(), timeout=self.cfg.bedtime.idle_warn_seconds)
            except asyncio.TimeoutError:
                self.speaker.say_now("Going quiet. I'm here when you need me.")
                await self.spotify.on_idle()
                return await self.input.get()  # sleep until a key finally comes

    # ================================================================== #
    # A single turn
    # ================================================================== #
    async def _run_turn(self, prompt: str) -> str | None:
        """Run one turn. Returns a follow-up prompt if the user interrupted
        with a new message mid-turn, else None."""
        self.log.user(prompt)
        self._turn_active = True
        consume = asyncio.ensure_future(self._consume(prompt))
        try:
            while True:
                get_input = asyncio.ensure_future(self.input.get())
                done, _ = await asyncio.wait(
                    {consume, get_input}, return_when=asyncio.FIRST_COMPLETED)
                if consume in done:
                    get_input.cancel()
                    return None
                get_input_result = get_input.result()
                handoff = await self._handle_turn_input(get_input_result, consume)
                if handoff is _CONTINUE:
                    continue
                return handoff  # None (stop) or a follow-up prompt string
        finally:
            self._turn_active = False

    async def _handle_turn_input(self, line: InputLine, consume: asyncio.Task):
        # Route a yes/no to a waiting confirmation gate first.
        if self._pending_confirm is not None and not self._pending_confirm.done():
            self._pending_confirm.set_result(_is_yes(line.text))
            return _CONTINUE
        if line.kind == "eof":
            await self._interrupt_and_drain(consume)
            self.running = False
            return None
        text = line.text.strip()
        if not text:
            return _CONTINUE
        if text.startswith("/"):
            stop = await self._handle_command(text[1:].strip(), in_turn=True, consume=consume)
            return None if stop else _CONTINUE
        # A new message mid-turn: interrupt and hand off to the next turn.
        await self._interrupt_and_drain(consume)
        return text

    async def _interrupt_and_drain(self, consume: asyncio.Task) -> None:
        await self.speaker.skip()
        await self.agent.interrupt()
        try:
            await consume
        except Exception:
            pass

    async def _consume(self, prompt: str) -> None:
        self._enter(State.WORKING)
        await self.spotify.on_working()
        pf = ProseFilter(self.cfg.speech.suppress_code_blocks)
        try:
            async for ev in self.agent.run(prompt):
                self._handle_event(ev, pf)
        except asyncio.CancelledError:
            raise
        for item in pf.flush():
            self.speaker.enqueue(item)
        await self.speaker.drain()

    def _handle_event(self, ev, pf: ProseFilter) -> None:
        self._dbg_event(ev)
        if isinstance(ev, ProseDelta):
            for item in pf.feed(ev.text):
                self.speaker.enqueue(item)
        elif isinstance(ev, ProseComplete):
            for item in pf.feed(ev.text):
                self.speaker.enqueue(item)
            for item in pf.flush():
                self.speaker.enqueue(item)
            self.log.assistant(ev.text)
        elif isinstance(ev, ToolStarted):
            self._tool_names[ev.tool_id] = ev.name
            brief = describe_tool_use(ev.name, ev.tool_input)
            fp = (ev.tool_input or {}).get("file_path", "")
            if brief:
                self.log.tool(ev.name, brief, fp)
                if self.cfg.speech.speak_tool_use and self.cfg.speech.verbosity != "quiet":
                    self.speaker.say_now(brief, kind="breadcrumb")
        elif isinstance(ev, ToolFinished):
            name = self._tool_names.get(ev.tool_id, "")
            if self.cfg.speech.speak_tool_results and self.cfg.speech.verbosity != "quiet":
                summary = summarize_tool_result(
                    name, ev.content, ev.is_error, self.cfg.speech.max_result_chars)
                if summary:
                    self.speaker.say_now(summary, kind="summary")
        elif isinstance(ev, TurnFinished):
            self._on_turn_finished(ev)
        elif isinstance(ev, RateLimitWarning):
            if self.cfg.context.speak_rate_limit_warnings and ev.status \
                    and ev.status.lower() != "allowed":
                self.audio.play("context-low")
                self.speaker.say_now(f"Rate limit heads up: {ev.message}." if ev.message
                                     else "Heads up, you're approaching a rate limit.")
        elif isinstance(ev, StreamError):
            self._enter(State.ERROR)
            self.audio.play("error")
            self.speaker.say_now(f"Something went wrong. {ev.message}")

    def _dbg_event(self, ev) -> None:
        if not self.debug:
            return
        if isinstance(ev, ProseDelta):
            sys.stderr.write(ev.text)
        elif isinstance(ev, ProseComplete):
            sys.stderr.write(f"\n[prose] {ev.text}\n")
        elif isinstance(ev, ToolStarted):
            sys.stderr.write(f"\n[tool→] {ev.name} {ev.tool_input}\n")
        elif isinstance(ev, ToolFinished):
            body = (ev.content or "")[:200]
            sys.stderr.write(f"[tool←] {'ERR ' if ev.is_error else ''}{body}\n")
        elif isinstance(ev, TurnFinished):
            sys.stderr.write(f"\n[turn done] err={ev.is_error} subtype={ev.subtype} "
                             f"session={ev.session_id}\n")
        elif isinstance(ev, StreamError):
            sys.stderr.write(f"\n[stream error] {ev.message}\n")
        sys.stderr.flush()

    def _on_turn_finished(self, ev: TurnFinished) -> None:
        for w in self.context.update(ev.usage):
            if w.earcon:
                self.audio.play("context-low")
            self.speaker.say_now(w.text)
        if ev.is_error:
            self.audio.play("error")
            # Speak the actual reason (e.g. "Not logged in, please run login")
            # so a screenless user hears what's wrong, not just a code.
            reason = _spoken_error(ev.result_text) or ev.subtype or "an unknown error"
            self.speaker.say_now(f"That didn't go through: {reason}.")
        else:
            self.audio.play("done")

    # ================================================================== #
    # Confirmation gate (called from the safety hook, inside the stream)
    # ================================================================== #
    async def _confirm(self, description: str, tool_name: str, tool_input: dict) -> bool:
        prev = self.state
        self._enter(State.NEEDS_CONFIRM)
        await self.spotify.on_speaking()
        self.audio.play("needs-confirm")
        loop = asyncio.get_running_loop()
        self._pending_confirm = loop.create_future()
        self.speaker.say_now(
            f"Claude wants to {description}. Should I allow it? Say yes or no.")
        await self.speaker.drain()
        away = asyncio.ensure_future(self._away_ping_after(description))
        try:
            approved = await self._pending_confirm
        finally:
            away.cancel()
            self._pending_confirm = None
        self.log.note(f"Confirm '{description}': {'allowed' if approved else 'denied'}")
        self.speaker.say_now("Okay, going ahead." if approved else "Okay, skipping that.")
        self._enter(prev)
        if prev == State.WORKING:
            await self.spotify.on_working()
        return approved

    async def _away_ping_after(self, description: str) -> None:
        """If a confirmation goes unanswered for a while, ping the phone."""
        if not self.notifier.enabled:
            return
        try:
            await asyncio.sleep(self.cfg.session.away_after_seconds)
        except asyncio.CancelledError:
            return
        await self.notifier.ping(f"Claude wants to {description}. Waiting for your yes/no.")

    # ================================================================== #
    # Commands
    # ================================================================== #
    async def _handle_command(self, cmd: str, in_turn: bool = False,
                              consume: asyncio.Task | None = None) -> bool:
        """Handle a /command. Returns True if it ended the current turn."""
        parts = cmd.split()
        name = parts[0].lower() if parts else ""
        arg = parts[1] if len(parts) > 1 else ""

        if name in ("quit", "exit", "q"):
            if in_turn and consume is not None:
                await self._interrupt_and_drain(consume)
            self.running = False
            return True
        if name in ("stop", "cancel"):
            if in_turn and consume is not None:
                await self._interrupt_and_drain(consume)
                self.speaker.say_now("Stopped.")
                return True
            await self.speaker.skip()
            return False
        if name in ("skip", "next"):
            await self.speaker.skip()
            return False
        if name in ("again", "replay"):
            await self.speaker.replay()
            return False
        if name in ("slower",):
            await self.speaker.replay(rate_scale=0.75)
            return False
        if name == "faster":
            self.tts.set_rate(self.tts.rate * 1.15)
            self.speaker.say_now(f"Speaking faster.")
            return False
        if name == "context":
            await self._speak_context()
            return False
        if name == "recap":
            self.speaker.say_now(self.log.recap())
            return False
        if name == "mode":
            await self._toggle_mode(arg or None)
            return False
        if name == "voice":
            self._switch_voice(arg)
            return False
        if name in ("help", "h", "?"):
            self.speaker.say_now(_HELP)
            return False
        self.speaker.say_now(f"Unknown command {name}.")
        return False

    async def _speak_context(self) -> None:
        # Prefer the SDK's exact figure if available (SDK driver only).
        try:
            client = getattr(self.agent, "_client", None)
            if client is not None and hasattr(client, "get_context_usage"):
                usage = await client.get_context_usage()
                pct = _extract_pct(usage)
                if pct is not None:
                    self.speaker.say_now(
                        f"You're using about {round(pct)} percent of the context window.")
                    return
        except Exception:
            pass
        self.speaker.say_now(self.context.report())

    async def _toggle_mode(self, forced: str | None = None) -> None:
        self.mode = forced or ("focused" if self.mode == "bedtime" else "bedtime")
        if self.mode == "bedtime":
            self.tts.set_volume(self.cfg.bedtime.wind_down_volume)
            if self.cfg.bedtime.wind_down_voice:
                self.tts.set_voice(self.cfg.bedtime.wind_down_voice)
            await self._set_permission_mode("plan")
            self.speaker.say_now("Bedtime mode. Planning only; I won't change files.")
        else:
            self.tts.set_volume(self.cfg.tts.volume)
            await self._set_permission_mode(self.cfg.agent.permission_mode or "default")
            self.speaker.say_now("Focused mode.")

    async def _set_permission_mode(self, mode: str) -> None:
        """Live permission-mode switch. SDK exposes set_permission_mode; the CLI
        takes it over the next turn (best-effort here)."""
        client = getattr(self.agent, "_client", None)
        if client is not None and hasattr(client, "set_permission_mode"):
            try:
                await client.set_permission_mode(mode)
            except Exception:
                pass
        self.cfg.agent.permission_mode = mode  # applied on the CLI's next turn

    def _switch_voice(self, backend: str) -> None:
        if not backend:
            self.speaker.say_now(f"Current voice is {self.tts.name}.")
            return
        from .config import TTSConfig
        newcfg = TTSConfig(**{**self.cfg.tts.__dict__, "backend": backend})
        try:
            new_tts = create_tts(newcfg)
        except Exception as e:
            self.speaker.say_now(f"Couldn't switch voice: {e}")
            return
        old = self.tts
        self.tts = new_tts
        self.speaker.tts = new_tts
        asyncio.ensure_future(old.close())
        self.speaker.say_now(f"Switched to the {new_tts.name} voice.")

    # ================================================================== #
    # State + audio glue
    # ================================================================== #
    def _enter(self, state: State) -> None:
        if state == self.state:
            return
        self.state = state
        if state == State.IDLE:
            self.audio.play("your-turn")

    async def _on_speech_start(self) -> None:
        if self.state != State.NEEDS_CONFIRM:
            self._enter(State.SPEAKING)
        await self.spotify.on_speaking()

    async def _on_speech_idle(self) -> None:
        # Speech queue drained: return the music to the thinking level if the
        # agent is still working, else go quiet.
        if self._turn_active:
            self._enter(State.WORKING)
            await self.spotify.on_working()
        else:
            await self.spotify.on_idle()

    def _request_barge_in(self) -> None:
        # Fired by the input buffer on any keystroke; only acts while speaking.
        if self.speaker.is_speaking and self._pending_confirm is None:
            asyncio.ensure_future(self.speaker.skip())

    def _prompt_text(self) -> str:
        if self.state == State.NEEDS_CONFIRM:
            return "yes/no › "
        return "› "


_CONTINUE = object()  # sentinel: keep looping inside a turn


def _spoken_error(result_text: str) -> str:
    """Turn a CLI error result into something worth hearing."""
    t = (result_text or "").strip()
    if not t:
        return ""
    low = t.lower()
    if "not logged in" in low or "/login" in low:
        return ("you're not logged in to Claude Code. Open a terminal, run "
                "claude, and sign in — no A P I key needed")
    # strip markdown-ish noise; keep it short
    return t.replace("·", ",")[:200]


def _is_yes(text: str) -> bool:
    return text.strip().lower() in ("y", "yes", "yeah", "yep", "sure", "ok", "okay", "do it", "go", "allow")


def _extract_pct(usage) -> float | None:
    for attr in ("percent_used", "utilization", "used_percent"):
        v = getattr(usage, attr, None)
        if v is not None:
            return float(v) * (100 if v <= 1 else 1)
    if isinstance(usage, dict):
        for k in ("percent_used", "utilization", "used_percent"):
            if k in usage:
                v = float(usage[k])
                return v * (100 if v <= 1 else 1)
    return None
