"""Keyboard input: an async prompt, barge-in, and control-chord hotkeys.

Input is keyboard-only (no speech recognition). One ``PromptSession`` stays live
the whole time, so the user can type the next message even while Claude is
working or speaking. **Barge-in** is wired to the buffer's ``on_text_changed``
event — the moment a key changes the line while TTS is talking, speech is cut.
This observes typing without hijacking key handling, so Enter/Backspace/arrows
keep working normally.

Instant actions that shouldn't wait for Enter use control chords (a plain
single key would collide with typing a message):

    Ctrl-K   skip / stop the current utterance
    Ctrl-R   replay the last response
    Ctrl-N   read the last response slower
    Ctrl-T   toggle bedtime / focused mode
    Ctrl-G   speak remaining context

Everything else is a typed line: a message, or a ``/command`` (see app.py).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass
class InputLine:
    kind: str          # "line" | "eof"
    text: str = ""


HotkeyAction = Callable[[], Awaitable[None] | None]


class InputController:
    def __init__(
        self,
        on_barge_in: Callable[[], None] | None = None,
        hotkeys: dict[str, HotkeyAction] | None = None,
        prompt_text: Callable[[], str] | None = None,
    ) -> None:
        self.queue: asyncio.Queue[InputLine] = asyncio.Queue()
        self._on_barge_in = on_barge_in
        self._hotkeys = hotkeys or {}
        self._prompt_text = prompt_text or (lambda: "› ")
        self._session = None
        self._running = False

    # ------------------------------------------------------------------ #
    def _build_bindings(self):
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()
        # chord name -> hotkey key used in self._hotkeys
        chords = {
            "c-k": "skip",
            "c-r": "replay",
            "c-n": "slower",
            "c-t": "mode",
            "c-g": "context",
        }

        def make(action_key):
            def _handler(event):
                action = self._hotkeys.get(action_key)
                if action is None:
                    return
                result = action()
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            return _handler

        for chord, action_key in chords.items():
            kb.add(chord)(make(action_key))
        return kb

    def _on_text_changed(self, _buf) -> None:
        if self._on_barge_in is not None:
            self._on_barge_in()

    async def run(self) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.patch_stdout import patch_stdout

        self._session = PromptSession(key_bindings=self._build_bindings())
        self._session.default_buffer.on_text_changed += self._on_text_changed
        self._running = True
        with patch_stdout():
            while self._running:
                try:
                    line = await self._session.prompt_async(self._prompt_text())
                except (EOFError, KeyboardInterrupt):
                    await self.queue.put(InputLine(kind="eof"))
                    break
                await self.queue.put(InputLine(kind="line", text=line))

    def stop(self) -> None:
        self._running = False

    async def get(self) -> InputLine:
        return await self.queue.get()

    async def ask_line(self, message: str) -> str:
        """Prompt once for a free-form answer (used by confirmation gates).

        This reuses the running session's app if present; otherwise it falls
        back to a fresh minimal prompt.
        """
        from prompt_toolkit import PromptSession
        session = self._session or PromptSession()
        return await session.prompt_async(message)
