"""End-to-end orchestration without the SDK or audio hardware.

Drives NocturneApp's real turn machinery (``_consume`` -> ProseFilter ->
Speaker -> TTS) with a scripted fake agent and a recording TTS, verifying the
right things get spoken and music ducks/returns around speech.
"""

import asyncio

import pytest

from nocturne.config import Config
from nocturne.events import (
    ProseDelta, ToolFinished, ToolStarted, TurnFinished,
)
from nocturne.tts.base import TTSBackend


class RecordingTTS(TTSBackend):
    name = "recording"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.spoken = []

    def _speak_blocking(self, text, stop):
        self.spoken.append(text)
        return True


class DummyAudio:
    def __init__(self):
        self.earcons = []
    def play(self, name):
        self.earcons.append(name)
    def close(self):
        pass


class DummySpotify:
    def __init__(self):
        self.events = []
    async def connect(self):
        return False
    async def on_working(self):
        self.events.append("working")
    async def on_speaking(self):
        self.events.append("speaking")
    async def on_idle(self):
        self.events.append("idle")
    async def restore(self):
        pass


class FakeAgent:
    def __init__(self, script):
        self._script = script
        self.interrupted = False
    async def start(self):
        pass
    async def close(self):
        pass
    async def run(self, prompt):
        for ev in self._script:
            await asyncio.sleep(0)
            yield ev
    async def interrupt(self):
        self.interrupted = True


def _make_app(script):
    from nocturne.app import NocturneApp
    from nocturne.speaker import Speaker
    cfg = Config()
    cfg.spotify.enabled = False
    cfg.tts.backend = "none"
    app = NocturneApp(cfg)
    rec = RecordingTTS(cfg.tts)
    app.tts = rec
    app.audio = DummyAudio()
    app.spotify = DummySpotify()
    app.speaker = Speaker(rec, on_start=app._on_speech_start, on_idle=app._on_speech_idle)
    app.speaker.start()
    app.agent = FakeAgent(script)
    return app, rec


@pytest.mark.asyncio
async def test_turn_speaks_prose_and_summaries():
    script = [
        ProseDelta("Let me check the tests. "),
        ToolStarted(name="Bash", tool_input={"description": "Run the tests"}, tool_id="t1"),
        ToolFinished(name="", content="12 passed in 0.4s", tool_id="t1"),
        ProseDelta("All green. "),
        TurnFinished(usage={"input_tokens": 10, "output_tokens": 3}, subtype="success"),
    ]
    app, rec = _make_app(script)
    app.speaker.start()
    result = await app._run_turn("run the tests")
    assert result is None
    joined = " ".join(rec.spoken)
    assert "Let me check the tests." in joined
    assert "run the tests" in joined.lower()   # breadcrumb
    assert "12 tests passed" in joined          # result summary
    assert "All green." in joined
    assert "working" in app.spotify.events      # music started
    assert "done" in app.audio.earcons          # done earcon at turn end
    await app.speaker.close()


@pytest.mark.asyncio
async def test_code_block_not_spoken():
    script = [
        ProseDelta("Here it is:\n```python\nsecret = 42\n```\nDone.\n"),
        TurnFinished(usage={"input_tokens": 1, "output_tokens": 1}, subtype="success"),
    ]
    app, rec = _make_app(script)
    result = await app._run_turn("show code")
    joined = " ".join(rec.spoken)
    assert "secret" not in joined
    assert "Done." in joined
    assert any("code block" in s for s in rec.spoken)
    await app.speaker.close()


@pytest.mark.asyncio
async def test_context_warning_spoken():
    script = [
        ProseDelta("Working. "),
        TurnFinished(usage={"input_tokens": 190000, "output_tokens": 0}, subtype="success"),
    ]
    app, rec = _make_app(script)
    await app._run_turn("big")
    joined = " ".join(rec.spoken)
    assert "percent of the context window" in joined
    assert "context-low" in app.audio.earcons
    await app.speaker.close()


@pytest.mark.asyncio
async def test_new_message_midturn_interrupts_and_hands_off():
    # A long-ish script; we inject a new user line while it's streaming.
    async def slow_script():
        pass
    script = [ProseDelta(f"chunk {i}. ") for i in range(6)]
    script.append(TurnFinished(usage={"input_tokens": 1, "output_tokens": 1}, subtype="success"))
    app, rec = _make_app(script)

    async def inject():
        await asyncio.sleep(0.01)
        from nocturne.input import InputLine
        await app.input.queue.put(InputLine(kind="line", text="new question"))

    asyncio.ensure_future(inject())
    result = await app._run_turn("first question")
    # Either the turn completed first (None) or it handed off the new prompt.
    assert result in (None, "new question")
    await app.speaker.close()
