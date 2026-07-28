"""Tests for the subscription CLI driver: raw stream-json translation, plus a
real subprocess round-trip against a fake `claude` that emits canned output."""

import sys
from pathlib import Path

import pytest

from nocturne.claude_cli import ClaudeCLISession, locate_cli
from nocturne.config import AgentConfig
from nocturne.events import (
    ProseComplete, ProseDelta, ToolFinished, ToolStarted, TurnFinished,
)

FAKE = Path(__file__).resolve().parent / "fake_claude.py"


def _session(partial=True):
    return ClaudeCLISession(AgentConfig(include_partial_messages=partial))


# --- pure translation ------------------------------------------------------ #
def test_stream_event_text_delta():
    s = _session()
    out = s._translate({"type": "stream_event", "event": {
        "type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}})
    assert out == [ProseDelta("hi")]


def test_assistant_tool_use_raw_json():
    s = _session()
    out = s._translate({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "Edit", "input": {"file_path": "a.py"}}]}})
    assert isinstance(out[0], ToolStarted)
    assert out[0].name == "Edit" and out[0].tool_id == "t1"


def test_assistant_text_skipped_when_partial():
    s = _session(partial=True)
    out = s._translate({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "streamed already"}]}})
    assert out == []


def test_assistant_text_when_not_partial():
    s = _session(partial=False)
    out = s._translate({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "whole"}]}})
    assert out == [ProseComplete("whole")]


def test_user_tool_result_raw_json():
    s = _session()
    out = s._translate({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "ok", "is_error": False}]}})
    assert isinstance(out[0], ToolFinished)
    assert out[0].content == "ok" and out[0].tool_id == "t1"


def test_result_message_captures_usage_and_session():
    s = _session()
    out = s._translate({"type": "result", "subtype": "success", "session_id": "abc",
                        "usage": {"input_tokens": 10}, "total_cost_usd": 0.02})
    assert isinstance(out[0], TurnFinished)
    assert out[0].session_id == "abc" and s.session_id == "abc"
    assert out[0].cost_usd == 0.02


def test_tool_result_list_content():
    s = _session()
    out = s._translate({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t2",
         "content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]}]}})
    assert out[0].content == "line1\nline2"


def test_locate_cli_explicit_valid(tmp_path):
    fake = tmp_path / "claude.exe"
    fake.write_text("x")
    assert locate_cli(str(fake)) == str(fake)


def test_locate_cli_explicit_missing_falls_through():
    # A bogus explicit path must NOT be returned verbatim; it falls through to
    # auto-detection (which finds the real CLI, or returns None on a bare box).
    assert locate_cli("Z:/nope/claude.exe") != "Z:/nope/claude.exe"


# --- real subprocess round-trip against the fake CLI ----------------------- #
class _FakeCLISession(ClaudeCLISession):
    def _build_argv(self):
        return [sys.executable, str(FAKE)]


@pytest.mark.asyncio
async def test_driver_round_trip_multi_turn():
    cfg = AgentConfig(claude_cli_path=sys.executable)  # any real file so start() proceeds
    s = _FakeCLISession(cfg)
    await s.start()
    try:
        # turn 1
        events = [ev async for ev in s.run("hello")]
        kinds = [type(e).__name__ for e in events]
        assert "ProseDelta" in kinds
        assert "ToolStarted" in kinds
        assert "ToolFinished" in kinds
        assert isinstance(events[-1], TurnFinished)
        prose = "".join(e.text for e in events if isinstance(e, ProseDelta))
        assert "You said hello." in prose
        assert s.session_id == "fake-123"

        # turn 2 on the SAME persistent process
        events2 = [ev async for ev in s.run("again")]
        assert isinstance(events2[-1], TurnFinished)
        prose2 = "".join(e.text for e in events2 if isinstance(e, ProseDelta))
        assert "again." in prose2
    finally:
        await s.close()
