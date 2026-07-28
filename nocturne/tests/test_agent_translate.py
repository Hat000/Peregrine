"""Translation of SDK messages -> Nocturne events, using stand-in objects that
mimic the SDK's class names and fields (verified against the installed SDK)."""

from nocturne.agent_loop import AgentSession
from nocturne.config import AgentConfig
from nocturne.events import (
    ProseComplete, ProseDelta, RateLimitWarning, ToolFinished, ToolStarted,
    TurnFinished,
)


def _session(partial=True):
    cfg = AgentConfig(include_partial_messages=partial)
    return AgentSession(cfg)


# --- fakes: names must match the real SDK classes -------------------------- #
class StreamEvent:
    def __init__(self, event):
        self.event = event


class TextBlock:
    def __init__(self, text):
        self.text = text


class ToolUseBlock:
    def __init__(self, id, name, input):
        self.id, self.name, self.input = id, name, input


class ToolResultBlock:
    def __init__(self, tool_use_id, content, is_error=False):
        self.tool_use_id, self.content, self.is_error = tool_use_id, content, is_error


class AssistantMessage:
    def __init__(self, content):
        self.content = content


class UserMessage:
    def __init__(self, content):
        self.content = content
        self.tool_use_result = None


class ResultMessage:
    def __init__(self, usage, session_id="s1", subtype="success"):
        self.usage, self.session_id, self.subtype = usage, session_id, subtype
        self.is_error = False
        self.result = "done"
        self.total_cost_usd = 0.01


class RateLimitInfo:
    def __init__(self, status, utilization):
        self.status, self.utilization = status, utilization
        self.resets_at = None


class RateLimitEvent:
    def __init__(self, info):
        self.rate_limit_info = info


def test_text_delta_becomes_prose_delta():
    s = _session()
    ev = StreamEvent({"type": "content_block_delta",
                      "delta": {"type": "text_delta", "text": "hi"}})
    out = s._translate(ev)
    assert out == [ProseDelta("hi")]


def test_input_json_delta_is_ignored():
    s = _session()
    ev = StreamEvent({"type": "content_block_delta",
                      "delta": {"type": "input_json_delta", "partial_json": "{"}})
    assert s._translate(ev) == []


def test_tool_use_block_from_assistant():
    s = _session()
    msg = AssistantMessage([ToolUseBlock("t1", "Edit", {"file_path": "a.py"})])
    out = s._translate(msg)
    assert isinstance(out[0], ToolStarted)
    assert out[0].name == "Edit" and out[0].tool_id == "t1"


def test_text_block_skipped_when_partial_streaming():
    s = _session(partial=True)
    msg = AssistantMessage([TextBlock("already streamed")])
    assert s._translate(msg) == []


def test_text_block_surfaced_when_not_streaming():
    s = _session(partial=False)
    msg = AssistantMessage([TextBlock("whole block")])
    out = s._translate(msg)
    assert out == [ProseComplete("whole block")]


def test_tool_result_from_user():
    s = _session()
    msg = UserMessage([ToolResultBlock("t1", "12 passed", is_error=False)])
    out = s._translate(msg)
    assert isinstance(out[0], ToolFinished)
    assert out[0].content == "12 passed" and out[0].tool_id == "t1"


def test_result_message_captures_usage_and_session():
    s = _session()
    msg = ResultMessage({"input_tokens": 10, "output_tokens": 5})
    out = s._translate(msg)
    assert isinstance(out[0], TurnFinished)
    assert out[0].usage["input_tokens"] == 10
    assert s.session_id == "s1"


def test_rate_limit_event():
    s = _session()
    msg = RateLimitEvent(RateLimitInfo("allowed_warning", 0.85))
    out = s._translate(msg)
    assert isinstance(out[0], RateLimitWarning)
    assert out[0].status == "allowed_warning"
    assert "85 percent" in out[0].message
