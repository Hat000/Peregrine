"""High-level events the agent loop emits to the orchestrator.

These decouple ``app.py`` from the exact SDK message shapes: ``agent_loop``
translates raw SDK messages / stream events into this small vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProseDelta:
    """A streamed chunk of assistant prose (feed to the sentence filter)."""
    text: str


@dataclass
class ProseComplete:
    """A whole assistant text block (used when partial streaming is off)."""
    text: str


@dataclass
class ToolStarted:
    """Claude invoked a tool. Emit a terse breadcrumb, never the arguments."""
    name: str
    tool_input: dict = field(default_factory=dict)
    tool_id: str = ""


@dataclass
class ToolFinished:
    """A tool returned. Summarise, never read verbatim."""
    name: str
    content: str
    is_error: bool = False
    tool_id: str = ""


@dataclass
class TurnFinished:
    """End of a turn (ResultMessage)."""
    usage: dict | None = None
    session_id: str = ""
    subtype: str = ""
    is_error: bool = False
    result_text: str = ""
    cost_usd: float | None = None


@dataclass
class RateLimitWarning:
    """Rate-limit status crossed into a warning band."""
    status: str = ""
    message: str = ""


@dataclass
class StreamError:
    """Something went wrong talking to the SDK."""
    message: str


AgentEvent = (
    ProseDelta | ProseComplete | ToolStarted | ToolFinished
    | TurnFinished | RateLimitWarning | StreamError
)
