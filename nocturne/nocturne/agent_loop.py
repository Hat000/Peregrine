"""The Claude Agent SDK wrapper.

Owns a persistent, multi-turn ``ClaudeSDKClient`` and translates its raw stream
(``StreamEvent`` text deltas, ``AssistantMessage`` tool-use blocks,
``UserMessage`` tool results, ``ResultMessage`` usage) into the small event
vocabulary in ``events.py``.

The SDK is imported lazily so the rest of Nocturne can be unit-tested without it
installed. Message handling is duck-typed and defensive: attributes are read
with ``getattr`` and ``.get`` so minor SDK shape changes degrade gracefully
rather than crashing the bedside app.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .config import AgentConfig
from .events import (
    AgentEvent, ProseComplete, ProseDelta, RateLimitWarning, StreamError,
    ToolFinished, ToolStarted, TurnFinished,
)


class AgentSession:
    def __init__(
        self,
        cfg: AgentConfig,
        *,
        hooks: dict | None = None,
        can_use_tool: Callable[..., Awaitable[Any]] | None = None,
        resume: str | None = None,
        continue_conversation: bool = False,
    ) -> None:
        self.cfg = cfg
        self._hooks = hooks
        self._can_use_tool = can_use_tool
        self._resume = resume
        self._continue = continue_conversation
        self._client = None
        self.session_id: str = ""
        self._partial = cfg.include_partial_messages

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        from claude_agent_sdk import ClaudeSDKClient  # lazy
        options = self._build_options()
        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None

    def _build_options(self):
        from claude_agent_sdk import ClaudeAgentOptions
        kwargs: dict[str, Any] = {
            "include_partial_messages": self.cfg.include_partial_messages,
        }
        if self.cfg.model:
            kwargs["model"] = self.cfg.model
        if self.cfg.cwd:
            kwargs["cwd"] = self.cfg.cwd
        if self.cfg.allowed_tools:
            kwargs["allowed_tools"] = list(self.cfg.allowed_tools)
        if self.cfg.permission_mode:
            kwargs["permission_mode"] = self.cfg.permission_mode
        if self.cfg.setting_sources is not None:
            kwargs["setting_sources"] = list(self.cfg.setting_sources)
        if self.cfg.system_prompt:
            # Append to the default Claude Code prompt rather than replacing it.
            kwargs["system_prompt"] = {
                "type": "preset", "preset": "claude_code",
                "append": self.cfg.system_prompt,
            }
        if self._hooks:
            kwargs["hooks"] = self._hooks
        if self._can_use_tool is not None:
            kwargs["can_use_tool"] = self._can_use_tool
        if self._resume:
            kwargs["resume"] = self._resume
        elif self._continue:
            kwargs["continue_conversation"] = True
        try:
            return ClaudeAgentOptions(**kwargs)
        except TypeError:
            # An unknown kwarg for this SDK version — drop the soft ones.
            for soft in ("system_prompt", "continue_conversation", "resume",
                         "setting_sources"):
                kwargs.pop(soft, None)
            return ClaudeAgentOptions(**kwargs)

    # ------------------------------------------------------------------ #
    async def run(self, prompt: str) -> AsyncIterator[AgentEvent]:
        """Send ``prompt`` and yield events until the turn's ResultMessage."""
        if self._client is None:
            yield StreamError("agent session not started")
            return
        try:
            await self._client.query(prompt)
        except Exception as e:
            yield StreamError(f"failed to send prompt: {e}")
            return
        try:
            async for msg in self._client.receive_response():
                for ev in self._translate(msg):
                    yield ev
                if isinstance_name(msg, "ResultMessage"):
                    return
        except Exception as e:
            yield StreamError(f"stream error: {e}")

    async def interrupt(self) -> None:
        """Interrupt the current turn. The active ``run()`` generator keeps
        draining until its ResultMessage, satisfying the SDK's drain rule."""
        if self._client is not None:
            try:
                await self._client.interrupt()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def _translate(self, msg: Any) -> list[AgentEvent]:
        cls = type(msg).__name__
        if cls == "StreamEvent":
            return self._from_stream_event(getattr(msg, "event", {}) or {})
        if cls == "AssistantMessage":
            return self._from_assistant(msg)
        if cls == "UserMessage":
            return self._from_user(msg)
        if cls == "ResultMessage":
            return [TurnFinished(
                usage=getattr(msg, "usage", None),
                session_id=_capture_session(self, getattr(msg, "session_id", "")),
                subtype=getattr(msg, "subtype", "") or "",
                is_error=bool(getattr(msg, "is_error", False)),
                result_text=getattr(msg, "result", "") or "",
                cost_usd=getattr(msg, "total_cost_usd", None),
            )]
        if cls == "RateLimitEvent":
            return [self._from_rate_limit(msg)]
        return []

    def _from_stream_event(self, event: dict) -> list[AgentEvent]:
        if event.get("type") == "content_block_delta":
            delta = event.get("delta", {}) or {}
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    return [ProseDelta(text)]
        return []

    def _from_assistant(self, msg: Any) -> list[AgentEvent]:
        out: list[AgentEvent] = []
        for block in getattr(msg, "content", []) or []:
            # SDK content blocks are dataclasses with no `.type` field, so we
            # discriminate by class name.
            bname = type(block).__name__
            if bname in ("ToolUseBlock", "ServerToolUseBlock"):
                out.append(ToolStarted(
                    name=getattr(block, "name", "") or "",
                    tool_input=getattr(block, "input", {}) or {},
                    tool_id=getattr(block, "id", "") or "",
                ))
            elif bname == "TextBlock" and not self._partial:
                # Only surface whole text blocks when we aren't already
                # streaming deltas (else we'd speak everything twice).
                out.append(ProseComplete(getattr(block, "text", "") or ""))
        return out

    def _from_user(self, msg: Any) -> list[AgentEvent]:
        out: list[AgentEvent] = []
        content = getattr(msg, "content", None)
        blocks = content if isinstance(content, list) else []
        for block in blocks:
            if type(block).__name__ in ("ToolResultBlock", "ServerToolResultBlock"):
                out.append(ToolFinished(
                    name="",  # tool name is not on the result; app tracks by id
                    content=_result_text(getattr(block, "content", "")),
                    is_error=bool(getattr(block, "is_error", False)),
                    tool_id=getattr(block, "tool_use_id", "") or "",
                ))
        # Some SDK versions attach a parsed result dict instead.
        tur = getattr(msg, "tool_use_result", None)
        if tur and not out:
            out.append(ToolFinished(name="", content=_result_text(tur),
                                    tool_id=getattr(msg, "parent_tool_use_id", "") or ""))
        return out

    def _from_rate_limit(self, msg: Any) -> RateLimitWarning:
        info = getattr(msg, "rate_limit_info", None)
        status = str(getattr(info, "status", "") or "")
        util = getattr(info, "utilization", None)
        resets = getattr(info, "resets_at", None)
        parts = []
        if util is not None:
            try:
                parts.append(f"at {round(float(util) * 100)} percent of your rate limit")
            except (TypeError, ValueError):
                pass
        if resets:
            parts.append(f"resets {resets}")
        return RateLimitWarning(status=status, message="; ".join(parts))


# --------------------------------------------------------------------------- #
def isinstance_name(obj: Any, name: str) -> bool:
    return type(obj).__name__ == name


def _capture_session(session: AgentSession, sid: str) -> str:
    if sid:
        session.session_id = sid
    return sid or session.session_id


def _result_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "content" in content:
            return _result_text(content["content"])
        if "text" in content:
            return str(content["text"])
        return json.dumps(content)[:4000]
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(getattr(item, "text", "") or "")
        return "\n".join(p for p in parts if p)
    return getattr(content, "text", "") or str(content)
