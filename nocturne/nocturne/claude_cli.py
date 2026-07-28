"""Drive the Claude Code CLI directly — on your subscription, no API key.

This is the point of Nocturne: it is an audio skin over the *same* Claude Code
you already use, not a separate app billed to API credits. We spawn the
subscription-authenticated ``claude`` binary in headless stream-json mode::

    claude -p --input-format stream-json --output-format stream-json \
           --include-partial-messages --verbose [--permission-mode ...] ...

and talk to it over stdin/stdout as newline-delimited JSON. The process stays
alive across turns (a persistent multi-turn session); each turn ends with a
``result`` message. Output messages are translated into the same ``events.py``
vocabulary the SDK driver emits, so the rest of the app is unchanged.

Auth: the spawned CLI uses whatever login the standalone ``claude`` has on disk
(from ``claude`` → ``/login`` or ``claude setup-token``). No ANTHROPIC_API_KEY
is set or needed; if the CLI isn't logged in it reports "Please run /login".
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any


def _debug() -> bool:
    return bool(os.environ.get("NOCTURNE_DEBUG"))


def _dbg(msg: str) -> None:
    if _debug():
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()

from .config import AgentConfig
from .events import (
    AgentEvent, ProseComplete, ProseDelta, RateLimitWarning, StreamError,
    ToolFinished, ToolStarted, TurnFinished,
)


def _package_claude_roots() -> list[Path]:
    """The Store/MSIX-packaged Claude desktop app virtualizes %APPDATA% into
    ``%LOCALAPPDATA%\\Packages\\Claude_<suffix>\\LocalCache\\Roaming\\Claude``, so
    the CLI it bundles lives there and is invisible to a plain %APPDATA% search.
    The package suffix is machine-specific; glob it."""
    lad = os.environ.get("LOCALAPPDATA")
    base = Path(lad) if lad else Path.home() / "AppData" / "Local"
    pkgs = base / "Packages"
    out: list[Path] = []
    try:
        for d in pkgs.glob("Claude*/LocalCache/Roaming/Claude"):
            if d.is_dir():
                out.append(d)
    except OSError:
        pass
    return out


def _search_roots() -> list[Path]:
    home = Path.home()
    roots: list[Path] = []
    for var in ("APPDATA", "LOCALAPPDATA"):
        v = os.environ.get(var)
        if v:
            roots.append(Path(v) / "Claude")
    roots += [
        home / "AppData" / "Roaming" / "Claude",
        home / "AppData" / "Local" / "Claude",
    ]
    roots += _package_claude_roots()
    roots += [
        home / ".claude",
        home / ".local" / "bin",
        Path("/usr/local/bin"), Path("/opt/homebrew/bin"),
    ]
    # de-dup, preserve order
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        s = str(r)
        if s not in seen:
            seen.add(s)
            out.append(r)
    return out


def locate_cli(explicit: str = "") -> str | None:
    """Find the claude executable, robustly.

    Order: explicit path -> CLAUDE_CODE_EXECPATH env -> PATH -> a recursive
    search under the known Claude install roots (newest claude.exe by mtime, so
    version-folder churn doesn't matter). If ``explicit`` is set but missing, we
    fall through to auto-detection rather than failing.
    """
    exe_name = "claude.exe" if os.name == "nt" else "claude"

    if explicit and Path(explicit).exists():
        return explicit
    envp = os.environ.get("CLAUDE_CODE_EXECPATH")
    if envp and Path(envp).exists():
        return envp
    for name in ("claude", "claude.exe", "claude.cmd"):
        w = shutil.which(name)
        if w:
            return w

    candidates: list[Path] = []
    for root in _search_roots():
        try:
            if not root.exists():
                continue
            direct = root / exe_name
            if direct.is_file():
                candidates.append(direct)
            candidates.extend(root.glob(f"**/{exe_name}"))
        except OSError:
            continue
    if candidates:
        return str(max(candidates, key=lambda p: _mtime(p)))
    return None


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def search_report() -> str:
    """Human-readable list of where we looked (for diagnostics)."""
    lines = [f"  CLAUDE_CODE_EXECPATH={os.environ.get('CLAUDE_CODE_EXECPATH') or '(unset)'}"]
    for r in _search_roots():
        lines.append(f"  {r}  {'[exists]' if r.exists() else '[missing]'}")
    return "\n".join(lines)


class ClaudeCLISession:
    """Interface-compatible with agent_loop.AgentSession, but backed by the
    subscription CLI."""

    def __init__(
        self,
        cfg: AgentConfig,
        *,
        resume: str | None = None,
        continue_conversation: bool = False,
        permission_prompt_tool: str | None = None,
        mcp_config: str | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self._resume = resume
        self._continue = continue_conversation
        self._permission_prompt_tool = permission_prompt_tool
        self._mcp_config = mcp_config
        self._extra_args = extra_args or []
        self._proc: asyncio.subprocess.Process | None = None
        self._partial = cfg.include_partial_messages
        self.session_id: str = ""
        self._cli_path: str = ""
        self._stderr_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ #
    def _build_argv(self) -> list[str]:
        argv = [
            self._cli_path, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
        ]
        if self._partial:
            argv.append("--include-partial-messages")
        if self.cfg.model:
            argv += ["--model", self.cfg.model]
        if self.cfg.permission_mode:
            argv += ["--permission-mode", self.cfg.permission_mode]
        if self.cfg.allowed_tools:
            argv += ["--allowed-tools", ",".join(self.cfg.allowed_tools)]
        if self.cfg.disallowed_tools:
            argv += ["--disallowed-tools", ",".join(self.cfg.disallowed_tools)]
        if self._permission_prompt_tool:
            argv += ["--permission-prompt-tool", self._permission_prompt_tool]
        if self._mcp_config:
            argv += ["--mcp-config", self._mcp_config]
        if self._resume:
            argv += ["--resume", self._resume]
        elif self._continue:
            argv.append("--continue")
        argv += self._extra_args
        return argv

    async def start(self) -> None:
        self._cli_path = locate_cli(self.cfg.claude_cli_path) or ""
        if not self._cli_path:
            raise RuntimeError(
                "Claude Code CLI not found. Set agent.claude_cli_path in "
                "nocturne.toml to your claude.exe. Searched:\n" + search_report())
        env = dict(os.environ)
        # We authenticate by the CLI's own subscription login, not an API key.
        env.pop("ANTHROPIC_API_KEY", None)
        cwd = self.cfg.cwd or None
        argv = self._build_argv()
        _dbg(f"[claude] spawn: {argv}")
        _dbg(f"[claude] cwd={cwd or os.getcwd()}")
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd, env=env,
        )
        _dbg(f"[claude] pid={self._proc.pid}")
        self._stderr_task = asyncio.ensure_future(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        try:
            async for line in self._proc.stderr:
                # Normally swallowed (surfaced via result.is_error); in debug we
                # echo it, because a silent stderr is where a stuck CLI hides.
                _dbg("[claude:stderr] " + line.decode("utf-8", "replace").rstrip())
        except Exception:
            pass

    async def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.is_closing():
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
        except Exception:
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=3)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    # ------------------------------------------------------------------ #
    async def run(self, prompt: str) -> AsyncIterator[AgentEvent]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            yield StreamError("claude CLI session not started")
            return
        try:
            wire = self._user_message(prompt)
            _dbg("[claude:stdin] " + wire)
            self._proc.stdin.write((wire + "\n").encode("utf-8"))
            await self._proc.stdin.drain()
            _dbg("[claude] prompt sent, waiting for output…")
        except Exception as e:
            yield StreamError(f"failed to send prompt: {e}")
            return
        try:
            async for raw in self._proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                _dbg("[claude:stdout] " + (line if len(line) < 600 else line[:600] + "…"))
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for ev in self._translate(msg):
                    yield ev
                if msg.get("type") == "result":
                    return
        except Exception as e:
            yield StreamError(f"stream error: {e}")
        # stdout closed without a result -> the process ended (e.g. not logged in).
        rc = self._proc.returncode
        _dbg(f"[claude] stdout closed with no result; returncode={rc}")
        if rc is not None and rc != 0:
            yield StreamError(
                f"the Claude Code CLI exited with code {rc} — "
                "run with NOCTURNE_DEBUG=1 to see its stderr")
        else:
            yield StreamError("the Claude Code session ended unexpectedly")

    def _user_message(self, prompt: str) -> str:
        return json.dumps({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
        })

    async def interrupt(self) -> None:
        """Best-effort turn interrupt via a control request; the run() loop
        drains to the next result regardless."""
        if self._proc is None or self._proc.stdin is None:
            return
        ctrl = json.dumps({
            "type": "control_request",
            "request_id": "nocturne-interrupt",
            "request": {"subtype": "interrupt"},
        })
        try:
            self._proc.stdin.write((ctrl + "\n").encode("utf-8"))
            await self._proc.stdin.drain()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _translate(self, msg: dict[str, Any]) -> list[AgentEvent]:
        mtype = msg.get("type")
        if mtype == "stream_event":
            return self._from_stream_event(msg.get("event", {}) or {})
        if mtype == "assistant":
            return self._from_assistant(msg.get("message", {}) or {})
        if mtype == "user":
            return self._from_user(msg.get("message", {}) or {})
        if mtype == "result":
            self.session_id = msg.get("session_id", "") or self.session_id
            return [TurnFinished(
                usage=msg.get("usage"),
                session_id=self.session_id,
                subtype=msg.get("subtype", "") or "",
                is_error=bool(msg.get("is_error", False)),
                result_text=msg.get("result", "") or "",
                cost_usd=msg.get("total_cost_usd"),
            )]
        if mtype == "system":
            self.session_id = msg.get("session_id", "") or self.session_id
        return []

    def _from_stream_event(self, event: dict) -> list[AgentEvent]:
        if event.get("type") == "content_block_delta":
            delta = event.get("delta", {}) or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                return [ProseDelta(delta["text"])]
        return []

    def _from_assistant(self, message: dict) -> list[AgentEvent]:
        out: list[AgentEvent] = []
        for block in message.get("content", []) or []:
            btype = block.get("type")
            if btype == "tool_use":
                out.append(ToolStarted(
                    name=block.get("name", "") or "",
                    tool_input=block.get("input", {}) or {},
                    tool_id=block.get("id", "") or "",
                ))
            elif btype == "text" and not self._partial:
                out.append(ProseComplete(block.get("text", "") or ""))
        return out

    def _from_user(self, message: dict) -> list[AgentEvent]:
        out: list[AgentEvent] = []
        content = message.get("content")
        blocks = content if isinstance(content, list) else []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                out.append(ToolFinished(
                    name="",
                    content=_result_text(block.get("content", "")),
                    is_error=bool(block.get("is_error", False)),
                    tool_id=block.get("tool_use_id", "") or "",
                ))
        return out


def _result_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return content.get("text", "") or json.dumps(content)[:4000]
    return str(content)
