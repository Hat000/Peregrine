"""Audible confirmation gates — you can't watch what Claude Code does.

Dangerous / irreversible tool calls are intercepted by a ``PreToolUse`` hook
that hands control to a spoken prompt and waits for a typed yes/no. If the user
declines, the tool call is denied and the agent is told why.

``is_dangerous`` returns a short spoken description when a call warrants a gate,
else ``None``. The heuristics are deliberately conservative: better to ask about
a harmless ``rm`` than to silently delete something while you're half asleep.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

# Bash command fragments that should always prompt.
_DANGER_BASH = [
    (re.compile(r"\brm\s+(-\w*\s+)*-\w*[rf]", re.I), "delete files with rm"),
    (re.compile(r"\brm\s+-[rf]", re.I), "delete files with rm"),
    (re.compile(r"\bgit\s+push\b.*(--force|-f)\b", re.I), "force-push with git"),
    (re.compile(r"\bgit\s+reset\s+--hard", re.I), "hard-reset the git tree"),
    (re.compile(r"\bgit\s+clean\s+-\w*f", re.I), "delete untracked files"),
    (re.compile(r"\b(sudo|doas)\b", re.I), "run a command as root"),
    (re.compile(r"\bdd\s+.*\bof=", re.I), "write a raw disk image with dd"),
    (re.compile(r"\bmkfs\b", re.I), "format a filesystem"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b", re.I), "power off or reboot"),
    (re.compile(r":\(\)\s*\{.*\};", re.S), "run a fork bomb"),
    (re.compile(r">\s*/dev/sd", re.I), "overwrite a disk device"),
    (re.compile(r"\bchmod\s+-R\b", re.I), "recursively change permissions"),
    (re.compile(r"\bchown\s+-R\b", re.I), "recursively change ownership"),
    (re.compile(r"\bcurl\b[^|]*\|\s*(sudo\s+)?(sh|bash)\b", re.I), "pipe a download into a shell"),
    (re.compile(r"\bwget\b[^|]*\|\s*(sudo\s+)?(sh|bash)\b", re.I), "pipe a download into a shell"),
    (re.compile(r"\bnpm\s+publish\b", re.I), "publish an npm package"),
    (re.compile(r"\b(pip|twine)\s+.*\bupload\b", re.I), "upload a Python package"),
    (re.compile(r"\bformat\b\s+[a-z]:", re.I), "format a Windows drive"),
    (re.compile(r"\bdel\s+/[sqf]", re.I), "delete files with del"),
    (re.compile(r"Remove-Item\b.*-Recurse", re.I), "recursively delete with Remove-Item"),
]

# File paths whose modification should prompt regardless of tool.
_SENSITIVE_PATH = re.compile(
    r"(^|[\\/])(\.env|\.git/config|id_rsa|id_ed25519|\.aws|\.ssh|credentials|"
    r"secrets?\.\w+|\.npmrc|\.pypirc)(\b|$)", re.I,
)


def is_dangerous(tool_name: str, tool_input: dict | None) -> str | None:
    """Return a spoken action description if this call warrants confirmation."""
    ti = tool_input or {}
    n = (tool_name or "").lower()

    if n in ("bash", "bashoutput"):
        cmd = str(ti.get("command", ""))
        for rx, label in _DANGER_BASH:
            if rx.search(cmd):
                return label
        return None

    if n in ("write", "edit", "multiedit", "notebookedit"):
        path = str(ti.get("file_path") or ti.get("path") or ti.get("notebook_path") or "")
        if _SENSITIVE_PATH.search(path):
            import os
            return f"modify the sensitive file {os.path.basename(path)}"
        return None

    # MCP tools that touch the outside world (email, messages, deletes, etc.).
    if n.startswith("mcp__") and any(
        k in n for k in ("send", "delete", "remove", "publish", "post", "create_event", "payment")
    ):
        return "run an external action that may not be reversible"

    return None


def build_confirmation_hook(
    confirm: Callable[[str, str, dict], Awaitable[bool]],
    enabled: bool = True,
) -> dict:
    """Build a ``hooks`` dict for ClaudeAgentOptions.

    ``confirm(description, tool_name, tool_input) -> bool`` is provided by the
    app; it performs the spoken NEEDS_CONFIRM prompt and returns the user's
    decision. Returns an empty dict when gating is disabled.
    """
    if not enabled:
        return {}

    try:
        from claude_agent_sdk import HookMatcher
    except Exception:
        return {}

    async def _pre_tool_use(input_data: dict, tool_use_id, context):
        name = input_data.get("tool_name", "")
        ti = input_data.get("tool_input", {}) or {}
        desc = is_dangerous(name, ti)
        if not desc:
            return {}
        approved = await confirm(desc, name, ti)
        decision = "allow" if approved else "deny"
        reason = ("Confirmed aloud by the user."
                  if approved else "The user declined this action out loud.")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
                "permissionDecisionReason": reason,
            }
        }

    return {"PreToolUse": [HookMatcher(hooks=[_pre_tool_use])]}
