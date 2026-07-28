"""Morning transcript — log the session to a timestamped markdown file.

Captures your prompts, Claude's prose, the tools it ran, and the turn results
so there's something to read in the morning. ``recap()`` returns a short spoken
summary of the session so far for the ``/recap`` command.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path


class SessionLog:
    def __init__(self, log_dir: str, enabled: bool = True, stamp: str | None = None) -> None:
        self.enabled = enabled
        self._counts: Counter[str] = Counter()
        self._files_touched: set[str] = set()
        self._turns = 0
        self._path: Path | None = None
        if enabled:
            d = Path(log_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
            self._path = d / f"nocturne-{stamp}.md"
            self._write(f"# Nocturne session — {stamp}\n\n")

    @property
    def path(self) -> Path | None:
        return self._path

    def _write(self, text: str) -> None:
        if self._path is None:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(text)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def user(self, text: str) -> None:
        self._turns += 1
        self._write(f"\n## You\n\n{text}\n")

    def assistant(self, text: str) -> None:
        if text.strip():
            self._write(f"\n## Claude\n\n{text}\n")

    def tool(self, name: str, brief: str, file_path: str = "") -> None:
        self._counts[name or "tool"] += 1
        if file_path:
            self._files_touched.add(file_path)
        self._write(f"- 🔧 `{name}` — {brief}\n")

    def note(self, text: str) -> None:
        self._write(f"\n> {text}\n")

    # ------------------------------------------------------------------ #
    def recap(self) -> str:
        if self._turns == 0:
            return "We haven't done anything yet this session."
        tools = sum(self._counts.values())
        top = ", ".join(f"{n} {t}" for t, n in self._counts.most_common(3)) or "no tools"
        files = len(self._files_touched)
        return (f"So far: {self._turns} exchanges, {tools} tool calls "
                f"({top}), and {files} files touched.")
