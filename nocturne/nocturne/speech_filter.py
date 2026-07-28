"""Speech filtering — turn a Claude Code stream into things worth hearing.

Three kinds of output need very different treatment for the ear:

* **Assistant prose** -> read in full (with markdown/code stripped out).
* **Tool use** (bash, edits) -> an optional terse breadcrumb, e.g. "editing
  config.py", never the arguments.
* **Tool results** (diffs, logs, file dumps) -> summarised, never read verbatim,
  e.g. "wrote 42 lines to motor.py", "12 tests passed".

Reading a code block aloud character-by-character is miserable, so fenced code /
diff / log blocks are suppressed and announced instead.

Everything here is pure and synchronous so it can be unit-tested without audio,
the SDK, or a network.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Speakable items
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpeechItem:
    """A single utterance the TTS layer can read."""

    text: str
    kind: str  # "prose" | "code_announce" | "breadcrumb" | "summary"


# --------------------------------------------------------------------------- #
# Streaming prose -> sentences, with code-fence suppression
# --------------------------------------------------------------------------- #
_SENTENCE_END = re.compile(r"([.!?])[\"')\]]?(\s+|$)")
# Common abbreviations that should NOT end a sentence.
_ABBREV = {
    "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "st", "no", "fig",
    "al", "approx", "cf", "ca", "vol", "pp",
}
_FENCE = re.compile(r"^\s*(```|~~~)")


class ProseFilter:
    """Streaming filter: feed it assistant ``text_delta`` chunks, get back
    complete, speakable sentences as soon as their boundary arrives.

    Fenced code blocks are held out of the spoken stream and replaced with a
    short announcement ("shared a 12-line code block") when they close.
    """

    def __init__(self, suppress_code_blocks: bool = True) -> None:
        self._suppress_code = suppress_code_blocks
        self._buf = ""          # unspoken prose accumulated so far
        self._pending_line = "" # partial current line (no newline yet)
        self._in_fence = False
        self._fence_marker = ""
        self._fence_lines = 0

    # -- public ---------------------------------------------------------- #
    def feed(self, delta: str) -> list[SpeechItem]:
        """Process a streamed chunk. Returns any items ready to speak now."""
        items: list[SpeechItem] = []
        self._pending_line += delta
        # Process every complete line; keep the trailing partial line pending.
        while "\n" in self._pending_line:
            line, self._pending_line = self._pending_line.split("\n", 1)
            items.extend(self._consume_line(line + "\n"))
        # A partial line can't close a fence, but if we're inside prose we can
        # still surface completed sentences from what we have so far. Hold back
        # any trailing partial line that might still grow into a fence marker
        # (e.g. "``" before the third backtick arrives) so code never leaks.
        if not self._in_fence and not self._pending_could_be_fence(self._pending_line):
            self._buf += self._pending_line
            self._pending_line = ""
            items.extend(self._drain_sentences(final=False))
        return items

    def flush(self) -> list[SpeechItem]:
        """Emit everything left over at end of turn."""
        items: list[SpeechItem] = []
        if self._pending_line:
            if self._in_fence:
                self._fence_lines += 1
            else:
                self._buf += self._pending_line
            self._pending_line = ""
        if self._in_fence:
            # An unterminated fence — announce whatever code we saw.
            items.append(self._close_fence())
        items.extend(self._drain_sentences(final=True))
        self._buf = ""
        return items

    # -- internals ------------------------------------------------------- #
    def _looks_like_fence(self, s: str) -> bool:
        return bool(_FENCE.match(s))

    def _pending_could_be_fence(self, s: str) -> bool:
        """True if the trailing partial line might still become a fence marker
        once more characters stream in (so we shouldn't speak it yet)."""
        t = s.lstrip()
        if t.startswith("```") or t.startswith("~~~"):
            return True
        return t in ("`", "``", "~", "~~")

    def _consume_line(self, line: str) -> list[SpeechItem]:
        items: list[SpeechItem] = []
        if self._suppress_code and _FENCE.match(line):
            if not self._in_fence:
                # Opening a fence: flush pending prose first so ordering holds.
                items.extend(self._drain_sentences(final=True))
                self._in_fence = True
                self._fence_marker = line.strip()[:3]
                self._fence_lines = 0
            else:
                # Closing fence.
                items.append(self._close_fence())
            return items
        if self._in_fence:
            self._fence_lines += 1
            return items
        self._buf += line
        items.extend(self._drain_sentences(final=False))
        return items

    def _close_fence(self) -> SpeechItem:
        n = self._fence_lines
        self._in_fence = False
        self._fence_marker = ""
        self._fence_lines = 0
        word = "line" if n == 1 else "lines"
        return SpeechItem(text=f"(shared a code block, {n} {word})", kind="code_announce")

    def _drain_sentences(self, final: bool) -> list[SpeechItem]:
        items: list[SpeechItem] = []
        while True:
            sentence, rest = _pop_sentence(self._buf)
            if sentence is None:
                break
            self._buf = rest
            spoken = _clean_for_speech(sentence)
            if spoken:
                items.append(SpeechItem(text=spoken, kind="prose"))
        if final and self._buf.strip():
            spoken = _clean_for_speech(self._buf)
            self._buf = ""
            if spoken:
                items.append(SpeechItem(text=spoken, kind="prose"))
        return items


def _pop_sentence(text: str) -> tuple[str | None, str]:
    """Return (sentence, remainder) if a complete sentence boundary exists,
    else (None, text). Also treats a blank line (paragraph break) as a
    boundary so bulleted / terse output still flows."""
    # Paragraph break first.
    para = re.search(r"\n[ \t]*\n", text)
    for m in _SENTENCE_END.finditer(text):
        end = m.end(1)
        # Skip abbreviations like "e.g." and decimals like "3.14".
        head = text[:end]
        last_word = re.split(r"[\s(]", head.rstrip("."))[-1].lower()
        if last_word in _ABBREV:
            continue
        # Decimal number: digit before '.' and digit after.
        if m.group(1) == "." and end < len(text) and text[end - 2:end - 1].isdigit() \
                and text[m.end():m.end() + 1].isdigit():
            continue
        boundary = m.end()
        if para and para.start() < boundary:
            break  # paragraph break comes first; handle it below
        return text[:boundary].strip(), text[boundary:]
    if para:
        return text[:para.start()].strip(), text[para.end():]
    return None, text


# --------------------------------------------------------------------------- #
# Markdown -> speakable prose
# --------------------------------------------------------------------------- #
_INLINE_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_IMG = re.compile(r"!\[([^\]]*)\]\((?:[^)]+)\)")
_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
_HEADER = re.compile(r"^\s{0,3}#{1,6}\s*")
_BULLET = re.compile(r"^\s*([-*+]|\d+\.)\s+")
_URL = re.compile(r"https?://\S+")


def _clean_for_speech(text: str) -> str:
    """Strip markdown noise that sounds bad read aloud."""
    text = _IMG.sub(lambda m: m.group(1) or "image", text)
    text = _LINK.sub(r"\1", text)                 # keep link text, drop URL
    text = _INLINE_CODE.sub(r"\1", text)          # drop backticks
    text = _BOLD_ITALIC.sub(r"\2", text)          # drop emphasis marks
    lines_out: list[str] = []
    for line in text.splitlines():
        line = _HEADER.sub("", line)
        line = _BULLET.sub("", line)
        lines_out.append(line)
    text = " ".join(l.strip() for l in lines_out if l.strip())
    text = _URL.sub("a link", text)
    text = text.replace("`", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# Tool use -> terse breadcrumb
# --------------------------------------------------------------------------- #
def describe_tool_use(name: str, tool_input: dict | None) -> str | None:
    """A short spoken breadcrumb for a tool call — never the arguments."""
    ti = tool_input or {}
    base = _basename(ti.get("file_path") or ti.get("path") or ti.get("notebook_path") or "")
    n = (name or "").lower()

    if n in ("edit", "multiedit", "notebookedit"):
        return f"editing {base}" if base else "editing a file"
    if n == "write":
        return f"writing {base}" if base else "writing a file"
    if n == "read":
        return f"reading {base}" if base else "reading a file"
    if n in ("bash", "bashoutput"):
        desc = (ti.get("description") or "").strip()
        if desc:
            return desc[0].lower() + desc[1:] if desc[0].isupper() else desc
        return "running a shell command"
    if n in ("grep", "glob"):
        return "searching the code"
    if n in ("websearch",):
        return "searching the web"
    if n in ("webfetch",):
        return "fetching a web page"
    if n in ("task", "agent"):
        return "delegating to a subagent"
    if n in ("todowrite", "taskcreate", "taskupdate"):
        return "updating the task list"
    if n.startswith("mcp__"):
        return "using an external tool"
    # Fallback: humanise the tool name.
    return f"using {re.sub(r'[_-]+', ' ', name).strip().lower()}" if name else None


# --------------------------------------------------------------------------- #
# Tool result -> summary (never verbatim)
# --------------------------------------------------------------------------- #
_PYTEST = re.compile(r"(\d+)\s+passed(?:.*?(\d+)\s+failed)?", re.I)
_PYTEST_FAIL = re.compile(r"(\d+)\s+failed", re.I)
_WROTE = re.compile(r"(?:wrote|created|updated|added)\D*(\d+)\s+lines?", re.I)


def summarize_tool_result(
    name: str,
    content: str,
    is_error: bool = False,
    max_chars: int = 2000,
) -> str | None:
    """Summarise a tool result. Diffs/logs/dumps are counted, not read."""
    text = (content or "").strip()
    if is_error:
        first = _first_line(text)
        return f"that {(_humanise(name))} call failed" + (f": {first}" if first and len(first) < 120 else "")
    if not text:
        return None

    # Test runs are worth reading the tally of.
    m = _PYTEST.search(text)
    if m:
        passed = m.group(1)
        failed = m.group(2) or (_PYTEST_FAIL.search(text) and _PYTEST_FAIL.search(text).group(1))
        if failed and int(failed) > 0:
            return f"{passed} tests passed, {failed} failed"
        return f"{passed} tests passed"

    n = (name or "").lower()
    if n in ("write", "edit", "multiedit"):
        m = _WROTE.search(text)
        lines = m.group(1) if m else str(_count_lines(text))
        return f"done, {lines} lines"

    # Short, non-code results can just be read.
    if len(text) <= 140 and not _looks_like_code(text):
        return _clean_for_speech(text)

    lines = _count_lines(text)
    word = "line" if lines == 1 else "lines"
    return f"returned {lines} {word} of output"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _basename(p: str) -> str:
    return os.path.basename(p.rstrip("/\\")) if p else ""


def _first_line(s: str) -> str:
    return s.splitlines()[0].strip() if s.strip() else ""


def _count_lines(s: str) -> int:
    return len([l for l in s.splitlines() if l.strip()]) or (1 if s.strip() else 0)


def _humanise(name: str) -> str:
    return re.sub(r"[_-]+", " ", name or "tool").strip().lower() or "tool"


def _looks_like_code(s: str) -> bool:
    hints = ("{", "};", "def ", "class ", "import ", "function ", "=>", "</", "/>", "\t")
    return any(h in s for h in hints) or s.count("\n") >= 2
