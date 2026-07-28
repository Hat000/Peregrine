"""Context-window tracking and audible warnings.

Reads token usage off each turn's ``ResultMessage`` and compares cumulative
context against the model's window (200k standard, 1M on newer models). Speaks
a warning once per threshold, and a distinct heads-up just before auto-
compaction so the user can choose to wrap up vs. let it compact. ``/context``
asks for the current figure on demand.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ContextConfig


@dataclass
class ContextWarning:
    text: str
    earcon: bool  # whether to precede with the context-low earcon
    kind: str     # "threshold" | "compaction"


class ContextTracker:
    def __init__(self, cfg: ContextConfig) -> None:
        self.window = max(1, cfg.window_tokens)
        self._thresholds = sorted(cfg.warn_thresholds)
        self._compact_at = cfg.warn_before_compaction_at
        self._fired: set[float] = set()
        self._compaction_fired = False
        self.current = 0

    # ------------------------------------------------------------------ #
    def update(self, usage: dict | None) -> list[ContextWarning]:
        """Feed a ResultMessage.usage dict; return any newly-tripped warnings."""
        if not usage:
            return []
        self.current = _context_size(usage)
        return self._check()

    def note_context_size(self, tokens: int) -> list[ContextWarning]:
        self.current = max(self.current, int(tokens))
        return self._check()

    def _check(self) -> list[ContextWarning]:
        out: list[ContextWarning] = []
        frac = self.fraction
        for t in self._thresholds:
            if frac >= t and t not in self._fired:
                self._fired.add(t)
                out.append(ContextWarning(
                    text=f"Heads up: you're at {round(frac * 100)} percent of the "
                         f"context window.",
                    earcon=True, kind="threshold",
                ))
        if not self._compaction_fired and frac >= self._compact_at:
            self._compaction_fired = True
            out.append(ContextWarning(
                text="You're close to automatic compaction. Wrap up the current "
                     "thread soon, or let it compact.",
                earcon=True, kind="compaction",
            ))
        return out

    # ------------------------------------------------------------------ #
    @property
    def fraction(self) -> float:
        return min(1.0, self.current / self.window)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.window - self.current)

    def report(self) -> str:
        """A spoken answer for the /context command."""
        pct_left = round((1 - self.fraction) * 100)
        return (f"You're using about {round(self.fraction * 100)} percent of the "
                f"context window. Roughly {pct_left} percent, or "
                f"{_k(self.remaining_tokens)}, remaining.")

    def reset(self) -> None:
        """Call after a compaction to start counting the fresh window."""
        self._fired.clear()
        self._compaction_fired = False
        self.current = 0


def _context_size(usage: dict) -> int:
    """Approximate live context size from an Anthropic usage dict. Input and
    cache tokens together are the prompt that was sent; output adds to the
    context the next turn will carry."""
    return (
        int(usage.get("input_tokens", 0) or 0)
        + int(usage.get("cache_read_input_tokens", 0) or 0)
        + int(usage.get("cache_creation_input_tokens", 0) or 0)
        + int(usage.get("output_tokens", 0) or 0)
    )


def _k(n: int) -> str:
    return f"{round(n / 1000)} thousand tokens" if n >= 1000 else f"{n} tokens"
