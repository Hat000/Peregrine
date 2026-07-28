"""Nocturne — a screenless, listen-and-type shell for Claude Code.

Audio is the only output channel. Every piece of session state (thinking,
speaking, done, error, needs-your-input, context-low) is conveyed by sound —
speech or a distinct earcon. See ``nocturne.states`` for the state machine that
drives that audio behaviour.
"""

__version__ = "0.1.0"
