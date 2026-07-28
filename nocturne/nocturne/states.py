"""The interaction state machine.

The whole app is a loop over these states. Each state owns an audio behaviour;
the canonical transition is::

    submit -> WORKING (music fades in) -> first sentence ready
           -> SPEAKING (music ducks out, TTS reads) -> IDLE (your-turn earcon)

Because music plays for exactly as long as real work takes, its duration is a
genuine "still thinking" signal during long agentic runs.
"""

from __future__ import annotations

import enum


class State(enum.Enum):
    """The five interaction states. Values double as earcon / log keys."""

    IDLE = "idle"
    """Waiting for the user to type. Silence; a soft your-turn earcon when
    control returns to them."""

    WORKING = "working"
    """A prompt was submitted and we are streaming until the first speakable
    sentence is ready. Music plays (fades in); optional terse breadcrumbs."""

    SPEAKING = "speaking"
    """Assistant prose is ready to read. Music ducks/pauses; TTS reads."""

    NEEDS_CONFIRM = "needs_confirm"
    """Claude wants a risky / irreversible action. Music pauses, urgent earcon,
    spoken prompt, waits for a typed yes/no."""

    ERROR = "error"
    """An SDK or tool error. Distinct error earcon + one-sentence summary."""


# States during which music (the thinking indicator) should be audible.
MUSIC_STATES = frozenset({State.WORKING})

# States during which the microphone-of-attention is the user's: we are waiting
# on a typed reply, so TTS should be quiet.
AWAITING_INPUT_STATES = frozenset({State.IDLE, State.NEEDS_CONFIRM})
