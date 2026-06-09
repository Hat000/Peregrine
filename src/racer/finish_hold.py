"""Post-finish recording hold -- keep the recording OPEN for ~0.5-1.0 s after the final gate so the
sim's TERMINAL ``RACE_STATUS`` (``finished`` + the recognized finish time) lands in the tlog BEFORE
``fly_vq1`` force-disarms and closes the recorder.

Why this exists: :meth:`racer.mission.Mission.run` exits the instant its GEOMETRIC ``_passed``
advances ``gate_index`` past the last gate -- which LEADS the sim's authoritative ``RACE_STATUS``
broadcast by a beat. ``fly_vq1`` used to force-disarm (closing the recorder) on that same tick, so
the terminal status never hit the tlog and :mod:`racer.race_outcome` -- our RL/eval grader --
self-certified only 5/6 gates with ``finished=False`` (the true 6/6 finish + ~35.3 s time were
GUI-confirmed only). This module is the tiny, FULLY-AUTONOMOUS (no human interaction) drain that
bridges that gap: pump (which records the inbound RACE_STATUS) + hold position until the terminal
status is captured, then return so the caller can disarm.

The decision logic is split into two PURE functions (:func:`sim_finish_confirmed`,
:func:`finish_drain_done`) so the completion behaviour is unit-tested without a live sim; the
:func:`drain_until_finish` loop keeps its I/O behind injected seams for the same reason.

Note on the race_status key: this consumes the LIVE :attr:`racer.mavlink_client.MavlinkClient.race_status`
(from ``parse_race_status``), whose finish-time key is ``race_finish_time_ns``. The OFFLINE
:mod:`racer.race_outcome` loader renames the same field to ``finish_ns`` -- don't confuse the two.
"""
from __future__ import annotations

import time
from typing import Callable


def sim_finish_confirmed(race_status: dict | None, n_gates: int | None) -> bool:
    """True once the sim's RACE_STATUS reports the race finished -- i.e. the terminal status we need
    in the recording has been received.

    Confirmed by ANY of the authoritative finish signals: the ``finished`` flag, a valid
    ``race_finish_time_ns`` (>= 0), or ``active_gate_index`` having advanced to/past ``n_gates``
    (the NEXT-gate pointer ran off the end -> every gate passed). ``race_status`` is the live
    :attr:`MavlinkClient.race_status` dict (or ``None`` before any sample).
    """
    if not race_status:
        return False
    if race_status.get("finished"):
        return True
    fin = race_status.get("race_finish_time_ns")
    if fin is not None and fin >= 0:
        return True
    active = race_status.get("active_gate_index")
    return active is not None and n_gates is not None and active >= n_gates


def finish_drain_done(elapsed_s: float, confirmed_elapsed_s: float | None, *,
                      max_hold_s: float, post_confirm_hold_s: float) -> bool:
    """Decide when the post-finish drain can STOP (-> the caller proceeds to force-disarm). Pure.

    Stops when EITHER:

    * the hard cap ``max_hold_s`` is reached -- the terminal status never arrived, so don't block
      the (autonomous) disarm forever; or
    * the finish was confirmed (``confirmed_elapsed_s`` = the elapsed time at first confirmation,
      or ``None`` if not yet) AND we've held ``post_confirm_hold_s`` more since -- a small flush
      margin so the fully-populated terminal sample (incl. the recognized time, which can lag the
      ``finished`` flag by a sample) is safely recorded.

    The cap always wins, so ``post_confirm_hold_s`` is best-effort within ``max_hold_s``.
    """
    if elapsed_s >= max_hold_s:
        return True
    if confirmed_elapsed_s is None:
        return False
    return (elapsed_s - confirmed_elapsed_s) >= post_confirm_hold_s


def drain_until_finish(step_once: Callable[[], None], race_status_fn: Callable[[], dict | None], *,
                       n_gates: int | None, max_hold_s: float = 1.0,
                       post_confirm_hold_s: float = 0.4,
                       now_fn: Callable[[], float] = time.monotonic) -> dict:
    """Pump + hold until the sim's terminal RACE_STATUS is captured (or ``max_hold_s`` elapses).

    ``step_once()`` performs ONE control tick. In ``fly_vq1`` it drains MAVLink (which RECORDS the
    inbound RACE_STATUS via the recorder's ``on_message``) and re-sends the FINISHED hold so the
    drone stays put while we wait. ``race_status_fn()`` returns the latest parsed RACE_STATUS dict.
    Both I/O seams + ``now_fn`` are injected so this loop is unit-tested with fakes.

    ``step_once`` is always called at least once (we always pump a tick before deciding). Returns a
    small report: ``{confirmed, confirmed_at_s, elapsed_s}``.
    """
    t0 = now_fn()
    confirmed_at: float | None = None
    elapsed = 0.0
    while True:
        step_once()
        elapsed = now_fn() - t0
        if confirmed_at is None and sim_finish_confirmed(race_status_fn(), n_gates):
            confirmed_at = elapsed
        if finish_drain_done(elapsed, confirmed_at, max_hold_s=max_hold_s,
                             post_confirm_hold_s=post_confirm_hold_s):
            break
    return {"confirmed": confirmed_at is not None, "confirmed_at_s": confirmed_at,
            "elapsed_s": elapsed}
