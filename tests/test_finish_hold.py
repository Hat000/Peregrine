"""Tests for the post-finish recording hold (bridges Mission's geometric FINISHED to the sim's
terminal RACE_STATUS so race_outcome can self-certify a full 6/6 finish).

The two decision functions are pure; ``drain_until_finish`` is driven through fake now/step/
race-status seams (no live sim) -- a fake "tick" advances a fake clock and, after a configured
time, starts returning a finished RACE_STATUS, exactly as the live pump would once the sim
broadcasts the finish.
"""
from __future__ import annotations

import pytest

from racer.finish_hold import drain_until_finish, finish_drain_done, sim_finish_confirmed


# -- sim_finish_confirmed -------------------------------------------------------------------

def test_finish_confirmed_none_or_empty_is_false():
    assert not sim_finish_confirmed(None, 6)
    assert not sim_finish_confirmed({}, 6)


def test_finish_confirmed_by_finished_flag():
    assert sim_finish_confirmed({"finished": True, "race_finish_time_ns": -1,
                                 "active_gate_index": 5}, 6)


def test_finish_confirmed_by_recognized_time():
    # finished flag absent/false but a valid recognized time present
    assert sim_finish_confirmed({"finished": False, "race_finish_time_ns": 35_300_000_000,
                                 "active_gate_index": 5}, 6)
    # the live parser ties finished to ns>=0, but guard the field independently anyway
    assert sim_finish_confirmed({"race_finish_time_ns": 0, "active_gate_index": 0}, 6)


def test_finish_confirmed_by_active_index_running_off_the_end():
    # NEXT-gate pointer advanced past the last gate -> all gates passed
    assert sim_finish_confirmed({"finished": False, "race_finish_time_ns": -1,
                                 "active_gate_index": 6}, 6)


def test_not_confirmed_mid_race():
    assert not sim_finish_confirmed({"finished": False, "race_finish_time_ns": -1,
                                     "active_gate_index": 5}, 6)


def test_not_confirmed_when_n_gates_unknown_and_only_active_signal():
    # without n_gates we can't use the active-index signal; no finish flag/time -> not confirmed
    assert not sim_finish_confirmed({"finished": False, "race_finish_time_ns": -1,
                                     "active_gate_index": 99}, None)


# -- finish_drain_done ----------------------------------------------------------------------

def test_drain_done_caps_when_never_confirmed():
    # not yet confirmed, under the cap -> keep holding
    assert not finish_drain_done(0.5, None, max_hold_s=1.0, post_confirm_hold_s=0.4)
    # not confirmed, at/over the cap -> stop anyway (don't block disarm forever)
    assert finish_drain_done(1.0, None, max_hold_s=1.0, post_confirm_hold_s=0.4)
    assert finish_drain_done(1.2, None, max_hold_s=1.0, post_confirm_hold_s=0.4)


def test_drain_done_holds_post_confirm_margin():
    # confirmed at t=0.30; need 0.40 more before stopping (values kept clear of the float boundary)
    assert not finish_drain_done(0.60, 0.30, max_hold_s=2.0, post_confirm_hold_s=0.40)  # +0.30 < 0.40
    assert finish_drain_done(0.80, 0.30, max_hold_s=2.0, post_confirm_hold_s=0.40)      # +0.50 >= 0.40


def test_drain_done_cap_overrides_post_confirm():
    # confirmed late at t=0.90; post-confirm margin would want 1.30 but the cap (1.0) wins
    assert finish_drain_done(1.0, 0.90, max_hold_s=1.0, post_confirm_hold_s=0.40)


# -- drain_until_finish (fake-seam loop) ----------------------------------------------------

def _harness(*, dt=0.1, finish_after_s=None, n_gates=6):
    """Build (now_fn, step_fn, race_status_fn, calls) fakes. Each step advances a fake clock by
    ``dt``; once the clock reaches ``finish_after_s`` the race_status reads finished."""
    clock = {"t": 0.0}
    calls = {"n": 0}

    def now():
        return clock["t"]

    def step():
        calls["n"] += 1
        clock["t"] += dt              # a real tick takes ~one control period of wall time

    def race_status():
        if finish_after_s is not None and clock["t"] >= finish_after_s:
            return {"finished": True, "race_finish_time_ns": 35_300_000_000,
                    "active_gate_index": n_gates}
        return {"finished": False, "race_finish_time_ns": -1, "active_gate_index": n_gates - 1}

    return now, step, race_status, calls


def test_drain_captures_finish_then_holds_margin():
    now, step, rs, calls = _harness(dt=0.1, finish_after_s=0.25)
    res = drain_until_finish(step, rs, n_gates=6, max_hold_s=2.0,
                             post_confirm_hold_s=0.15, now_fn=now)
    assert res["confirmed"] is True
    # finish first reads at t≈0.30 (the first tick at/after 0.25); +0.15 margin -> stop ≈0.50
    assert res["confirmed_at_s"] == pytest.approx(0.30, abs=1e-9)
    assert res["elapsed_s"] == pytest.approx(0.50, abs=1e-9)
    assert calls["n"] == 5


def test_drain_caps_out_when_finish_never_arrives():
    now, step, rs, calls = _harness(dt=0.1, finish_after_s=None)
    res = drain_until_finish(step, rs, n_gates=6, max_hold_s=0.55,
                             post_confirm_hold_s=0.15, now_fn=now)
    assert res["confirmed"] is False
    assert res["confirmed_at_s"] is None
    assert res["elapsed_s"] >= 0.55          # ran to the cap
    assert calls["n"] == 6                   # ticks 0.1..0.6; 0.6 is first >= 0.55


def test_drain_always_pumps_at_least_once():
    # even a zero cap pumps one tick before deciding -> we never disarm without draining a frame
    now, step, rs, calls = _harness(dt=0.1, finish_after_s=None)
    res = drain_until_finish(step, rs, n_gates=6, max_hold_s=0.0,
                             post_confirm_hold_s=0.4, now_fn=now)
    assert calls["n"] == 1
    assert res["confirmed"] is False


def test_drain_stops_promptly_when_finish_already_present():
    # finish visible from the very first tick (e.g. we exited the loop on RACE_STATUS finished):
    # confirm immediately, hold only the post-confirm margin, then stop.
    now, step, rs, calls = _harness(dt=0.1, finish_after_s=0.0)
    res = drain_until_finish(step, rs, n_gates=6, max_hold_s=2.0,
                             post_confirm_hold_s=0.25, now_fn=now)
    assert res["confirmed"] is True
    assert res["confirmed_at_s"] == pytest.approx(0.10, abs=1e-9)   # first tick
    assert res["elapsed_s"] == pytest.approx(0.40, abs=1e-9)        # +0.25 margin, next tick >= 0.35
