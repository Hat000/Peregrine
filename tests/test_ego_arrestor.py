"""EGO post-gate STARE-BRAKE arrestor (replay-ratchet ARRESTOR) — state machine + command bounds.

Pins the fly_ego arrestor contract [replay-ratchet ARRESTOR, 2026-07-18], built from the
ratchet-P0.3 evidence (handoff REPORT4: N=5 settled-launch policy flights banked gates {2,2,1,2,2},
all died too-fast/too-high entering gate 2, 0/5 contact-free). The arrestor takes over after a
banked gate, bleeds speed while keeping the next gate centered, and hands a slow/level/in-view state
back to the stateless policy — re-spawning it into its training distribution mid-flight.

Unit-tested as a PURE object (injectable clock, no sim):
  * OFF => never returns a command (the byte-identical policy-path guarantee);
  * gate trigger (fire-once, only on a LISTED gate's pass) + never on the pad;
  * speed trigger + post-handback refractory (no chatter);
  * handback on slow AND gate-in-view AND ~level, held for the dwell;
  * every abort path (rel_up drift / gate-lost / timeout), all fail-OPEN to the policy;
  * command bounds + signs: yaw clamp + stare direction, nose-up pitch brake + cap + level stop,
    vision-vertical collective hold + [0.7,1.4] g clamp, roll == 0.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

import fly_rl  # noqa: E402

_HOVER = fly_rl._HOVER_THRUST      # 0.2656 g-units -> [0,1] collective


def _mk(enabled=True, after_gates=(1,), speed_hi=0.0, max_s=4.0, yaw_clamp=0.7, **kw):
    return fly_rl.EgoArrestor(enabled=enabled, after_gates=after_gates, speed_hi=speed_hi,
                              max_s=max_s, yaw_clamp=yaw_clamp, hover_collective=_HOVER, **kw)


def _drive_to_pass(a, *, gate=1, now=0.0, **step_kw):
    """Walk gate_index gate-1 -> gate -> gate+1 (the 'passed `gate`' transition) and return the
    ArrestCommand from the engage tick. Defaults keep the drone airborne + gate not-yet-seen."""
    base = dict(ground_active=False, speed_est=6.0, bearing_rad=None, rel_up=None,
                obs_pitch=-0.4, obs_roll=0.0, det_fresh=False)
    base.update(step_kw)
    a.step(now=now, gate_index=gate - 1, **base)
    a.step(now=now, gate_index=gate, **base)
    return a.step(now=now, gate_index=gate + 1, **base)


# --------------------------------------------------------------------------- OFF == byte-identical
def test_disabled_never_commands():
    """OFF: step returns None for EVERY input incl. a listed-gate pass + hot speed, so the loop's
    `if arr is not None` is always False and the policy path runs byte-identical."""
    a = _mk(enabled=False, after_gates=(1,), speed_hi=3.0)
    assert a.enabled is False
    # a listed-gate pass AND a hot speed both would engage if enabled — assert they do not.
    out = _drive_to_pass(a, speed_est=20.0)
    assert out is None
    assert a.phase == "policy"
    assert a.arrest_id == 0


# --------------------------------------------------------------------------- gate trigger
def test_gate_trigger_engages_on_listed_pass():
    a = _mk(after_gates=(1,))
    out = _drive_to_pass(a, gate=1)
    assert out is not None
    assert out.event == "ENGAGE"
    assert "after gate 1" in out.reason
    assert a.phase == "arrest"
    assert a.arrest_id == 1


def test_gate_trigger_ignores_unlisted_pass():
    """after_gates={1}: passing gate 0 (0->1) must NOT engage."""
    a = _mk(after_gates=(1,))
    base = dict(ground_active=False, speed_est=6.0, bearing_rad=None, rel_up=None,
                obs_pitch=-0.4, obs_roll=0.0, det_fresh=False)
    assert a.step(now=0.0, gate_index=0, **base) is None   # prev=None
    assert a.step(now=0.0, gate_index=1, **base) is None   # 0->1 = passed gate 0 (unlisted)
    assert a.phase == "policy"


def test_gate_trigger_never_on_the_pad():
    """ground_active (takeoff assist still owns the pad) suppresses the gate trigger."""
    a = _mk(after_gates=(1,))
    out = _drive_to_pass(a, ground_active=True)
    assert out is None
    assert a.phase == "policy"


def test_gate_trigger_fires_once():
    """A listed gate engages ONCE: after a handback, a second pass of the same gate does not re-fire."""
    a = _mk(after_gates=(1,))
    out = _drive_to_pass(a, gate=1)
    assert out is not None and a.phase == "arrest"
    # force a handback (slow + in-view + level, held for the dwell)
    hb = dict(ground_active=False, speed_est=1.0, bearing_rad=0.0, rel_up=0.0,
              obs_pitch=-0.31, obs_roll=0.0, det_fresh=True)
    a.step(now=10.0, gate_index=2, **hb)
    assert a.step(now=10.4, gate_index=2, **hb) is None      # handback
    assert a.phase == "policy"
    # a fresh 1->2 transition must NOT re-engage (gate 1 already consumed)
    base = dict(ground_active=False, speed_est=6.0, bearing_rad=None, rel_up=None,
                obs_pitch=-0.4, obs_roll=0.0, det_fresh=False)
    a.step(now=11.0, gate_index=1, **base)
    assert a.step(now=11.0, gate_index=2, **base) is None
    assert a.phase == "policy"


# --------------------------------------------------------------------------- speed trigger
def test_speed_trigger_and_refractory():
    a = _mk(after_gates=(), speed_hi=5.0, refractory_s=2.0)
    base = dict(ground_active=False, bearing_rad=None, rel_up=None,
                obs_pitch=-0.4, obs_roll=0.0, det_fresh=True)
    # below the cap: no engage
    assert a.step(now=0.0, gate_index=3, speed_est=4.0, **base) is None
    # above the cap: engage
    out = a.step(now=0.1, gate_index=3, speed_est=6.0, **base)
    assert out is not None and out.event == "ENGAGE" and "speed" in out.reason
    # hand back (slow + in-view + level, dwell); arms the refractory at now
    hb = dict(base, bearing_rad=0.0, rel_up=0.0, obs_pitch=-0.31)
    a.step(now=1.0, gate_index=3, speed_est=1.0, **hb)
    assert a.step(now=1.4, gate_index=3, speed_est=1.0, **hb) is None
    assert a.phase == "policy"
    # within the refractory (< 2 s after handback): hot speed does NOT re-engage
    assert a.step(now=2.0, gate_index=3, speed_est=9.0, **base) is None
    assert a.phase == "policy"
    # after the refractory: re-engages
    out2 = a.step(now=3.5, gate_index=3, speed_est=9.0, **base)
    assert out2 is not None and out2.event == "ENGAGE"


# --------------------------------------------------------------------------- handback
def test_handback_requires_all_three_and_dwell():
    # (all times stay < max_s=4 s so the timeout-abort never preempts the handback under test)
    a = _mk(after_gates=(1,), handback_dwell_s=0.3)
    _drive_to_pass(a, gate=1)                                 # engage at now=0.0
    good = dict(ground_active=False, gate_index=2, speed_est=1.0, bearing_rad=0.0,
                rel_up=0.0, obs_pitch=-0.31, obs_roll=0.0, det_fresh=True)
    # missing gate-in-view -> no handback (keeps braking)
    assert a.step(now=0.5, **{**good, "det_fresh": False}) is not None
    # not slow -> no handback
    assert a.step(now=0.6, **{**good, "speed_est": 5.0}) is not None
    # not level (nose-down past the window) -> no handback
    assert a.step(now=0.7, **{**good, "obs_pitch": -0.9}) is not None
    assert a.phase == "arrest"
    # all three, but the dwell has not elapsed -> still braking
    assert a.step(now=1.0, **good) is not None
    assert a.step(now=1.1, **good) is not None
    # dwell elapsed -> handback (None), policy resumes
    assert a.step(now=1.4, **good) is None
    assert a.phase == "policy"


def test_handback_dwell_resets_on_a_break():
    """A tick that breaks the condition resets the dwell clock (anti-chatter)."""
    a = _mk(after_gates=(1,), handback_dwell_s=0.3)
    _drive_to_pass(a, gate=1)                                 # engage at now=0.0
    good = dict(ground_active=False, gate_index=2, speed_est=1.0, bearing_rad=0.0,
                rel_up=0.0, obs_pitch=-0.31, obs_roll=0.0, det_fresh=True)
    a.step(now=0.5, **good)                                   # dwell starts at 0.5
    assert a.step(now=0.7, **{**good, "det_fresh": False}) is not None   # break -> reset
    a.step(now=0.8, **good)                                   # dwell restarts at 0.8
    assert a.step(now=1.0, **good) is not None                # only 0.2 s -> not yet
    assert a.step(now=1.15, **good) is None                   # 0.35 s -> handback


# --------------------------------------------------------------------------- aborts (fail-open)
def test_abort_rel_up_drift():
    a = _mk(after_gates=(1,), rel_up_drift_m=2.0)
    _drive_to_pass(a, gate=1, rel_up=5.0)                     # entry rel_up latched at 5.0
    br = dict(ground_active=False, gate_index=2, speed_est=6.0, bearing_rad=0.0,
              obs_pitch=-0.4, obs_roll=0.0, det_fresh=True)
    assert a.step(now=0.1, rel_up=6.5, **br) is not None      # 1.5 m drift < 2 -> still braking
    assert a.step(now=0.2, rel_up=8.0, **br) is None          # 3.0 m drift > 2 -> ABORT (fail-open)
    assert a.phase == "policy"


def test_abort_gate_lost():
    a = _mk(after_gates=(1,), gate_lost_s=1.0)
    _drive_to_pass(a, gate=1)                                 # engage, gate never seen (rel_up None)
    br = dict(ground_active=False, gate_index=2, speed_est=6.0, bearing_rad=None,
              rel_up=None, obs_pitch=-0.4, obs_roll=0.0, det_fresh=False)
    assert a.step(now=0.5, **br) is not None                  # 0.5 s lost < 1 -> grace, still braking
    assert a.step(now=1.2, **br) is None                      # 1.2 s lost > 1 -> ABORT
    assert a.phase == "policy"


def test_abort_timeout():
    a = _mk(after_gates=(1,), max_s=4.0)
    _drive_to_pass(a, gate=1)
    # keep gate in view (no gate-lost) + hot (no handback) + stable rel_up (no drift): only timeout
    br = dict(ground_active=False, gate_index=2, speed_est=6.0, bearing_rad=0.0,
              rel_up=3.0, obs_pitch=-0.4, obs_roll=0.0, det_fresh=True)
    assert a.step(now=2.0, **br) is not None
    assert a.step(now=3.9, **br) is not None
    assert a.step(now=4.2, **br) is None                      # > max_s -> ABORT
    assert a.phase == "policy"


# --------------------------------------------------------------------------- command bounds + signs
def test_yaw_stare_sign_and_clamp():
    """Gate LEFT (bearing>0) => nose LEFT => wz<0 in FRD; magnitude clamped to the yaw clamp."""
    a = _mk(after_gates=(1,), yaw_clamp=0.7, k_yaw=1.5)
    # engage while already seeing a gate hard to the LEFT (bearing large +)
    out = _drive_to_pass(a, gate=1, bearing_rad=1.2, rel_up=2.0, det_fresh=True)
    assert out is not None
    assert out.rate_frd[2] == pytest.approx(-0.7)             # -clip(1.5*1.2)= -0.7 (gate left -> nose left)
    # gate to the RIGHT (bearing<0) => nose RIGHT => wz>0
    out_r = a.step(now=0.05, ground_active=False, gate_index=2, speed_est=6.0,
                   bearing_rad=-0.2, rel_up=2.0, obs_pitch=-0.4, obs_roll=0.0, det_fresh=True)
    assert out_r.rate_frd[2] == pytest.approx(0.2 * 1.5)      # +0.3, within the clamp


def test_pitch_brake_nose_up_cap_and_level_stop():
    a = _mk(after_gates=(1,), k_brake=0.1, pitch_rate_cap=0.8, pitch_level_rad=-0.05)
    # fast forward, well below level -> nose-UP (wy>0), clamped to the cap
    out = _drive_to_pass(a, gate=1, speed_est=20.0, obs_pitch=-0.5, det_fresh=True, rel_up=2.0)
    assert out.rate_frd[1] == pytest.approx(0.8)              # clip(0.1*20,0,0.8) nose-up
    assert out.rate_frd[1] > 0.0
    # at/above level -> hard-stop further nose-up (camera-out-the-bottom guard)
    out_l = a.step(now=0.05, ground_active=False, gate_index=2, speed_est=20.0, bearing_rad=0.0,
                   rel_up=2.0, obs_pitch=-0.02, obs_roll=0.0, det_fresh=True)
    assert out_l.rate_frd[1] == 0.0
    # roll is always commanded 0
    assert out_l.rate_frd[0] == 0.0


def test_collective_altitude_hold_and_bounds():
    # rel_up_drift_m huge so the DRIFT ABORT does not preempt the collective-CLAMP under test
    # (in normal operation the 2 m drift abort fires long before g reaches [0.7,1.4]; the clamp is a
    # safety backstop). k_z=0.15: within +-2 m of the entry ref, g stays in [0.7,1.3] (no clamp).
    a = _mk(after_gates=(1,), k_z=0.15, collective_lo_g=0.7, collective_hi_g=1.4,
            rel_up_drift_m=1000.0)
    out0 = _drive_to_pass(a, gate=1, rel_up=5.0, det_fresh=True, bearing_rad=0.0)  # entry ref = 5.0
    assert out0.normed_g == pytest.approx(1.0)               # at entry: drift 0 -> hover
    assert out0.collective_01 == pytest.approx(fly_rl._clip01(1.0 * _HOVER))
    # drone SANK (gate now higher, rel_up up 1 m) -> climb (>hover)
    up = a.step(now=0.05, ground_active=False, gate_index=2, speed_est=3.0, bearing_rad=0.0,
                rel_up=6.0, obs_pitch=-0.4, obs_roll=0.0, det_fresh=True)
    assert up.normed_g == pytest.approx(1.0 + 0.15 * 1.0)    # 1.15 g
    # a large positive drift saturates at the high bound
    hi = a.step(now=0.06, ground_active=False, gate_index=2, speed_est=3.0, bearing_rad=0.0,
                rel_up=5.0 + 100.0, obs_pitch=-0.4, obs_roll=0.0, det_fresh=True)
    assert hi.normed_g == pytest.approx(1.4)                 # clamped to collective_hi_g
    assert hi.collective_01 == pytest.approx(fly_rl._clip01(1.4 * _HOVER))
    # a large negative drift saturates at the low bound
    lo = a.step(now=0.07, ground_active=False, gate_index=2, speed_est=3.0, bearing_rad=0.0,
                rel_up=5.0 - 100.0, obs_pitch=-0.4, obs_roll=0.0, det_fresh=True)
    assert lo.normed_g == pytest.approx(0.7)
    assert lo.collective_01 == pytest.approx(fly_rl._clip01(0.7 * _HOVER))


def test_yaw_zero_when_no_bearing():
    """No slot0 estimate yet (bearing None) -> hold heading (wz=0), still brake + hover-hold."""
    a = _mk(after_gates=(1,))
    out = _drive_to_pass(a, gate=1, bearing_rad=None, rel_up=None, speed_est=6.0)
    assert out.rate_frd[2] == 0.0
    assert out.normed_g == pytest.approx(1.0)               # no altitude ref -> hover
    assert out.rate_frd[1] > 0.0                             # still braking


def test_rate_frd_is_length3_float():
    a = _mk(after_gates=(1,))
    out = _drive_to_pass(a, gate=1, bearing_rad=0.3, rel_up=2.0, det_fresh=True)
    assert isinstance(out.rate_frd, np.ndarray)
    assert out.rate_frd.shape == (3,)
    assert 0.0 <= out.collective_01 <= 1.0
