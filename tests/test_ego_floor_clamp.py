"""EGO deploy-side FLOOR / descent fence for the low gates (gates 4/5/6 near the ground).

Pins the fly_ego floor-clamp contract [ego-deploy 2026-07-12]:

  * THE PROBLEM: the low gates sit close to the pad; the policy dives to reach them and can
    OVERSHOOT into the floor (gate contact = INVALID run + crash). EgoFloorClamp arrests a
    NEAR-GROUND sink by flooring the EMITTED collective to the takeoff-assist thrust floor
    (>= hover), WITHOUT pinning altitude — above the clamp height it is a bit-identical
    passthrough, so the drone still descends toward a low gate.

  * ONE-SIDED, arrest-only: it fires ONLY when (estimated height above pad) < clamp_m AND the
    drone is descending; it only ever RAISES thrust (max with the policy's own output), never
    forces a climb when high, and never touches roll/pitch/yaw. Descent = nav down-velocity > 0
    (primary), or — when velocity is missing/non-finite — the policy commanding below hover
    (fallback). An ASCENDING drone is never fenced.

  * The state object (EgoFloorClamp) is unit-tested as a pure object: the g-units floor + its
    [0,1] collective conversion, the pad-datum latch + NED height convention, above-clamp and
    ascending passthrough, the velocity/fallback descent detection, one-sided never-lower, the
    obs[8] feedback reflecting the ARRESTED value, blind-safety on missing altitude, the
    rising-edge log, and the disabled (clamp=0) byte-identical no-op.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

import fly_rl  # noqa: E402

_HOVER = fly_rl._HOVER_THRUST                       # 0.2656 g-units -> [0,1] collective


def _mk(clamp_m=0.5, floor_g=1.10):
    """Match the deploy wiring: floor_g REUSES the takeoff-assist thrust floor (--ego-assist-thrust,
    default 1.10 g); hover_collective = _HOVER_THRUST; hover_g = 1.0 (the 'descending' threshold)."""
    return fly_rl.EgoFloorClamp(clamp_m=clamp_m, hover_collective=_HOVER, floor_g=floor_g)


def _passthru_coll(polN):
    return float(np.clip(polN * fly_rl._HOVER_THRUST, 0.0, 1.0))


# --------------------------------------------------------------------------- arrest (the core case)
def test_below_clamp_and_descending_raises_to_floor():
    """Below the clamp height + descending (nav down-vel > 0) + sub-floor thrust -> the emitted
    thrust is raised to the floor (>= hover) and the wire collective follows."""
    a = _mk(clamp_m=0.5, floor_g=1.10)
    a.apply(1.10, nav_z=0.0, vel_down=0.0)                  # latch pad datum at z0 = 0 (on the pad)
    emitN, coll, on = a.apply(0.25, nav_z=-0.2, vel_down=+0.5)   # 0.2 m up (< 0.5), sinking
    assert on is True
    assert emitN == pytest.approx(1.10)                    # raised to the g-units floor (>= hover 1.0)
    assert coll == pytest.approx(fly_rl._clip01(1.10 * _HOVER))
    assert emitN >= 1.0                                    # 'at least a hover level'


def test_arrest_on_the_latch_tick_when_on_the_pad():
    """On the FIRST finite-z tick the datum latches and height == 0 < clamp; a descending sub-floor
    command is arrested immediately (defensive: no un-fenced first tick)."""
    a = _mk(clamp_m=0.4)
    emitN, coll, on = a.apply(0.30, nav_z=0.0, vel_down=+0.3)
    assert on is True and emitN == pytest.approx(1.10)


# --------------------------------------------------------------------------- above clamp / ascending
def test_above_clamp_is_untouched():
    """Above the clamp height the fence is a bit-identical passthrough (so the drone can descend
    toward a low gate), regardless of a downward velocity or a sub-hover thrust."""
    a = _mk(clamp_m=0.5)
    a.apply(1.10, nav_z=0.0, vel_down=0.0)                  # latch z0 = 0
    emitN, coll, on = a.apply(0.25, nav_z=-2.0, vel_down=+0.9)   # 2.0 m up: well above the fence
    assert on is False
    assert emitN == pytest.approx(0.25)
    assert coll == pytest.approx(_passthru_coll(0.25))


def test_ascending_is_never_fenced():
    """Below the clamp but ASCENDING (nav down-vel < 0) -> untouched, even with a sub-hover thrust."""
    a = _mk(clamp_m=0.5)
    a.apply(1.10, nav_z=0.0, vel_down=0.0)                  # latch z0 = 0
    emitN, coll, on = a.apply(0.25, nav_z=-0.2, vel_down=-0.5)   # 0.2 m up but climbing
    assert on is False and emitN == pytest.approx(0.25)


def test_level_hover_below_clamp_is_untouched():
    """Below the clamp, exactly level (vel_down == 0, not > 0) with an at-hover thrust -> untouched
    (the fence arrests a SINK, it does not pin altitude)."""
    a = _mk(clamp_m=0.5)
    a.apply(1.0, nav_z=0.0, vel_down=0.0)                   # latch z0 = 0
    emitN, coll, on = a.apply(1.0, nav_z=-0.2, vel_down=0.0)
    assert on is False and emitN == pytest.approx(1.0)


# --------------------------------------------------------------------------- one-sided (never lower)
def test_policy_at_or_above_floor_passes_through():
    """One-sided: when the policy already commands at/above the floor, a descending below-clamp
    tick must NOT lower it."""
    a = _mk(clamp_m=0.5, floor_g=1.10)
    a.apply(1.10, nav_z=0.0, vel_down=0.0)                  # latch z0 = 0
    emitN, coll, on = a.apply(2.5, nav_z=-0.2, vel_down=+0.5)    # below clamp + sinking, but strong thrust
    assert on is False                                     # nothing to arrest
    assert emitN == pytest.approx(2.5)
    assert coll == pytest.approx(_passthru_coll(2.5))


def test_between_hover_and_floor_while_sinking_is_raised_to_floor():
    """A slightly-over-hover command (1.05 g) that is still sinking (momentum) below the clamp is
    raised to the floor (1.10) — one-sided max, never a reduction."""
    a = _mk(clamp_m=0.5, floor_g=1.10)
    a.apply(1.10, nav_z=0.0, vel_down=0.0)                  # latch z0 = 0
    emitN, _, on = a.apply(1.05, nav_z=-0.2, vel_down=+0.4)
    assert on is True and emitN == pytest.approx(1.10)


# --------------------------------------------------------------------------- obs[8] feedback
def test_obs8_feedback_is_the_arrested_value():
    """The FIRST return (fed back as obs[8] next tick) is the ACTUALLY emitted g-units value, not
    the raw sub-floor policy output."""
    a = _mk(clamp_m=0.5)
    a.apply(1.10, nav_z=0.0, vel_down=0.0)                  # latch z0 = 0
    emitN, _, _ = a.apply(0.20, nav_z=-0.1, vel_down=+0.6)
    assert emitN == pytest.approx(1.10)                    # NOT 0.20


def test_collective_floor_never_exceeds_one():
    """A large floor_g must still clip the wire collective to [0,1]."""
    a = _mk(clamp_m=0.5, floor_g=5.0)
    a.apply(5.0, nav_z=0.0, vel_down=0.0)                   # latch z0 = 0 (policy above floor -> no arrest)
    _, coll, on = a.apply(0.1, nav_z=-0.1, vel_down=+0.5)
    assert on is True and coll == pytest.approx(1.0)


# --------------------------------------------------------------------------- velocity fallback
def test_fallback_missing_velocity_uses_sub_hover_thrust():
    """When velocity is unavailable (None), 'descending' falls back to the policy commanding below
    hover (1.0 g): a sub-hover thrust below the clamp is arrested."""
    a = _mk(clamp_m=0.5)
    a.apply(1.10, nav_z=0.0, vel_down=None)                 # latch z0 = 0
    emitN, _, on = a.apply(0.30, nav_z=-0.2, vel_down=None)
    assert on is True and emitN == pytest.approx(1.10)


def test_fallback_missing_velocity_above_hover_not_fenced():
    """Fallback: with no velocity and a thrust at/above hover (but below the floor), the drone is
    not treated as descending -> untouched (avoid fencing a non-sinking hold)."""
    a = _mk(clamp_m=0.5, floor_g=1.10)
    a.apply(1.10, nav_z=0.0, vel_down=None)                 # latch z0 = 0
    emitN, _, on = a.apply(1.05, nav_z=-0.2, vel_down=None)
    assert on is False and emitN == pytest.approx(1.05)


def test_nonfinite_velocity_falls_back_like_missing():
    """A NaN velocity is treated as unavailable (fallback to the thrust proxy), never as +/-."""
    a = _mk(clamp_m=0.5)
    a.apply(1.10, nav_z=0.0, vel_down=np.nan)               # latch z0 = 0
    emitN, _, on = a.apply(0.30, nav_z=-0.2, vel_down=np.nan)
    assert on is True and emitN == pytest.approx(1.10)


# --------------------------------------------------------------------------- blind-safety on altitude
def test_missing_altitude_never_fences():
    """No altitude estimate -> the datum never latches and the fence never fires (never fence blind)."""
    a = _mk(clamp_m=0.5)
    emitN, coll, on = a.apply(0.20, nav_z=None, vel_down=+1.0)
    assert on is False and emitN == pytest.approx(0.20)
    assert a._z0 is None                                   # no datum latched from a missing z


def test_nonfinite_altitude_never_latches_or_fences():
    """A NaN nav-z must not latch a NaN datum and must not fire the fence."""
    a = _mk(clamp_m=0.5)
    emitN, _, on = a.apply(0.20, nav_z=np.nan, vel_down=+1.0)
    assert on is False and emitN == pytest.approx(0.20) and a._z0 is None


def test_pad_datum_latches_once_and_defines_height():
    """The datum is the FIRST finite nav-z and never moves; height = z0 - z (NED down-positive)."""
    a = _mk(clamp_m=0.5)
    a.apply(1.10, nav_z=-3.0, vel_down=0.0)                 # first finite z -> z0 = -3.0
    assert a._z0 == pytest.approx(-3.0)
    # 3.4 m 'below' the datum in NED (z = 0.4 -> height = -3.0 - 0.4 = -3.4 < clamp): descending arrests
    emitN, _, on = a.apply(0.25, nav_z=0.4, vel_down=+0.5)
    assert on is True and emitN == pytest.approx(1.10)
    # a later NaN z does not move the datum
    a.apply(0.25, nav_z=np.nan, vel_down=+0.5)
    assert a._z0 == pytest.approx(-3.0)


# --------------------------------------------------------------------------- disabled = byte-identical
def test_disabled_is_bit_identical_passthrough():
    """--ego-floor-clamp 0: pass-through, floor_on False, collective bit-identical to what
    policy_step computes for the same normed thrust, and NO log emitted — for any altitude/velocity."""
    a = _mk(clamp_m=0.0)
    logs = []
    for polN in (0.05, 0.25, 1.0, 1.88, 3.765):
        emitN, coll, on = a.apply(polN, nav_z=-0.1, vel_down=+2.0, log=logs.append)
        assert on is False
        assert emitN == pytest.approx(polN)
        assert coll == pytest.approx(_passthru_coll(polN))
    assert a.enabled is False and logs == []               # disabled never fences or logs
    assert a._z0 is None                                   # disabled never even latches a datum


# --------------------------------------------------------------------------- logging (rising edge)
def test_log_fires_once_per_arrest_onset():
    """Exactly one ARREST line at the onset; it does NOT repeat while the arrest continues, and
    re-fires after a passthrough gap."""
    a = _mk(clamp_m=0.5)
    logs = []
    a.apply(1.10, nav_z=0.0, vel_down=0.0, log=logs.append)          # latch, no arrest
    a.apply(0.25, nav_z=-0.2, vel_down=+0.5, log=logs.append)        # ARREST (onset) -> 1 log
    a.apply(0.25, nav_z=-0.2, vel_down=+0.5, log=logs.append)        # still arresting -> no new log
    assert len(logs) == 1 and logs[0].startswith("[ego-floor] ARREST")
    a.apply(0.25, nav_z=-2.0, vel_down=+0.5, log=logs.append)        # above clamp -> passthrough gap
    a.apply(0.25, nav_z=-0.2, vel_down=+0.5, log=logs.append)        # ARREST again -> a new log
    assert len(logs) == 2


# --------------------------------------------------------------------------- source guard
def test_single_instantiation_only_on_ego_path():
    """The floor clamp is wired into _fly_ego only (source guard against accidental plumbing into a
    non-ego path)."""
    src = (_RL / "fly_rl.py").read_text(encoding="utf-8")
    assert src.count("EgoFloorClamp(") == 1                          # constructed exactly once
    ego_start = src.index("def _fly_ego(")
    ego_end = src.index("def _fly_armed(")
    assert ego_start < src.index("EgoFloorClamp(") < ego_end         # ...inside _fly_ego
