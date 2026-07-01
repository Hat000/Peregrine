"""FEEDFORWARD-OWNS-VERTICAL — the A19c alt-hold bang-bang ROOT fix (the vertical twin of
ff_owns_horizontal).

THE BUG (live VQ2, A19c, 2026-07-01): the decoupled controller's altitude-hold thrust
    thrust = hover + kp_alt*(z - z_t) + kd_alt*(vel[2] - vz_t)
SLAMMED bang-bang between its clip rails (0.05 <-> 0.60) every 2-4 ticks for the whole active
flight. In pursuit ``sp.position_ned`` is None, so ``kp_alt*(z - z_t)`` collapses to the CONSTANT
``kp_alt*alt_offset_m`` and the ONLY varying driver is ``kd_alt*(vel[2] - vz_t)`` with kd_alt=3.0.
``vel[2] = nav.velocity_ned[2]`` is the estimator's DEAD-RECKONED vertical velocity: on the VQ2
wire there is no baro and ODOMETRY is blocked, so vz is pure IMU-accel integration (noisy +
drifting). Damping the collective against that rail-slammed the thrust -> a net asymmetric climb
OVER the acquired gate -> gate lost -> the 360 yaw-search.

THE FIX: ``Controller.ff_owns_vertical`` -- when set AND ``accel_ned`` is present, the vertical
channel does NOT damp against ``vel[2]``. It keeps a position loop on the trustworthy floor-
corrected z, routes the vertical-align ``vz_t`` through a RAMPING z_target, and damps against a
low-passed FINITE DIFFERENCE of the (bounded, corrected) z -- a trustworthy vertical rate that
cannot drift the collective into the rails. Default OFF => VQ1 / case-A byte-identical. vq2_case_c
flips it ON via ``controller_overrides``.

[VQ2 slow-is-smooth, A19c, 2026-07-01]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import NavState, Setpoint  # noqa: E402
from racer.deploy_profile import vq2_case_c  # noqa: E402
from racer.gate_seeker import make_seeker_controller  # noqa: E402

DT_NS = int(1e9 / 12)          # ~12 Hz, the choked VQ2 loop rate
LO, HI = 0.05, 0.6             # make_seeker_controller alt-hold clip rails


def _drifting_vz_arc(n=40, seed=0):
    """A held altitude (z ~= const) but a DEAD-RECKONED vel[2] that DRIFTS + jitters -- the state-
    denied VQ2 wire, where vz is pure IMU integration. This is the poison that fed kd_alt."""
    rng = np.random.default_rng(seed)
    z = -2.0 + 0.02 * rng.standard_normal(n)              # true altitude ~ held (small noise)
    # dead-reckoned vz: a slow drift + tick jitter, decorrelated from the (held) true z-rate
    vz = np.cumsum(0.05 * rng.standard_normal(n)) + 0.4 * rng.standard_normal(n)
    t = np.arange(n) * DT_NS
    return z, vz, t


def _pursuit_setpoint(t_ns, vz_t):
    """Pursuit-style setpoint: bounded forward feedforward (accel_ned) + a vertical-align vz carried
    in velocity_ned (horizontal ZERO), exactly what _feedforward_command emits (position_ned None)."""
    vned = np.array([0.0, 0.0, vz_t]) if vz_t != 0.0 else None
    return Setpoint(sim_time_ns=int(t_ns),
                    accel_ned=1.2 * np.array([1.0, 0.0, 0.0]),
                    velocity_ned=vned, yaw=0.0)


def _run_alt_hold(ctrl, z, vz, t, vz_t=0.0):
    """Replay the (z, vz) sequence through the alt-hold at level attitude; return the thrust trace."""
    out = np.zeros_like(z)
    for i in range(len(z)):
        nav = NavState(sim_time_ns=int(t[i]),
                       position_ned=np.array([0.0, 0.0, z[i]]),
                       velocity_ned=np.array([0.0, 0.0, vz[i]]),
                       roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3))
        out[i] = float(ctrl.command(nav, _pursuit_setpoint(t[i], vz_t)).thrust)
    return out


def _rail_flips(thr):
    """Count adjacent-tick lo<->hi rail slams (the bang-bang signature)."""
    return int(sum(
        1 for i in range(1, len(thr))
        if (abs(thr[i - 1] - LO) < 1e-9 and abs(thr[i] - HI) < 1e-9)
        or (abs(thr[i - 1] - HI) < 1e-9 and abs(thr[i] - LO) < 1e-9)
    ))


def test_flag_off_reproduces_the_bangbang():
    """Byte-identity / bug guard: with the fix OFF (VQ1 / case-A default), damping the DRIFTING
    dead-reckoned vel[2] STILL rail-slams the collective (the documented A19c bug) -- proving the
    flag is the ONLY behavioural change."""
    z, vz, t = _drifting_vz_arc()
    ctrl = make_seeker_controller(ff_owns_vertical=False)
    thr = _run_alt_hold(ctrl, z, vz, t)
    # the poison rail-slams: thrust hits BOTH rails and flips between them repeatedly
    assert np.any(np.abs(thr - LO) < 1e-9) and np.any(np.abs(thr - HI) < 1e-9), thr
    assert _rail_flips(thr) >= 5, ("expected a rail-slam on the OFF path", _rail_flips(thr))


def test_ff_owns_vertical_kills_the_bangbang():
    """With the fix ON, damping is against the low-passed finite-diff of the trustworthy z (NOT the
    drifting vel[2]), so a held altitude produces a STABLE thrust around hover with NO rail-slam."""
    z, vz, t = _drifting_vz_arc()
    ctrl = make_seeker_controller(ff_owns_vertical=True)
    thr = _run_alt_hold(ctrl, z, vz, t)
    # no bang-bang: essentially no lo<->hi rail flips
    assert _rail_flips(thr) <= 1, ("ff_owns_vertical rail-slammed", _rail_flips(thr), thr)
    # and after the first-tick latch the thrust sits in a sane band around hover (not pinned to a rail)
    hover = ctrl.hover_thrust
    assert abs(float(np.median(thr[1:])) - hover) < 0.12, (np.median(thr[1:]), hover)
    assert float(np.std(thr[1:])) < 0.5 * float(np.std(_run_alt_hold(
        make_seeker_controller(ff_owns_vertical=False), z, vz, t))), "ON should be far calmer than OFF"


def test_ff_owns_vertical_tracks_commanded_vertical():
    """A commanded vertical-align vz_t must still move altitude: a SINK (vz_t>0) cuts thrust below a
    hold, a CLIMB (vz_t<0) raises it -- the A5-blocker-1 vertical-align intent is preserved through
    the ramping z_target (no fictional-velocity feedback)."""
    z = np.full(24, -2.0)                 # held altitude (isolate the vz_t effect)
    vz = np.zeros(24)                     # no dead-reckoned drift here (isolate vz_t)
    t = np.arange(24) * DT_NS
    hold = np.mean(_run_alt_hold(make_seeker_controller(ff_owns_vertical=True), z, vz, t, vz_t=0.0))
    sink = np.mean(_run_alt_hold(make_seeker_controller(ff_owns_vertical=True), z, vz, t, vz_t=+0.3))
    climb = np.mean(_run_alt_hold(make_seeker_controller(ff_owns_vertical=True), z, vz, t, vz_t=-0.3))
    assert sink < hold < climb, (sink, hold, climb)


def test_off_path_byte_identical():
    """OFF-path byte-identity: with the flag OFF, the alt-hold thrust equals the EXACT legacy formula
    ``hover + kp_alt*(z - (z - alt_offset)) + kd_alt*(vel[2] - vz_t)`` computed independently, for an
    arbitrary (drifting) vz -- the OFF path is untouched by the fix."""
    z, vz, t = _drifting_vz_arc(seed=3)
    ctrl = make_seeker_controller(ff_owns_vertical=False)   # default gains == today's flight code
    thr = _run_alt_hold(ctrl, z, vz, t, vz_t=0.0)
    for i in range(len(z)):
        # legacy: position term collapses to kp_alt*alt_offset_m (alt_offset=0 -> 0); tilt_comp at
        # level attitude divides by cos(0)*cos(0)=1 (no-op); then clip to the rails.
        z_t = float(z[i]) - ctrl.alt_offset_m
        expect = ctrl.hover_thrust + ctrl.kp_alt * (z[i] - z_t) + ctrl.kd_alt * (vz[i] - 0.0)
        expect = float(np.clip(expect, ctrl.alt_thrust_lo, ctrl.alt_thrust_hi))
        assert abs(thr[i] - expect) < 1e-9, (i, thr[i], expect)


def test_default_is_off():
    """The Controller default (and thus VQ1 / every offline path) leaves ff_owns_vertical OFF."""
    from racer.controller import Controller
    assert Controller().ff_owns_vertical is False


def test_vq2_profile_enables_the_fix():
    """The vq2_case_c profile must ship the fix ON (beside ff_owns_horizontal)."""
    ov = vq2_case_c().controller_overrides
    assert ov.get("ff_owns_vertical") is True
    assert ov.get("ff_owns_horizontal") is True     # the horizontal twin stays ON too
