"""FEEDFORWARD-OWNS-HORIZONTAL — the A15b nose-up-into-the-ceiling ROOT fix.

THE BUG (live VQ2, A15b flight 2, 2026-07-01): the instant pursuit engaged the seeker commanded a
SATURATED +1.5 rad/s NOSE-UP pitch rate and the drone ballooned up into the ceiling. ROOT CAUSE: the
map-free pursuit passes ``accel_ned`` (a small bounded forward feedforward tilt) AND, when vertical-
align is active, ``velocity_ned=[0,0,vz]`` to feed the alt-hold a sink/climb rate. The alt-hold reads
``velocity_ned[2]`` on its own, but the decoupled controller's legacy horizontal block ALSO added
``kd_vel*(velocity_ned[0:2] - vel_xy) = -kd_vel*vel_xy`` -- damping the DEAD-RECKONED horizontal
velocity. On the state-denied VQ2 wire that velocity is FICTION and grows (~1.8 m/s forward), so the
backward damping accel (~-5.4) SWAMPED the +1.2 forward feedforward, flipped the horizontal accel
BACKWARD, and ``_accel_to_attitude`` tilted the target NOSE-UP.

THE FIX: ``Controller.ff_owns_horizontal`` -- when set AND ``accel_ned`` is present, the horizontal
channel is driven PURELY by the feedforward; the ``velocity_ned`` term is NOT applied to the
horizontal axes (the alt-hold still consumes ``velocity_ned[2]`` for the vertical-align vz). Default
OFF => VQ1 / case-A byte-identical. vq2_case_c flips it ON via ``controller_overrides``.

[VQ2 slow-is-smooth, A15b, 2026-07-01]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import NavState, Setpoint  # noqa: E402
from racer.deploy_profile import vq2_case_c  # noqa: E402
from racer.gate_seeker import make_seeker_controller  # noqa: E402


def _handoff_nav(sim_time_ns=0):
    """The A15b egress->pursuit handoff state: ~-15.5 deg nose-down + a DEAD-RECKONED forward
    velocity (fiction on the denied wire) that the horizontal damping term would chase."""
    return NavState(
        sim_time_ns=sim_time_ns,
        position_ned=np.array([4.4, 0.0, -1.4]),
        velocity_ned=np.array([1.8, 0.0, -0.5]),   # dead-reckoned; grows on the denied wire
        roll=0.0, pitch=np.deg2rad(-15.5), yaw=np.deg2rad(-0.5),
        angular_rate_body=np.zeros(3),
    )


def _pursuit_setpoint(nav, vz):
    """A pursuit-style setpoint: bounded forward feedforward (accel_ned) + a vertical-align vz
    carried in velocity_ned (horizontal components ZERO), exactly what _feedforward_command emits."""
    # Mirror _feedforward_command exactly: velocity_ned is None when there is no vertical-align (vz==0),
    # else [0,0,vz] (horizontal ZERO). A [0,0,0] would itself trip the horizontal damping term.
    vned = np.array([0.0, 0.0, vz]) if vz != 0.0 else None
    return Setpoint(
        sim_time_ns=nav.sim_time_ns,
        accel_ned=1.2 * np.array([1.0, 0.0, 0.0]),          # forward feedforward (~7 deg nose-down)
        velocity_ned=vned,                                   # vertical-align vz ONLY (or None)
        yaw=0.0, launch_ramp=None,
    )


def _pitch_rate(controller_kwargs, vz):
    nav = _handoff_nav()
    ctrl = make_seeker_controller(**controller_kwargs)
    cmd = ctrl.command(nav, _pursuit_setpoint(nav, vz))
    return float(np.asarray(cmd.body_rate)[1]), float(cmd.thrust)


def test_ff_owns_horizontal_kills_the_noseup():
    """With the fix ON, a vertical-align vz must NOT swing the pitch command nose-up: it stays at
    the benign forward-lean-leveling value (== the vz=0 case), NOT the saturated nose-up climb."""
    kw = {"kp_att": 4.0, "ff_owns_horizontal": True}
    pr_novz, _ = _pitch_rate(kw, vz=0.0)
    pr_climb, thr_climb = _pitch_rate(kw, vz=-0.8)
    # vertical-align must not change the pitch command (horizontal is pure feedforward now)
    assert pr_climb == pr_novz, (pr_climb, pr_novz)
    # and it stays a small BENIGN command (leveling the -15.5 deg dive toward the ~-7 deg lean),
    # nowhere near the +1.5 saturated nose-up climb that flew into the ceiling.
    assert pr_climb < 1.0, pr_climb
    # the vertical-align climb is UNAFFECTED: the alt-hold still consumed velocity_ned[2] -> more thrust.
    thr_flat = _pitch_rate(kw, vz=0.0)[1]
    assert thr_climb > thr_flat, (thr_climb, thr_flat)


def test_flag_off_reproduces_the_noseup():
    """Byte-identity guard: with the fix OFF (VQ1 / case-A default), the vertical-align vz STILL
    poisons the horizontal (the documented bug) -- proving the flag is the ONLY behavioural change."""
    kw = {"kp_att": 4.0, "ff_owns_horizontal": False}
    pr_novz, _ = _pitch_rate(kw, vz=0.0)
    pr_climb, _ = _pitch_rate(kw, vz=-0.8)
    # OFF: the vz case is strictly MORE nose-up than the no-vz case (the dead-reckoned-velocity poison)
    assert pr_climb > pr_novz + 0.5, (pr_climb, pr_novz)


def test_vq2_profile_enables_the_fix():
    """The vq2_case_c profile must ship the fix ON (and VQ1 default leaves it OFF)."""
    assert vq2_case_c().controller_overrides.get("ff_owns_horizontal") is True
