"""Pitch-sign faithfulness pin for the AHRS attitude estimate -> CTBR control chain.

WHY THIS EXISTS (VQ2 A8 climb-and-retreat diagnosis, 2026-06-30)
---------------------------------------------------------------
A8 flew NOSE-UP + throttle-down, climbing backward into the ceiling/back-wall. Three
independent analyses converged on a WRONG live ATTITUDE ESTIMATE (an AHRS/ESKF pitch or
gyro_y sign flip) rather than seeker logic: with a CORRECT estimate the seeker commands
nose-DOWN/forward, but a pitch estimate biased nose-DOWN (while truly level) makes the
controller "correct" by commanding nose-UP -> the real nose pitches up, thrust collapses.

A gyro/attitude sign error is DETERMINISTICALLY findable offline: feed a known angular
velocity / known tilt and check the integrated attitude (and the resulting control
command) move the correct way. This test does exactly that across the four sign seams:

  SEAM 2  gyro_y -> ESKF pitch       : +gyro_y (body-Y rate) integrates to +pitch (nose-up).
  SEAM 4  accel  -> seed / tilt fix  : a nose-up specific-force levels to +pitch (nose-up).
  SEAM 3  AHRS TRUE -> re-encode      : the ODOMETRY-wire re-encode (q_true*CONJ) decodes to
                                        the SAME pitch the controller's alias expects.
  SEAM 1->3 end-to-end                : a CORRECT nose-up estimate yields a RESTORING
                                        (nose-DOWN) pitch-rate command; the INVERTED estimate
                                        yields the AMPLIFYING (nose-UP) A8 signature.

If any seam silently inverts pitch, the corresponding assert fails and localises the bug.
Pure numpy/scipy; torch-free. [A8 attitude-sign audit 2026-06-30]
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.ahrs.ahrs_adapter import AHRSAttitudeSource
from racer.ahrs.eskf import ESKFAHRS
from racer.contracts import NavState, Setpoint
from racer.frames import (
    ODO_QUAT_TRUE_CONJ_WXYZ,
    R_world_from_body,
    euler_from_quat_wxyz,
)
from racer.gate_seeker import make_seeker_controller

G = 9.80665
G_NED = np.array([0.0, 0.0, G])
DT = 0.005


def _quat_true_wxyz(roll, pitch, yaw):
    """TRUE body(FRD)->world(NED) quaternion (w,x,y,z) from aerospace ZYX Euler."""
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return np.array([w, x, y, z])


def _sf_body(roll, pitch, yaw, lin_accel_world=None):
    """IMU specific force in body FRD for a given attitude (rest reading = [0,0,-g])."""
    R_wb = R_world_from_body(roll, pitch, yaw)
    a_world = np.zeros(3) if lin_accel_world is None else np.asarray(lin_accel_world, float)
    return R_wb.T @ (a_world - G_NED)


# ===========================================================================
# SEAM 2 — gyro_y (pitch rate) integrates to the correct-sign pitch
# ===========================================================================
def test_gyro_y_pulse_integrates_nose_up():
    """A KNOWN positive gyro_y (body-Y rate) must integrate to POSITIVE (nose-up) pitch,
    with the correct MAGNITUDE (omega*t). An inverted gyro_y seam would give -pitch."""
    eskf = ESKFAHRS(gyro_noise_std=0.0, gyro_bias_std=0.0, accel_gate_alpha=0.0,
                    accel_chi2_thresh=0.0, accel_freefall_tol_lo=-1.0,
                    accel_freefall_tol_hi=-1.0)
    eskf.reset()
    wy = 0.5                         # +0.5 rad/s about body-Y
    n = 80                           # 0.4 s -> expect pitch ~ +0.20 rad
    for _ in range(n):
        eskf._predict(np.array([0.0, wy, 0.0]), DT)
    _, pitch, _ = euler_from_quat_wxyz(eskf.q_wxyz)
    expected = wy * n * DT
    assert pitch > 0.0, f"gyro_y +0.5 gave pitch={pitch:+.4f} (INVERTED — nose-up seam flipped)"
    assert abs(pitch - expected) < 1e-3, f"pitch {pitch:+.4f} != expected {expected:+.4f}"


# ===========================================================================
# SEAM 4 — accel-only leveling settles at the correct-sign tilt
# ===========================================================================
@pytest.mark.parametrize("true_pitch_deg", [+15.0, -15.0])
def test_accel_seed_and_tilt_sign(true_pitch_deg):
    """level_seed_from_accel AND the ESKF accel update must recover the TRUE tilt SIGN
    (a nose-up specific-force -> +pitch). Both directions checked so a sign flip can't hide."""
    theta = np.deg2rad(true_pitch_deg)
    sf = _sf_body(0.0, theta, 0.0)

    # (a) closed-form seed
    q_seed = AHRSAttitudeSource.level_seed_from_accel(sf)
    _, p_seed, _ = euler_from_quat_wxyz(q_seed)
    assert np.sign(p_seed) == np.sign(theta), f"seed pitch sign flipped: {np.rad2deg(p_seed):+.2f}"
    assert abs(p_seed - theta) < 1e-6

    # (b) ESKF accel update converging from a LEVEL seed
    eskf = ESKFAHRS(gyro_noise_std=0.0, gyro_bias_std=0.0, accel_gate_alpha=10.0,
                    accel_chi2_thresh=0.0)
    eskf.reset()
    for _ in range(2000):
        eskf.step(np.zeros(3), sf, DT)
    _, p_conv, _ = euler_from_quat_wxyz(eskf.q_wxyz)
    assert np.sign(p_conv) == np.sign(theta), f"accel-update pitch sign flipped: {np.rad2deg(p_conv):+.2f}"
    assert abs(p_conv - theta) < np.deg2rad(0.5)


# ===========================================================================
# SEAM 3 — the ODOMETRY-wire re-encode preserves the pitch the controller expects
# ===========================================================================
@pytest.mark.parametrize("true_pitch_deg", [+15.0, -15.0])
def test_reencode_preserves_controller_pitch(true_pitch_deg):
    """_step_ahrs re-encodes the TRUE attitude as q_true*CONJ and hands euler_from_quat_wxyz(that)
    to the controller. The decoded pitch must keep the TRUE sign (it is a R_y(pi) conjugation, which
    is a pitch-PRESERVING involution) so the controller's feedback is not silently inverted."""
    theta = np.deg2rad(true_pitch_deg)
    q_true = _quat_true_wxyz(0.0, theta, 0.0)
    q_odo = q_true * ODO_QUAT_TRUE_CONJ_WXYZ           # what _step_ahrs caches
    _, p_ctrl, _ = euler_from_quat_wxyz(q_odo)         # what the controller receives as nav.pitch
    assert np.sign(p_ctrl) == np.sign(theta), f"re-encode flipped pitch: {np.rad2deg(p_ctrl):+.2f}"
    assert abs(p_ctrl - theta) < 1e-9


# ===========================================================================
# SEAM 1->3 — end-to-end: correct estimate is RESTORING; inverted is the A8 signature
# ===========================================================================
def _nav_for_true_pitch(true_pitch_deg):
    """A NavState carrying the controller-convention attitude the AHRS delivers for a TRUE
    nose-up/down pitch (q_true*CONJ decoded), at hover with zero rate."""
    theta = np.deg2rad(true_pitch_deg)
    q_odo = _quat_true_wxyz(0.0, theta, 0.0) * ODO_QUAT_TRUE_CONJ_WXYZ
    r, p, y = euler_from_quat_wxyz(q_odo)
    return NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                    roll=r, pitch=p, yaw=y, angular_rate_body=np.zeros(3))


def test_correct_nose_up_estimate_commands_restoring_pitch_rate():
    """The load-bearing assert: with a CORRECT nose-up attitude estimate and a level-hover target,
    the seeker controller must command a RESTORING pitch-rate (nose-DOWN). With body_rate_sign[1]=+1,
    nose-DOWN/forward is a NEGATIVE pitch-rate command. A positive (nose-UP) command here would be
    the A8 climb-and-retreat bug."""
    ctrl = make_seeker_controller()
    assert ctrl.body_rate_sign[1] == 1.0, "pitch-axis command sign assumption changed — re-derive"

    nav_up = _nav_for_true_pitch(+15.0)                       # truly nose-up
    sp = Setpoint(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3), yaw=nav_up.yaw)
    cmd = ctrl.command(nav_up, sp)
    assert cmd.body_rate[1] < -0.01, (
        f"nose-up estimate -> pitch-rate {cmd.body_rate[1]:+.4f} is NOT restoring (A8: should be < 0)")


def test_inverted_estimate_reproduces_a8_signature():
    """CONTROL: an INVERTED estimate (reads nose-DOWN while truly level) makes the controller
    command nose-UP (positive pitch-rate) — the A8 climb signature. This proves the test above is
    sensitive to the exact failure mode, i.e. it would FAIL if the real estimate were inverted."""
    ctrl = make_seeker_controller()
    nav_bad = _nav_for_true_pitch(-25.0)                     # estimate says nose-DOWN
    sp = Setpoint(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3), yaw=nav_bad.yaw)
    cmd = ctrl.command(nav_bad, sp)
    assert cmd.body_rate[1] > 0.01, "inverted-estimate control case did not produce the nose-up signature"
