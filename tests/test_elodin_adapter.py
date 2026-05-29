import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.atmosphere import altitude_to_pressure_hpa
from racer.contracts import ControlCommand, ControlMode
from racer.elodin_adapter import (
    R_FRD_FROM_FLU,
    R_NED_FROM_ENU,
    RCParams,
    attitude_command_to_rc,
    controlcommand_to_rc,
    frame_from_rgba,
    sensorupdate_to_state,
)

_IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


def _su(quat_xyzw=_IDENTITY_QUAT_XYZW, pos_enu=(0, 0, 0), vel_enu=(0, 0, 0),
        gyro=(0, 0, 0), accel=(0, 0, 9.80665), **kw):
    world_pos = np.concatenate([quat_xyzw, np.asarray(pos_enu, float)])
    world_vel = np.concatenate([np.zeros(3), np.asarray(vel_enu, float)])
    return sensorupdate_to_state(t=1.0, world_pos=world_pos, world_vel=world_vel,
                                 gyro=np.asarray(gyro, float), accel=np.asarray(accel, float), **kw)


def test_frame_matrices_are_proper_rotations():
    for R in (R_NED_FROM_ENU, R_FRD_FROM_FLU):
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_identity_quat_faces_east_yaw90():
    # Elodin identity quat = body FLU aligned with world ENU -> forward points East.
    # In NED, East is yaw = +90 deg.
    state, _ = _su()
    assert state.yaw == pytest.approx(np.pi / 2)
    assert state.roll == pytest.approx(0.0, abs=1e-9)
    assert state.pitch == pytest.approx(0.0, abs=1e-9)


def test_enu_position_velocity_to_ned():
    state, gt = _su(pos_enu=(1, 2, 3), vel_enu=(4, 5, 6), expose_ground_truth=True)
    # [E,N,U] -> [N,E,-U]
    np.testing.assert_allclose(gt.position_ned, [2, 1, -3])
    np.testing.assert_allclose(gt.velocity_ned, [5, 4, -6])
    np.testing.assert_allclose(state.position_ned, [2, 1, -3])  # exposed here


def test_ground_truth_hidden_by_default():
    state, gt = _su(pos_enu=(1, 2, 3))
    assert state.position_ned is None  # official-sim discipline
    assert state.velocity_ned is None
    np.testing.assert_allclose(gt.position_ned, [2, 1, -3])  # still available for eval


def test_accel_gravity_sign_flu_to_frd():
    # Rest accel in FLU is +Z (up) 1g; in FRD (Z down) it must read -Z.
    state, _ = _su(accel=(0, 0, 9.80665))
    np.testing.assert_allclose(state.accel_body, [0, 0, -9.80665])


def test_gyro_and_mag_flu_to_frd():
    state, _ = _su(gyro=(0.1, 0.2, 0.3), mag=(1.0, 0.0, 0.0))
    np.testing.assert_allclose(state.angular_rate_body, [0.1, -0.2, -0.3])
    np.testing.assert_allclose(state.mag_body, [1.0, 0.0, 0.0])  # x unchanged by FLU->FRD


def test_mag_none_when_absent():
    state, _ = _su()
    assert state.mag_body is None


def test_baro_altitude_becomes_pressure():
    state, _ = _su(baro=30.0)
    assert state.baro_pressure_hpa == pytest.approx(altitude_to_pressure_hpa(30.0))


def test_attitude_recovers_a_known_rotation():
    # Build a known FRD->NED attitude, express it as an Elodin (FLU->ENU) quat, and
    # confirm the adapter recovers the original roll/pitch/yaw.
    yaw, pitch, roll = 0.3, -0.2, 0.1
    r_ned_frd = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    r_enu_flu = R_NED_FROM_ENU.T @ r_ned_frd @ R_FRD_FROM_FLU.T
    quat_xyzw = Rotation.from_matrix(r_enu_flu).as_quat()
    state, _ = _su(quat_xyzw=quat_xyzw)
    assert (state.roll, state.pitch, state.yaw) == pytest.approx((roll, pitch, yaw))


def test_frame_rgba_to_bgr():
    rgba = np.zeros((360, 640, 4), dtype=np.uint8)
    rgba[..., 0] = 10  # R
    rgba[..., 1] = 20  # G
    rgba[..., 2] = 30  # B
    frame = frame_from_rgba(rgba, frame_id=7, sim_time_ns=123)
    assert frame.image_bgr.shape == (360, 640, 3)
    assert frame.frame_id == 7 and frame.sim_time_ns == 123
    np.testing.assert_array_equal(frame.image_bgr[0, 0], [30, 20, 10])  # B, G, R


def test_attitude_command_to_rc_center_and_saturation():
    p = RCParams()
    neutral = attitude_command_to_rc(0.0, 0.0, 0.0, 0.5, p)
    assert (neutral.roll, neutral.pitch, neutral.yaw) == (1500, 1500, 1500)
    assert neutral.throttle == 1500 and neutral.arm == p.arm_pwm
    # Beyond max angle saturates the stick.
    hard = attitude_command_to_rc(10 * p.max_angle_rad, 0.0, 0.0, 1.0, p)
    assert hard.roll == 2000 and hard.throttle == 2000
    assert attitude_command_to_rc(0, 0, 0, 0.0, p).throttle == 1000
    assert attitude_command_to_rc(0, 0, 0, 0.5, p, armed=False).arm == p.disarm_pwm


def test_position_control_not_supported_on_rig():
    cmd = ControlCommand(mode=ControlMode.POSITION, position_ned=np.zeros(3))
    with pytest.raises(NotImplementedError):
        controlcommand_to_rc(cmd)
