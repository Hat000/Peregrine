import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.contracts import ControlCommand, ControlMode, NavState, Setpoint
from racer.controller import _G, Controller
from racer.mavlink_client import _POS_IGNORE_PX, _POS_IGNORE_VX, _pos_type_mask


def _R_wb(cmd: ControlCommand) -> np.ndarray:
    q = cmd.attitude_quat_wxyz                     # (w,x,y,z)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _body_up_world(cmd: ControlCommand) -> np.ndarray:
    """Direction the rotor thrust points, in world NED (body -z expressed in world)."""
    return _R_wb(cmd) @ np.array([0.0, 0.0, -1.0])


# -- floor: position / velocity passthrough --------------------------------
def test_position_mode_passthrough():
    c = Controller(mode=ControlMode.POSITION)
    sp = Setpoint(sim_time_ns=42, position_ned=np.array([1.0, 2.0, -3.0]), yaw=0.5)
    cmd = c.command(NavState(sim_time_ns=0), sp)
    assert cmd.mode is ControlMode.POSITION
    assert cmd.sim_time_ns == 42
    np.testing.assert_array_equal(cmd.position_ned, [1.0, 2.0, -3.0])
    assert cmd.yaw == 0.5 and cmd.velocity_ned is None
    # mavlink_client would use position + yaw, ignore velocity.
    m = _pos_type_mask(cmd.position_ned, cmd.velocity_ned, cmd.accel_ned, cmd.yaw, cmd.yaw_rate)
    assert not (m & _POS_IGNORE_PX) and (m & _POS_IGNORE_VX)


def test_velocity_mode_passthrough():
    c = Controller(mode=ControlMode.VELOCITY)
    sp = Setpoint(velocity_ned=np.array([4.0, 0.0, 0.0]), yaw_rate=0.2)
    cmd = c.command(NavState(sim_time_ns=0), sp)
    assert cmd.mode is ControlMode.VELOCITY
    np.testing.assert_array_equal(cmd.velocity_ned, [4.0, 0.0, 0.0])
    m = _pos_type_mask(cmd.position_ned, cmd.velocity_ned, cmd.accel_ned, cmd.yaw, cmd.yaw_rate)
    assert (m & _POS_IGNORE_PX) and not (m & _POS_IGNORE_VX)


# -- upgrade: geometric attitude control -----------------------------------
def test_attitude_hover_is_level_at_hover_thrust():
    c = Controller(mode=ControlMode.ATTITUDE, hover_thrust=0.5)
    cmd = c.command(NavState(sim_time_ns=0), Setpoint(accel_ned=np.zeros(3), yaw=0.0))
    assert cmd.mode is ControlMode.ATTITUDE
    assert Rotation.from_quat([*cmd.attitude_quat_wxyz[1:], cmd.attitude_quat_wxyz[0]]).magnitude() < 1e-6
    np.testing.assert_allclose(_body_up_world(cmd), [0.0, 0.0, -1.0], atol=1e-9)  # thrust straight up
    assert cmd.thrust == pytest.approx(0.5)


@pytest.mark.parametrize("a_des", [
    [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [-1.5, 1.0, 0.0], [0.0, 0.0, -2.0], [1.0, -1.0, -1.0],
])
def test_attitude_reproduces_desired_thrust_direction(a_des):
    # The key invariant (independent of the uncalibrated throttle scale): the realized
    # attitude points the thrust along (a_des - g). Magnitudes here stay under the tilt clamp.
    c = Controller(mode=ControlMode.ATTITUDE)
    cmd = c.command(NavState(sim_time_ns=0), Setpoint(accel_ned=np.array(a_des), yaw=0.3))
    expected = np.array(a_des) - np.array([0.0, 0.0, _G])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(_body_up_world(cmd), expected, atol=1e-9)


def test_attitude_honors_requested_yaw():
    c = Controller(mode=ControlMode.ATTITUDE)
    cmd = c.command(NavState(sim_time_ns=0), Setpoint(accel_ned=np.zeros(3), yaw=0.7))
    yaw = Rotation.from_quat([*cmd.attitude_quat_wxyz[1:], cmd.attitude_quat_wxyz[0]]).as_euler("ZYX")[0]
    assert yaw == pytest.approx(0.7)


def test_forward_accel_tilts_thrust_north():
    # Accelerating north (+x) must tilt the thrust so its horizontal component points north.
    c = Controller(mode=ControlMode.ATTITUDE)
    cmd = c.command(NavState(sim_time_ns=0), Setpoint(accel_ned=np.array([3.0, 0.0, 0.0]), yaw=0.0))
    up = _body_up_world(cmd)
    assert up[0] > 0.0                                   # leans north
    assert up[1] == pytest.approx(0.0, abs=1e-9)         # no east/west lean
    assert up[2] < 0.0                                   # still mostly up


def test_tilt_is_clamped():
    c = Controller(mode=ControlMode.ATTITUDE, max_tilt_rad=np.deg2rad(30.0))
    cmd = c.command(NavState(sim_time_ns=0), Setpoint(accel_ned=np.array([100.0, 0.0, 0.0]), yaw=0.0))
    up = _body_up_world(cmd)
    tilt = np.arccos(np.clip(up @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0))
    assert tilt == pytest.approx(np.deg2rad(30.0), abs=1e-6)   # clamped to the limit
    assert up[0] > 0.0                                         # still leaning the right way


def test_position_error_drives_attitude():
    # Attitude mode with a position setpoint north of the drone -> thrust tilts north.
    c = Controller(mode=ControlMode.ATTITUDE)
    nav = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3))
    cmd = c.command(nav, Setpoint(position_ned=np.array([10.0, 0.0, 0.0]), yaw=0.0))
    assert _body_up_world(cmd)[0] > 0.0


def test_freefall_setpoint_gives_zero_thrust():
    c = Controller(mode=ControlMode.ATTITUDE)
    cmd = c.command(NavState(sim_time_ns=0), Setpoint(accel_ned=np.array([0.0, 0.0, _G])))  # a_des == g
    assert cmd.thrust == 0.0
    assert Rotation.from_quat([*cmd.attitude_quat_wxyz[1:], cmd.attitude_quat_wxyz[0]]).magnitude() < 1e-6


def test_body_rate_mode_not_implemented_yet():
    c = Controller(mode=ControlMode.BODY_RATE)
    with pytest.raises(NotImplementedError):
        c.command(NavState(sim_time_ns=0), Setpoint(accel_ned=np.zeros(3)))
