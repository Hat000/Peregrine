import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.contracts import ControlCommand, ControlMode, NavState, Setpoint
from racer.controller import _G, Controller, level_hold_body_rate
from racer.frames import R_world_from_body
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


def test_clamped_tilt_preserves_vertical_thrust_no_skyward_launch():
    # [red-team 2026-05-30] A horizontal accel beyond the tilt budget must NOT keep the full,
    # un-clamped thrust magnitude at the clamped 45/30-deg angle -- the surplus vertical
    # component would rocket the drone up on every hard turn/brake. Altitude priority: the
    # realized thrust's vertical projection still equals the hover requirement (g), and the
    # command is no longer saturated to 1.0.
    c = Controller(mode=ControlMode.ATTITUDE, hover_thrust=0.5, max_tilt_rad=np.deg2rad(30.0))
    cmd = c.command(NavState(sim_time_ns=0), Setpoint(accel_ned=np.array([100.0, 0.0, 0.0]), yaw=0.0))
    cos_tilt = float(_body_up_world(cmd) @ np.array([0.0, 0.0, -1.0]))
    # thrust = clip(hover * f_mag / g); reconstruct the vertical specific force it produces.
    vertical_sf = (cmd.thrust * _G / 0.5) * cos_tilt
    assert vertical_sf == pytest.approx(_G, rel=1e-6)   # holds altitude, doesn't climb
    assert cmd.thrust < 1.0                              # old code clipped to 1.0 (over-thrust)


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


# -- CTBR / BODY_RATE (the ACRO control path, 2026-06-02) ------------------
def _R_wb_from_nav(nav):
    from racer.frames import R_world_from_body

    return R_world_from_body(nav.roll, nav.pitch, nav.yaw)


def test_body_rate_hover_is_zero_rate():
    # Already level + at the desired heading with a zero-accel setpoint -> no rotation needed.
    c = Controller(mode=ControlMode.BODY_RATE, hover_thrust=0.489)
    cmd = c.command(NavState(sim_time_ns=0, roll=0.0, pitch=0.0, yaw=0.0),
                    Setpoint(accel_ned=np.zeros(3), yaw=0.0))
    assert cmd.mode is ControlMode.BODY_RATE
    np.testing.assert_allclose(cmd.body_rate, np.zeros(3), atol=1e-9)
    assert cmd.thrust == pytest.approx(0.489)


def test_body_rate_reduces_attitude_error():
    # A tilted/yawed drone commanded to level: integrating the commanded body-rate forward must
    # SHRINK the geodesic attitude error (the definitive sign/frame check for the inner loop).
    c = Controller(mode=ControlMode.BODY_RATE, kp_att=4.0)
    nav = NavState(sim_time_ns=0, roll=0.1, pitch=0.3, yaw=0.2)
    cmd = c.command(nav, Setpoint(accel_ned=np.zeros(3), yaw=0.0))   # desired = level, yaw 0 (R_des=I)
    R_cur = _R_wb_from_nav(nav)
    R_next = R_cur @ Rotation.from_rotvec(cmd.body_rate * 0.02).as_matrix()  # body-rate integration
    before = Rotation.from_matrix(R_cur).magnitude()
    after = Rotation.from_matrix(R_next).magnitude()
    assert after < before                                            # error shrank


def test_body_rate_thrust_matches_attitude_path():
    # CTBR reuses the same desired attitude + collective thrust as the ATTITUDE law.
    sp = Setpoint(accel_ned=np.array([2.0, -1.0, -1.0]), yaw=0.3)
    nav = NavState(sim_time_ns=0, roll=0.05, pitch=-0.1, yaw=0.3)
    br = Controller(mode=ControlMode.BODY_RATE, hover_thrust=0.489, thrust_slope_mps2=25.9)
    at = Controller(mode=ControlMode.ATTITUDE, hover_thrust=0.489, thrust_slope_mps2=25.9)
    assert br.command(nav, sp).thrust == pytest.approx(at.command(nav, sp).thrust)


def test_body_rate_is_clamped():
    # A large heading error must not demand an unbounded rate.
    c = Controller(mode=ControlMode.BODY_RATE, kp_att=4.0, max_body_rate_rps=2.0)
    cmd = c.command(NavState(sim_time_ns=0, yaw=0.0), Setpoint(accel_ned=np.zeros(3), yaw=3.0))
    assert np.linalg.norm(cmd.body_rate) == pytest.approx(2.0, rel=1e-6)


def test_body_rate_sign_maps_to_sim_convention():
    # This sim inverts roll+yaw body-rate commands (first contact). The body_rate_sign
    # calibration flips them on output; identity (default) leaves the pure law unchanged.
    nav = NavState(sim_time_ns=0, roll=0.1, pitch=0.3, yaw=0.2)
    sp = Setpoint(accel_ned=np.array([1.0, -0.5, 0.0]), yaw=0.0)
    base = Controller(mode=ControlMode.BODY_RATE, kp_att=4.0).command(nav, sp).body_rate
    flipped = Controller(mode=ControlMode.BODY_RATE, kp_att=4.0,
                         body_rate_sign=np.array([-1.0, 1.0, -1.0])).command(nav, sp).body_rate
    np.testing.assert_allclose(flipped, base * np.array([-1.0, 1.0, -1.0]))


# -- singularity guards [red-team 2026-05-30] ------------------------------
def test_straight_down_accel_recovers_upright_not_inverted():
    # A desired accel pointing straight down past gravity makes the required thrust point
    # straight DOWN -- the antipode of WORLD_UP, where the tilt-clamp axis is undefined. The
    # old `if n > 1e-9` guard skipped the clamp and let an inverted command through. The fix
    # clamps via an arbitrary horizontal axis and recovers toward upright.
    c = Controller(mode=ControlMode.ATTITUDE)   # default max_tilt 45 deg
    cmd = c.command(NavState(sim_time_ns=0), Setpoint(accel_ned=np.array([0.0, 0.0, 20.0]), yaw=0.0))
    assert np.all(np.isfinite(cmd.attitude_quat_wxyz))      # no NaNs
    assert _body_up_world(cmd)[2] < 0.0                     # thrust points UP, not into the ground


def test_horizontal_thrust_aligned_with_heading_no_nan():
    # b3 (body-down) parallel to the heading vector zeros the cross product b3 x x_c. Lift the
    # tilt clamp (max_tilt > 90 deg) so we actually reach that singularity, and check the b2
    # fallback keeps the attitude quaternion finite instead of emitting NaNs.
    c = Controller(mode=ControlMode.ATTITUDE, max_tilt_rad=np.deg2rad(120.0))
    cmd = c.command(NavState(sim_time_ns=0), Setpoint(accel_ned=np.array([10.0, 0.0, _G]), yaw=0.0))
    q = cmd.attitude_quat_wxyz
    assert np.all(np.isfinite(q))                            # no NaNs from the zero cross product
    assert np.linalg.norm(q) == pytest.approx(1.0)          # a well-formed unit quaternion
    assert np.all(np.isfinite(_body_up_world(cmd)))         # ...and a usable thrust direction


# -- measured throttle map + safety bounds (innerloop_step 2026-06-02) ------
def test_measured_thrust_slope_affine_map():
    # innerloop_step: hover_thrust=0.489, slope=25.9 (m/s^2)/thrust. A 2 m/s^2 climb demand
    # => thrust = hover + a_up/slope. The placeholder |f|/g would over-thrust.
    c = Controller(mode=ControlMode.ATTITUDE, hover_thrust=0.489, thrust_slope_mps2=25.9)
    hover = c.command(NavState(sim_time_ns=0), Setpoint(accel_ned=np.zeros(3)))
    assert hover.thrust == pytest.approx(0.489, abs=1e-6)            # hover unchanged
    climb = c.command(NavState(sim_time_ns=0), Setpoint(accel_ned=np.array([0.0, 0.0, -2.0])))
    assert climb.thrust == pytest.approx(0.489 + 2.0 / 25.9, abs=1e-3)
    placeholder = Controller(mode=ControlMode.ATTITUDE, hover_thrust=0.489).command(
        NavState(sim_time_ns=0), Setpoint(accel_ned=np.array([0.0, 0.0, -2.0])))
    assert placeholder.thrust > climb.thrust                        # placeholder over-thrusts


def test_max_accel_caps_tilt():
    # A huge position error must not saturate to the tilt clamp: with max_accel 3 m/s^2 the
    # realized lean corresponds to a 3 m/s^2 horizontal demand (atan(3/g)~17 deg), NOT 45 deg.
    c = Controller(mode=ControlMode.ATTITUDE, kp_pos=1.5, max_accel_mps2=3.0)
    nav = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3))
    up = _body_up_world(c.command(nav, Setpoint(position_ned=np.array([1000.0, 0.0, 0.0]), yaw=0.0)))
    assert up[0] / (-up[2]) == pytest.approx(3.0 / _G, rel=1e-3)     # capped to 3, not clamped to 45 deg


def test_max_pos_error_bounds_pursuit():
    # The position-error clamp makes a far carrot and a near-but-still-far carrot demand the SAME
    # (bounded) acceleration -> a 24 m gate carrot can't drive a violent tilt.
    c = Controller(mode=ControlMode.ATTITUDE, kp_pos=1.0, max_pos_error_m=2.0)
    nav = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3))
    far = _body_up_world(c.command(nav, Setpoint(position_ned=np.array([100.0, 0.0, 0.0]), yaw=0.0)))
    near = _body_up_world(c.command(nav, Setpoint(position_ned=np.array([10.0, 0.0, 0.0]), yaw=0.0)))
    np.testing.assert_allclose(far, near, atol=1e-9)                 # both clamped to 2 m error
    assert far[0] / (-far[2]) == pytest.approx(2.0 / _G, rel=1e-3)   # demand = kp*2 = 2 m/s^2


# -- level_hold_body_rate: the re-level primitive used by the rate-step sysid probe --------
_SIM_SIGN = np.array([-1.0, 1.0, -1.0])   # measured sim actuation convention (roll+yaw inverted)


def test_level_hold_zero_error_commands_zero_rate():
    yaw = -3.10
    sent = level_hold_body_rate(0.0, 0.0, yaw, yaw, np.zeros(3),
                                kp=2.0, kd=0.4, body_rate_sign=_SIM_SIGN, max_rate=5.0)
    np.testing.assert_allclose(sent, np.zeros(3), atol=1e-9)


def test_level_hold_is_negative_feedback_toward_level():
    # THE safety-critical property: the TRUE body rate the sim would produce (= body_rate_sign *
    # sent, under the inversion + unit gain) must point along the attitude error toward level,
    # i.e. it reduces the tilt. A wrong sign here makes the hold DIVERGE (the early tumbles).
    yaw = -3.10
    roll, pitch = 0.12, -0.31      # rolled right + nose-down (the −17.8° resting pitch)
    kp = 2.0
    sent = level_hold_body_rate(roll, pitch, yaw, yaw, np.zeros(3),
                                kp=kp, kd=0.0, body_rate_sign=_SIM_SIGN, max_rate=10.0)
    actual_true = _SIM_SIGN * sent     # what the sim actually rotates at (sign inversion, unit gain)
    R_err = R_world_from_body(roll, pitch, yaw).T @ R_world_from_body(0.0, 0.0, yaw)
    err = Rotation.from_matrix(R_err).as_rotvec()
    np.testing.assert_allclose(actual_true, kp * err, atol=1e-9)   # rate ∝ error-to-level
    assert actual_true[1] > 0.0        # nose-down -> true pitch-up rate (re-levels)


def test_level_hold_damping_opposes_measured_rate():
    yaw = 0.0
    meas = np.array([0.0, 1.5, 0.0])   # spinning in pitch
    undamped = level_hold_body_rate(0.0, -0.2, yaw, yaw, np.zeros(3),
                                    kp=2.0, kd=0.0, body_rate_sign=_SIM_SIGN, max_rate=10.0)
    damped = level_hold_body_rate(0.0, -0.2, yaw, yaw, meas,
                                  kp=2.0, kd=0.5, body_rate_sign=_SIM_SIGN, max_rate=10.0)
    # damping subtracts kd*meas in the trusted frame (before the sign map). Pitch sign is +1,
    # so a positive measured pitch rate lowers the commanded pitch magnitude.
    assert abs(damped[1]) < abs(undamped[1])


def test_level_hold_clamps_to_max_rate():
    sent = level_hold_body_rate(0.0, 1.2, 0.0, 0.0, np.zeros(3),     # huge tilt -> would saturate
                                kp=10.0, kd=0.0, body_rate_sign=_SIM_SIGN, max_rate=2.0)
    assert np.linalg.norm(sent) == pytest.approx(2.0, abs=1e-9)
