"""TOPP-RA spike: time-optimal speed profiling behind the Setpoint seam."""
import numpy as np

from racer.contracts import Gate, Setpoint
from racer.speed_profile import Trajectory, time_optimal_profile, waypoints_from_gates


def _speed(traj: Trajectory) -> np.ndarray:
    return np.linalg.norm(traj.vel, axis=1)


def test_straight_line_respects_limits_and_endpoints():
    # 20 m straight; dist to reach v_max=8 under a_max=4 is 8 m < 10 m, so it hits the cap.
    traj = time_optimal_profile(
        np.array([[0, 0, 0], [20, 0, 0]], float), v_max=8.0, a_max=4.0, v_start=0.0, v_end=0.0
    )
    sp = _speed(traj)
    assert sp.max() <= 8.0 + 1e-6           # never exceeds v_max
    assert sp.max() > 7.9                    # actually reaches it
    assert sp[0] < 1e-6 and sp[-1] < 0.2     # honours v_start / v_end
    assert np.all(np.diff(traj.t) > 0)       # time strictly increasing
    assert np.linalg.norm(traj.accel, axis=1).max() <= 4.0 + 0.4   # a_max (finite-diff slack)


def test_vmax_cap_on_long_straight():
    traj = time_optimal_profile(np.array([[0, 0, 0], [50, 0, 0]], float), v_max=5.0, a_max=10.0)
    sp = _speed(traj)
    assert sp.max() <= 5.0 + 1e-6
    assert sp.max() > 4.9                     # cruises at the cap


def test_corner_forces_slowdown():
    # Right-angle through three points: the profile must brake for the corner, then re-accelerate.
    traj = time_optimal_profile(
        np.array([[0, 0, 0], [12, 0, 0], [12, 12, 0]], float), v_max=12.0, a_max=4.0
    )
    sp = _speed(traj)
    n = len(sp)
    assert sp[n // 3 : 2 * n // 3].min() < 0.8 * sp.max()    # mid-path dip
    # Lateral accel stayed within the envelope (corner cap kept kappa*v^2 <= a_max).
    assert np.linalg.norm(traj.accel, axis=1).max() <= 4.0 + 0.6


def test_exploits_straights_vs_corner_speed():
    # The point of time-optimality: peak speed on straights is well above the corner speed,
    # so the run is faster than crawling the whole path at the corner-limited speed.
    traj = time_optimal_profile(
        np.array([[0, 0, 0], [15, 0, 0], [15, 15, 0], [30, 15, 0]], float),
        v_max=12.0, a_max=5.0, v_start=2.0, v_end=2.0,
    )
    sp = _speed(traj)
    assert sp.max() > 1.5 * sp.min()
    t_crawl = traj.length_m / max(sp.min(), 0.5)     # naive: whole path at the slowest speed
    assert traj.duration < t_crawl


def test_setpoint_at_seam():
    wp = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], float)
    traj = time_optimal_profile(wp, v_max=8.0, a_max=4.0)

    sp0 = traj.setpoint_at(0.0)
    assert isinstance(sp0, Setpoint)
    np.testing.assert_allclose(sp0.position_ned, wp[0], atol=0.3)
    np.testing.assert_allclose(traj.setpoint_at(traj.duration).position_ned, wp[-1], atol=0.3)

    mid = traj.setpoint_at(0.5 * traj.duration)
    assert mid.velocity_ned is not None and mid.accel_ned is not None and mid.yaw is not None
    # Out-of-range queries clamp rather than raise.
    assert traj.setpoint_at(-5.0).position_ned is not None
    assert traj.setpoint_at(traj.duration + 5.0).position_ned is not None


def test_yaw_follows_travel_direction():
    # Heading north then turning east -> yaw goes from ~0 to ~pi/2.
    traj = time_optimal_profile(
        np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], float), v_max=6.0, a_max=4.0
    )
    assert abs(traj.yaw[0]) < 0.2
    assert abs(traj.yaw[-1] - np.pi / 2) < 0.2


def test_from_gate_map():
    # Plugs into the Gate map contract: centres -> waypoints -> profile.
    gates = [
        Gate(gate_id=0, position_ned=np.array([0.0, 0.0, -2.0]), R_world_gate=np.eye(3)),
        Gate(gate_id=1, position_ned=np.array([10.0, 0.0, -2.0]), R_world_gate=np.eye(3)),
        Gate(gate_id=2, position_ned=np.array([20.0, 6.0, -2.0]), R_world_gate=np.eye(3)),
    ]
    wp = waypoints_from_gates(gates)
    assert wp.shape == (3, 3)
    traj = time_optimal_profile(wp, v_max=9.0, a_max=4.0)
    assert traj.duration > 0 and np.all(np.diff(traj.t) > 0)
    assert _speed(traj).max() <= 9.0 + 1e-6


def test_duplicate_waypoints_are_tolerated():
    traj = time_optimal_profile(
        np.array([[0, 0, 0], [10, 0, 0], [10, 0, 0], [20, 0, 0]], float), v_max=6.0, a_max=4.0
    )
    assert traj.duration > 0
