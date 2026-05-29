import numpy as np
import pytest

from racer.contracts import DroneState
from racer.frames import R_world_from_body
from racer.state_estimator import GRAVITY_NED, LinearKF, make_nav_state


def _trajectory(t: float):
    """Analytic NED path with exact derivatives (a smooth horizontal arc + descent)."""
    p = np.array([2.0 * np.sin(t), 1.5 * (1.0 - np.cos(t)), -1.0 - 0.3 * t])
    v = np.array([2.0 * np.cos(t), 1.5 * np.sin(t), -0.3])
    a = np.array([-2.0 * np.sin(t), 1.5 * np.cos(t), 0.0])
    return p, v, a


def _run(seed=0, blind=None, accel_noise=0.1, vis_noise=0.05, R_wb=None, T=4.0, dt=0.01):
    """Simulate IMU-rate predict + sparse vision updates; return per-step position error."""
    if R_wb is None:
        R_wb = np.eye(3)
    rng = np.random.default_rng(seed)
    p0, v0, _ = _trajectory(0.0)
    kf = LinearKF.initialize(p0, v0, pos_std=0.05, vel_std=0.05, accel_noise_std=accel_noise)
    errs = []
    for k in range(1, int(T / dt) + 1):
        t = k * dt
        p, v, a = _trajectory(t)
        # Draw BOTH noise streams every step so runs with/without vision stay
        # comparable (only the update decision differs, not the RNG sequence).
        accel_n = rng.normal(0, accel_noise, 3)
        vis_n = rng.normal(0, vis_noise, 3)
        kf.predict(R_wb.T @ (a - GRAVITY_NED) + accel_n, R_wb, dt)   # body specific force
        do_vis = (k % 4 == 0)
        if blind is not None and blind[0] <= t <= blind[1]:
            do_vis = False
        if do_vis:
            kf.update_position(p + vis_n, (vis_noise**2) * np.eye(3))
        errs.append(float(np.linalg.norm(kf.position - p)))
    return np.array(errs)


def test_predict_at_rest_no_drift():
    # The gravity sign trap: rest specific force [0,0,-g] must yield zero world accel.
    kf = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.0, vel_std=0.0)
    rest = np.array([0.0, 0.0, -9.80665])
    for _ in range(500):
        kf.predict(rest, np.eye(3), 0.01)
    np.testing.assert_allclose(kf.position, np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(kf.velocity, np.zeros(3), atol=1e-9)


def test_predict_constant_accel_is_exact():
    kf = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.0, vel_std=0.0)
    a_world = np.array([1.0, 0.0, 0.0])
    f_body = a_world - GRAVITY_NED  # R_wb = I
    for _ in range(200):           # 2.0 s at dt=0.01
        kf.predict(f_body, np.eye(3), 0.01)
    np.testing.assert_allclose(kf.position, [2.0, 0.0, 0.0], atol=1e-9)   # 0.5*a*t^2
    np.testing.assert_allclose(kf.velocity, [2.0, 0.0, 0.0], atol=1e-9)   # a*t


def test_position_update_pulls_toward_measurement_and_shrinks_cov():
    kf = LinearKF.initialize(np.array([5.0, 0.0, 0.0]), np.zeros(3), pos_std=2.0, vel_std=1.0)
    before = np.trace(kf.P)
    kf.update_position(np.zeros(3), 0.01 * np.eye(3))
    assert np.linalg.norm(kf.position) < 5.0          # moved toward the measurement
    assert np.trace(kf.P) < before                    # more certain


def test_velocity_and_baro_updates():
    kf = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=1.0, vel_std=2.0)
    kf.update_velocity(np.array([3.0, 0.0, 0.0]), 0.01 * np.eye(3))
    assert kf.velocity[0] > 2.0
    kf.update_position_z(-7.0, 0.01)                   # baro: pz toward -7
    assert kf.position[2] < -3.0


def test_tracks_trajectory_with_vision():
    errs = _run(seed=1)
    assert np.sqrt(np.mean(errs**2)) < 0.1            # RMS position error
    assert errs[-1] < 0.1


def test_blind_segment_drift_bounded_and_recovers():
    errs = _run(seed=2, blind=(1.5, 2.5))
    dt = 0.01
    window = errs[int(1.5 / dt) - 1:int(2.5 / dt)]
    assert window.max() < 0.3                          # dead-reckoning stays bounded over 1 s blind
    assert errs[-1] < 0.12                             # recovers once vision returns


def test_vision_beats_pure_dead_reckoning():
    # At a realistic IMU noise level, dead-reckoning the whole 4 s drifts ~0.3-0.5 m
    # while vision holds it to a few cm. (A clean IMU drifts little over ~1 s, which is
    # why short blind segments are tolerable -- see the blind-segment test.)
    rms = lambda e: float(np.sqrt(np.mean(e**2)))
    full = _run(seed=7, accel_noise=0.5)                       # vision throughout
    none = _run(seed=7, accel_noise=0.5, blind=(0.0, 1e9))    # never any vision (same noise)
    assert rms(full) < 0.1
    assert none[-1] > 0.2                                      # dead-reckoning drifts away
    assert rms(none) > 2.5 * rms(full)                        # vision clearly bounds the drift


def test_tracks_under_constant_yaw():
    # Exercises the R_world_body path: drone yawed 60 deg the whole run.
    R_wb = R_world_from_body(0.0, 0.0, np.deg2rad(60.0))
    errs = _run(seed=3, R_wb=R_wb)
    assert np.sqrt(np.mean(errs**2)) < 0.1


def test_make_nav_state_carries_attitude_and_cov():
    kf = LinearKF.initialize(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
    ds = DroneState(sim_time_ns=12345, roll=0.1, pitch=-0.2, yaw=0.3,
                    angular_rate_body=np.array([0.01, 0.02, 0.03]))
    nav = make_nav_state(kf, ds, time_since_vision_update_s=0.05)
    np.testing.assert_allclose(nav.position_ned, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(nav.velocity_ned, [0.1, 0.2, 0.3])
    assert (nav.roll, nav.pitch, nav.yaw) == (0.1, -0.2, 0.3)
    assert nav.sim_time_ns == 12345
    assert nav.pos_vel_covariance.shape == (6, 6)
    assert nav.time_since_vision_update_s == 0.05
