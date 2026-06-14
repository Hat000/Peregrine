"""Estimator -> obs wiring faithfulness — G1 (C2 step 4, BLUEPRINT §1.2/item 4).

Pins the seam that sources the 17-dim policy obs pos_g/vel_g from the ESTIMATOR (the gate-relative +L
estimate) instead of the given wire pose:
  - the wiring is a PURE pos/vel substitution -> when the estimator pos/vel equal the wire's, the obs is
    BYTE-IDENTICAL to the shipped build_obs (no layout corruption; the inc7 17-dim contract is intact);
  - feeding the gate-relative estimate p = gate_map - L_seen delivers pos_g == R_w2g @ (+L_seen) to
    <=1e-5 (the +L convention), and the -L sign (p = gate_map + L_seen) BREAKS by ~24 m (the negative
    control) -- the same contract as tests/test_obs_sign_faithfulness, now through the estimator wiring.

Imports fly_rl (the shipped obs builder), which pulls torch -- skipped if torch is absent.
[C2-ESTIMATOR-CHAIN 2026-06-13]
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("torch")   # fly_rl imports torch at module top

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

from fly_rl import (  # noqa: E402
    _FLIP,
    _GATE_POS_ZUP,
    _GATE_YAW_ZUP,
    _gate_rotmat_w2g,
    build_obs,
    make_gate_map,
)
from racer.contracts import DroneState, Gate, GatePose, NavState  # noqa: E402
from racer.estimator_obs import estimator_obs, estimator_state_for_obs  # noqa: E402
from racer.frames import R_camera_from_body, R_world_from_body  # noqa: E402
from racer.localization import gate_pose_to_world_position  # noqa: E402

_PLUS_L_TOL = 1e-5
_FLIP_BREAK_M = 10.0
_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _ds(pos, vel, q=_LEVEL_Q):
    return DroneState(sim_time_ns=0, orientation_ned_wxyz=np.asarray(q, float),
                      angular_rate_body=np.zeros(3), accel_body=np.array([0.0, 0.0, -9.80665]),
                      position_ned=np.asarray(pos, float), velocity_ned=np.asarray(vel, float))


def _nav(pos, vel):
    return NavState(sim_time_ns=0, position_ned=np.asarray(pos, float),
                    velocity_ned=np.asarray(vel, float))


def test_wiring_is_identity_when_estimator_matches_wire():
    # The wiring is a pure pos/vel substitution: estimator==wire -> obs byte-identical to build_obs.
    rng = np.random.default_rng(1)
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    for _ in range(20):
        pos = rng.normal(0, 20, 3)
        vel = rng.normal(0, 5, 3)
        q = Rotation.random(random_state=rng).as_quat()        # xyzw -> wxyz
        ds = _ds(pos, vel, q=[q[3], q[0], q[1], q[2]])
        tg = int(rng.integers(0, len(_GATE_POS_ZUP)))
        direct = build_obs(ds, tg, 0.3, gate_map=gm)
        wired = estimator_obs(ds, _nav(pos, vel), tg, 0.3, gate_map=gm)
        np.testing.assert_array_equal(wired, direct)           # byte-identical (no layout corruption)


def test_estimator_state_for_obs_substitutes_pos_vel_keeps_attitude():
    ds = _ds([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], q=[0.5, 0.5, 0.5, 0.5])
    nav = _nav([10.0, 20.0, 30.0], [40.0, 50.0, 60.0])
    shim = estimator_state_for_obs(ds, nav)
    np.testing.assert_array_equal(shim.position_ned, nav.position_ned)   # estimator pos
    np.testing.assert_array_equal(shim.velocity_ned, nav.velocity_ned)   # estimator vel
    np.testing.assert_array_equal(shim.orientation_ned_wxyz, ds.orientation_ned_wxyz)  # given attitude
    np.testing.assert_array_equal(shim.angular_rate_body, ds.angular_rate_body)         # given rates


def _plus_minus_L_residuals(n=1000, seed=4242):
    """For n gate-4 sightings, route the gate-relative estimate p=gate-L through the ESTIMATOR wiring
    and return (max|pos_g - R_w2g@(+L)|  -- the +L identity residual,
                max|pos_g - R_w2g@(-L)|  -- the -L sign-flip negative control, ~24 m at the far band)."""
    rng = np.random.default_rng(seed)
    tg = 4
    gate_true_ned = _GATE_POS_ZUP[tg].copy() * _FLIP
    yaw_gate = float(_GATE_YAW_ZUP[tg])
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    R_w2g = _gate_rotmat_w2g(yaw_gate)
    max_plus, max_break = 0.0, 0.0
    for _ in range(n):
        drone = gate_true_ned - np.array([rng.uniform(4, 12), rng.uniform(-1, 1), rng.uniform(-1, 1)])
        roll, pitch, yaw = rng.normal(0, 0.15), rng.normal(-0.3, 0.1), rng.normal(np.pi, 0.1)
        R_wb = R_world_from_body(roll, pitch, yaw)
        R_wc = R_wb @ R_camera_from_body().T
        L_ned = gate_true_ned - drone                          # +L
        t_cam_gate = R_wc.T @ L_ned
        gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t_cam_gate,
                      reproj_error_px=0.0, gate_id=tg, covariance=None, n_corners=4)
        gate_obj = Gate(gate_id=tg, position_ned=gate_true_ned, R_world_gate=np.eye(3))
        # the gate-relative estimate the KF converges to: p = gate.position_ned - L  (== abs fix pos)
        p_est, _ = gate_pose_to_world_position(gp, gate_obj, R_wb, attitude_noise_std=0.0,
                                               fix_cov_floor_std=0.0)
        L_seen_zup = (R_wc @ t_cam_gate) * _FLIP
        # CORRECT estimator-sourced obs: pos_g must be R_w2g @ (+L_seen).
        pos_g = estimator_obs(_ds(p_est, np.zeros(3)), _nav(p_est, np.zeros(3)), tg, 0.0,
                              gate_map=gm)[:3].astype(np.float64)
        max_plus = max(max_plus, float(np.max(np.abs(pos_g - R_w2g @ (+L_seen_zup)))))
        max_break = max(max_break, float(np.max(np.abs(pos_g - R_w2g @ (-L_seen_zup)))))
    return max_plus, max_break


def test_estimator_sourced_obs_is_plus_L_and_minus_L_breaks():
    max_plus, max_break = _plus_minus_L_residuals()
    assert max_plus <= _PLUS_L_TOL, (
        f"estimator-sourced pos_g diverges from R_w2g @ (+L_seen) by {max_plus:.3e} m; +L broken.")
    assert max_break >= _FLIP_BREAK_M, (
        f"the -L sign-flip control only misses by {max_break:.3e} m from the estimator-sourced pos_g; "
        f"the +L/-L control is not discriminating (expected ~24 m).")
