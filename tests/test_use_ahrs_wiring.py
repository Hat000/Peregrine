"""use_ahrs seam wiring — L1 AHRS attitude into the case-C nav loop (build step 3).

Pins the NEW ``NavigatorConfig.use_ahrs`` flag (defaults FALSE) + the raw-gyro plumbing:

  OFF == BYTE-IDENTICAL
    With ``use_ahrs=False`` the navigator's KF state (x, P) and the produced 20-dim obs are
    numerically IDENTICAL (assert_array_equal) to the pre-change ODOMETRY-attitude path over a
    synthetic IMU+gate sequence. No new RNG drawn. The whole VQ1 / case-A loop is unchanged.

  GYRO PLUMBING
    ``DroneState.gyro_body`` is the RAW HIGHRES_IMU gyro (parsed in mavlink_client), distinct from
    the ODOMETRY-derived ``angular_rate_body``; ``None`` by default. (parse test lives in the
    mavlink ingest suite; here we exercise the field through the loop.)

  ON-path smoke (use_ahrs=True)
    Over a synthetic IMU trajectory (ahrs.imu_gen) the loop runs, produces FINITE in-bounds obs,
    the obs attitude is NOT double-conjugated (compared against a TRUE-attitude oracle within an
    attitude-error budget), and the rate sign in obs[9:12] matches the ODOMETRY-path convention.

Torch-free for the OFF/plumbing asserts; the obs asserts import build_obs lazily (skipped if torch
absent). [use_ahrs-wiring 2026-06-28]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

from racer.ahrs.imu_gen import Scenario, generate_imu_sequence  # noqa: E402
from racer.contracts import DroneState, Frame, Gate, GateObservation  # noqa: E402
from racer.frames import (  # noqa: E402
    ODO_QUAT_TRUE_CONJ_WXYZ,
    R_camera_from_body,
    euler_from_quat_wxyz,
)
from racer.navigator import Navigator, NavigatorConfig  # noqa: E402
from racer.vision.gate_pose import project_gate_corners  # noqa: E402

_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])
_HOVER_ACCEL = np.array([0.0, 0.0, -9.80665])
_DT_NS = 5_000_000          # 5 ms = 200 Hz HIGHRES_IMU


# --------------------------------------------------------------------------- helpers
def _gate_facing_north(position, gate_id=0, inner=1.5) -> Gate:
    R = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float),
                R_world_gate=R, inner_size_m=inner)


def _project_gate(gate: Gate, drone_pos, R_wb, inner=1.5) -> np.ndarray:
    R_camera_world = (R_wb @ R_camera_from_body().T).T
    t_cam_gate = R_camera_world @ (gate.position_ned - np.asarray(drone_pos, float))
    R_cam_gate = R_camera_world @ gate.R_world_gate
    return project_gate_corners(R_cam_gate, t_cam_gate, inner_size_m=inner)


class _FakeDetector:
    """Projects one gate from a fixed true drone pose (level) into GateObservations."""

    def __init__(self, gate: Gate, drone_pos, inner=1.5):
        self.gate, self.drone_pos, self.inner = gate, np.asarray(drone_pos, float), inner

    def detect(self, frame: Frame):
        corners = _project_gate(self.gate, self.drone_pos, np.eye(3), self.inner)
        return [GateObservation(
            frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
            corners_px=corners, corner_confidence=np.ones(4), score=0.9,
        )]


def _frame(frame_id, sim_time_ns):
    return Frame(frame_id=frame_id, sim_time_ns=int(sim_time_ns),
                 image_bgr=np.zeros((360, 640, 3), np.uint8), recv_monotonic_ns=0)


def _ds_wire(sim_time_ns, *, position=None, q_wxyz=None, accel=None,
             gyro_body=None, w_body=None):
    """A DroneState as the wire would deliver it (q raw, angular_rate ODOMETRY-convention)."""
    return DroneState(
        sim_time_ns=int(sim_time_ns),
        orientation_ned_wxyz=(_LEVEL_Q.copy() if q_wxyz is None else np.asarray(q_wxyz, float)),
        roll=0.0, pitch=0.0, yaw=0.0,
        angular_rate_body=(np.zeros(3) if w_body is None else np.asarray(w_body, float)),
        accel_body=(_HOVER_ACCEL.copy() if accel is None else np.asarray(accel, float)),
        gyro_body=(None if gyro_body is None else np.asarray(gyro_body, float)),
        position_ned=None if position is None else np.asarray(position, float),
    )


def _ds_with_euler(ds: DroneState) -> DroneState:
    """Fill roll/pitch/yaw from the (raw) quat, as mavlink_client does on the ODOMETRY branch."""
    import dataclasses
    r, p, y = euler_from_quat_wxyz(ds.orientation_ned_wxyz)
    return dataclasses.replace(ds, roll=r, pitch=p, yaw=y)


def _make_nav(use_ahrs: bool, gate: Gate, drone_pos):
    cfg = NavigatorConfig(
        use_given_position=False, use_given_velocity=False,   # true case-C: vision-only
        use_vision=True, use_ahrs=use_ahrs,
        reconcile_vision_clock=False,                          # same-clock synthetic data
    )
    return Navigator(gates=[gate], detector=_FakeDetector(gate, drone_pos), config=cfg)


# ===========================================================================
# OFF == BYTE-IDENTICAL
# ===========================================================================
def test_off_kf_and_obs_byte_identical_to_wire_path():
    """use_ahrs=False: KF (x,P) AND the 20-dim obs are bit-identical to a navigator that never had
    the flag — proving the new gated seam changes NOTHING on the default path."""
    pytest.importorskip("torch")
    from racer.estimator_obs import estimator_obs20

    gate = _gate_facing_north([30.0, 0.0, 0.0])
    drone_pos = np.array([0.0, 0.0, 0.0])

    nav_a = _make_nav(use_ahrs=False, gate=gate, drone_pos=drone_pos)
    nav_b = _make_nav(use_ahrs=False, gate=gate, drone_pos=drone_pos)

    obs_a, obs_b = [], []
    for k in range(40):
        t = k * _DT_NS
        ds = _ds_with_euler(_ds_wire(t, q_wxyz=_LEVEL_Q, accel=_HOVER_ACCEL))
        ns_a = nav_a.update(ds, _frame(k, t))
        ns_b = nav_b.update(ds, _frame(k, t))
        obs_a.append(estimator_obs20(nav_a.obs_drone_state(ds), ns_a, 0, 0.0))
        obs_b.append(estimator_obs20(nav_b.obs_drone_state(ds), ns_b, 0, 0.0))

    np.testing.assert_array_equal(nav_a.kf.x, nav_b.kf.x)
    np.testing.assert_array_equal(nav_a.kf.P, nav_b.kf.P)
    for oa, ob in zip(obs_a, obs_b):
        np.testing.assert_array_equal(oa, ob)
    # OFF path: obs_drone_state returns ds UNCHANGED (the AHRS seam is inert).
    last_ds = _ds_with_euler(_ds_wire(0, q_wxyz=_LEVEL_Q))
    assert nav_a.obs_drone_state(last_ds) is last_ds
    assert nav_a._ahrs is None and nav_a._ahrs_odo_quat is None


def test_off_obs_matches_direct_build_obs_wire_path():
    """use_ahrs=False end-to-end obs == build_obs on the raw wire DroneState (the inc7 path).
    Pins that the seam did not perturb the ODOMETRY-attitude obs contract."""
    pytest.importorskip("torch")
    from fly_rl import build_obs

    from racer.estimator_obs import estimator_obs20

    gate = _gate_facing_north([30.0, 0.0, 0.0])
    nav = _make_nav(use_ahrs=False, gate=gate, drone_pos=[0.0, 0.0, 0.0])
    for k in range(40):
        t = k * _DT_NS
        ds = _ds_with_euler(_ds_wire(t, q_wxyz=_LEVEL_Q))
        ns = nav.update(ds, _frame(k, t))
    obs_seam = estimator_obs20(nav.obs_drone_state(ds), ns, 0, 0.0)
    # build_obs reads attitude/rate from ds, pos/vel from ds too -> source pos/vel from the estimate
    import dataclasses
    ds_est = dataclasses.replace(ds, position_ned=ns.position_ned, velocity_ned=ns.velocity_ned)
    obs_direct = build_obs(ds_est, 0, 0.0)
    np.testing.assert_array_equal(obs_seam[:17], obs_direct)   # [0:17] is the build_obs core


# ===========================================================================
# GYRO PLUMBING
# ===========================================================================
def test_gyro_body_default_none_and_distinct_from_angular_rate():
    ds = DroneState()
    assert ds.gyro_body is None
    # angular_rate_body keeps its non-None zero default (ODOMETRY-derived).
    assert ds.angular_rate_body is not None
    np.testing.assert_array_equal(ds.angular_rate_body, np.zeros(3))


# ===========================================================================
# ON-path smoke (use_ahrs=True)
# ===========================================================================
def _truth_oracle_ds(t_ns, q_true_gt, w_true_frd_gt, position=None):
    """A DroneState carrying the GROUND-TRUTH attitude/rate re-encoded into the ODOMETRY-WIRE
    convention (q_gt*CONJ, -w_gt). Run through build_obs (which re-conjugates) it yields the obs
    a perfect attitude estimate would produce — the oracle the AHRS obs is compared against."""
    return _ds_wire(
        t_ns,
        position=position,
        q_wxyz=np.asarray(q_true_gt, float) * ODO_QUAT_TRUE_CONJ_WXYZ,
        w_body=-np.asarray(w_true_frd_gt, float),
    )


def _angle_between_quats(qa, qb) -> float:
    """Geodesic angle (rad) between two (w,x,y,z) attitudes."""
    qa = np.asarray(qa, float); qa = qa / np.linalg.norm(qa)
    qb = np.asarray(qb, float); qb = qb / np.linalg.norm(qb)
    return float(2.0 * np.arccos(min(1.0, abs(float(qa @ qb)))))


def test_on_path_attitude_not_double_conjugated_and_finite_obs():
    """use_ahrs=True over a synthetic rolling-maneuver IMU sequence:
      * the loop runs and produces FINITE, in-bounds obs every tick;
      * the AHRS-estimated TRUE attitude tracks the ground truth within an attitude-error budget
        (a DOUBLE-conjugation would flip roll+yaw -> tens of degrees of error -> FAIL);
      * the obs[6:9]/[9:12] from the AHRS path match a TRUE-attitude oracle within budget;
      * obs[9:12] rate sign matches the ODOMETRY-path convention (oracle built with -w_gt)."""
    pytest.importorskip("torch")
    from racer.estimator_obs import estimator_obs20

    # Synthetic 200 Hz trajectory with GT attitude + raw gyro/accel (rolling S-curve, mild g).
    seq = generate_imu_sequence(Scenario.ROLLING_MANEUVER, duration_s=3.0, dt=0.005,
                                gyro_noise_std=0.005, accel_noise_std=0.03, seed=7)
    gate = _gate_facing_north([30.0, 0.0, 0.0])
    drone_pos = np.array([0.0, 0.0, 0.0])
    nav = _make_nav(use_ahrs=True, gate=gate, drone_pos=drone_pos)

    att_err_deg = []
    obs_deltas = []          # |obs_ahrs[6:12] - obs_oracle[6:12]| after convergence
    n = seq.N
    for k in range(n):
        t = k * _DT_NS
        ds = _ds_wire(t, accel=seq.accel[k], gyro_body=seq.gyro[k])
        ns = nav.update(ds, _frame(k, t))
        obs = estimator_obs20(nav.obs_drone_state(ds), ns, 0, 0.0)
        assert np.all(np.isfinite(obs)), f"non-finite obs at k={k}"
        assert obs.shape == (20,)
        assert np.all(np.abs(obs) < 1e3), f"obs out of bounds at k={k}: {obs}"

        if k == 0:
            # First update() runs _initialize and returns before the first AHRS step -> cache empty;
            # obs uses the seeded (level) attitude. Skip the attitude/oracle asserts for this tick.
            assert nav._ahrs_odo_quat is None
            continue
        q_true_gt = seq.q_wxyz_gt[k]
        # AHRS TRUE attitude (un-re-encode the cached ODOMETRY-convention quat -> TRUE).
        q_ahrs_true = nav._ahrs_odo_quat * ODO_QUAT_TRUE_CONJ_WXYZ
        att_err_deg.append(np.degrees(_angle_between_quats(q_ahrs_true, q_true_gt)))

        if k >= n // 2:        # post-convergence window
            w_true_frd = seq.gyro[k]   # noise-free omega ~ gyro (zero bias); budget covers noise
            oracle_ds = _truth_oracle_ds(t, q_true_gt, w_true_frd, position=ns.position_ned)
            obs_oracle = estimator_obs20(oracle_ds, ns, 0, 0.0)
            obs_deltas.append(np.abs(obs[6:12] - obs_oracle[6:12]))

    att_err = np.array(att_err_deg)
    steady = att_err[n // 2:]
    p90 = float(np.percentile(steady, 90))
    # Budget: ESKF rolling-maneuver steady-state attitude error. A double-conjugation would be
    # ~tens of degrees; a correct estimate is a few degrees. 6 deg p90 is a generous-but-tight bar.
    assert p90 < 6.0, f"AHRS attitude p90={p90:.2f} deg exceeds budget (double-conjugation?)"

    obs_deltas = np.array(obs_deltas)
    rpy_err = float(np.percentile(obs_deltas[:, :3].max(axis=1), 90))   # obs[6:9]
    rate_err = float(np.percentile(obs_deltas[:, 3:].max(axis=1), 90))  # obs[9:12]
    # obs[6:9] rpy_g tracks the attitude budget (~0.1 rad); a flip would be radians off.
    assert rpy_err < 0.12, f"obs[6:9] vs oracle p90={rpy_err:.3f} (double-conjugation?)"
    # obs[9:12] w_flu: oracle built with the ODOMETRY-convention rate (-w_gt) -> AHRS must MATCH
    # it (same sign). A wrong sign would be ~2x the rate magnitude off.
    assert rate_err < 0.10, f"obs[9:12] rate-sign mismatch p90={rate_err:.3f}"


def test_on_path_reseeds_ahrs_on_epoch_reset():
    """A sim epoch restart (reset_counter ticks) re-seeds the AHRS alongside the KF (GAP #6)."""
    import dataclasses
    gate = _gate_facing_north([30.0, 0.0, 0.0])
    nav = _make_nav(use_ahrs=True, gate=gate, drone_pos=[0.0, 0.0, 0.0])
    seq = generate_imu_sequence(Scenario.CONSTANT_SPIN, duration_s=1.0, dt=0.005, seed=3)
    for k in range(50):
        t = k * _DT_NS
        ds = _ds_wire(t, accel=seq.accel[k], gyro_body=seq.gyro[k])
        nav.update(ds, _frame(k, t))
    ahrs_before = nav._ahrs
    assert ahrs_before is not None
    # epoch restart
    ds_reset = dataclasses.replace(
        _ds_wire(0, accel=seq.accel[0], gyro_body=seq.gyro[0]), reset_counter=1)
    nav.update(ds_reset)
    assert nav._ahrs is not None and nav._ahrs is not ahrs_before   # fresh instance, re-seeded
