"""Mag-free vision YAW + Z corrections into the ESKF + C2 navigator (magfree-vision-yaw-scope §4).

VQ2 carries NO magnetometer + NO barometer, so the ESKF accel update is yaw-blind (eskf.py:301)
and double-integrated accel z drifts: BOTH yaw and z must be pinned by VISION. This suite pins the
three NEW gated channels wired here (all default OFF):

  ESKF.update_yaw — a scalar world-yaw pseudo-measurement (the missing yaw observer).
    * FD-PIN the body-delta_phi-vs-world-yaw Jacobian (a wrong sign/projection makes the yaw update
      FIGHT roll/pitch).
    * ANGLE-WRAP the innovation (a near-2*pi raw difference must be the small correction it is).
    * yaw drifts on a gyro-z bias WITHOUT update_yaw, stays bounded WITH it.
    * roll/pitch unaffected by a yaw update.

  Navigator.use_vp_yaw / use_gate_bearing_yaw / use_floor_height — the wiring.
    * OFF == BYTE-IDENTICAL: all flags default False -> KF (x, P) numerically identical
      (assert_array_equal) to a navigator that never had the flags, over a synthetic IMU+gate
      sequence; no new RNG.
    * ON synthetic: a yaw-drifting run where gyro-only yaw diverges but the VP-yaw correction holds
      it bounded; a floor-height z correction reduces z error; branch-disambiguation picks the
      nearest 90-deg branch.

Torch-free throughout (the wiring touches KF/ESKF state, not the obs builder). [magfree-vision-yaw 2026-06-29]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.ahrs.eskf import (  # noqa: E402
    ESKFAHRS,
    _omega_exp_wxyz,
    _quat_multiply_wxyz,
    _quat_to_R_wxyz,
    _normalize_quat,
    _wrap_angle,
    _yaw_jacobian_dphi,
)
from racer.ahrs.imu_gen import Scenario, generate_imu_sequence  # noqa: E402
from racer.contracts import DroneState, Frame, Gate, GateObservation  # noqa: E402
from racer.frames import R_camera_from_body, euler_from_quat_wxyz  # noqa: E402
from racer.navigator import Navigator, NavigatorConfig  # noqa: E402
from racer.vision.gate_pose import project_gate_corners  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])
_HOVER_ACCEL = np.array([0.0, 0.0, -9.80665])
_DT_NS = 5_000_000          # 5 ms = 200 Hz HIGHRES_IMU


# =========================================================================== ESKF.update_yaw
def _set_attitude(f: ESKFAHRS, roll, pitch, yaw) -> np.ndarray:
    q = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()  # x,y,z,w
    f._q = _normalize_quat(np.array([q[3], q[0], q[1], q[2]]))
    return _quat_to_R_wxyz(f._q)


def test_yaw_jacobian_fd_pinned_at_tilt():
    """FD-PIN the body-delta_phi -> world-yaw Jacobian at a NON-LEVEL attitude (where the naive
    ``R^T e_z`` shortcut is WRONG). The exact atan2 form must match finite differences."""
    f = ESKFAHRS()
    for (r, p, y) in [(0.0, 0.0, 0.0), (-0.2, 0.3, 0.6), (0.4, -0.5, -1.2)]:
        R = _set_attitude(f, r, p, y)

        def yaw_of(qq):
            return float(Rotation.from_matrix(_quat_to_R_wxyz(qq)).as_euler("ZYX")[0])

        psi0 = yaw_of(f._q)
        H = _yaw_jacobian_dphi(R)
        eps = 1e-6
        H_fd = np.zeros(3)
        for i in range(3):
            dphi = np.zeros(3); dphi[i] = eps
            dq = _omega_exp_wxyz(dphi, 1.0)          # rotation by |dphi| about dphi (injection convention)
            qp = _normalize_quat(_quat_multiply_wxyz(f._q, dq))
            H_fd[i] = (_wrap_angle(yaw_of(qp) - psi0)) / eps
        np.testing.assert_allclose(H, H_fd, atol=1e-5,
                                   err_msg=f"yaw Jacobian FD mismatch at r,p,y={r,p,y}")
    # At level the Jacobian is the [0,0,1] shortcut.
    R = _set_attitude(f, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(_yaw_jacobian_dphi(R), [0.0, 0.0, 1.0], atol=1e-9)


def test_update_yaw_corrects_yaw_predominantly():
    """A yaw pseudo-measurement moves YAW toward the datum, and moves roll/pitch MUCH less (the
    exact Jacobian confines the correction predominantly to the yaw DOF; a wrong Jacobian would let
    a yaw datum slosh into roll/pitch comparably)."""
    f = ESKFAHRS()
    f._P = np.diag([0.05, 0.05, 0.05, 1e-6, 1e-6, 1e-6])
    _set_attitude(f, 0.1, -0.15, 0.0)
    r0, p0, y0 = euler_from_quat_wxyz(f.q_wxyz)
    f.update_yaw(np.deg2rad(10.0), np.deg2rad(2.0))
    r1, p1, y1 = euler_from_quat_wxyz(f.q_wxyz)
    dyaw = abs(np.rad2deg(y1 - y0))
    droll = abs(np.rad2deg(r1 - r0))
    dpitch = abs(np.rad2deg(p1 - p0))
    assert dyaw > 3.0, "yaw did not move toward the datum"
    assert y1 > y0, "yaw moved the wrong way (sign error)"
    # Yaw is the dominant DOF the update moves; roll/pitch slosh is small relative to it.
    assert droll < 0.4 * dyaw, f"roll slosh {droll:.2f} too large vs dyaw {dyaw:.2f} (Jacobian wrong)"
    assert dpitch < 0.4 * dyaw, f"pitch slosh {dpitch:.2f} too large vs dyaw {dyaw:.2f} (Jacobian wrong)"


def test_update_yaw_innovation_wraps():
    """The innovation wraps: measured +179 deg vs estimated -179 deg is a small +2 deg correction,
    NOT a ~358 deg catastrophic yank."""
    f = ESKFAHRS()
    f._P = np.diag([0.3, 0.3, 0.3, 1e-6, 1e-6, 1e-6])
    _set_attitude(f, 0.0, 0.0, np.deg2rad(-179.0))
    f.update_yaw(np.deg2rad(179.0), np.deg2rad(2.0))
    _, _, y = euler_from_quat_wxyz(f.q_wxyz)
    # Correction should be a small step toward +179 (i.e. MORE negative, wrapping past -180), never a
    # huge swing toward 0. Yaw stays near +-180.
    assert abs(abs(np.rad2deg(y)) - 180.0) < 10.0, f"wrap failed: yaw={np.rad2deg(y):.1f} deg"


def test_update_yaw_noop_on_bad_inputs():
    f = ESKFAHRS()
    P0 = f._P.copy(); q0 = f.q_wxyz.copy()
    f.update_yaw(np.nan, 0.05)
    f.update_yaw(0.3, 0.0)            # non-positive std
    f.update_yaw(0.3, -1.0)
    np.testing.assert_array_equal(f._P, P0)
    np.testing.assert_array_equal(f.q_wxyz, q0)


def test_yaw_drifts_without_update_bounded_with_it():
    """THE point: with a gyro-z bias, yaw is a free integrator and drifts; periodic update_yaw bounds
    it. Roll/pitch stay pinned by the accel either way."""
    bias_z = np.deg2rad(6.0)   # rad/s yaw-axis gyro bias (unobservable inertially -> integrates freely)
    dt = 0.005
    n = 1200                   # 6 s -> ~36 deg of free yaw drift

    def run(with_yaw_update):
        f = ESKFAHRS(gyro_noise_std=0.02)            # a little more Q on the yaw axis -> live gain
        f.reset(_LEVEL_Q)
        y_hist = []
        for k in range(n):
            gyro = np.array([0.0, 0.0, bias_z])      # pure biased yaw rate, drone actually level
            f.step(gyro, _HOVER_ACCEL, dt)
            if with_yaw_update and k % 7 == 0:       # ~30 Hz vision yaw fix at the true yaw (0)
                f.update_yaw(0.0, np.deg2rad(2.0))
            if k > n // 2:
                y_hist.append(euler_from_quat_wxyz(f.q_wxyz)[2])
        return euler_from_quat_wxyz(f.q_wxyz), np.array(y_hist)

    (r_off, p_off, y_off), _ = run(False)
    (r_on, p_on, y_on), y_hist_on = run(True)
    assert abs(np.rad2deg(y_off)) > 30.0, "yaw should drift far without the correction"
    # WITH the correction the yaw is BOUNDED (a steady-state lag, not an unbounded ramp). Compare the
    # corrected steady-state yaw against the runaway open-loop drift: it must be a small fraction of it.
    steady = float(np.rad2deg(np.mean(np.abs(y_hist_on))))
    assert steady < 8.0, f"corrected yaw should stay bounded-small (got {steady:.1f} deg mean)"
    assert steady < 0.3 * abs(np.rad2deg(y_off)), "correction did not meaningfully bound the drift"
    # roll/pitch leveled in both
    for v in (r_off, p_off, r_on, p_on):
        assert abs(np.rad2deg(v)) < 1.0


# =========================================================================== Navigator wiring helpers
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


def _frame(frame_id, sim_time_ns, image=None):
    img = np.zeros((360, 640, 3), np.uint8) if image is None else image
    return Frame(frame_id=frame_id, sim_time_ns=int(sim_time_ns),
                 image_bgr=img, recv_monotonic_ns=0)


def _ds_wire(sim_time_ns, *, position=None, q_wxyz=None, accel=None, gyro_body=None,
             active_gate_index=None):
    return DroneState(
        sim_time_ns=int(sim_time_ns),
        orientation_ned_wxyz=(_LEVEL_Q.copy() if q_wxyz is None else np.asarray(q_wxyz, float)),
        roll=0.0, pitch=0.0, yaw=0.0,
        angular_rate_body=np.zeros(3),
        accel_body=(_HOVER_ACCEL.copy() if accel is None else np.asarray(accel, float)),
        gyro_body=(None if gyro_body is None else np.asarray(gyro_body, float)),
        position_ned=None if position is None else np.asarray(position, float),
        active_gate_index=active_gate_index,
    )


def _make_nav(gate, drone_pos, **flags):
    cfg = NavigatorConfig(
        use_given_position=False, use_given_velocity=False,
        use_vision=True, use_ahrs=flags.pop("use_ahrs", True),
        reconcile_vision_clock=False,
        **flags,
    )
    return Navigator(gates=[gate], detector=_FakeDetector(gate, drone_pos), config=cfg)


# =========================================================================== OFF == BYTE-IDENTICAL
def test_off_kf_byte_identical_use_ahrs():
    """All three new flags default False -> KF (x,P) bit-identical to a navigator built without them,
    on the use_ahrs path, over a synthetic IMU+gate sequence."""
    gate = _gate_facing_north([30.0, 0.0, 0.0])
    drone_pos = np.array([0.0, 0.0, 0.0])
    seq = generate_imu_sequence(Scenario.ROLLING_MANEUVER, duration_s=2.0, dt=0.005, seed=5)

    nav_a = _make_nav(gate, drone_pos, use_ahrs=True)   # new flags all default OFF
    nav_b = _make_nav(gate, drone_pos, use_ahrs=True)
    # sanity: defaults are OFF
    assert not nav_a.config.use_vp_yaw and not nav_a.config.use_floor_height
    assert not nav_a.config.use_gate_bearing_yaw

    for k in range(seq.N):
        t = k * _DT_NS
        ds = _ds_wire(t, accel=seq.accel[k], gyro_body=seq.gyro[k], active_gate_index=0)
        nav_a.update(ds, _frame(k, t))
        nav_b.update(ds, _frame(k, t))
    np.testing.assert_array_equal(nav_a.kf.x, nav_b.kf.x)
    np.testing.assert_array_equal(nav_a.kf.P, nav_b.kf.P)
    # AHRS attitude identical too (no yaw correction touched it).
    np.testing.assert_array_equal(nav_a._ahrs.q_wxyz, nav_b._ahrs.q_wxyz)


def test_off_kf_byte_identical_case_a_path():
    """With use_ahrs OFF (VQ1 / case-A) the new flags can do NOTHING (no ESKF to correct); KF state is
    bit-identical regardless of the flag values. Pins that the channels are inert without the AHRS."""
    gate = _gate_facing_north([30.0, 0.0, 0.0])
    drone = [0.0, 0.0, 0.0]
    nav_a = _make_nav(gate, drone, use_ahrs=False)
    nav_b = _make_nav(gate, drone, use_ahrs=False,
                      use_vp_yaw=True, use_floor_height=True, use_gate_bearing_yaw=True)
    for k in range(40):
        t = k * _DT_NS
        ds = _ds_wire(t, q_wxyz=_LEVEL_Q, accel=_HOVER_ACCEL, active_gate_index=0)
        import dataclasses
        r, p, y = euler_from_quat_wxyz(ds.orientation_ned_wxyz)
        ds = dataclasses.replace(ds, roll=r, pitch=p, yaw=y)
        nav_a.update(ds, _frame(k, t))
        nav_b.update(ds, _frame(k, t))
    np.testing.assert_array_equal(nav_a.kf.x, nav_b.kf.x)
    np.testing.assert_array_equal(nav_a.kf.P, nav_b.kf.P)
    assert nav_a._ahrs is None and nav_b._ahrs is None


# =========================================================================== ON synthetic
def _striped_floor_grid_frame() -> np.ndarray:
    """A synthetic BGR frame with bright near-horizontal lines in the lower image (a floor-grid
    surrogate) so estimate_floor_height returns a finite height. Smoke-level: exercises the wiring +
    quality gate, NOT a metric-accuracy target."""
    img = np.zeros((360, 640, 3), np.uint8)
    for v in range(305, 360, 6):
        img[v, 40:600, :] = 220
    return img


def test_on_vp_yaw_bounds_drift_branch_disambiguation():
    """ON use_vp_yaw: a stubbed VP heading at the TRUE yaw holds the ESKF yaw bounded under a gyro-z
    bias that diverges without it; the mod-90 branch nearest the gyro estimate is selected."""
    gate = _gate_facing_north([30.0, 0.0, 0.0])
    drone = [0.0, 0.0, 0.0]
    bias_z = np.deg2rad(8.0)   # large yaw-axis bias -> clear drift contrast over the run
    n = 600                    # 3 s -> ~24 deg free drift

    # Stub estimate_heading: return a HeadingEstimate at TRUE yaw 0, wrapped mod-90 (=0), all branches.
    import racer.navigator as navmod
    from racer.vision.heading_vp import HeadingEstimate

    def fake_heading(image_bgr, roll, pitch, **kw):
        branches = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
        return HeadingEstimate(heading_mod90_rad=0.0, quality=0.9, n_support=40,
                               vp_px=np.array([320.0, 180.0]), horizontality=0.05,
                               branch_headings_rad=branches)

    def run(use_vp):
        nav = _make_nav(gate, drone, use_ahrs=True, use_vp_yaw=use_vp, vp_yaw_min_quality=0.3)
        for k in range(n):
            t = k * _DT_NS
            gyro = np.array([0.0, 0.0, bias_z])
            ds = _ds_wire(t, accel=_HOVER_ACCEL, gyro_body=gyro, active_gate_index=0)
            nav.update(ds, _frame(k, t))
        return euler_from_quat_wxyz(nav._ahrs.q_wxyz)

    orig = navmod.estimate_heading
    try:
        navmod.estimate_heading = fake_heading
        r_on, p_on, y_on = run(True)
        r_off, p_off, y_off = run(False)
    finally:
        navmod.estimate_heading = orig

    assert abs(np.rad2deg(y_off)) > 15.0, "yaw should drift far with VP off"
    assert abs(np.rad2deg(y_on)) < 5.0, f"VP yaw should bound the drift (got {np.rad2deg(y_on):.1f} deg)"
    assert abs(np.rad2deg(y_on)) < 0.3 * abs(np.rad2deg(y_off)), "VP yaw did not meaningfully bound drift"


def test_on_vp_yaw_rejects_ambiguous_branch():
    """If the nearest branch is implausibly far from the current yaw estimate, the update is SKIPPED
    (no silent 90-deg flip)."""
    gate = _gate_facing_north([30.0, 0.0, 0.0])
    nav = _make_nav(gate, [0.0, 0.0, 0.0], use_ahrs=True, use_vp_yaw=True,
                    vp_yaw_branch_max_rad=np.deg2rad(20.0))
    import racer.navigator as navmod
    from racer.vision.heading_vp import HeadingEstimate

    def fake_heading(image_bgr, roll, pitch, **kw):
        # branches all >20 deg from yaw~0 (offset 45 deg): nearest is 45 deg away -> reject.
        off = np.deg2rad(45.0)
        branches = np.array([off, off + np.pi / 2, off + np.pi, off - np.pi / 2])
        return HeadingEstimate(heading_mod90_rad=off, quality=0.9, n_support=40,
                               vp_px=np.array([320.0, 180.0]), horizontality=0.05,
                               branch_headings_rad=branches)

    orig = navmod.estimate_heading
    try:
        navmod.estimate_heading = fake_heading
        for k in range(20):
            t = k * _DT_NS
            ds = _ds_wire(t, accel=_HOVER_ACCEL, gyro_body=np.zeros(3), active_gate_index=0)
            nav.update(ds, _frame(k, t))
    finally:
        navmod.estimate_heading = orig
    assert nav.vision_diag.n_vp_yaw_applied == 0
    assert nav.vision_diag.n_vp_yaw_rejected > 0


def test_on_floor_height_reduces_z_error():
    """ON use_floor_height: a stubbed floor height at the true height pulls the KF z toward truth vs a
    z that has drifted. Smoke-level (the channel wiring + the z-pull), not a metric target."""
    gate = _gate_facing_north([30.0, 0.0, 5.0])
    drone = [0.0, 0.0, 0.0]
    true_height = 3.0     # camera 3 m above floor; floor at z=+3 NED -> camera z = 0
    import racer.navigator as navmod
    from racer.vision.floor_height import FloorHeightEstimate

    def fake_floor(image_bgr, roll, pitch, **kw):
        return FloorHeightEstimate(height_m=true_height, quality=0.9, n_support=6, std_m=0.10)

    nav = _make_nav(gate, drone, use_ahrs=True, use_floor_height=True,
                    floor_height_min_quality=0.3, floor_camera_height_ref_m=3.0)
    # Inject a z drift into the KF so there is an error to reduce.
    orig = navmod.estimate_floor_height
    try:
        navmod.estimate_floor_height = fake_floor
        # one init tick
        ds0 = _ds_wire(0, accel=_HOVER_ACCEL, gyro_body=np.zeros(3))
        nav.update(ds0)
        nav.kf.x[2] = 2.0     # corrupt z by +2 m
        z_before = nav.kf.x[2]
        for k in range(1, 30):
            t = k * _DT_NS
            ds = _ds_wire(t, accel=_HOVER_ACCEL, gyro_body=np.zeros(3))
            nav.update(ds, _frame(k, t))
        z_after = nav.kf.x[2]
    finally:
        navmod.estimate_floor_height = orig
    # target z (camera above floor) = floor_ref(3.0) - height(3.0) = 0.0
    assert abs(z_after - 0.0) < abs(z_before - 0.0), "floor-height update did not reduce z error"
    assert nav.vision_diag.n_floor_z_applied > 0


def test_on_floor_height_composes_with_rewind_kf():
    """The floor-height z fix routes through _apply_pos_fix so it works on the RewindKF (OOSM) path too
    (which has NO update_position_z). Guards the missing-method bug."""
    gate = _gate_facing_north([30.0, 0.0, 5.0])
    import racer.navigator as navmod
    from racer.vision.floor_height import FloorHeightEstimate

    def fake_floor(image_bgr, roll, pitch, **kw):
        return FloorHeightEstimate(height_m=3.0, quality=0.9, n_support=6, std_m=0.10)

    nav = _make_nav(gate, [0.0, 0.0, 0.0], use_ahrs=True, use_floor_height=True,
                    use_rewind_kf=True, use_gate_relative=True, vision_latency_const_s=0.0,
                    floor_height_min_quality=0.3, floor_camera_height_ref_m=3.0)
    orig = navmod.estimate_floor_height
    try:
        navmod.estimate_floor_height = fake_floor
        for k in range(20):
            t = k * _DT_NS
            ds = _ds_wire(t, accel=_HOVER_ACCEL, gyro_body=np.zeros(3))
            nav.update(ds, _frame(k, t))   # must not raise
    finally:
        navmod.estimate_floor_height = orig
    assert nav.n_floor_z_total > 0
    assert np.all(np.isfinite(nav.kf.x))


def test_on_floor_height_gated_on_quality_and_std():
    """A low-quality or large-std floor estimate is rejected (the near-horizon ill-conditioning gate)."""
    gate = _gate_facing_north([30.0, 0.0, 0.0])
    nav = _make_nav(gate, [0.0, 0.0, 0.0], use_ahrs=True, use_floor_height=True,
                    floor_height_min_quality=0.5, floor_height_max_std_m=0.3)
    import racer.navigator as navmod
    from racer.vision.floor_height import FloorHeightEstimate

    def low_q(image_bgr, roll, pitch, **kw):
        return FloorHeightEstimate(height_m=3.0, quality=0.2, n_support=3, std_m=0.1)

    def big_std(image_bgr, roll, pitch, **kw):
        return FloorHeightEstimate(height_m=3.0, quality=0.9, n_support=6, std_m=0.9)

    orig = navmod.estimate_floor_height
    for stub in (low_q, big_std):
        try:
            navmod.estimate_floor_height = stub
            nav2 = _make_nav(gate, [0.0, 0.0, 0.0], use_ahrs=True, use_floor_height=True,
                             floor_height_min_quality=0.5, floor_height_max_std_m=0.3)
            for k in range(10):
                t = k * _DT_NS
                ds = _ds_wire(t, accel=_HOVER_ACCEL, gyro_body=np.zeros(3))
                nav2.update(ds, _frame(k, t))
            assert nav2.vision_diag.n_floor_z_applied == 0
            assert nav2.vision_diag.n_floor_z_rejected > 0
        finally:
            navmod.estimate_floor_height = orig


def _true_pose_for(gate: Gate, drone_pos, true_yaw=0.0) -> "GatePoseT":
    """A GatePose whose t_cam_gate is the TRUE camera-frame gate direction from a drone at
    (drone_pos, true_yaw, level) -- i.e. the OBSERVATION a perfect detector would yield. Independent of
    the ESKF estimate (the detection does not know the drone's drifted yaw)."""
    from racer.contracts import GatePose
    R_wb = Rotation.from_euler("ZYX", [true_yaw, 0.0, 0.0]).as_matrix()
    R_camera_world = R_wb @ R_camera_from_body().T
    t_cam_gate = R_camera_world.T @ (gate.position_ned - np.asarray(drone_pos, float))
    R_cam_gate = R_camera_world.T @ gate.R_world_gate
    return GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=R_cam_gate, t_cam_gate=t_cam_gate,
                    reproj_error_px=0.1, gate_id=gate.gate_id, n_corners=4)


GatePoseT = object  # annotation alias (avoids a top-level import that other tests don't need)


def test_gate_bearing_yaw_measures_true_yaw_offaxis():
    """UNIT: the gate-bearing lock derives the CORRECT world yaw NON-CIRCULARLY -- from the OBSERVED
    camera-frame lever direction (flip-safe; NOT the PnP rotation) compared against the known gate's
    world bearing. With the ESKF yaw drifted but the observation + KF position + gate truth known, the
    injected datum pulls yaw toward the TRUE value (0). A circular implementation (bearing from the
    estimate) would inject the current estimate and never correct -- this guards against that."""
    gate = _gate_facing_north([25.0, 8.0, 0.0])     # off to the side -> real yaw leverage
    nav = _make_nav(gate, [0.0, 0.0, 0.0], use_ahrs=True, use_gate_bearing_yaw=True,
                    gate_bearing_min_offaxis_rad=np.deg2rad(5.0))
    nav.update(_ds_wire(0, accel=_HOVER_ACCEL, gyro_body=np.zeros(3), active_gate_index=0))
    nav.kf.x[:3] = np.array([0.0, 0.0, 0.0])          # KF position == truth (origin)
    # Drive the ESKF yaw away from 0 (accumulated drift).
    nav._ahrs.eskf.update_yaw(np.deg2rad(12.0), np.deg2rad(0.5))
    y_before = euler_from_quat_wxyz(nav._ahrs.q_wxyz)[2]
    assert abs(np.rad2deg(y_before)) > 8.0
    nav._active_gate_index = 0
    pose = _true_pose_for(gate, [0.0, 0.0, 0.0], true_yaw=0.0)   # observation from the TRUE (yaw=0) pose
    for _ in range(60):
        # Mimic the predict step that re-grows the yaw-axis covariance between vision fixes (otherwise P
        # collapses after the first update and the gain dies -- which is exactly what the in-plane STATE
        # floor / live predict prevent in the loop).
        nav._ahrs.eskf._P[:3, :3] += np.eye(3) * (np.deg2rad(0.5) ** 2)
        nav._apply_gate_bearing_yaw(pose, gate)
    y_after = euler_from_quat_wxyz(nav._ahrs.q_wxyz)[2]
    assert nav.n_gate_bearing_yaw_total > 0, "gate-bearing yaw never fired off-axis"
    assert abs(np.rad2deg(y_after)) < abs(np.rad2deg(y_before)), "lock did not pull yaw toward truth"
    assert abs(np.rad2deg(y_after)) < 6.0, f"lock did not converge yaw to ~0 (got {np.rad2deg(y_after):.1f})"


def test_on_gate_bearing_yaw_skips_head_on():
    """A head-on active gate (dead-ahead, no off-boresight azimuth) has no yaw leverage and is
    SKIPPED (avoids a near-zero-leverage update collapsing the yaw covariance)."""
    gate = _gate_facing_north([30.0, 0.0, 0.0])    # dead ahead -> azimuth ~0
    nav = _make_nav(gate, [0.0, 0.0, 0.0], use_ahrs=True, use_gate_bearing_yaw=True,
                    gate_bearing_min_offaxis_rad=np.deg2rad(8.0))
    for k in range(30):
        t = k * _DT_NS
        ds = _ds_wire(t, accel=_HOVER_ACCEL, gyro_body=np.zeros(3), active_gate_index=0)
        nav.update(ds, _frame(k, t))
    assert nav.n_gate_bearing_yaw_total == 0          # never applied (head-on, no leverage)
    assert nav.vision_diag.n_gate_bearing_yaw_rejected > 0   # last tick associated + skipped on leverage


def test_active_gate_index_threaded_and_none_safe():
    """active_gate_index threads onto DroneState (default None) and a None / out-of-range index makes
    the gate-bearing yaw lock no-op (no crash)."""
    assert DroneState().active_gate_index is None
    gate = _gate_facing_north([30.0, 0.0, 0.0])   # dead-ahead -> fixes ARE accepted (isolates the guard)
    nav = _make_nav(gate, [0.0, 0.0, 0.0], use_ahrs=True, use_gate_bearing_yaw=True)
    for k in range(20):
        t = k * _DT_NS
        # No active_gate_index (None) -> the lock must not fire and must not crash even with good fixes.
        ds = _ds_wire(t, accel=_HOVER_ACCEL, gyro_body=np.zeros(3), active_gate_index=None)
        nav.update(ds, _frame(k, t))
    assert nav.n_vision_fixes > 0, "fixes should be accepted (dead-ahead) to isolate the None-guard"
    assert nav.n_gate_bearing_yaw_total == 0


# =========================================================================== recon-frame smoke (optional)
_RECON = ROOT / "handoff" / "vq2-recon-2026-06-29" / "frames" / "curated"


@pytest.mark.skipif(not _RECON.exists(), reason="recon frames not present")
def test_recon_frame_vp_yaw_runs_through_loop():
    """SMOKE: a real recon frame drives the use_vp_yaw path end-to-end without crashing and produces
    finite KF/AHRS state. Not a heading-accuracy assertion (recon frames are fixtures, not targets)."""
    cv2 = pytest.importorskip("cv2")
    png = _RECON / "02_gate_deadahead_from_startpad.png"
    if not png.exists():
        pytest.skip("recon fixture missing")
    img = cv2.imread(str(png))
    assert img is not None
    gate = _gate_facing_north([30.0, 0.0, 0.0])
    nav = _make_nav(gate, [0.0, 0.0, 0.0], use_ahrs=True, use_vp_yaw=True, use_floor_height=True)
    for k in range(5):
        t = k * _DT_NS
        ds = _ds_wire(t, accel=_HOVER_ACCEL, gyro_body=np.zeros(3), active_gate_index=0)
        nav.update(ds, _frame(k, t, image=img))
    assert np.all(np.isfinite(nav.kf.x)) and np.all(np.isfinite(nav._ahrs.q_wxyz))
