"""Case-C VQ2 foundation: the 3 P0 navigator bugs (BLUEPRINT §1.5).

These pin the foundation the gate-relative case-C pipeline rests on:
  P0-a  the cold-start seed is gated on ``use_given_position`` (config), NOT on
        ``position_ned`` presence -- so a true case-C run is genuinely vision-only and not
        secretly anchored to the ground-truth pose (every "case C" test was secretly case A).
  P0-b  ``time_since_vision_update_s`` is measured on the IMU master clock: a learned
        ``delta_epoch`` reconciles the camera/server epoch, and a predict-forward fallback
        stamps a constant calibrated age when no usable capture stamp exists.
  P0-c  case-C velocity = the KF pos/vel coupling, exported on NavState.velocity_ned +
        pos_vel_covariance[3:6,3:6] (no new vision-velocity surface; d4v was REFUTED).

Torch-free (navigator + KF are numpy/scipy only). [P0-CASEC-FOUNDATION 2026-06-13]
"""
import numpy as np

from racer.contracts import DroneState, Frame, Gate, GateObservation
from racer.frames import R_camera_from_body
from racer.navigator import Navigator, NavigatorConfig
from racer.vision.gate_pose import project_gate_corners

_HOVER_ACCEL = np.array([0.0, 0.0, -9.80665])
_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# helpers (mirror tests/test_navigator.py)
# ---------------------------------------------------------------------------
def _ds(sim_time_ns, position=None, velocity=None, *, reset_counter=0, recv_monotonic_ns=0):
    return DroneState(
        sim_time_ns=int(sim_time_ns),
        recv_monotonic_ns=int(recv_monotonic_ns),
        orientation_ned_wxyz=_LEVEL_Q.copy(),
        accel_body=_HOVER_ACCEL.copy(),
        position_ned=None if position is None else np.asarray(position, float),
        velocity_ned=None if velocity is None else np.asarray(velocity, float),
        reset_counter=reset_counter,
    )


def _frame(frame_id, sim_time_ns, recv_monotonic_ns=0):
    return Frame(frame_id=frame_id, sim_time_ns=int(sim_time_ns),
                 image_bgr=np.zeros((360, 640, 3), np.uint8),
                 recv_monotonic_ns=int(recv_monotonic_ns))


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
    """Projects one gate from a fixed true drone pose into GateObservations."""

    def __init__(self, gate: Gate, drone_pos, inner=1.5):
        self.gate, self.drone_pos, self.inner = gate, np.asarray(drone_pos, float), inner

    def detect(self, frame: Frame):
        corners = _project_gate(self.gate, self.drone_pos, np.eye(3), self.inner)
        return [GateObservation(
            frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
            corners_px=corners, corner_confidence=np.ones(4), score=0.9,
        )]


def _run_casec_vision(nav: Navigator, *, frame_epoch_offset=0, n=60):
    """Init at origin (case C) then drive n converging vision ticks; return the last NavState.
    The IMU clock is k*1e7 ns; the camera frame epoch is shifted by frame_epoch_offset."""
    nav.update(_ds(0), _frame(0, 0 + frame_epoch_offset))
    ns = None
    for k in range(1, n + 1):
        t = k * 10_000_000
        ns = nav.update(_ds(t), _frame(k, t + frame_epoch_offset))
    return ns


# ===========================================================================
# P0-a  cold-start seed gated on the config flag, NOT on wire presence
# ===========================================================================
def test_casec_seed_is_origin_when_position_on_wire_but_flag_off():
    # THE gate to validating anything case-C: LOCAL_POSITION_NED is broadcasting a (ground-truth)
    # pose, but use_given_position=False -> the seed MUST be the origin at pos_std=5.0 (P[0,0]=25),
    # not the GT pose at the tight given_pos_std. Without the fix this was a hidden case-A seed.
    nav = Navigator(gates=[], detector=None,
                    config=NavigatorConfig(use_given_position=False))
    nav.update(_ds(0, position=[100.0, -50.0, -20.0]))      # GT pose ON the wire
    assert nav.kf is not None
    np.testing.assert_array_equal(nav.kf.position, [0.0, 0.0, 0.0])   # ORIGIN, not the GT pose
    np.testing.assert_allclose(np.diag(nav.kf.P[:3, :3]), [25.0, 25.0, 25.0])  # pos_std=5.0


def test_casec_seed_does_not_use_given_velocity_when_flag_off():
    # Symmetric leak: velocity on the wire must NOT seed the KF in case C (use_given_velocity off).
    nav = Navigator(gates=[], detector=None,
                    config=NavigatorConfig(use_given_position=False, use_given_velocity=False))
    nav.update(_ds(0, position=[10.0, 0.0, 0.0], velocity=[7.0, 0.0, 0.0]))   # GT vel on the wire
    np.testing.assert_array_equal(nav.kf.velocity, [0.0, 0.0, 0.0])           # not seeded from GT


def test_caseA_seed_still_uses_given_pose_when_flag_on():
    # Back-compat: case A is unchanged -- given pose seeds the KF at the tight given_pos_std.
    nav = Navigator(gates=[], detector=None,
                    config=NavigatorConfig(use_given_position=True, given_pos_std=0.05))
    nav.update(_ds(0, position=[100.0, -50.0, -20.0]))
    np.testing.assert_array_equal(nav.kf.position, [100.0, -50.0, -20.0])
    np.testing.assert_allclose(np.diag(nav.kf.P[:3, :3]), [0.05**2] * 3)


# ===========================================================================
# P0-b  TIMESYNC: tsv on the IMU master clock + predict-forward fallback
# ===========================================================================
def test_timesync_reconciles_cross_epoch_vision_clock():
    # The camera frame epoch trails the IMU master epoch by D = 5 s. The buggy code stamped a fix
    # at the raw camera time obs.sim and computed tsv = imu_now - obs.sim ~= +5 s (a FRESH fix
    # reads as 5 s stale). With reconciliation, delta_epoch = -D is learned once and the fix is
    # stamped on the IMU clock -> a fresh fix reads tsv ~= 0.
    D = 5_000_000_000
    gate = _gate_facing_north([9.0, 0.0, -2.5], gate_id=0)
    nav = Navigator(gates=[gate], detector=_FakeDetector(gate, [1.0, -0.5, -2.0]),
                    config=NavigatorConfig(use_given_position=False, use_given_velocity=False))
    ns = _run_casec_vision(nav, frame_epoch_offset=-D)
    assert nav.n_vision_fixes > 0
    assert nav._delta_epoch_ns == -D                       # epoch offset learned once, recv-paired
    assert ns.time_since_vision_update_s < 0.05            # fresh on the IMU clock (NOT ~5 s)


def test_predict_forward_stamps_constant_age_without_capture_stamp():
    # Predict-forward fallback: reconciliation OFF, so the chain leans on a constant calibrated
    # age. Every frame carries sim_time_ns=0 (NO usable capture stamp) yet a fix still applies and
    # tsv reads the configured age -- the path that works before a live TIMESYNC trace exists.
    L_const = 0.08
    gate = _gate_facing_north([9.0, 0.0, -2.5], gate_id=0)
    nav = Navigator(gates=[gate], detector=_FakeDetector(gate, [1.0, -0.5, -2.0]),
                    config=NavigatorConfig(use_given_position=False, use_given_velocity=False,
                                           reconcile_vision_clock=False,
                                           vision_latency_const_s=L_const))
    # IMU clock advances (k*1e7); the camera stamp is pinned at 0 (unusable / no TIMESYNC).
    nav.update(_ds(0), _frame(0, 0))
    ns = None
    for k in range(1, 61):
        ns = nav.update(_ds(k * 10_000_000), _frame(k, 0))
    assert nav.n_vision_fixes > 0
    assert nav._delta_epoch_ns is None                     # reconciliation never engaged
    assert abs(ns.time_since_vision_update_s - L_const) < 5e-3   # constant calibrated age


def test_reset_relearns_delta_epoch():
    # delta_epoch must be re-learned after a sim epoch restart (reset()).
    gate = _gate_facing_north([9.0, 0.0, -2.5], gate_id=0)
    nav = Navigator(gates=[gate], detector=_FakeDetector(gate, [1.0, -0.5, -2.0]),
                    config=NavigatorConfig(use_given_position=False, use_given_velocity=False))
    _run_casec_vision(nav, frame_epoch_offset=-1_000_000_000, n=5)
    assert nav._delta_epoch_ns == -1_000_000_000
    nav.reset()
    assert nav._delta_epoch_ns is None                     # dropped; next frame re-learns it


# ===========================================================================
# P0-c  velocity = KF pos/vel coupling, exported interface (no new surface)
# ===========================================================================
def test_casec_velocity_is_kf_pos_vel_coupling_export():
    # Case-C velocity is the KF pos/vel coupling (vision is position-only). Pin the exported
    # interface: NavState.velocity_ned == kf.velocity (the obs vel_g = R_w2g @ vel consumes it)
    # and pos_vel_covariance is the full 6x6 KF P whose [3:6,3:6] block is the velocity covariance.
    gate = _gate_facing_north([9.0, 0.0, -2.5], gate_id=0)
    nav = Navigator(gates=[gate], detector=_FakeDetector(gate, [1.0, -0.5, -2.0]),
                    config=NavigatorConfig(use_given_position=False, use_given_velocity=False))
    ns = _run_casec_vision(nav)
    assert nav.n_vision_fixes > 0
    np.testing.assert_array_equal(ns.velocity_ned, nav.kf.velocity)         # the velocity export
    assert ns.pos_vel_covariance.shape == (6, 6)
    np.testing.assert_array_equal(ns.pos_vel_covariance[3:6, 3:6], nav.kf.P[3:6, 3:6])  # vel cov
    vblk = ns.pos_vel_covariance[3:6, 3:6]
    assert np.all(np.isfinite(vblk)) and np.allclose(vblk, vblk.T)          # finite, symmetric
