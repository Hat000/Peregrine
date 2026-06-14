"""Navigator C2 integration — RewindKF wrap + gate-relative in-plane +L fix + confidence export.

Drives the FULL case-C navigator (origin seed, vision-only) with the C2 pipeline ON and pins:
  - self.kf is the RewindKF OOSM proxy in case-C mode (bare LinearKF when off -> no VQ1 regression);
  - gate-relative fixes are applied (n_rel_applied > 0) and the estimate converges to the true pose;
  - the NavState gate-frame confidence export (nav_inplane_sigma / nav_along_sigma) goes finite once a
    gate-relative fix anchors a gate frame, and is tight in-plane;
  - the relative-innovation gate REJECTS an in-plane depth-flip the absolute estimate then ignores.

Torch-free (navigator + KF + localization are numpy/scipy only). [C2-ESTIMATOR-CHAIN 2026-06-13]
"""
import numpy as np

from racer.contracts import DroneState, Frame, Gate, GateObservation, GatePose
from racer.frames import R_camera_from_body
from racer.kf_rewind import RewindKF
from racer.navigator import Navigator, NavigatorConfig
from racer.vision.gate_pose import project_gate_corners

_HOVER_ACCEL = np.array([0.0, 0.0, -9.80665])
_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])
_TRUE_POS = np.array([1.0, -0.5, -2.0])


def _ds(sim_time_ns, position=None, velocity=None, *, reset_counter=0, recv_monotonic_ns=0):
    return DroneState(
        sim_time_ns=int(sim_time_ns), recv_monotonic_ns=int(recv_monotonic_ns),
        orientation_ned_wxyz=_LEVEL_Q.copy(), accel_body=_HOVER_ACCEL.copy(),
        position_ned=None if position is None else np.asarray(position, float),
        velocity_ned=None if velocity is None else np.asarray(velocity, float),
        reset_counter=reset_counter)


def _frame(frame_id, sim_time_ns):
    return Frame(frame_id=frame_id, sim_time_ns=int(sim_time_ns),
                 image_bgr=np.zeros((360, 640, 3), np.uint8), recv_monotonic_ns=0)


def _gate_facing_north(position, gate_id=0, inner=1.5) -> Gate:
    R = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float),
                R_world_gate=R, inner_size_m=inner)


class _FakeDetector:
    def __init__(self, gate: Gate, drone_pos, inner=1.5):
        self.gate, self.drone_pos, self.inner = gate, np.asarray(drone_pos, float), inner

    def detect(self, frame: Frame):
        R_camera_world = (np.eye(3) @ R_camera_from_body().T).T
        t_cam_gate = R_camera_world @ (self.gate.position_ned - self.drone_pos)
        R_cam_gate = R_camera_world @ self.gate.R_world_gate
        corners = project_gate_corners(R_cam_gate, t_cam_gate, inner_size_m=self.inner)
        return [GateObservation(frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
                                corners_px=corners, corner_confidence=np.ones(4), score=0.9)]


def _casec_config(**kw):
    return NavigatorConfig(use_given_position=False, use_given_velocity=False,
                           use_rewind_kf=True, use_gate_relative=True, **kw)


def _run(nav, n=80):
    nav.update(_ds(0), _frame(0, 0))
    ns = None
    for k in range(1, n + 1):
        ns = nav.update(_ds(k * 10_000_000), _frame(k, k * 10_000_000))
    return ns


def _build_nav(**cfg_kw):
    gate = _gate_facing_north([9.0, 0.0, -2.5], gate_id=0)
    return Navigator(gates=[gate], detector=_FakeDetector(gate, _TRUE_POS), config=_casec_config(**cfg_kw)), gate


def test_casec_uses_rewind_kf_and_applies_gate_relative_fixes():
    nav, _ = _build_nav()
    ns = _run(nav)
    assert isinstance(nav.kf, RewindKF)                       # OOSM proxy in case-C mode
    assert nav.vision_diag.n_rel_applied > 0 or nav.n_vision_fixes > 0
    # the estimate converges to the true (vision-only) pose
    np.testing.assert_allclose(nav.kf.position, _TRUE_POS, atol=0.3)
    np.testing.assert_allclose(ns.position_ned, nav.kf.position)


def test_confidence_export_goes_finite_and_tight_after_gate_relative_fix():
    nav, _ = _build_nav()
    ns = _run(nav)
    assert nav._last_fix_gate_R is not None                   # a gate-relative fix anchored a frame
    assert np.isfinite(ns.nav_inplane_sigma) and np.isfinite(ns.nav_along_sigma)
    assert 0.0 < ns.nav_inplane_sigma < 1.0                   # tight in-plane (confidence channel feeder)
    # along-track is looser than in-plane (the absolute fix + IMU own depth)
    assert ns.nav_along_sigma >= ns.nav_inplane_sigma


def test_vq1_path_unchanged_when_c2_off():
    # C2 OFF (default) -> bare LinearKF, no RewindKF wrap (the VQ1/case-A path is untouched).
    gate = _gate_facing_north([9.0, 0.0, -2.5], gate_id=0)
    nav = Navigator(gates=[gate], detector=_FakeDetector(gate, _TRUE_POS),
                    config=NavigatorConfig(use_given_position=False, use_given_velocity=False))
    _run(nav)
    assert not isinstance(nav.kf, RewindKF)
    assert nav.vision_diag.n_rel_applied == 0                 # no gate-relative augment


def test_relative_innovation_gate_rejects_inplane_flip():
    # Warm the estimator to a tight prior, then feed a crafted DEPTH-FLIP (large in-plane lever throw).
    # The relative-innovation gate must REJECT it and leave the estimate intact.
    nav, gate = _build_nav()
    _run(nav, n=80)
    x_before = nav.kf.x.copy()
    rel_rej_before = nav.vision_diag.n_rel_rejected
    # a flipped PnP: the true lever is gate - true_pos (~8 m along +N); throw +3 m in-plane (E).
    L_true = gate.position_ned - _TRUE_POS
    L_flip = L_true + np.array([0.0, 3.0, 0.0])               # 3 m E throw (in-plane), as a depth-flip
    R_wc = np.eye(3) @ R_camera_from_body().T
    flipped = GatePose(frame_id=999, sim_time_ns=0, R_cam_gate=np.eye(3),
                       t_cam_gate=R_wc.T @ L_flip, reproj_error_px=0.4, gate_id=0,
                       covariance=None, n_corners=4)
    nav.vision_diag.n_rel_rejected = rel_rej_before           # reset the live-tick diag for a clean read
    nav._apply_gate_relative_fix(flipped, gate, np.eye(3), nav.kf.now_ns)
    assert nav.vision_diag.n_rel_rejected == rel_rej_before + 1   # rejected
    np.testing.assert_array_equal(nav.kf.x, x_before)            # estimate untouched by the rejected fix


def test_clean_relative_fix_is_accepted_after_warmup():
    nav, gate = _build_nav()
    _run(nav, n=80)
    rel_app_before = nav.vision_diag.n_rel_applied
    L_true = gate.position_ned - _TRUE_POS
    R_wc = np.eye(3) @ R_camera_from_body().T
    clean = GatePose(frame_id=998, sim_time_ns=0, R_cam_gate=np.eye(3),
                     t_cam_gate=R_wc.T @ L_true, reproj_error_px=0.4, gate_id=0,
                     covariance=None, n_corners=4)
    nav.vision_diag.n_rel_applied = rel_app_before
    nav._apply_gate_relative_fix(clean, gate, np.eye(3), nav.kf.now_ns)
    assert nav.vision_diag.n_rel_applied == rel_app_before + 1   # a clean in-band fix is accepted
