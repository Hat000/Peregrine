"""B1 bearing-range channel — span-derived attitude-independent along-track range fix.

Pins the perception-l2 Upgrade-B1 contract:
  - ``apparent_range_from_gate_span`` recovers the metric range from the gate's apparent inner-square
    pixel SPAN (known 1.5 m) to high accuracy, head-on at several ranges;
  - the range channel is ATTITUDE-INDEPENDENT: a moderate tilt of the gate-in-camera (the ε_vert-style
    boresight perturbation that corrupts the +L lever's depth) barely moves the span-derived range;
  - ``gate_range_fix`` builds an anisotropic 3-DOF measurement that is tight along-track / very loose
    in-plane (an effective 1-DOF range update) and is +L-sign-consistent (drone sits BEHIND the gate);
  - wired behind ``NavigatorConfig.use_range_channel`` the range fix REDUCES the KF along-track error
    vs the loose GATE_REL_ALONG_SIGMA prior on a synthetic biased-depth case;
  - flag OFF (default) is NUMERICALLY BYTE-IDENTICAL to main: no new RNG, no obs change, no KF delta.

Torch-free (navigator + KF + localization are numpy/scipy only). [perception-l2 B1 2026-06-28]
"""
import numpy as np
from scipy.spatial.transform import Rotation

from racer.contracts import DroneState, Frame, Gate, GateObservation
from racer.frames import R_camera_from_body
from racer.localization import (
    FIX_COV_FLOOR_STD,
    GATE_RANGE_INPLANE_STD,
    apparent_range_from_gate_span,
    gate_range_fix,
    gate_range_sigma,
)
from racer.navigator import Navigator, NavigatorConfig
from racer.vision.gate_pose import project_gate_corners

# ---------------------------------------------------------------------------------------------------
# (a) apparent_range_from_gate_span accuracy on synthetic corner geometry
# ---------------------------------------------------------------------------------------------------

def test_apparent_range_exact_head_on_several_ranges():
    # A head-on square at depth Z must recover range == Z (the per-axis half-extent law makes this exact).
    for Z in (5.0, 8.0, 10.0, 15.0, 20.0, 30.0):
        px = project_gate_corners(np.eye(3), np.array([0.0, 0.0, Z]))
        r = apparent_range_from_gate_span(px)
        assert r is not None
        assert abs(r - Z) / Z < 5e-3, f"range@{Z} = {r:.4f}, err {abs(r - Z) / Z * 100:.3f}% > 0.5%"


def test_apparent_range_needs_four_corners():
    px = project_gate_corners(np.eye(3), np.array([0.0, 0.0, 10.0]))
    assert apparent_range_from_gate_span(px[:3]) is None      # 3-corner P3P set: no reliable span
    assert apparent_range_from_gate_span(px[:2]) is None


def test_apparent_range_degenerate_span_returns_none():
    # All corners collapsed to one point (zero span) -> None, not a divide-by-zero / inf.
    collapsed = np.tile(np.array([320.0, 180.0]), (4, 1))
    assert apparent_range_from_gate_span(collapsed) is None


def test_apparent_range_custom_intrinsics_and_inner_size():
    # Self-consistent under a different focal length + inner size: project then recover.
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]])
    inner = 2.0
    px = project_gate_corners(np.eye(3), np.array([0.0, 0.0, 12.0]), inner_size_m=inner, camera_matrix=K)
    r = apparent_range_from_gate_span(px, inner_size_m=inner, camera_matrix=K)
    assert abs(r - 12.0) < 0.05


# ---------------------------------------------------------------------------------------------------
# (a') ATTITUDE INDEPENDENCE — the whole point of B1 (robust to the ε_vert boresight bias)
# ---------------------------------------------------------------------------------------------------

def test_apparent_range_robust_to_gate_tilt():
    # Tilting the gate-in-camera (a roll/pitch perturbation, the boresight-bias regime) shifts the
    # centroid pixel but must barely move the subtended SPAN -> range changes < 0.1 m at 10 m for 15 deg.
    Z = 10.0
    base = apparent_range_from_gate_span(project_gate_corners(np.eye(3), np.array([0.0, 0.0, Z])))
    for ang_deg in (5.0, 10.0, 15.0):
        a = np.deg2rad(ang_deg)
        R = Rotation.from_euler("xyz", [a, a, 0.0]).as_matrix()
        px = project_gate_corners(R, np.array([0.0, 0.0, Z]))
        r = apparent_range_from_gate_span(px)
        assert abs(r - base) < 0.4, f"tilt {ang_deg}deg moved range by {abs(r - base):.3f} m"
    # explicitly small at 10 deg vertical-only (the ε_vert axis) — the scope-doc robustness claim. A
    # pure single-axis tilt foreshortens one edge, so the span shifts ~ (1 - cos(tilt)) -> tiny: at 10
    # deg / 10 m it is well under 0.1 m. (Contrast a 10-deg ATTITUDE error on the +L lever: ~1.7 m.)
    a = np.deg2rad(10.0)
    px = project_gate_corners(Rotation.from_euler("x", a).as_matrix(), np.array([0.0, 0.0, Z]))
    assert abs(apparent_range_from_gate_span(px) - base) < 0.1


# ---------------------------------------------------------------------------------------------------
# (b) gate_range_fix covariance shape + +L sign
# ---------------------------------------------------------------------------------------------------

def _gate_facing_north(position, gate_id=0, inner=1.5) -> Gate:
    R = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])   # gate +Z = world +N
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float),
                R_world_gate=R, inner_size_m=inner)


def test_gate_range_fix_cov_is_tight_along_track_loose_in_plane():
    gate = _gate_facing_north([20.0, 0.0, -2.0])
    range_m = 8.0
    sig = gate_range_sigma(range_m)
    z, cov = gate_range_fix(range_m, gate, np.eye(3), sigma_range=sig)
    # eigen-decompose: along-track eigenvalue = sig^2 + floor^2, two in-plane eigenvalues >> 1000 m^2.
    eig = np.sort(np.linalg.eigvalsh(cov))
    expected_along = sig ** 2 + FIX_COV_FLOOR_STD ** 2
    assert abs(eig[0] - expected_along) < 1e-9                 # tight along-track
    assert eig[1] > 1000.0 and eig[2] > 1000.0                # both in-plane >> 1000 m^2
    assert abs(eig[1] - GATE_RANGE_INPLANE_STD ** 2) < 1e-6


def test_gate_range_fix_sign_is_plus_L_consistent(monkeypatch):
    # The fix places the drone range_m BEHIND the gate along its normal — the same direction the +L
    # lever does (gate - L). For a north-facing gate, behind = SOUTH (smaller N). Sign must match +L.
    # Zero the deployed boresight vert-offset bake (frames.BORESIGHT, vert_offset_m=-0.25) so the exact-
    # vector identity tests the SIGN/convention, not the calibration offset (same as test_obs_sign_*).
    import racer.frames as _F
    monkeypatch.setattr(_F, "BORESIGHT", _F.BoresightCorrection())
    gate = _gate_facing_north([20.0, 0.0, -2.0])
    range_m = 8.0
    z, _ = gate_range_fix(range_m, gate, np.eye(3))
    # drone is range_m south of the gate along +N normal -> z_N = 20 - 8 = 12, lateral/vert unchanged.
    np.testing.assert_allclose(z, np.array([12.0, 0.0, -2.0]), atol=1e-9)
    # the along-track displacement (drone - gate) . normal must be NEGATIVE (behind), == +L convention.
    along = float((z - gate.position_ned) @ gate.normal_ned)
    assert along < 0.0 and abs(along + range_m) < 1e-9


def test_gate_range_sigma_grows_with_range_with_floor():
    from racer.localization import GATE_RANGE_SIGMA_FLOOR
    assert gate_range_sigma(0.1) == GATE_RANGE_SIGMA_FLOOR     # near-field clamps to the floor
    assert gate_range_sigma(100.0) > gate_range_sigma(10.0)    # grows with range


# ---------------------------------------------------------------------------------------------------
# Navigator harness (mirrors test_navigator_gate_relative.py)
# ---------------------------------------------------------------------------------------------------

_HOVER_ACCEL = np.array([0.0, 0.0, -9.80665])
_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])
_TRUE_POS = np.array([1.0, -0.5, -2.0])


def _ds(sim_time_ns, position=None, velocity=None):
    return DroneState(
        sim_time_ns=int(sim_time_ns), recv_monotonic_ns=0,
        orientation_ned_wxyz=_LEVEL_Q.copy(), accel_body=_HOVER_ACCEL.copy(),
        position_ned=None if position is None else np.asarray(position, float),
        velocity_ned=None if velocity is None else np.asarray(velocity, float))


def _frame(frame_id, sim_time_ns):
    return Frame(frame_id=frame_id, sim_time_ns=int(sim_time_ns),
                 image_bgr=np.zeros((360, 640, 3), np.uint8), recv_monotonic_ns=0)


class _FakeDetector:
    """Projects the gate from the drone's TRUE pose, optionally with a constant depth (boresight) bias
    baked into the lever — to mimic ε_vert corrupting the +L along-track estimate while the SPAN (and
    thus the B1 range channel) stays honest."""

    def __init__(self, gate: Gate, drone_pos, inner=1.5, depth_bias_m=0.0):
        self.gate, self.drone_pos, self.inner = gate, np.asarray(drone_pos, float), inner
        self.depth_bias_m = depth_bias_m

    def detect(self, frame: Frame):
        R_camera_world = (np.eye(3) @ R_camera_from_body().T).T
        lever_world = self.gate.position_ned - self.drone_pos
        if self.depth_bias_m:
            # shift the apparent gate along the gate normal (a depth bias on the lever, NOT the span)
            lever_world = lever_world + self.depth_bias_m * self.gate.normal_ned
        t_cam_gate = R_camera_world @ lever_world
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


# ---------------------------------------------------------------------------------------------------
# (c) flag OFF == byte-identical (numerical identity, no new RNG)
# ---------------------------------------------------------------------------------------------------

def test_range_channel_off_is_byte_identical_to_default():
    # Two case-C navigators on the SAME synthetic stream: one default (flag implicitly OFF), one with
    # use_range_channel=False explicitly. The KF state must be NUMERICALLY IDENTICAL (no new RNG path).
    gate = _gate_facing_north([9.0, 0.0, -2.5], gate_id=0)
    nav_a = Navigator(gates=[gate], detector=_FakeDetector(gate, _TRUE_POS), config=_casec_config())
    nav_b = Navigator(gates=[gate], detector=_FakeDetector(gate, _TRUE_POS),
                      config=_casec_config(use_range_channel=False))
    _run(nav_a)
    _run(nav_b)
    np.testing.assert_array_equal(nav_a.kf.x, nav_b.kf.x)         # exact equality
    np.testing.assert_array_equal(nav_a.kf.P, nav_b.kf.P)
    assert nav_a.vision_diag.n_range_applied == 0                 # the channel never ran
    assert nav_b.vision_diag.n_range_applied == 0


def test_range_channel_flag_on_without_gate_relative_is_inert():
    # use_range_channel=True but use_gate_relative=False -> the range channel must NOT participate
    # (it augments the gate-relative path). Byte-identical to the same config with the flag off.
    gate = _gate_facing_north([9.0, 0.0, -2.5], gate_id=0)
    base = NavigatorConfig(use_given_position=False, use_given_velocity=False, use_rewind_kf=True)
    nav_off = Navigator(gates=[gate], detector=_FakeDetector(gate, _TRUE_POS), config=base)
    on = NavigatorConfig(use_given_position=False, use_given_velocity=False, use_rewind_kf=True,
                         use_range_channel=True)
    nav_on = Navigator(gates=[gate], detector=_FakeDetector(gate, _TRUE_POS), config=on)
    _run(nav_off)
    _run(nav_on)
    np.testing.assert_array_equal(nav_off.kf.x, nav_on.kf.x)
    assert nav_on.vision_diag.n_range_applied == 0


# ---------------------------------------------------------------------------------------------------
# (d) range fix REDUCES along-track error vs the loose prior on a synthetic biased-depth case
# ---------------------------------------------------------------------------------------------------

def test_range_channel_reduces_along_track_error_under_depth_bias():
    # A constant +0.6 m depth bias on the +L lever (the ε_vert-style boresight error) drags the
    # along-track (gate-normal) estimate. The span-derived range channel is bias-FREE, so turning it
    # on must reduce the along-track estimate error vs the loose-prior (channel-off) baseline.
    gate = _gate_facing_north([12.0, 0.0, -2.0], gate_id=0)
    depth_bias = 0.6

    nav_off = Navigator(gates=[gate], detector=_FakeDetector(gate, _TRUE_POS, depth_bias_m=depth_bias),
                        config=_casec_config())
    nav_on = Navigator(gates=[gate], detector=_FakeDetector(gate, _TRUE_POS, depth_bias_m=depth_bias),
                       config=_casec_config(use_range_channel=True))
    _run(nav_off, n=120)
    _run(nav_on, n=120)

    n_hat = gate.normal_ned
    err_off = abs(float((nav_off.kf.position - _TRUE_POS) @ n_hat))
    err_on = abs(float((nav_on.kf.position - _TRUE_POS) @ n_hat))
    assert nav_on.vision_diag.n_range_applied > 0                 # the channel actually fired
    assert err_on < err_off, f"range channel did not help: on={err_on:.3f} off={err_off:.3f}"
    # and it meaningfully shrinks the bias-induced along-track error
    assert err_on < 0.6 * err_off + 1e-6


def test_range_channel_converges_without_bias():
    # No bias: range channel ON must still converge to the true pose (it does not break the clean case).
    gate = _gate_facing_north([9.0, 0.0, -2.5], gate_id=0)
    nav = Navigator(gates=[gate], detector=_FakeDetector(gate, _TRUE_POS),
                    config=_casec_config(use_range_channel=True))
    _run(nav)
    assert nav.vision_diag.n_range_applied > 0
    np.testing.assert_allclose(nav.kf.position, _TRUE_POS, atol=0.3)
