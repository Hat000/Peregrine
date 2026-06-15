"""PROPOSED regression test for the DUAL-FORM (angular + metric) boresight calibration. NOT in tests/
this session (no src edits); lands in tests/ WITH proposed_calib_dualform.patch. Extends the audit's
proposed_test_boresight_calibration.py to the metric form + both estimators.

Pins, in order of importance:
  1. DEFAULT byte-identity: frames.BORESIGHT all-zero => R_camera_from_body() bit-identical to the
     20deg-only mount AND the localization +L lever bit-identical => stack stays VQ1 byte-identical.
                                                                              (needs the src patch)
  2. ANGULAR fields compose into R_camera_from_body(); METRIC field shifts the lever by exactly
     R_wb @ [0,0,vert_offset_m] and PRESERVES +L (direction unchanged).        (needs the src patch)
  3. SENSITIVITY: angular m_v ~ -range*tan(eps) (range-proportional); metric m_v ~ vert_offset
     (range-FLAT). Matched render/decode round-trips < 1 mm.                    (runs today)
  4. ESTIMATORS recover an injected value: #15 IoU-BO recovers (pitch, roll); #16 level-hover WLS
     regression recovers (intercept=metric, slope=angular) and discriminates.   (runs today)

Run standalone:  py -3.13 -m pytest handoff/.../scratch-calib-v2/proposed_test_calib_dualform.py -q
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from racer import frames as F
from racer.contracts import Gate, GateObservation, GatePose
from racer.vision.gate_pose import GATE_INNER_SIZE_M, estimate_gate_pose, project_gate_corners

DEG = np.pi / 180.0


# ---- local reimplementations of the PROPOSED patched mount + lever (so 3/4 run before the patch) ----
def _mount(pitch_rad: float = 0.0, roll_rad: float = 0.0) -> np.ndarray:
    R_tilt = Rotation.from_euler("Y", -(F.CAMERA_PITCH_RAD + pitch_rad)).as_matrix()
    R_roll = Rotation.from_euler("X", -roll_rad).as_matrix()
    return F._R_CAMERA_FROM_TILTED_BODY @ R_roll @ R_tilt


def _gate(range_m: float, az_deg: float = 0.0, el_deg: float = 0.0) -> Gate:
    az, el = az_deg * DEG, el_deg * DEG
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])
    Zc = d / np.linalg.norm(d)
    Yc = np.array([0.0, 0.0, 1.0]); Yc = Yc - Zc * (Yc @ Zc); Yc /= np.linalg.norm(Yc)
    return Gate(gate_id=0, position_ned=range_m * d,
                R_world_gate=np.column_stack([np.cross(Yc, Zc), Yc, Zc]), inner_size_m=GATE_INNER_SIZE_M)


def _fix(gate, R_wb, *, r_pitch=0.0, r_roll=0.0, r_voff=0.0, d_pitch=0.0, d_roll=0.0, d_voff=0.0):
    """Render through the physical camera (mount pitch/roll + optical-centre offset r_voff), decode
    with the (d_*) correction. Returns the full world fix error vec (recovered drone - truth)."""
    drone = np.zeros(3)
    cam_c = drone + R_wb @ np.array([0.0, 0.0, r_voff])
    R_cw_r = (R_wb @ _mount(r_pitch, r_roll).T).T
    corners = project_gate_corners(R_cw_r @ gate.R_world_gate, R_cw_r @ (gate.position_ned - cam_c),
                                   gate.inner_size_m)
    R_dec = _mount(d_pitch, d_roll)
    prior = GatePose(0, 0, (R_wb @ R_dec.T).T @ gate.R_world_gate,
                     (R_wb @ R_dec.T).T @ (gate.position_ned - drone), 0.0)
    gp = estimate_gate_pose(GateObservation(0, 0, corners_px=corners, corner_confidence=np.ones(4)),
                            prior=prior, weighted_refine=False)
    assert gp is not None and np.all(np.isfinite(gp.t_cam_gate))
    pos = gate.position_ned - (R_wb @ R_dec.T) @ gp.t_cam_gate - R_wb @ np.array([0.0, 0.0, d_voff])
    return pos - drone


def _vert(err, gate):
    return float(err @ gate.R_world_gate[:, 1])


def _try_fix(rng, *, r_pitch=0.0, r_roll=0.0, r_voff=0.0, d_pitch=0.0, d_roll=0.0, d_voff=0.0):
    """Round-trip helper robust to cv2 SOLVEPNP_IPPE_SQUARE's NaN edge on near-frontal quads: retry
    across a few probe geometries (the round-trip identity is geometry-exact at any RESOLVED pose, so
    a transient IPPE failure is just 'try another frame' -- exactly the navigator's coast behaviour)."""
    for az, el in [(0.0, 0.0), (0.0, 10.0), (8.0, 0.0), (15.0, 6.0), (20.0, 10.0)]:
        try:
            return _fix(_gate(rng, az, el), np.eye(3), r_pitch=r_pitch, r_roll=r_roll, r_voff=r_voff,
                        d_pitch=d_pitch, d_roll=d_roll, d_voff=d_voff)
        except AssertionError:
            continue
    raise AssertionError("IPPE_SQUARE unresolved at all probe geometries")


# ============================ 3. SENSITIVITY + ROUND-TRIP (run today) ============================
@pytest.mark.parametrize("eps_deg", [-0.56, -0.25, 0.25, 0.56])
def test_angular_sensitivity_is_range_proportional(eps_deg):
    """Angular: m_v ~ -range*tan(eps); the back-out ANGLE is range-flat (== angular signature)."""
    for r in (12.0, 22.0, 30.0):
        vd = _vert(_fix(_gate(r), np.eye(3), r_pitch=eps_deg * DEG), _gate(r))
        assert vd == pytest.approx(-r * np.tan(eps_deg * DEG), abs=2e-3)


def test_metric_sensitivity_is_range_flat():
    """Metric: m_v ~ vert_offset, CONSTANT across range (== metric signature). Dead-on (el=0) so
    gate-Y == world-down => m_v == voff exactly; dead-frontal is IPPE's robust case for a uniform shift."""
    voff = -0.215
    vds = [_vert(_fix(_gate(r), np.eye(3), r_voff=voff), _gate(r)) for r in (14.0, 22.0, 30.0)]
    for vd in vds:
        assert vd == pytest.approx(voff, abs=3e-3)
    assert max(vds) - min(vds) < 3e-3            # range-FLAT (the discriminator vs angular)


@pytest.mark.parametrize("r_pitch,r_roll,r_voff", [(0.0, 0.0, 0.0), (0.56 * DEG, 0.0, 0.0),
                                                   (0.0, 0.4 * DEG, 0.0), (0.0, 0.0, -0.215),
                                                   (0.3 * DEG, 0.4 * DEG, -0.1)])
@pytest.mark.parametrize("rng", [14.0, 22.0])
def test_matched_correction_round_trips(r_pitch, r_roll, r_voff, rng):
    """Render and decode at the SAME (pitch, roll, vert_offset) correction -> round-trips < 1 mm."""
    err = _try_fix(rng, r_pitch=r_pitch, r_roll=r_roll, r_voff=r_voff,
                   d_pitch=r_pitch, d_roll=r_roll, d_voff=r_voff)
    assert float(np.linalg.norm(err)) < 1e-3


# ============================ 4. ESTIMATORS recover an injected value (run today) ================
def _quad_iou(a, b):
    a = np.ascontiguousarray(a, np.float32); b = np.ascontiguousarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _project(gate, R_wb, pitch, roll):
    R_cw = (R_wb @ _mount(pitch, roll).T).T
    return project_gate_corners(R_cw @ gate.R_world_gate, R_cw @ gate.position_ned, gate.inner_size_m)


def test_iou_bo_recovers_pitch_and_roll():
    """#15 IoU-BO (here a fast grid+polish on the IoU objective; production uses GP-EI, validated in
    scratch-calib-v2/iou_bo_calib.py). Off-axis scene so ROLL is observable (degenerate head-on)."""
    R_wb = np.eye(3)
    pt, rt = 0.56 * DEG, 0.30 * DEG
    scene = [_gate(r, az) for az in (-20.0, -8.0, 8.0, 20.0) for r in (12.0, 26.0)]
    detected = [(g, _project(g, R_wb, pt, rt)) for g in scene]

    def neg_iou(x):
        pr, rr = x[0] * DEG, x[1] * DEG
        return -float(np.mean([_quad_iou(_project(g, R_wb, pr, rr), d) for g, d in detected]))

    grid = np.linspace(-1.2, 1.2, 13)
    best = min(((a, b) for a in grid for b in grid), key=lambda x: neg_iou(x))
    res = minimize(neg_iou, np.array(best), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 300})
    assert res.x[0] == pytest.approx(0.56, abs=0.05)      # pitch deg
    assert res.x[1] == pytest.approx(0.30, abs=0.10)      # roll deg (weaker; looser tol)
    assert -res.fun > 0.999                                # IoU ~ 1 at the recovered extrinsic


@pytest.mark.parametrize("eps_deg,voff,want", [(0.56, 0.0, "ANGULAR"), (0.0, -0.215, "METRIC"),
                                               (0.30, -0.10, "MIXED")])
def test_levelhover_regression_recovers_and_discriminates(eps_deg, voff, want):
    """#16 noiseless level-hover WLS: m_v ~ intercept(metric) + slope(angular)*range. Recovers both
    and classifies the form (the angular-vs-metric discriminator)."""
    ranges = np.array([10.0, 14.0, 18.0, 22.0, 26.0, 30.0])
    mv = np.array([_vert(_fix(_gate(r), np.eye(3), r_pitch=eps_deg * DEG, r_voff=voff), _gate(r))
                   for r in ranges])
    X = np.column_stack([np.ones_like(ranges), ranges])
    icpt, slope = np.linalg.lstsq(X, mv, rcond=None)[0]
    assert icpt == pytest.approx(voff, abs=2e-3)
    assert np.rad2deg(-np.arctan(slope)) == pytest.approx(eps_deg, abs=0.02)
    m_sig, a_sig = abs(icpt) > 0.05, abs(slope) > np.tan(0.05 * DEG)
    got = "ANGULAR" if a_sig and not m_sig else "METRIC" if m_sig and not a_sig else \
          "MIXED" if a_sig and m_sig else "NEGLIGIBLE"
    assert got == want


# ============================ 1 + 2. PATCHED-SRC pins (skip until the patch lands) ===============
_HAS = hasattr(F, "BORESIGHT")
_skip = pytest.mark.skipif(not _HAS, reason="frames.BORESIGHT not present until proposed_calib_dualform.patch lands")


@_skip
def test_unified_default_zero_is_byte_identical_mount(monkeypatch):
    # INVARIANT (not the shipped default): a ZERO BoresightCorrection -> the mount is bit-identical to
    # the 20deg-only mount. The SHIPPED frames.BORESIGHT is now the METRIC bake (vert_offset_m=-0.25,
    # boresight-closure-2026-06-14); that is a +L-lever translation that does NOT touch the mount, so
    # R_camera_from_body() is byte-identical to 20deg-only REGARDLESS of vert_offset_m. We pin BOTH:
    #   (a) the structural dual-form invariant under an explicit zero correction, AND
    #   (b) that the SHIPPED metric bake still leaves the mount byte-identical (metric != mount).
    expected = F._R_CAMERA_FROM_TILTED_BODY @ Rotation.from_euler("Y", -F.CAMERA_PITCH_RAD).as_matrix()
    assert np.array_equal(F.R_camera_from_body(), expected)      # shipped metric bake: mount untouched
    monkeypatch.setattr(F, "BORESIGHT", F.BoresightCorrection())  # explicit zero correction
    assert (F.BORESIGHT.pitch_rad, F.BORESIGHT.roll_rad, F.BORESIGHT.vert_offset_m) == (0.0, 0.0, 0.0)
    assert np.array_equal(F.R_camera_from_body(), expected)      # bit-identical, not just allclose


@_skip
def test_angular_fields_compose_into_mount(monkeypatch):
    monkeypatch.setattr(F, "BORESIGHT", F.BoresightCorrection(pitch_rad=0.56 * DEG, roll_rad=0.4 * DEG))
    assert np.allclose(F.R_camera_from_body(), _mount(0.56 * DEG, 0.4 * DEG))


@_skip
def test_metric_field_shifts_lever_and_preserves_plus_L(monkeypatch):
    from racer.localization import gate_pose_to_world_position
    g = _gate(20.0, az_deg=5.0); R_wb = np.eye(3)
    R_cw = (R_wb @ F.R_camera_from_body().T).T
    corners = project_gate_corners(R_cw @ g.R_world_gate, R_cw @ g.position_ned, g.inner_size_m)
    gp = estimate_gate_pose(GateObservation(0, 0, corners_px=corners, corner_confidence=np.ones(4)),
                            weighted_refine=False)
    # Baseline at an EXPLICIT zero correction (the shipped frames.BORESIGHT is now the metric bake,
    # vert_offset_m=-0.25; this test pins the lever MECHANICS, not the deployed calibration).
    monkeypatch.setattr(F, "BORESIGHT", F.BoresightCorrection())
    pos0, _ = gate_pose_to_world_position(gp, g, R_wb)            # vert_offset = 0 (explicit)
    monkeypatch.setattr(F, "BORESIGHT", F.BoresightCorrection(vert_offset_m=-0.215))
    pos1, _ = gate_pose_to_world_position(gp, g, R_wb)
    # +L preserved: the ONLY change is the additive metric term -R_wb@[0,0,voff] (direction unchanged)
    assert np.allclose(pos1 - pos0, -R_wb @ np.array([0.0, 0.0, -0.215]))
