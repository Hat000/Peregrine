import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.contracts import GateObservation, GatePose
from racer.frames import CAMERA_INTRINSICS_K
from racer.vision.gate_pose import (
    GATE_INNER_SIZE_M,
    PRIOR_DISAMBIG_MAX_RATIO,
    _rotation_geodesic,
    _solve,
    estimate_gate_pose,
    gate_object_points,
    project_gate_corners,
)

# Head-on gate: the gate frame coincides with the camera optical frame (IPPE-native
# convention is X-right, Y-down, Z-downrange, so a head-on gate is R_cam_gate = I).
_BASE = np.eye(3)


def _obs(R, t, frame_id=7, sim_time_ns=123, gate_id=None) -> GateObservation:
    return GateObservation(
        frame_id=frame_id,
        sim_time_ns=sim_time_ns,
        corners_px=project_gate_corners(R, np.asarray(t, float)),
        gate_id=gate_id,
    )


def test_object_points_scale():
    pts = gate_object_points(GATE_INNER_SIZE_M)
    assert pts.shape == (4, 3)
    # Inner-square side length between adjacent corners == 1.5 m.
    assert np.linalg.norm(pts[0] - pts[1]) == pytest.approx(GATE_INNER_SIZE_M)
    assert np.allclose(pts[:, 2], 0.0)  # coplanar


def test_round_trip_tilted_no_prior():
    R_true = _BASE @ Rotation.from_euler("y", 0.45).as_matrix()
    t_true = np.array([0.3, -0.2, 5.0])
    pose = estimate_gate_pose(_obs(R_true, t_true))
    assert pose is not None
    np.testing.assert_allclose(pose.t_cam_gate, t_true, atol=1e-3)
    assert _rotation_geodesic(pose.R_cam_gate, R_true) < 2e-3
    assert pose.reproj_error_px < 1e-2
    # A clearly non-frontal gate is unambiguous: the wrong branch has larger error.
    assert pose.ambiguity_ratio is not None and pose.ambiguity_ratio > 1.0
    # Provenance is carried through.
    assert pose.frame_id == 7 and pose.sim_time_ns == 123


def test_round_trip_many_poses():
    rng = np.random.default_rng(0)
    for _ in range(12):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.2, 0.5)  # clearly non-frontal -> unambiguous
        R_true = _BASE @ Rotation.from_rotvec(axis * angle).as_matrix()
        t_true = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(3.0, 8.0)])
        pose = estimate_gate_pose(_obs(R_true, t_true))
        assert pose is not None
        np.testing.assert_allclose(pose.t_cam_gate, t_true, atol=1e-2)
        assert _rotation_geodesic(pose.R_cam_gate, R_true) < 5e-3


def test_range_property_matches_translation():
    R_true = _BASE @ Rotation.from_euler("y", 0.3).as_matrix()
    t_true = np.array([1.0, 0.5, 6.0])
    pose = estimate_gate_pose(_obs(R_true, t_true))
    assert pose is not None
    assert pose.range_m == pytest.approx(np.linalg.norm(t_true), abs=1e-2)


def test_prior_ignored_when_geometry_unambiguous():
    # [review 2B] A clearly tilted gate has one IPPE solution that fits far better (high
    # ambiguity_ratio). A STALE prior pointing at the wrong/flipped branch must NOT override
    # the obviously-correct low-reprojection pose -- this is the bug the gating fixes.
    R_true = _BASE @ Rotation.from_euler("y", 0.45).as_matrix()
    t_true = np.array([0.0, 0.0, 5.0])
    corners = project_gate_corners(R_true, t_true)
    cands = _solve(gate_object_points(), corners, CAMERA_INTRINSICS_K)
    assert len(cands) >= 2
    best = min(cands, key=lambda c: c[2])
    other = max(cands, key=lambda c: c[2])
    assert _rotation_geodesic(best[0], other[0]) > 1e-2          # distinct branches

    obs = GateObservation(frame_id=1, sim_time_ns=0, corners_px=corners)
    prior = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=other[0],
                     t_cam_gate=other[1], reproj_error_px=other[2])
    pose = estimate_gate_pose(obs, prior=prior)
    # The true (low-reproj) branch wins despite the misleading prior.
    assert _rotation_geodesic(pose.R_cam_gate, best[0]) < 1e-6
    assert _rotation_geodesic(pose.R_cam_gate, R_true) < 5e-3


def test_prior_breaks_tie_when_ambiguous():
    # [review 2B] A near-frontal gate with measurement noise: both IPPE solutions fit
    # nearly as well (ambiguity_ratio below the threshold), so the prior decides the branch.
    rng = np.random.default_rng(0)
    R_true = _BASE @ Rotation.from_euler("y", 0.06).as_matrix()
    t_true = np.array([0.0, 0.0, 5.0])
    corners = project_gate_corners(R_true, t_true) + rng.normal(0, 1.0, (4, 2))
    cands = _solve(gate_object_points(), corners, CAMERA_INTRINSICS_K)
    assert len(cands) >= 2
    e = sorted(c[2] for c in cands)
    assert 1.0 < e[1] / e[0] < PRIOR_DISAMBIG_MAX_RATIO         # genuinely ambiguous regime
    best = min(cands, key=lambda c: c[2])
    other = max(cands, key=lambda c: c[2])
    assert _rotation_geodesic(best[0], other[0]) > 1e-2

    obs = GateObservation(frame_id=1, sim_time_ns=0, corners_px=corners)
    # No prior -> lowest-reproj branch.
    assert _rotation_geodesic(estimate_gate_pose(obs).R_cam_gate, best[0]) < 1e-6
    # Prior near the other branch -> the prior breaks the tie.
    prior = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=other[0],
                     t_cam_gate=other[1], reproj_error_px=other[2])
    assert _rotation_geodesic(estimate_gate_pose(obs, prior=prior).R_cam_gate, other[0]) < 1e-6


@pytest.mark.parametrize("missing", [0, 1, 2, 3])
def test_three_corner_pose_with_prior(missing):
    # [review 2C] A clipped gate (one corner out of frame) still yields a pose via the P3P
    # fallback, disambiguated by a temporal prior -> near-exact recovery.
    R_true = _BASE @ Rotation.from_euler("y", 0.3).as_matrix()
    t_true = np.array([0.4, -0.2, 5.0])
    corners4 = project_gate_corners(R_true, t_true)
    keep = [i for i in range(4) if i != missing]
    obs = GateObservation(frame_id=2, sim_time_ns=5,
                          corners_px=corners4[keep], corner_ids=np.array(keep))
    prior = GatePose(frame_id=1, sim_time_ns=0, R_cam_gate=R_true,
                     t_cam_gate=t_true, reproj_error_px=0.0)
    pose = estimate_gate_pose(obs, prior=prior)
    assert pose is not None and pose.n_corners == 3
    np.testing.assert_allclose(pose.t_cam_gate, t_true, atol=1e-3)
    assert _rotation_geodesic(pose.R_cam_gate, R_true) < 1e-3
    assert pose.covariance is None        # covariance only on the 4-corner IPPE path
    assert pose.ambiguity_ratio is None   # P3P branches can't be error-ranked


def test_three_corner_pose_no_prior_is_close():
    # Without a prior, P3P picks the most head-on cheirality-valid pose: a usable degraded
    # fallback (not exact). Callers at gate transit should supply a temporal/map prior.
    R_true = _BASE @ Rotation.from_euler("y", 0.12).as_matrix()
    t_true = np.array([0.1, 0.0, 5.0])
    corners4 = project_gate_corners(R_true, t_true)
    keep = [0, 1, 2]
    obs = GateObservation(frame_id=2, sim_time_ns=5,
                          corners_px=corners4[keep], corner_ids=np.array(keep))
    pose = estimate_gate_pose(obs)
    assert pose is not None and pose.n_corners == 3
    assert _rotation_geodesic(pose.R_cam_gate, R_true) < 0.1     # within ~6 deg
    np.testing.assert_allclose(pose.t_cam_gate, t_true, atol=0.1)


def test_four_corner_with_permuted_ids():
    # corner_ids let a detector hand corners in any order; estimate_gate_pose reorders them
    # to the canonical IPPE_SQUARE order before solving.
    R_true = _BASE @ Rotation.from_euler("y", 0.3).as_matrix()
    t_true = np.array([0.2, 0.1, 6.0])
    corners = project_gate_corners(R_true, t_true)
    perm = [2, 0, 3, 1]
    obs = GateObservation(frame_id=1, sim_time_ns=0,
                          corners_px=corners[perm], corner_ids=np.array(perm))
    pose = estimate_gate_pose(obs)
    assert pose is not None and pose.n_corners == 4
    np.testing.assert_allclose(pose.t_cam_gate, t_true, atol=1e-2)
    assert _rotation_geodesic(pose.R_cam_gate, R_true) < 5e-3


def test_corner_perturbation_covariance():
    R_true = _BASE @ Rotation.from_euler("y", 0.4).as_matrix()
    t_true = np.array([0.2, 0.1, 6.0])
    pose = estimate_gate_pose(
        _obs(R_true, t_true),
        compute_covariance=True,
        corner_sigma_px=1.0,
        n_samples=40,
        rng=np.random.default_rng(1),
    )
    assert pose is not None
    cov = pose.covariance
    assert cov is not None and cov.shape == (6, 6)
    assert np.allclose(cov, cov.T)                 # symmetric
    assert np.all(np.diag(cov) >= 0.0)             # valid variances
    tz_std = float(np.sqrt(cov[2, 2]))             # depth is the dominant uncertainty
    assert 0.0 < tz_std < 0.5
    # The point estimate itself is unchanged by the covariance computation.
    np.testing.assert_allclose(pose.t_cam_gate, t_true, atol=1e-2)


def test_nan_corners_return_none():
    corners = project_gate_corners(_BASE @ Rotation.from_euler("y", 0.3).as_matrix(),
                                   np.array([0.0, 0.0, 5.0]))
    corners[0, 0] = np.nan
    obs = GateObservation(frame_id=1, sim_time_ns=0, corners_px=corners)
    assert estimate_gate_pose(obs) is None


def test_degenerate_geometry_does_not_raise():
    # Collinear corners cannot form a square; must fail gracefully (None), not raise.
    corners = np.array([[100.0, 100.0], [150.0, 150.0], [200.0, 200.0], [250.0, 250.0]])
    obs = GateObservation(frame_id=1, sim_time_ns=0, corners_px=corners)
    result = estimate_gate_pose(obs)
    assert result is None or isinstance(result, GatePose)


def test_projector_rejects_gate_behind_camera():
    with pytest.raises(ValueError):
        project_gate_corners(_BASE, np.array([0.0, 0.0, -1.0]))
