import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.contracts import GateObservation, GatePose
from racer.frames import CAMERA_INTRINSICS_K
from racer.vision.gate_pose import (
    GATE_INNER_SIZE_M,
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


def test_prior_selects_the_consistent_branch():
    # A tilted gate yields two IPPE solutions; the prior must pick the matching one.
    R_true = _BASE @ Rotation.from_euler("y", 0.35).as_matrix()
    t_true = np.array([0.0, 0.0, 5.0])
    corners = project_gate_corners(R_true, t_true)
    cands = _solve(gate_object_points(), corners, CAMERA_INTRINSICS_K)
    assert len(cands) >= 2  # genuinely ambiguous geometry

    best = min(cands, key=lambda c: c[2])
    other = max(cands, key=lambda c: c[2])
    assert _rotation_geodesic(best[0], other[0]) > 1e-2  # the two branches differ

    obs = GateObservation(frame_id=1, sim_time_ns=0, corners_px=corners)

    # No prior -> lowest-reprojection-error branch (the true pose).
    no_prior = estimate_gate_pose(obs)
    assert _rotation_geodesic(no_prior.R_cam_gate, best[0]) < 1e-6

    # Prior near the *other* branch -> that branch is returned instead.
    prior = GatePose(
        frame_id=0, sim_time_ns=0,
        R_cam_gate=other[0], t_cam_gate=other[1], reproj_error_px=other[2],
    )
    with_prior = estimate_gate_pose(obs, prior=prior)
    assert _rotation_geodesic(with_prior.R_cam_gate, other[0]) < 1e-6


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
