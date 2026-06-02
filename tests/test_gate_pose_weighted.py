"""Robust confidence-weighted PnP refinement (gate_pose._refine_pose via estimate_gate_pose)."""
import numpy as np
from scipy.spatial.transform import Rotation

from racer.contracts import GateObservation
from racer.vision.gate_pose import (
    _rotation_geodesic,
    estimate_gate_pose,
    project_gate_corners,
)


def _truth():
    R = Rotation.from_euler("xyz", [0.15, -0.20, 0.10]).as_matrix()
    t = np.array([0.30, -0.20, 4.0])
    return R, t


def _obs(corners, conf, ids=None):
    return GateObservation(frame_id=0, sim_time_ns=0, corners_px=corners,
                           corner_ids=ids, corner_confidence=conf)


def test_clean_equal_confidence_is_a_noop():
    # Exact corners, equal confidence -> refinement must not move the (already optimal) IPPE pose.
    R, t = _truth()
    px = project_gate_corners(R, t)
    pose = estimate_gate_pose(_obs(px, np.ones(4)), weighted_refine=True)
    assert np.linalg.norm(pose.t_cam_gate - t) < 1e-2
    assert _rotation_geodesic(pose.R_cam_gate, R) < 1e-3


def test_low_confidence_corner_is_downweighted():
    # One corner is badly localised AND flagged low-confidence: weighting should beat the
    # equal-trust (refine-off) pose and land close to truth.
    R, t = _truth()
    px = project_gate_corners(R, t)
    bad = px.copy()
    bad[2] += np.array([18.0, -12.0])
    conf = np.array([0.95, 0.95, 0.20, 0.95])
    err_w = np.linalg.norm(estimate_gate_pose(_obs(bad, conf), weighted_refine=True).t_cam_gate - t)
    err_u = np.linalg.norm(estimate_gate_pose(_obs(bad, conf), weighted_refine=False).t_cam_gate - t)
    assert err_w < err_u
    assert err_w < 0.20


def test_huber_rejects_a_confident_outlier():
    # A gross outlier with HIGH confidence: confidence weighting alone wouldn't catch it, the
    # Huber residual kernel must.
    R, t = _truth()
    px = project_gate_corners(R, t)
    bad = px.copy()
    bad[1] += np.array([26.0, 24.0])
    pose = estimate_gate_pose(_obs(bad, np.full(4, 0.95)), weighted_refine=True)
    assert np.linalg.norm(pose.t_cam_gate - t) < 0.35


def test_covariance_grows_as_confidence_drops():
    R, t = _truth()
    px = project_gate_corners(R, t)
    hi = estimate_gate_pose(_obs(px, np.ones(4)), weighted_refine=True, compute_covariance=True)
    lo = estimate_gate_pose(_obs(px, np.full(4, 0.30)), weighted_refine=True, compute_covariance=True)
    assert hi.covariance is not None and lo.covariance is not None
    assert hi.covariance.shape == (6, 6)
    assert np.trace(lo.covariance) > np.trace(hi.covariance)


def test_refine_disabled_matches_legacy_path():
    # weighted_refine=False must reproduce the plain IPPE pose (no confidence used).
    R, t = _truth()
    px = project_gate_corners(R, t)
    pose = estimate_gate_pose(_obs(px, np.ones(4)), weighted_refine=False)
    assert np.linalg.norm(pose.t_cam_gate - t) < 1e-6
    assert _rotation_geodesic(pose.R_cam_gate, R) < 1e-6


def test_no_confidence_skips_refinement_safely():
    # An obs without per-corner confidence must still solve (no refine, no crash).
    R, t = _truth()
    px = project_gate_corners(R, t)
    pose = estimate_gate_pose(GateObservation(frame_id=0, sim_time_ns=0, corners_px=px))
    assert pose is not None and np.linalg.norm(pose.t_cam_gate - t) < 1e-6
