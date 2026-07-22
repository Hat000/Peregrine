"""Outer-corner fusion (2026-07-05): 8-keypoint models' OUTER square joins the pose fit.

Covers the three seams the feature touches:
  contracts  — GateObservation optional outer fields validate;
  detector   — outer corners ride along; the outer->inner homography RESCUE emits an observation
               where the legacy path dropped the detection; kill-switch restores the discard;
  gate_pose  — the joint (inner+outer) refinement beats inner-only accuracy on synthetic ground
               truth; 3-inner + confident-outer upgrades P3P -> outer-IPPE (n_corners=4); an obs
               with NO outer (or use_outer=False) reproduces the legacy pose exactly.

All ground truth is synthesised with gate_pose.project_gate_corners (the same projector the rest
of the suite trusts), so these tests run laptop-only: no model, no GPU, no ultralytics.
"""
from __future__ import annotations

import numpy as np
import pytest

from racer.contracts import Frame, GateObservation
from racer.vision.detector import (
    _derive_inner_from_outer,
    observations_from_keypoints,
)
from racer.vision.gate_pose import (
    GATE_INNER_SIZE_M,
    GATE_OUTER_SIZE_M,
    estimate_gate_pose,
    project_gate_corners,
)


def _rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _random_pose(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A plausible gate pose in front of the camera (frontal-ish view, race-band range)."""
    R = _rot(rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3))
    t = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.0, 1.0), rng.uniform(4.0, 16.0)])
    return R, t


def _obs(inner_px, conf=None, outer_px=None, outer_conf=None, ids=None) -> GateObservation:
    return GateObservation(
        frame_id=0, sim_time_ns=0,
        corners_px=np.asarray(inner_px, dtype=np.float64),
        corner_ids=None if ids is None else np.asarray(ids, dtype=int),
        corner_confidence=None if conf is None else np.asarray(conf, dtype=np.float64),
        outer_corners_px=None if outer_px is None else np.asarray(outer_px, dtype=np.float64),
        outer_corner_confidence=None if outer_conf is None else np.asarray(outer_conf, dtype=np.float64),
    )


# ---------------------------------------------------------------------------- contracts
def test_contract_outer_shape_validated():
    inner = np.zeros((4, 2))
    with pytest.raises(AssertionError):
        _obs(inner, outer_px=np.zeros((3, 2)))                  # outer must be (4,2)
    with pytest.raises(AssertionError):
        GateObservation(frame_id=0, sim_time_ns=0, corners_px=inner,
                        outer_corner_confidence=np.ones(4))     # conf without corners
    o = _obs(inner, outer_px=np.zeros((4, 2)), outer_conf=np.ones(4))
    assert o.outer_corners_px.shape == (4, 2)


# ---------------------------------------------------------------------------- detector core
def test_outer_rides_along_and_legacy_path_unchanged():
    frame = Frame(frame_id=7, sim_time_ns=1, image_bgr=np.zeros((360, 640, 3), np.uint8))
    inner = np.tile(np.array([[100.0, 100], [200, 100], [200, 200], [100, 200]]), (1, 1, 1))
    conf = np.full((1, 4), 0.9)
    outer = inner * 1.5
    oconf = np.full((1, 4), 0.8)
    with_outer = observations_from_keypoints(frame, inner, conf, np.array([0.9]),
                                             outer_xy=outer, outer_conf=oconf)
    assert len(with_outer) == 1
    assert np.allclose(with_outer[0].outer_corners_px, outer[0])
    assert np.allclose(with_outer[0].outer_corner_confidence, oconf[0])
    # legacy call (no outer kwargs) == byte-compatible 4-kpt behaviour
    legacy = observations_from_keypoints(frame, inner, conf, np.array([0.9]))
    assert len(legacy) == 1 and legacy[0].outer_corners_px is None
    assert np.allclose(legacy[0].corners_px, with_outer[0].corners_px)


def test_homography_derive_recovers_exact_inner():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R, t = _random_pose(rng)
        inner_true = project_gate_corners(R, t, GATE_INNER_SIZE_M)
        outer_true = project_gate_corners(R, t, GATE_OUTER_SIZE_M)
        derived = _derive_inner_from_outer(outer_true)
        assert derived is not None
        # noiseless outer corners -> exact inner corners (same plane homography; float32 H ~1e-4 px)
        assert np.abs(derived - inner_true).max() < 1e-3


def test_outer_rescue_emits_observation():
    rng = np.random.default_rng(1)
    R, t = _random_pose(rng)
    inner_true = project_gate_corners(R, t, GATE_INNER_SIZE_M)
    outer_true = project_gate_corners(R, t, GATE_OUTER_SIZE_M)
    frame = Frame(frame_id=1, sim_time_ns=0, image_bgr=np.zeros((360, 640, 3), np.uint8))
    kxy = inner_true[None]
    # Only 2 inner corners usable. Confidences are put well below ANY plausible threshold rather
    # than just under the default, so this test keeps exercising the RESCUE and does not silently
    # turn into a 3-corner P3P case if KPT_CONF_THRESH_DEFAULT moves again (it did, 0.5 -> 0.2).
    kconf = np.array([[0.01, 0.02, 0.9, 0.9]])
    # legacy: dropped
    assert observations_from_keypoints(frame, kxy, kconf, np.array([0.9])) == []
    # with confident outer: rescued via the plane homography. partial_rescue is disabled so this
    # pins the ORIGINAL 4-outer rescue specifically, not the newer any-4-keypoints path.
    out = observations_from_keypoints(frame, kxy, kconf, np.array([0.9]), partial_rescue=False,
                                      outer_xy=outer_true[None], outer_conf=np.full((1, 4), 0.9))
    assert len(out) == 1
    assert np.abs(out[0].corners_px - inner_true).max() < 1e-3   # derived == true (noiseless)
    assert float(out[0].corner_confidence[0]) == pytest.approx(0.10)
    # rescued obs must survive PnP and land near the true pose
    pose = estimate_gate_pose(out[0])
    assert pose is not None
    assert abs(pose.range_m - float(np.linalg.norm(t))) < 0.25


def test_rescue_rejects_degenerate_outer():
    frame = Frame(frame_id=1, sim_time_ns=0, image_bgr=np.zeros((360, 640, 3), np.uint8))
    kxy = np.zeros((1, 4, 2))
    kconf = np.zeros((1, 4))
    collinear = np.array([[[10.0, 10], [12, 10], [14, 10], [16, 10]]])   # zero-area quad
    out = observations_from_keypoints(frame, kxy, kconf, np.array([0.9]),
                                      outer_xy=collinear, outer_conf=np.full((1, 4), 0.9))
    assert out == []


# ---------------------------------------------------------------------------- gate_pose fusion
def test_fused_pose_beats_inner_only():
    """Monte-Carlo: with realistic keypoint noise, the joint inner+outer fit must cut both the
    mean range error and the pose scatter vs the inner-only fit (the whole point of the fusion)."""
    rng = np.random.default_rng(42)
    err_inner, err_fused = [], []
    for _ in range(200):
        R, t = _random_pose(rng)
        inner = project_gate_corners(R, t, GATE_INNER_SIZE_M) + rng.normal(0, 1.2, (4, 2))
        outer = project_gate_corners(R, t, GATE_OUTER_SIZE_M) + rng.normal(0, 0.8, (4, 2))
        conf = np.full(4, 0.9)
        obs = _obs(inner, conf=conf, outer_px=outer, outer_conf=conf)
        r_true = float(np.linalg.norm(t))
        p_i = estimate_gate_pose(obs, use_outer=False)
        p_f = estimate_gate_pose(obs, use_outer=True)
        assert p_i is not None and p_f is not None
        err_inner.append(abs(p_i.range_m - r_true))
        err_fused.append(abs(p_f.range_m - r_true))
    err_inner, err_fused = np.array(err_inner), np.array(err_fused)
    # fused must be meaningfully better on average AND in spread (jitter proxy), with margin
    assert err_fused.mean() < 0.85 * err_inner.mean(), (err_fused.mean(), err_inner.mean())
    assert err_fused.std() < 0.95 * err_inner.std(), (err_fused.std(), err_inner.std())
    # and never catastrophically worse
    assert err_fused.max() < max(1.5 * err_inner.max(), 1.0)


def test_outer_assisted_three_corner_upgrades_p3p():
    rng = np.random.default_rng(7)
    R, t = _random_pose(rng)
    inner = project_gate_corners(R, t, GATE_INNER_SIZE_M) + rng.normal(0, 0.8, (4, 2))
    outer = project_gate_corners(R, t, GATE_OUTER_SIZE_M) + rng.normal(0, 0.6, (4, 2))
    ids = np.array([0, 1, 2])                       # UL missing (clipped)
    obs3 = _obs(inner[:3], conf=np.full(3, 0.9), ids=ids,
                outer_px=outer, outer_conf=np.full(4, 0.9))
    pose = estimate_gate_pose(obs3)
    assert pose is not None
    assert pose.n_corners == 4                       # outer-assisted: NOT the x9-inflated P3P class
    assert abs(pose.range_m - float(np.linalg.norm(t))) < 0.35
    # same 3 corners WITHOUT outer: legacy P3P class
    p3p = estimate_gate_pose(_obs(inner[:3], conf=np.full(3, 0.9), ids=ids))
    assert p3p is None or p3p.n_corners == 3


def test_inconsistent_outer_discarded():
    """The consistency gate: outer corners that tell a DIFFERENT geometric story than the inner
    square (decoys / degraded outer head — an unresolvable 4-vs-4 split for robust weighting) are
    discarded, and the pose must equal the proven inner-only fit exactly."""
    rng = np.random.default_rng(11)
    R, t = _random_pose(rng)
    inner = project_gate_corners(R, t, GATE_INNER_SIZE_M)
    ctr = inner.mean(axis=0)
    decoy = ctr + 1.4 * (inner - ctr)              # image-space dilation != the physical 2.72 m square
    conf = np.full(4, 0.9)
    fused = estimate_gate_pose(_obs(inner, conf=conf, outer_px=decoy, outer_conf=conf))
    plain = estimate_gate_pose(_obs(inner, conf=conf))
    assert fused is not None and plain is not None
    assert np.allclose(fused.t_cam_gate, plain.t_cam_gate)
    assert np.allclose(fused.R_cam_gate, plain.R_cam_gate)


def test_no_outer_is_exactly_legacy():
    """The kill-switch invariant: use_outer=False on an outer-carrying obs == an obs with no outer
    at all — bitwise-identical R, t (the pre-fusion pipeline)."""
    rng = np.random.default_rng(3)
    R, t = _random_pose(rng)
    inner = project_gate_corners(R, t, GATE_INNER_SIZE_M) + rng.normal(0, 1.0, (4, 2))
    outer = project_gate_corners(R, t, GATE_OUTER_SIZE_M) + rng.normal(0, 1.0, (4, 2))
    conf = np.full(4, 0.9)
    with_outer = _obs(inner, conf=conf, outer_px=outer, outer_conf=conf)
    without = _obs(inner, conf=conf)
    p_off = estimate_gate_pose(with_outer, use_outer=False)
    p_none = estimate_gate_pose(without)
    assert np.allclose(p_off.R_cam_gate, p_none.R_cam_gate)
    assert np.allclose(p_off.t_cam_gate, p_none.t_cam_gate)
    assert p_off.reproj_error_px == pytest.approx(p_none.reproj_error_px)
