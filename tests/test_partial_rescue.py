"""Partial-corner rescue: recover a CROPPED gate from any 4 usable keypoints (2026-07-21).

The inner (1.5 m) and outer (2.72 m) squares are concentric and COPLANAR, so all 8 keypoints
sample ONE plane-to-image homography. A gate half out of frame that still shows two inner corners
and their two outer corners therefore determines the full inner square -- including the corners
that are off-frame. These tests pin the geometry, the degeneracy guards, and (critically) that the
pre-existing emission path is byte-unchanged.
"""
import numpy as np
import pytest

from racer.contracts import Frame
from racer.vision.detector import (
    N_CORNERS,
    _PARTIAL_RESCUE_CONF,
    observations_from_keypoints,
)
from racer.vision.gate_pose import (
    GATE_OUTER_SIZE_M,
    estimate_gate_pose,
    gate_object_points,
    inner_from_partial_keypoints,
    project_gate_corners,
)

IMG_WH = (640, 360)


def _frame(fid=1):
    return Frame(frame_id=fid, sim_time_ns=fid * 1000, image_bgr=np.zeros((360, 640, 3), np.uint8),
                 recv_monotonic_ns=0, jpeg_bytes=None)


def _project8(R, t):
    """The 8 keypoints (inner 4 then outer 4) of a gate at pose (R, t)."""
    inner = project_gate_corners(R, t)
    outer = project_gate_corners(R, t, inner_size_m=GATE_OUTER_SIZE_M)
    return np.vstack([inner, outer])


def _pose(yaw=0.0, pitch=0.0, t=(0.0, 0.0, 6.0)):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Ry @ Rx, np.asarray(t, float)


# --------------------------------------------------------------------------- geometry

@pytest.mark.parametrize("yaw,pitch,t", [
    (0.0, 0.0, (0.0, 0.0, 6.0)),
    (0.5, 0.2, (1.5, -0.5, 4.0)),
    (-0.7, -0.25, (-2.0, 1.0, 9.0)),
])
def test_two_corner_pairs_recover_the_inner_square_exactly(yaw, pitch, t):
    """2 adjacent corners + their 2 outer corners == the whole gate, to sub-pixel."""
    R, tv = _pose(yaw, pitch, t)
    kp = _project8(R, tv)
    truth = kp[:N_CORNERS]
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        usable = np.zeros(8, bool)
        usable[[a, b, a + 4, b + 4]] = True
        got = inner_from_partial_keypoints(kp, usable, img_w=IMG_WH[0])
        assert got is not None, f"pairs {a},{b} failed"
        assert np.allclose(got, truth, atol=1e-6), f"pairs {a},{b} max {np.abs(got-truth).max()}"


def test_recovers_corners_that_are_off_frame():
    """The point of the exercise: the derived corners may legitimately lie outside the image."""
    R, tv = _pose(0.35, 0.1, (0.9, -0.35, 2.2))     # close + oblique => gate crops out
    kp = _project8(R, tv)
    inner = kp[:N_CORNERS]
    off = [(x < 0 or x > IMG_WH[0] or y < 0 or y > IMG_WH[1]) for x, y in inner]
    assert any(off), "fixture must actually crop a corner off-frame"
    usable = np.zeros(8, bool)
    usable[[0, 1, 4, 5]] = True
    got = inner_from_partial_keypoints(kp, usable, img_w=IMG_WH[0])
    assert got is not None
    assert np.allclose(got, inner, atol=1e-3)


def test_mixed_keypoint_sets_work_not_just_pairs():
    """Any 4 in general position -- e.g. 3 inner + 1 outer, or mixed inner/outer across corners."""
    R, tv = _pose(0.3, -0.15, (0.5, 0.2, 5.0))
    kp = _project8(R, tv)
    truth = kp[:N_CORNERS]
    for idx in ([0, 1, 2, 5], [0, 1, 5, 6], [1, 2, 6, 7], [2, 3, 4, 5]):
        usable = np.zeros(8, bool)
        usable[idx] = True
        got = inner_from_partial_keypoints(kp, usable, img_w=IMG_WH[0])
        assert got is not None, f"set {idx} failed"
        assert np.allclose(got, truth, atol=1e-3), f"set {idx}"


def test_three_collinear_points_are_rejected():
    """4 points are not enough if THREE of them are collinear -- the homography is underdetermined
    and OpenCV happily returns an arbitrary member of the solution family. All 8 keypoints lie on
    the gate's two diagonals, so e.g. {inner_LL, outer_LL, outer_UR} + one more is degenerate."""
    R, tv = _pose(0.3, -0.15, (0.5, 0.2, 5.0))
    kp = _project8(R, tv)
    for idx in ([0, 4, 5, 6], [0, 2, 4, 5], [1, 3, 5, 6], [0, 2, 6, 4]):
        usable = np.zeros(8, bool)
        usable[idx] = True
        assert inner_from_partial_keypoints(kp, usable, img_w=IMG_WH[0]) is None, f"set {idx}"


# --------------------------------------------------------------------------- guards

def test_fewer_than_four_keypoints_is_rejected():
    R, tv = _pose()
    kp = _project8(R, tv)
    for n in (0, 1, 2, 3):
        usable = np.zeros(8, bool)
        usable[:n] = True
        assert inner_from_partial_keypoints(kp, usable, img_w=IMG_WH[0]) is None


def test_two_opposite_corner_pairs_are_rejected_as_collinear():
    """A pure opposite pair puts all 4 points on ONE diagonal -- a 2-D homography is undefined.

    Measured consequence of NOT guarding this: routing such sets through an intersection solver
    gave 10.4 px median centre error (vs 0.21 px for a cross-ratio treatment), because keypoint
    noise makes nominally-parallel rays cross at an arbitrary point. Reject, don't guess."""
    R, tv = _pose(0.0, 0.0, (0.0, 0.0, 8.0))
    kp = _project8(R, tv)
    for a in (0, 1):
        usable = np.zeros(8, bool)
        usable[[a, a + 2, a + 4, a + 6]] = True
        assert inner_from_partial_keypoints(kp, usable, img_w=IMG_WH[0]) is None


def test_vanishing_line_geometry_is_rejected_not_wrapped():
    """Near edge-on, a derived corner crosses the vanishing line: w -> 0 then NEGATIVE and the
    corner WRAPS to the far side of the image. Must return None rather than a plausible-looking
    but meaningless pixel (the silent-label-poisoning failure the gate labeler hit)."""
    R, tv = _pose(np.radians(88.5), 0.0, (0.0, 0.0, 1.6))
    kp = _project8(R, tv)
    usable = np.zeros(8, bool)
    usable[[0, 1, 4, 5]] = True
    got = inner_from_partial_keypoints(kp, usable, img_w=IMG_WH[0])
    assert got is None or np.abs(got).max() <= 4.0 * IMG_WH[0]


def test_non_finite_input_is_rejected():
    R, tv = _pose()
    kp = _project8(R, tv)
    kp[4] = [np.nan, np.nan]
    usable = np.zeros(8, bool)
    usable[[0, 1, 4, 5]] = True
    assert inner_from_partial_keypoints(kp, usable, img_w=IMG_WH[0]) is None


# --------------------------------------------------------------------------- emission path

def _arrays(kp, kconf, oconf):
    return (kp[None, :N_CORNERS, :], np.asarray(kconf)[None, :], np.array([0.9]),
            kp[None, N_CORNERS:, :], np.asarray(oconf)[None, :])


def test_rescue_emits_where_the_pipeline_used_to_drop():
    R, tv = _pose(0.3, 0.1, (0.8, -0.3, 3.0))
    kp = _project8(R, tv)
    xy, kc, sc, oxy, oc = _arrays(kp, [0.9, 0.9, 0.0, 0.0], [0.9, 0.9, 0.0, 0.0])
    kw = dict(bboxes_xywh=None, outer_xy=oxy, outer_conf=oc, image_wh=IMG_WH)

    assert observations_from_keypoints(_frame(), xy, kc, sc, partial_rescue=False, **kw) == []
    got = observations_from_keypoints(_frame(), xy, kc, sc, partial_rescue=True, **kw)
    assert len(got) == 1
    o = got[0]
    assert np.allclose(o.corners_px, kp[:N_CORNERS], atol=1e-6)
    assert o.corner_ids is None
    assert np.allclose(o.corner_confidence, _PARTIAL_RESCUE_CONF)
    # outer measurements are already baked into the derived corners; re-offering them would
    # double-count the same measurement in the joint refine
    assert o.outer_corners_px is None
    pose = estimate_gate_pose(o)
    assert pose is not None
    assert np.linalg.norm(pose.t_cam_gate - tv) < 0.05


def test_rescue_is_marked_and_reports_reduced_trust():
    """A rescue must NOT reach consumers looking like an ordinary 4-corner fix.

    The RL obs contract keys off exactly these two fields: n_corners ("3 => trust less") and a
    None visible_area_ratio ("cropped gate -> mask on it"). A rescued gate IS cropped and IS ~3x
    noisier, so both must say so -- otherwise the policy weights it like a full-quality fix."""
    R, tv = _pose(0.3, 0.1, (0.8, -0.3, 3.0))
    kp = _project8(R, tv)
    xy, kc, sc, oxy, oc = _arrays(kp, [0.9, 0.9, 0.0, 0.0], [0.9, 0.9, 0.0, 0.0])
    o = observations_from_keypoints(_frame(), xy, kc, sc, bboxes_xywh=None, outer_xy=oxy,
                                    outer_conf=oc, partial_rescue=True, image_wh=IMG_WH)[0]
    assert o.derived_corners is True
    assert o.inner_area_px is None
    assert o.visible_area_ratio is None
    assert estimate_gate_pose(o).n_corners == 3


def test_ordinary_observations_keep_full_trust():
    """The marking must not leak onto measured detections."""
    R, tv = _pose(0.25, 0.1, (0.6, -0.2, 5.5))
    kp = _project8(R, tv)
    xy, kc, sc, oxy, oc = _arrays(kp, [0.9] * 4, [0.9] * 4)
    o = observations_from_keypoints(_frame(), xy, kc, sc, bboxes_xywh=None, outer_xy=oxy,
                                    outer_conf=oc, partial_rescue=True, image_wh=IMG_WH)[0]
    assert o.derived_corners is False
    assert o.inner_area_px is not None
    assert o.visible_area_ratio is not None
    assert estimate_gate_pose(o).n_corners == 4


def test_border_clamped_keypoints_are_excluded():
    """ultralytics clips keypoints to the image, so a border-pinned point is not a measurement.
    Four keypoints that are all clamped must NOT be treated as a usable set."""
    kp = np.array([[0.0, 0.0], [639.99, 0.0], [639.99, 359.99], [0.0, 359.99],
                   [0.0, 0.0], [639.99, 0.0], [639.99, 359.99], [0.0, 359.99]])
    xy, kc, sc, oxy, oc = _arrays(kp, [0.9, 0.9, 0.0, 0.0], [0.9, 0.9, 0.0, 0.0])
    got = observations_from_keypoints(_frame(), xy, kc, sc, bboxes_xywh=None, outer_xy=oxy,
                                      outer_conf=oc, partial_rescue=True, image_wh=IMG_WH)
    assert got == []


def test_rescue_requires_image_wh():
    """Without the image size we cannot tell a clamped keypoint from a real one -> stay conservative."""
    R, tv = _pose(0.3, 0.1, (0.8, -0.3, 3.0))
    kp = _project8(R, tv)
    xy, kc, sc, oxy, oc = _arrays(kp, [0.9, 0.9, 0.0, 0.0], [0.9, 0.9, 0.0, 0.0])
    assert observations_from_keypoints(_frame(), xy, kc, sc, bboxes_xywh=None, outer_xy=oxy,
                                       outer_conf=oc, partial_rescue=True, image_wh=None) == []


@pytest.mark.parametrize("kconf,oconf", [
    ([0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9]),      # full 4-corner emission
    ([0.9, 0.9, 0.9, 0.0], [0.9, 0.9, 0.9, 0.9]),      # 3-corner + outer-assisted
    ([0.9, 0.9, 0.0, 0.0], [0.9, 0.9, 0.9, 0.9]),      # the legacy 4-outer rescue
    ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),      # nothing usable -> still dropped
])
def test_preexisting_emissions_are_unchanged(kconf, oconf):
    """The rescue is additive: it may only fire where the pipeline previously emitted NOTHING."""
    R, tv = _pose(0.25, 0.1, (0.6, -0.2, 5.5))
    kp = _project8(R, tv)
    xy, kc, sc, oxy, oc = _arrays(kp, kconf, oconf)
    kw = dict(bboxes_xywh=None, outer_xy=oxy, outer_conf=oc, image_wh=IMG_WH)
    before = observations_from_keypoints(_frame(), xy, kc, sc, partial_rescue=False, **kw)
    after = observations_from_keypoints(_frame(), xy, kc, sc, partial_rescue=True, **kw)
    if not before:
        return                                     # rescue is free to add here
    assert len(after) == len(before)
    for a, b in zip(before, after):
        assert np.array_equal(np.asarray(a.corners_px), np.asarray(b.corners_px))
        assert np.array_equal(np.asarray(a.corner_confidence), np.asarray(b.corner_confidence))
        assert (a.outer_corners_px is None) == (b.outer_corners_px is None)


def test_four_keypoint_model_is_untouched():
    """A 4-keypoint (inner-only) model has no outer corners -> the rescue can never engage."""
    R, tv = _pose()
    inner = project_gate_corners(R, tv)
    xy = inner[None, :, :]
    kc = np.array([[0.9, 0.9, 0.0, 0.0]])
    got = observations_from_keypoints(_frame(), xy, kc, np.array([0.9]), bboxes_xywh=None,
                                      partial_rescue=True, image_wh=IMG_WH)
    assert got == []


def test_derived_pose_matches_the_full_corner_pose():
    """The rescue must not shift the gate: same pose as the untruncated observation."""
    R, tv = _pose(0.4, -0.2, (1.0, 0.4, 4.5))
    kp = _project8(R, tv)
    xy, kc, sc, oxy, oc = _arrays(kp, [0.9, 0.9, 0.0, 0.0], [0.9, 0.9, 0.0, 0.0])
    o = observations_from_keypoints(_frame(), xy, kc, sc, bboxes_xywh=None, outer_xy=oxy,
                                    outer_conf=oc, partial_rescue=True, image_wh=IMG_WH)[0]
    got = estimate_gate_pose(o)
    full_xy, full_kc, full_sc, full_oxy, full_oc = _arrays(kp, [0.9] * 4, [0.9] * 4)
    ref_obs = observations_from_keypoints(_frame(), full_xy, full_kc, full_sc, bboxes_xywh=None,
                                          outer_xy=full_oxy, outer_conf=full_oc, image_wh=IMG_WH)[0]
    ref = estimate_gate_pose(ref_obs)
    assert got is not None and ref is not None
    assert np.linalg.norm(got.t_cam_gate - ref.t_cam_gate) < 0.05


def test_object_points_convention_is_the_shared_one():
    """Pin the assumption the whole derivation rests on: outer/inner = 2.72/1.5, concentric."""
    inner, outer = gate_object_points(), gate_object_points(GATE_OUTER_SIZE_M)
    assert np.allclose(inner.mean(axis=0), 0.0)
    assert np.allclose(outer.mean(axis=0), 0.0)
    ratio = np.linalg.norm(outer, axis=1) / np.linalg.norm(inner, axis=1)
    assert np.allclose(ratio, GATE_OUTER_SIZE_M / 1.5)
