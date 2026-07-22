"""Tests for the additive gate-CENTRE keypoint label (M+1's 5th keypoint).

The load-bearing properties: the centre is the EXACT projected gate origin (diagonal intersection of
the inner corners) even when corners are off-frame; it rides the augmentation warp because it is a
projective invariant; its visibility is the in-frame test on the centre itself; and the sidecar
round-trips. These are exactly the failure modes the M+1 brief flags as silent poison.
"""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from racer.frames import CAMERA_INTRINSICS_K
from racer.vision.gate_pose import project_gate_corners
from racer.vision.blender_gen.contract import IMAGE_HEIGHT, IMAGE_WIDTH, V_OFF, V_VIS
from racer.vision.blender_gen.center_label import (
    centre_from_inner_px,
    centre_visibility,
    decode_centre_row,
    encode_centre_row,
    frame_centre_rows,
    gate_centre_row,
)
from racer.vision.blender_gen.geometry import GateRender


def _projected_centre(R, t):
    """The true projected gate-frame origin (0,0,0) -- the definition of the gate centre."""
    c = CAMERA_INTRINSICS_K @ np.asarray(t, float)
    return c[:2] / c[2]


def test_centre_equals_projected_origin_in_frame():
    # a plain frontal gate: the diagonal intersection must equal the projected gate origin.
    R = np.eye(3)
    t = np.array([0.4, -0.2, 6.0])
    inner = project_gate_corners(R, t, 1.5)
    cx, cy, v = centre_from_inner_px(inner)
    assert np.allclose([cx, cy], _projected_centre(R, t), atol=1e-6)
    assert v == V_VIS


def test_centre_exact_even_with_offframe_corners():
    """The whole reason M+1 exists: a CROPPED gate whose inner corners leave the frame but whose
    centre is still visible. The diagonal intersection must land on the true projected centre to
    sub-pixel -- NOT drift toward the visible corners, NOT clamp to the border."""
    R = cv2.Rodrigues(np.array([0.12, 0.28, 0.0]))[0]
    t = np.array([1.3, -0.6, 2.5])
    inner = project_gate_corners(R, t, 1.5)
    n_in = int(((inner[:, 0] >= 0) & (inner[:, 0] <= IMAGE_WIDTH - 1)
                & (inner[:, 1] >= 0) & (inner[:, 1] <= IMAGE_HEIGHT - 1)).sum())
    assert n_in < 4, "test pose must actually crop a corner off-frame"
    true_c = _projected_centre(R, t)
    assert 0 <= true_c[0] <= IMAGE_WIDTH - 1 and 0 <= true_c[1] <= IMAGE_HEIGHT - 1
    cx, cy, v = centre_from_inner_px(inner)
    assert np.allclose([cx, cy], true_c, atol=1e-4)      # exact despite off-frame corners
    assert v == V_VIS


def test_centre_offframe_gets_v_off():
    """Looking THROUGH the gate: the centre itself leaves the frame -> v=0 (the ~11% case)."""
    # inner corners pushed so the diagonal crossing lands past the right edge.
    inner = np.array([[600, 300], [900, 300], [900, 60], [620, 60]], float)  # LL,LR,UR,UL
    cx, cy, v = centre_from_inner_px(inner)
    assert cx > IMAGE_WIDTH - 1
    assert v == V_OFF
    # the coordinate is preserved (unclamped) so the off-frame ground truth survives
    assert centre_visibility(cx, cy) == V_OFF


def test_centre_is_projective_invariant_under_warp():
    """The centre must ride the albumentations geometric warp. It does BECAUSE it is a projective
    invariant: for any homography H (affine and perspective augments are homographies),
    intersect(H @ corners) == H @ intersect(corners). Verify to sub-pixel for a real perspective H,
    which is what guarantees the post-augment centre stays aligned with the warped image."""
    inner = np.array([[120, 250], [360, 250], [360, 60], [120, 60]], float)
    c0 = np.array(centre_from_inner_px(inner)[:2])
    # a non-affine perspective homography (mimics A.Perspective)
    H = np.array([[1.02, 0.05, 12.0],
                  [-0.03, 0.97, -8.0],
                  [1.2e-4, 3.0e-4, 1.0]])
    def warp(pts):
        p = np.c_[pts, np.ones(len(pts))] @ H.T
        return p[:, :2] / p[:, 2:3]
    warped_inner = warp(inner)
    c1 = np.array(centre_from_inner_px(warped_inner)[:2])
    assert np.allclose(c1, warp(c0[None])[0], atol=1e-4)


def test_sidecar_row_roundtrip():
    cx, cy, v = 512.4, 173.9, V_VIS
    row = encode_centre_row(cx, cy, v)
    assert len(row.split()) == 3
    dcx, dcy, dv = decode_centre_row(row)
    assert dv == v and abs(dcx - cx) < 0.05 and abs(dcy - cy) < 0.05


def test_sidecar_offframe_is_unclamped():
    """An off-frame centre must NOT be clamped in the sidecar -- the builder re-clamps for the pose
    row, but the lossless sidecar preserves the true off-frame ground truth for evaluation."""
    row = encode_centre_row(820.0, 175.0, V_OFF)          # x > W
    dcx, _dcy, dv = decode_centre_row(row)
    assert dcx > IMAGE_WIDTH and dv == V_OFF


def _gate_render(R, t):
    inner = project_gate_corners(R, t, 1.5)
    outer = project_gate_corners(R, t, 2.72)
    return GateRender(gate_id=1, R_cam_gate=R, t_cam_gate=t, keypoints_px=inner, outer_px=outer,
                      bbox_xywh=np.array([0, 0, 10, 10], float),
                      visibility=np.full(4, V_VIS), visible=True)


def test_gate_centre_row_reads_keypoints_px():
    """gate_centre_row must read gr.keypoints_px (the post-augment unclamped corners), so the centre
    stays aligned with the pose row that is written from the SAME GateRender."""
    R = cv2.Rodrigues(np.array([0.05, 0.1, 0.0]))[0]
    t = np.array([0.3, -0.1, 5.0])
    gr = _gate_render(R, t)
    dcx, dcy, dv = decode_centre_row(gate_centre_row(gr))
    assert np.allclose([dcx, dcy], _projected_centre(R, t), atol=0.05)
    assert dv == V_VIS


def test_frame_centre_rows_is_one_per_gate_in_order():
    """The sidecar must be 1:1 and in order with the pose rows, or the builder pairs gate A's
    corners with gate B's centre."""
    grs = [_gate_render(np.eye(3), np.array([0.0, 0.0, 5.0])),
           _gate_render(np.eye(3), np.array([2.0, 0.5, 8.0]))]
    rows = frame_centre_rows(grs)
    assert len(rows) == len(grs)
    # first gate centred in frame -> its centre row is the image centre
    dcx, dcy, dv = decode_centre_row(rows[0])
    assert abs(dcx - IMAGE_WIDTH / 2) < 1.0 and abs(dcy - IMAGE_HEIGHT / 2) < 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
