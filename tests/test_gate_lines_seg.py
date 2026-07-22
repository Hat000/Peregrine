"""The segmentation front-end for the line solver: masks -> edge lines -> gate centre.

The line/homography half is already pinned by test_gate_lines. What is new here is the MASK path,
so these tests render masks from a known homography and check the solver recovers that gate's
centre -- including the case the whole second path exists for, a gate cropped by the frame edge.

Run: python -m pytest tests/test_gate_lines_seg.py -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.vision.gate_lines import (  # noqa: E402
    GATE_INNER_HALF,
    GATE_OUTER_HALF,
    centre_from_seg_masks,
    pair_gate_instances,
    segments_from_mask,
)

IMG_WH = (640, 360)


def _pose_H(yaw=0.3, pitch=-0.15, t=(0.4, 0.2, 6.0)):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ \
        np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    K = np.array([[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1.0]])
    H = K @ np.column_stack([R[:, 0], R[:, 1], np.asarray(t, float)])
    return H / H[2, 2]


def _px(H, X, Y):
    v = H @ np.array([X, Y, 1.0])
    return v[:2] / v[2]


def _quad(H, half):
    return np.array([_px(H, -half, half), _px(H, half, half),
                     _px(H, half, -half), _px(H, -half, -half)])


def _mask(H, half, wh=IMG_WH):
    m = np.zeros((wh[1], wh[0]), np.uint8)
    cv2.fillPoly(m, [np.round(_quad(H, half)).astype(np.int32)], 1)
    return m.astype(bool)


def test_segments_from_mask_finds_four_edges():
    segs = segments_from_mask(_mask(_pose_H(), GATE_OUTER_HALF))
    assert len(segs) == 4, f"a fully-visible square has 4 edges, got {len(segs)}"


def test_speck_mask_is_ignored():
    m = np.zeros((360, 640), bool)
    m[10:16, 10:16] = True
    assert segments_from_mask(m) == []


def test_centre_from_both_masks():
    for yaw, pitch, t in ((0.0, 0.0, (0.0, 0.0, 6.0)),
                          (0.4, -0.2, (1.0, 0.3, 4.0)),
                          (-0.5, 0.25, (-1.2, -0.4, 8.0))):
        H = _pose_H(yaw, pitch, t)
        got = centre_from_seg_masks(_mask(H, GATE_OUTER_HALF), _mask(H, GATE_INNER_HALF), IMG_WH)
        assert got is not None, f"no fit for {yaw},{pitch},{t}"
        assert np.linalg.norm(got[0] - _px(H, 0.0, 0.0)) < 2.0


def test_frame_mask_alone_is_enough():
    """An oblique gate can show no open interior; 4 outer edges still determine the plane."""
    H = _pose_H(0.25, -0.1, (0.5, 0.2, 5.0))
    got = centre_from_seg_masks(_mask(H, GATE_OUTER_HALF), None, IMG_WH)
    assert got is not None
    assert np.linalg.norm(got[0] - _px(H, 0.0, 0.0)) < 2.0


def test_offframe_outer_corners_still_fit():
    """The keypoint path's failure mode, handled: the OUTER corners leave the frame (ultralytics
    would clamp them to the border, 12.3 px of pure error), while the inner square stays visible.
    Lines are fixed by the in-frame pixels they pass through, so the plane still solves."""
    H = _pose_H(0.0, 0.0, (2.5, 0.0, 3.6))
    outer, inner = _mask(H, GATE_OUTER_HALF), _mask(H, GATE_INNER_HALF)
    assert outer[:, -1].any(), "outer must actually run off the right edge"
    assert not inner[:, -1].any(), "inner must stay inside the frame"
    got = centre_from_seg_masks(outer, inner, IMG_WH)
    assert got is not None
    assert np.linalg.norm(got[0] - _px(H, 0.0, 0.0)) < 8.0


def test_cropped_on_two_edges_is_a_known_gap():
    """Pins the documented limitation (see homography_from_lines' KNOWN GAP), so that if a future
    change makes this case fit, the gap note gets revisited rather than quietly going stale.

    Cropping on two edges leaves ONE line per square per pencil, and a square only contributes when
    both its extremes are present. Two attempts to relax that were measured NET-NEGATIVE on real
    frames; the sign is not recoverable from the lines alone."""
    H = _pose_H(0.0, 0.0, (2.6, 0.9, 3.2))
    outer = _mask(H, GATE_OUTER_HALF)
    assert outer[:, -1].any() and outer[-1, :].any(), "fixture must be cropped on TWO edges"
    assert centre_from_seg_masks(outer, _mask(H, GATE_INNER_HALF), IMG_WH) is None


def test_pairing_matches_opening_to_its_containing_frame():
    """Two gates in view: each opening must attach to the frame it sits INSIDE, not the nearer one."""
    Ha = _pose_H(0.0, 0.0, (-1.4, 0.0, 6.0))
    Hb = _pose_H(0.0, 0.0, (1.4, 0.0, 6.0))
    pairs = pair_gate_instances([_mask(Ha, GATE_OUTER_HALF), _mask(Hb, GATE_OUTER_HALF)],
                                [_mask(Hb, GATE_INNER_HALF), _mask(Ha, GATE_INNER_HALF)])
    assert len(pairs) == 2 and all(o is not None for _, o in pairs)
    for f, o in pairs:
        ys, xs = np.nonzero(o)
        assert f[int(ys.mean()), int(xs.mean())], "opening attached to the wrong frame"


def test_frame_without_an_opening_is_still_returned():
    pairs = pair_gate_instances([_mask(_pose_H(), GATE_OUTER_HALF)], [])
    assert len(pairs) == 1 and pairs[0][1] is None
