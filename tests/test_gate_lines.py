"""Line-based gate solver: pin the geometry that the classical front-end feeds.

The mask front-end is known-fragile (see gate_lines' module docstring); these tests cover the part
that is validated -- line construction, the canonical sign that makes signed point-line distance
meaningful, border rejection, and homography-from-lines recovering the true gate centre.
"""
import cv2
import numpy as np

from racer.vision.gate_lines import (
    GATE_INNER_HALF,
    GATE_OUTER_HALF,
    _line,
    _merge_collinear,
    _on_border,
    homography_from_lines,
)

IMG_WH = (640, 360)


def _plane_to_img(H, X, Y):
    v = H @ np.array([X, Y, 1.0])
    return v[:2] / v[2]


def _segments_for(H, half):
    """The 4 edges of the square at +-half, as image segments (two points each)."""
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    px = [_plane_to_img(H, X, Y) for X, Y in corners]
    return [(px[i], px[(i + 1) % 4]) for i in range(4)]


def _pose_H(yaw=0.3, pitch=-0.15, t=(0.4, 0.2, 6.0)):
    """Gate-plane -> image homography for a gate at the given pose (pinhole, fx=fy=320)."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ \
        np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    K = np.array([[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1.0]])
    M = np.column_stack([R[:, 0], R[:, 1], np.asarray(t, float)])
    H = K @ M
    return H / H[2, 2]


def test_line_sign_is_canonical():
    """l and -l are the same line, so signed point-line distance is meaningless until the sign is
    pinned. Without this, opposite edges (equidistant from the centroid, opposite sides) merge."""
    a = _line(np.array([10.0, 10.0]), np.array([100.0, 10.0]))
    b = _line(np.array([100.0, 10.0]), np.array([10.0, 10.0]))   # same line, reversed
    assert np.allclose(a, b, atol=1e-9)
    assert a[1] >= -1e-12


def test_merge_collinear_keeps_opposite_edges_apart():
    ref = np.array([50.0, 50.0])
    left = _line(np.array([10.0, 0.0]), np.array([10.0, 100.0]))
    right = _line(np.array([90.0, 0.0]), np.array([90.0, 100.0]))
    dup = _line(np.array([10.3, 20.0]), np.array([10.3, 80.0]))   # same edge as `left`
    kept = _merge_collinear([left, right, dup], ref)
    assert len(kept) == 2, "opposite edges must survive; only the duplicate collapses"


def test_border_segments_are_rejected():
    w, h = IMG_WH
    assert _on_border(np.array([0.0, 5.0]), np.array([0.0, 300.0]), w, h)
    assert _on_border(np.array([5.0, h - 1.0]), np.array([600.0, h - 1.0]), w, h)
    assert not _on_border(np.array([50.0, 40.0]), np.array([300.0, 200.0]), w, h)


def test_homography_from_lines_recovers_the_centre():
    """The whole point: 4+ identified edge lines fix the plane, so the centre needs no corner."""
    for yaw, pitch, t in ((0.0, 0.0, (0.0, 0.0, 6.0)),
                          (0.4, -0.2, (1.0, 0.3, 4.0)),
                          (-0.5, 0.25, (-1.2, -0.4, 8.0))):
        H = _pose_H(yaw, pitch, t)
        got = homography_from_lines(_segments_for(H, GATE_INNER_HALF),
                                    _segments_for(H, GATE_OUTER_HALF), image_wh=IMG_WH)
        assert got is not None, f"no homography for {yaw},{pitch},{t}"
        truth = _plane_to_img(H, 0.0, 0.0)
        assert np.linalg.norm(_plane_to_img(got, 0.0, 0.0) - truth) < 1.0


def test_outer_edges_alone_are_enough():
    """A cropped gate often shows no closed opening; 4 outer edges still determine the plane."""
    H = _pose_H(0.25, -0.1, (0.5, 0.2, 5.0))
    got = homography_from_lines([], _segments_for(H, GATE_OUTER_HALF), image_wh=IMG_WH)
    assert got is not None
    assert np.linalg.norm(_plane_to_img(got, 0.0, 0.0) - _plane_to_img(H, 0.0, 0.0)) < 1.0


def test_too_few_lines_returns_none():
    H = _pose_H()
    assert homography_from_lines(_segments_for(H, GATE_INNER_HALF)[:1], [], image_wh=IMG_WH) is None
    assert homography_from_lines([], [], image_wh=IMG_WH) is None
