"""Hull-quad mask fit + diagonal-intersection centre: the alternative to the line solver.

The line path's tests pin its geometry; these pin what is DIFFERENT here. Two claims are being
made and each gets a test that could fail:

  * the centre of a projected square is EXACTLY the intersection of its diagonals, for any
    homography, including when that intersection lands off-screen;
  * a hull reduced to 4 vertices is insensitive to boundary raggedness, which is the specific defect
    of an upsampled yolo-seg prototype mask.

⚠ These are SYNTHETIC fixtures and they are not the evidence that matters. On this codebase a
line-solver "improvement" passed its unit tests while costing 12x coverage on real frames
(82.4% -> 6.8%). scripts/eval_centre_ab.py is the arbiter; this file only stops the geometry from
silently rotting.

Run: python -m pytest tests/test_quad_centre.py -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.vision.gate_lines import (  # noqa: E402
    GATE_INNER_HALF,
    GATE_OUTER_HALF,
    centre_from_masks_quad,
    centre_from_quad,
    quad_from_mask,
    quad_from_mask_ex,
)

IMG_WH = (640, 360)


def _pose_H(yaw=0.3, pitch=-0.15, t=(0.4, 0.2, 6.0)):
    """Gate-plane -> image homography for a camera looking at a gate at ``t``."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ \
        np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    K = np.array([[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1.0]])
    H = K @ np.column_stack([R[:, 0], R[:, 1], np.asarray(t, float)])
    return H / H[2, 2]


def _plane_rot(ang):
    """Rotate the gate IN ITS OWN PLANE (a square symmetry, so the centre is untouched) -- used to
    get a corner-cut crop instead of a whole-side crop."""
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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


def test_diagonals_recover_the_plane_origin_exactly():
    """The whole claim, over random homographies: no PnP, no intrinsics, no corner identity."""
    rng = np.random.default_rng(20260722)
    worst = 0.0
    for _ in range(300):
        H = _pose_H(float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-0.7, 0.7)),
                    (float(rng.uniform(-3, 3)), float(rng.uniform(-2, 2)),
                     float(rng.uniform(1.5, 20.0))))
        c = centre_from_quad(_quad(H, GATE_INNER_HALF))
        assert c is not None
        worst = max(worst, float(np.linalg.norm(c - _px(H, 0.0, 0.0))))
    assert worst < 1e-6, f"diagonal intersection is not exact: {worst} px"


def test_corner_relabelling_does_not_move_the_centre():
    """Any cyclic relabelling names the same two diagonals, so corner IDENTITY is not needed."""
    q = _quad(_pose_H(0.5, -0.3, (1.0, 0.4, 5.0)), GATE_INNER_HALF)
    base = centre_from_quad(q)
    for k in (1, 2, 3):
        assert np.linalg.norm(centre_from_quad(np.roll(q, k, axis=0)) - base) < 1e-9
    assert np.linalg.norm(centre_from_quad(q[::-1]) - base) < 1e-9   # and reversed winding


def test_centre_off_screen_is_still_solved():
    """The case the keypoint path cannot represent AT ALL: the gate centre is outside the image."""
    H = _pose_H(0.0, 0.0, (3.4, 0.6, 3.0))
    truth = _px(H, 0.0, 0.0)
    assert not (0 <= truth[0] < IMG_WH[0]), f"fixture must be off-screen, got {truth}"
    c = centre_from_quad(_quad(H, GATE_INNER_HALF))
    assert c is not None and np.linalg.norm(c - truth) < 1e-6


def test_edge_on_square_has_no_centre():
    """Diagonals parallel = the centre is genuinely at infinity. Refuse it, do not return a number."""
    q = np.array([[100.0, 100.0], [300.0, 100.0], [300.0, 200.0], [100.0, 200.0]])
    q = np.array([q[0], q[1], q[0] + (q[1] - q[0]) * 2.0, q[1] + (q[1] - q[0]) * 2.0])
    assert centre_from_quad(q) is None


def test_clean_mask_recovers_the_corners():
    for yaw, pitch, t in ((0.0, 0.0, (0.0, 0.0, 6.0)),
                          (0.4, -0.2, (1.0, 0.3, 4.0)),
                          (-0.5, 0.25, (-1.2, -0.4, 8.0))):
        H = _pose_H(yaw, pitch, t)
        got = quad_from_mask(_mask(H, GATE_INNER_HALF))
        assert got is not None, f"no quad for {yaw},{pitch},{t}"
        # order-free comparison: each true corner must have a fitted corner near it
        for p in _quad(H, GATE_INNER_HALF):
            assert np.min(np.linalg.norm(got - p, axis=1)) < 2.0


def _ragged(mask, rng, band_px=5, p_flip=0.45):
    """Chew the boundary the way an upsampled prototype mask is chewed: flip pixels at random in a
    band straddling the true edge, so the outline becomes a coastline instead of a line."""
    m = mask.astype(np.uint8)
    k = np.ones((band_px, band_px), np.uint8)
    band = (cv2.dilate(m, k) - cv2.erode(m, k)).astype(bool)
    flip = band & (rng.random(m.shape) < p_flip)
    out = mask.copy()
    out[flip] = ~out[flip]
    return out


def test_ragged_boundary_still_recovers_corners_and_centre():
    """The reason the hull is used at all: every boundary pixel votes, so the wobble averages out.

    Corners tolerate more error than the centre because the convex hull is an OUTER bound -- noise
    can only push a corner outward. It pushes every side outward roughly equally, so the CENTRE
    error stays far smaller than the corner error, which is the property being relied on."""
    rng = np.random.default_rng(7)
    corner_worst, centre_worst = 0.0, 0.0
    for yaw, pitch, t in ((0.0, 0.0, (0.0, 0.0, 6.0)),
                          (0.35, -0.2, (0.8, 0.3, 5.0)),
                          (-0.45, 0.2, (-1.0, -0.3, 7.0))):
        H = _pose_H(yaw, pitch, t)
        got = quad_from_mask(_ragged(_mask(H, GATE_INNER_HALF), rng))
        assert got is not None
        for p in _quad(H, GATE_INNER_HALF):
            corner_worst = max(corner_worst, float(np.min(np.linalg.norm(got - p, axis=1))))
        centre_worst = max(centre_worst,
                           float(np.linalg.norm(centre_from_quad(got) - _px(H, 0.0, 0.0))))
    assert corner_worst < 8.0, f"corner drift {corner_worst:.2f} px under a 5 px ragged band"
    assert centre_worst < 3.0, f"centre drift {centre_worst:.2f} px under a 5 px ragged band"


def test_cropped_corner_is_reconstructed_not_fitted():
    """A hull cut by the frame edge carries a BORDER edge. Deleting it first extends the two real
    gate edges until they meet, which puts the corner back where the gate actually is -- rather than
    on the image border, which is the keypoint path's characteristic error."""
    # Rotated 45 deg in its own plane so the frame edge cuts ONE corner off, not a whole side --
    # with a whole side gone only 3 edges remain and nothing can be reconstructed (test below).
    H = _pose_H(0.0, 0.0, (3.2, 0.0, 4.0)) @ _plane_rot(np.pi / 4)
    m = _mask(H, GATE_INNER_HALF)
    assert m[:, -1].any(), "fixture must actually run off the right edge"
    got = quad_from_mask_ex(m)
    assert got is not None
    quad, n_border = got
    assert n_border == 0, "the border edge should have been merged away"
    assert quad[:, 0].max() > IMG_WH[0] + 15, "the reconstructed corner must sit OUTSIDE the frame"
    assert np.linalg.norm(centre_from_quad(quad) - _px(H, 0.0, 0.0)) < 4.0


def test_three_visible_edges_are_reported_as_clipped():
    """The method's honest limit: with only 3 gate edges visible the clipped hull is ALREADY a quad,
    nothing is merged, and the fit is the trapezoid. It must SAY so rather than pass as a clean fit."""
    H = _pose_H(0.0, 0.0, (2.6, 0.0, 3.2))
    got = quad_from_mask_ex(_mask(H, GATE_INNER_HALF))
    assert got is not None
    quad, n_border = got
    assert n_border >= 1, "a 3-edge crop must be flagged"
    err = float(np.linalg.norm(centre_from_quad(quad) - _px(H, 0.0, 0.0)))
    assert err > 2.0, "fixture is meant to be biased; if it is not, the limit has moved"
    # ...and a caller that refuses clipped fits gets nothing rather than a biased centre
    assert centre_from_masks_quad(_mask(H, GATE_OUTER_HALF), _mask(H, GATE_INNER_HALF),
                                  IMG_WH, max_border_edges=0) is None


def test_masks_quad_prefers_the_opening_and_falls_back_to_the_frame():
    H = _pose_H(0.3, -0.15, (0.5, 0.2, 5.5))
    truth = _px(H, 0.0, 0.0)
    both = centre_from_masks_quad(_mask(H, GATE_OUTER_HALF), _mask(H, GATE_INNER_HALF), IMG_WH)
    assert both is not None and np.linalg.norm(both[0] - truth) < 3.0
    # the opening's quad, not the frame's: its corners are the INNER square's
    for p in _quad(H, GATE_INNER_HALF):
        assert np.min(np.linalg.norm(both[1] - p, axis=1)) < 3.0
    alone = centre_from_masks_quad(_mask(H, GATE_OUTER_HALF), None, IMG_WH)
    assert alone is not None and np.linalg.norm(alone[0] - truth) < 3.0


def test_insane_centre_is_refused():
    """Same bound the line path applies: a collapsed fit reports a 'centre' tens of widths out, and
    without this it counts as coverage."""
    m = np.zeros((360, 640), bool)
    m[178:182, 100:540] = True          # a sliver: its quad is fine, its diagonals are near-parallel
    got = centre_from_masks_quad(m, None, IMG_WH)
    if got is not None:
        c = got[0]
        assert -3 * 640 <= c[0] <= 3 * 640 and abs(c[1] - 180) <= 3 * 360


def test_degenerate_inputs_return_none():
    assert quad_from_mask(np.zeros((360, 640), bool)) is None            # empty
    speck = np.zeros((360, 640), bool)
    speck[10:16, 10:16] = True
    assert quad_from_mask(speck) is None                                 # under the area floor
    line = np.zeros((360, 640), bool)
    line[180, 100:540] = True
    assert quad_from_mask(line) is None                                  # zero-area hull
    assert quad_from_mask(np.zeros((5, 360, 640), bool)) is None         # not 2-D
    assert centre_from_quad(np.zeros((3, 2))) is None                    # not 4 points
    assert centre_from_quad(np.full((4, 2), np.nan)) is None
    assert centre_from_quad(np.zeros((4, 2))) is None                    # all coincident
    assert centre_from_masks_quad(np.zeros((360, 640), bool), None, IMG_WH) is None
