"""Tests for MODEL FITTING (racer.vision.gate_model_fit).

⚠ THE LIMIT OF THESE TESTS, stated up front so nobody mistakes a green suite for evidence. Every
case here is SYNTHETIC: the mask is rasterised from the very model being fitted, so a passing
round-trip proves the geometry, the parameterisation and the search are self-consistent -- and
nothing about real predicted masks. This repo has been burned by exactly that gap once already: a
line-solver change passed its unit tests while causing a 12x REGRESSION on real frames (coverage
82.4% -> 6.8%). The evidence that counts is scripts/eval_centre_ab.py on the hand-labelled real
val split; these tests exist to stop a refactor silently breaking the machinery underneath it.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from racer.frames import CAMERA_INTRINSICS_K as K
from racer.vision.gate_model_fit import (
    DEFAULT_MODEL,
    FLAT_MODEL,
    GATE_DEPTH_M,
    GateModel,
    _clip_convex,
    centre_from_masks_model,
    fit_gate_model,
    project_silhouette,
    range_from_apparent_size,
)

W, H = 640, 360


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def rot(rx=0.0, ry=0.0, rz=0.0):
    return cv2.Rodrigues(np.array([rx, ry, rz], float))[0]


def masks_at(R, t, model=DEFAULT_MODEL, w=W, h=H):
    """Rasterise the model's true silhouette -- an IDEAL segmentation of that pose."""
    got = project_silhouette(R, t, K, model)
    assert got is not None, "test pose is unprojectable"
    fp, op = got
    fm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(fm, [np.asarray(fp).round().astype(np.int32)], 1)
    om = np.zeros((h, w), np.uint8)
    if len(op) >= 3:
        cv2.fillPoly(om, [np.asarray(op).round().astype(np.int32)], 1)
    return fm.astype(bool), om.astype(bool)


def centre_of(R, t):
    c = K @ np.asarray(t, float)
    return c[:2] / c[2]


def roughen(mask, seed=0, amp=3.0):
    """Boundary noise shaped like a yolo-seg prototype mask: a smooth random field pushing the
    silhouette in and out by a few pixels. NOT salt-and-pepper -- the real failure mode is a
    low-frequency wobble ('coastlines'), which is exactly what steers a per-segment line fit."""
    m8 = mask.astype(np.uint8) * 255
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, 1.0, (max(mask.shape[0] // 8, 2), max(mask.shape[1] // 8, 2))).astype(np.float32)
    n = cv2.resize(n, mask.shape[::-1], interpolation=cv2.INTER_CUBIC)
    d = (cv2.distanceTransform(m8, cv2.DIST_L2, 3)
         - cv2.distanceTransform(255 - m8, cv2.DIST_L2, 3))
    return (d + n * amp) > 0


# ------------------------------------------------------------------------------------------------
# the model itself
# ------------------------------------------------------------------------------------------------
def test_depth_constant_matches_the_blender_generator():
    """The duplicated 0.26 m must equal the canonical value it was copied from.

    gate_model_fit hard-codes GATE_DEPTH_M rather than importing blender_gen.contract, so that a
    flight-path module does not drag in the dataset/render stack. This is the guard that makes the
    duplicate safe: if the generator's depth ever changes, the label and the fitted model would
    silently describe different solids, and only this assert would notice."""
    from racer.vision.blender_gen.contract import GATE_DEPTH_M as CANON

    assert GATE_DEPTH_M == CANON


def test_model_vertices_match_the_blender_mesh():
    """The 16 fitted vertices ARE the 16 vertices Blender renders -- same set, same frame."""
    from racer.vision.blender_gen.bpy_scene import _gate_frame_geometry

    verts, _ = _gate_frame_geometry(DEFAULT_MODEL.inner_m, DEFAULT_MODEL.outer_m,
                                    DEFAULT_MODEL.depth_m)
    a = np.array(sorted(map(tuple, np.round(np.asarray(verts, float), 9))))
    b = np.array(sorted(map(tuple, np.round(DEFAULT_MODEL.verts, 9))))
    assert a.shape == (16, 3) and np.allclose(a, b)


def test_flat_model_is_the_two_projected_squares():
    """depth=0 must collapse to the flat concentric squares -- the shape today's seg labels encode."""
    R, t = rot(0.0, 0.25), np.array([0.4, -0.2, 6.0])
    fp, op = project_silhouette(R, t, K, FLAT_MODEL)
    h = FLAT_MODEL.outer_m / 2.0
    sq = np.array([[-h, h, 0.], [h, h, 0.], [h, -h, 0.], [-h, -h, 0.]])
    cam = sq @ R.T + t
    uv = (cam @ K.T)[:, :2] / (cam @ K.T)[:, 2:3]
    assert len(fp) == 4 and len(op) == 4
    # same point SET (convex hull reorders)
    assert np.allclose(np.array(sorted(map(tuple, np.round(fp, 4)))),
                       np.array(sorted(map(tuple, np.round(uv, 4)))), atol=1e-3)


def test_depth_makes_the_silhouette_bigger_and_the_hole_smaller():
    """The whole reason the 3-D model exists, and it is NOT only an off-axis effect.

    Even head-on, the prism's FRONT face is d/2 = 0.13 m nearer than the mid-plane the flat labels
    use, so it projects LARGER; and the see-through hole is limited by the BACK rim, which is d/2
    further away, so it projects SMALLER. Off-axis the gap widens further because the camera starts
    to see the side walls. Both effects are in the same direction, always -- which is exactly the
    systematic error the flat-quad seg labels carry at close range."""
    def area(p):
        p = np.asarray(p, float)
        return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                               - np.dot(p[:, 1], np.roll(p[:, 0], -1))))

    t = np.array([0.0, 0.0, 3.0])
    f0, o0 = project_silhouette(np.eye(3), t, K, DEFAULT_MODEL)
    ff, of = project_silhouette(np.eye(3), t, K, FLAT_MODEL)
    # head-on: the ratio is exactly the depth parallax on each square
    assert area(f0) / area(ff) == pytest.approx((3.0 / (3.0 - 0.13)) ** 2, rel=1e-3)
    assert area(o0) / area(of) == pytest.approx((3.0 / (3.0 + 0.13)) ** 2, rel=1e-3)

    # Off-axis the SIDE WALLS become visible, and the signature of that is the SHAPE, not the area:
    # the silhouette of the prism stops being a quadrilateral. (Area alone is not a clean test --
    # both squares foreshorten at once, so the depth/flat area ratio does not move monotonically.)
    R = rot(0.0, 0.5)
    f1, o1 = project_silhouette(R, t, K, DEFAULT_MODEL)
    f2, o2 = project_silhouette(R, t, K, FLAT_MODEL)
    assert len(f0) == 4 and len(ff) == 4                 # head-on: the back face hides exactly
    assert len(f1) > 4, "off-axis the outer silhouette must expose side walls"
    assert len(f2) == 4, "the flat model can never do that -- it has no side walls"
    assert area(f1) > area(f2) and area(o1) < area(o2)


def test_clip_convex_matches_an_independent_sutherland_hodgman():
    """gate_model_fit uses cv2.intersectConvexConvex for speed; this pins it to the textbook
    algorithm so a cv2 version bump cannot change the fitted shape unnoticed."""
    def ref(sub, clip):
        out = [np.asarray(p, float) for p in sub]
        n = len(clip)
        a2 = sum(clip[i][0] * clip[(i + 1) % n][1] - clip[(i + 1) % n][0] * clip[i][1]
                 for i in range(n))
        s = 1.0 if a2 >= 0 else -1.0
        for i in range(n):
            a, b = clip[i], clip[(i + 1) % n]
            e = b - a
            nxt = []
            for j in range(len(out)):
                cur, prv = out[j], out[j - 1]
                dc = s * (e[0] * (cur[1] - a[1]) - e[1] * (cur[0] - a[0]))
                dp = s * (e[0] * (prv[1] - a[1]) - e[1] * (prv[0] - a[0]))
                if dc >= 0:
                    if dp < 0:
                        nxt.append(prv + (cur - prv) * (dp / (dp - dc)))
                    nxt.append(cur)
                elif dp >= 0:
                    nxt.append(prv + (cur - prv) * (dp / (dp - dc)))
            out = nxt
            if not out:
                return np.empty((0, 2))
        return np.asarray(out, float)

    def area(p):
        p = np.asarray(p, float)
        if len(p) < 3:
            return 0.0
        return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                               - np.dot(p[:, 1], np.roll(p[:, 0], -1))))

    rng = np.random.default_rng(11)
    for _ in range(25):
        a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float) + rng.uniform(-3, 3, (4, 2))
        b = a + rng.uniform(-4, 4, (1, 2))
        assert area(_clip_convex(a, b)) == pytest.approx(area(ref(a, b)), abs=1e-3)


# ------------------------------------------------------------------------------------------------
# the fit
# ------------------------------------------------------------------------------------------------
def test_synthetic_pose_round_trip_recovers_the_centre():
    """Project the model at a known pose, rasterise, fit, recover the centre. The headline claim."""
    rng = np.random.default_rng(4)
    errs = []
    for _ in range(12):
        d = float(rng.uniform(3.0, 15.0))
        R = rot(rng.uniform(-.25, .25), rng.uniform(-.5, .5), rng.uniform(-.2, .2))
        t = np.array([rng.uniform(-.3, .3) * d, rng.uniform(-.2, .2) * d, d])
        fm, om = masks_at(R, t)
        if fm.sum() < 800:
            continue
        fit = fit_gate_model(fm, om, (W, H))
        assert fit is not None
        errs.append(float(np.linalg.norm(fit.centre_px - centre_of(R, t))))
    assert len(errs) >= 8
    # Sub-pixel MEDIAN. Not sub-pixel on every case: the objective is a rasterised IoU, so it is
    # piecewise constant and the last fraction of a pixel is genuinely unobservable.
    assert float(np.median(errs)) < 1.0, errs
    assert max(errs) < 6.0, errs


def test_an_off_screen_centre_is_emitted_not_collapsed_into_the_frame():
    """THE POINT OF THE METHOD. The hull-quad path structurally cannot emit an off-screen centre --
    once the centre leaves the frame the far edge is gone, the clipped hull is already a
    quadrilateral, nothing merges, and the fit is the visible trapezoid. The model's centre projects
    wherever the model puts it, so it can and must leave the frame."""
    R, t = rot(0.10, 0.0), np.array([0.1, 2.4, 3.4])
    c_true = centre_of(R, t)
    assert not (0 <= c_true[1] < H), f"test is not exercising an off-screen centre: {c_true}"
    fm, om = masks_at(R, t)
    assert fm.sum() > 2000
    fit = fit_gate_model(fm, om, (W, H))
    assert fit is not None
    assert not (0 <= fit.centre_px[1] < H), "fit collapsed the centre back into the frame"
    assert float(np.linalg.norm(fit.centre_px - c_true)) < 100.0


def test_a_cropped_gate_is_biased_inward_from_a_hull_quad_init():
    """THE HONEST LIMIT, pinned so a future change cannot quietly claim it away.

    On a cropped gate the fit is systematically biased back TOWARD the image. Measured on IDEAL
    masks rasterised from the model itself, so this is not a segmentation failure -- and it is not
    an evaluation-budget failure either: the search settles at IoU ~0.87-0.92 where the TRUE pose
    scores ~0.95-0.99, and raising the budget from 120 to 960 changes the answer by 1.2 px. It is a
    genuine LOCAL OPTIMUM reached from a hull-quad init that is itself 48-89 px out, because a
    clipped quad is the visible trapezoid rather than the gate.

    Two attempts to search past it were measured and BOTH made it worse -- a larger step (some fits
    collapsed to IoU 0.44) and a translation pre-search (p90 151.8 px vs 141.4 on real frames). The
    thing that DOES fix it is a better init, not a better search: see the next test.

    The assertion is deliberately weak -- it pins the DIRECTION and a generous bound, not a number
    to tune against."""
    R, t = rot(0.0, 0.0), np.array([3.4, 0.1, 3.0])
    c_true = centre_of(R, t)
    fm, om = masks_at(R, t)
    fit = fit_gate_model(fm, om, (W, H))
    assert fit is not None
    assert fit.centre_px[0] < c_true[0]                  # biased back toward the image
    assert float(np.linalg.norm(fit.centre_px - c_true)) < 130.0


def test_the_line_solver_init_fixes_the_cropped_case_on_ideal_masks():
    """``line_init=True`` seeds the search from the line solver's plane, which CAN solve a 3-edge
    crop (it has a second concentric square with a known metric ratio). On ideal masks that turns
    the previous test's failure into a near-exact fit.

    ⚠ AND YET IT IS OFF BY DEFAULT, which is the whole lesson of this file. On the 54 hand-labelled
    REAL gates the same switch is a wash (median 19.7 px with, 19.1 without; p90 136.7 vs 141.4),
    because the line solver's plane is itself unreliable when the masks are ragged -- that is its
    documented failure mode. A synthetic win of 48 px -> 4 px that does not appear on real frames is
    exactly the trap this repo has been caught by before, so the real measurement decides."""
    R, t = rot(0.0, 0.0), np.array([0.1, 2.0, 3.0])
    c_true = centre_of(R, t)
    fm, om = masks_at(R, t)
    plain = fit_gate_model(fm, om, (W, H))
    seeded = fit_gate_model(fm, om, (W, H), line_init=True)
    assert plain is not None and seeded is not None
    assert seeded.init_kind == "line-H"
    assert float(np.linalg.norm(seeded.centre_px - c_true)) < 10.0
    assert float(np.linalg.norm(plain.centre_px - c_true)) > 25.0
    assert seeded.score > plain.score


def test_cropped_mask_still_fits():
    """A gate cut by the image edge: most of the silhouette is gone, but the pose is still
    determined by the two concentric squares' known metric ratio."""
    R, t = rot(0.05, 0.2), np.array([0.9, 0.3, 2.2])
    fm, om = masks_at(R, t)
    assert fm[:, -1].any() or fm[:, 0].any() or fm[0].any() or fm[-1].any(), "mask is not cropped"
    fit = fit_gate_model(fm, om, (W, H))
    assert fit is not None
    assert float(np.linalg.norm(fit.centre_px - centre_of(R, t))) < 25.0


def test_gate_straddling_the_optical_plane_is_clipped_not_rejected():
    """Near-plane clipping, worth 7 of 54 gates on the real val split. A 2.72 m gate at <1 m has
    outer corners BESIDE and BEHIND the camera; rejecting the pose because one of 16 vertices is
    behind the optical plane threw away exactly the frames this path exists to serve."""
    R, t = rot(0.0, 0.9), np.array([0.2, 0.0, 0.75])
    cam = DEFAULT_MODEL.verts @ R.T + t
    assert cam[:, 2].min() < 0.0, "test pose does not actually straddle the optical plane"
    got = project_silhouette(R, t, K, DEFAULT_MODEL)
    assert got is not None and len(got[0]) >= 3

    fm, om = masks_at(R, t)
    fit = fit_gate_model(fm, om, (W, H))
    assert fit is not None


def test_gate_entirely_behind_the_camera_is_rejected():
    R, t = np.eye(3), np.array([0.0, 0.0, -5.0])
    assert project_silhouette(R, t, K, DEFAULT_MODEL) is None


def test_ragged_mask_does_not_derail_the_fit():
    """yolo-seg boundaries are ragged BY CONSTRUCTION (low-res prototype mask, upsampled) and
    mask-mAP barely penalises it. An AREA fit should average that wobble out rather than let one
    wandering boundary segment steer the answer -- that is the whole argument for this method."""
    R, t = rot(0.05, 0.3, -0.1), np.array([0.8, -0.3, 6.0])
    fm, om = masks_at(R, t)
    clean = fit_gate_model(fm, om, (W, H))
    assert clean is not None
    errs = []
    for seed in range(5):
        fit = fit_gate_model(roughen(fm, seed), roughen(om, seed + 100), (W, H))
        assert fit is not None
        errs.append(float(np.linalg.norm(fit.centre_px - centre_of(R, t))))
    assert float(np.median(errs)) < 12.0, errs


def test_degenerate_and_empty_masks_return_none():
    empty = np.zeros((H, W), bool)
    assert fit_gate_model(empty, None, (W, H)) is None
    assert fit_gate_model(None, None, (W, H)) is None
    assert centre_from_masks_model(empty, empty, (W, H)) is None
    speck = np.zeros((H, W), bool)
    speck[100:105, 100:105] = True                 # 25 px, below the 200 px^2 floor every arm uses
    assert fit_gate_model(speck, None, (W, H)) is None
    # a mask that is not a gate at all: a thin bar. It must not come back as a confident gate.
    bar = np.zeros((H, W), bool)
    bar[178:182, 40:600] = True
    got = fit_gate_model(bar, None, (W, H))
    assert got is None or got.score < 0.6


def test_opening_only_and_frame_only_both_work():
    """The seg model does not always predict both classes (an oblique gate's opening can close up),
    so neither may be assumed present."""
    R, t = rot(0.0, 0.2), np.array([0.0, 0.0, 7.0])
    fm, om = masks_at(R, t)
    assert fit_gate_model(fm, None, (W, H)) is not None
    assert fit_gate_model(None, om, (W, H)) is not None


# ------------------------------------------------------------------------------------------------
# what the fit emits, and which parts survive the 2-fold ambiguity
# ------------------------------------------------------------------------------------------------
def test_rel_position_is_range_times_bearing_through_the_centre_pixel():
    """The emitted rel-position must be built from the two AMBIGUITY-SAFE quantities -- the bearing
    ray through the fitted centre pixel and the range from apparent size -- and NOT from the fitted
    translation, whose magnitude is the one part of it the two IPPE branches can disagree about."""
    R, t = rot(0.05, 0.3), np.array([1.2, -0.4, 8.0])
    fm, om = masks_at(R, t)
    fit = fit_gate_model(fm, om, (W, H))
    assert fit is not None
    ray = np.array([(fit.centre_px[0] - K[0, 2]) / K[0, 0],
                    (fit.centre_px[1] - K[1, 2]) / K[1, 1], 1.0])
    ray /= np.linalg.norm(ray)
    assert np.allclose(fit.bearing_unit, ray, atol=1e-9)
    assert np.allclose(fit.rel_position_m, fit.range_m * ray, atol=1e-9)
    assert fit.range_from_size_ok
    # and it must be the right answer, not merely self-consistent
    assert abs(fit.range_m - float(np.linalg.norm(t))) < 0.6


def test_range_from_apparent_size_is_tilt_sign_invariant():
    """The 2-fold ambiguity flips the SIGN of the tilt, not its magnitude. Range built from
    apparent size therefore cannot see the difference -- which is the entire claim that makes it
    ambiguity-safe. (||t|| is not tested here because it is not claimed to be safe.)"""
    t = np.array([0.0, 0.0, 6.0])
    for tilt in (0.15, 0.35, 0.55):
        a = range_from_apparent_size(rot(0.0, tilt), t, K)
        b = range_from_apparent_size(rot(0.0, -tilt), t, K)
        assert a == pytest.approx(b, rel=1e-9)


def test_centre_is_invariant_to_the_gates_own_90_degree_symmetry():
    """A square annulus is unchanged by a quarter turn about its own axis, so relabelling which
    corner is 'lower-left' cannot move the centre. This is why the hull quad's arbitrary cyclic
    vertex order is safe to hand to IPPE."""
    t = np.array([0.5, -0.3, 7.0])
    R0 = rot(0.05, 0.25, 0.1)
    base = None
    for k in range(4):
        R = R0 @ rot(0.0, 0.0, k * np.pi / 2)
        fm, om = masks_at(R, t)
        fit = fit_gate_model(fm, om, (W, H))
        assert fit is not None
        if base is None:
            base = fit.centre_px
        assert float(np.linalg.norm(fit.centre_px - base)) < 2.0


def test_budget_is_respected():
    """A hard evaluation cap is what makes the worst-case wall time predictable in a flight loop."""
    R, t = rot(0.0, 0.3), np.array([0.0, 0.0, 6.0])
    fm, om = masks_at(R, t)
    for b in (20, 60, 120):
        fit = fit_gate_model(fm, om, (W, H), budget=b)
        assert fit is not None and fit.n_evals <= b + 2


def test_a_wrong_size_model_scores_worse():
    """Sanity that the score is actually about the KNOWN gate: fitting a model with the wrong
    metric ratio to a correct silhouette must not explain it as well."""
    R, t = rot(0.0, 0.2), np.array([0.0, 0.0, 6.0])
    fm, om = masks_at(R, t)
    good = fit_gate_model(fm, om, (W, H))
    bad = fit_gate_model(fm, om, (W, H), model=GateModel(inner_m=1.5, outer_m=1.9, depth_m=0.26))
    assert good is not None and bad is not None
    assert good.score > bad.score + 0.02
