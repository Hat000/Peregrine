"""MODEL FITTING -- fit the KNOWN 3-D gate to the predicted masks, instead of inferring the gate
FROM the mask boundary.

THE ARGUMENT. Both existing mask->centre paths (``gate_lines.homography_from_lines`` and
``gate_lines.quad_from_mask``) start by asking the mask a question it cannot answer: "which of your
boundary segments is a real gate edge?". On a cropped or ragged region that question is genuinely
ambiguous -- and yolo-seg boundaries are ragged BY CONSTRUCTION (a low-resolution prototype mask,
upsampled; mask-mAP barely penalises it -- 0.9855 on the v2 model while the outlines look like
coastlines), so nothing in training pushes back on the raggedness either.

Fitting AREA never asks that question. Take the gate we already know -- 1.5 m opening, 2.72 m outer
square, 0.26 m deep -- search over its 6-DoF pose, and maximise the overlap between its PROJECTED
SILHOUETTE and the predicted masks. Every pixel of both class masks votes on one pose. The emitted
centre is then read off the fitted model, which makes it exact by construction and, crucially, still
defined when it projects OUTSIDE the image -- the configuration the quad path cannot represent at
all (once the centre leaves the frame the far edge is gone, the clipped hull is already a
quadrilateral, nothing merges, and the fit is the visible trapezoid, biased inward).

WHAT IS AMBIGUITY-SAFE AND WHAT IS NOT -- read this before consuming any output.
  * CENTRE PIXEL: safe. Every relabelling confusion of a square acts as H -> H@R with R fixing the
    plane origin, so the projected centre is unchanged; and IPPE's 2-fold ambiguity is ambiguous
    precisely BECAUSE both poses project the same points to the same pixels.
  * BEARING: safe, and for the same reason. The two ambiguity solutions have PARALLEL translations
    (they must -- the gate origin projects to the same pixel in both), so the back-projected ray
    through the fitted centre pixel is identical either way.
  * RANGE: safe when taken from apparent SIZE (see :func:`range_from_apparent_size`), because the
    ambiguity flips the SIGN of the tilt, not its magnitude, so both branches foreshorten
    identically. NOT taken from ``||t||``: the fitted translation's magnitude is the one part of the
    translation the two branches can disagree about. Both are emitted (``range_m`` vs
    ``t_norm_m``) so a consumer can see the difference rather than inherit it silently.
  * ORIENTATION (the gate normal / relative yaw): the ONLY genuinely ambiguous output left. That is
    what the 0.26 m depth is asked to disambiguate here -- see ``score_pose`` and the ambiguity
    experiment in ``scripts/eval_centre_ab.py --ambiguity``.

NOT WIRED INTO FLIGHT. This is a third A/B arm next to the line solver and the hull quad; the
keypoint path remains the only emitter. ``homography_from_lines``, ``centre_from_seg_masks`` and the
quad functions are deliberately untouched so the comparison stays honest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from racer.frames import CAMERA_INTRINSICS_K
from racer.vision.gate_lines import quad_from_mask_ex
from racer.vision.gate_pose import GATE_INNER_SIZE_M, GATE_OUTER_SIZE_M

# Gate frame DEPTH. Canonical source: racer.vision.blender_gen.contract.GATE_DEPTH_M (spec
# VADR-TS-002 sec 3.7: 260 mm), realised as a mesh by blender_gen.bpy_scene._gate_frame_geometry.
# Duplicated rather than imported so this module -- a candidate for the flight loop -- does not drag
# in the whole blender_gen package (which pulls the dataset/render stack). tests/test_model_fit.py
# asserts the two agree, so the duplicate cannot drift silently.
GATE_DEPTH_M = 0.26

# --- numerical guards ---------------------------------------------------------------------------
# A vertex closer than this to the optical plane cannot be projected: the perspective divide
# explodes and the "silhouette" wraps to the far side of the image. Poses that put any model vertex
# there are rejected outright rather than scored on garbage.
_MIN_VERTEX_Z_M = 0.05
# A projected vertex further out than this many image widths means the pose is at/over the gate
# plane's vanishing line. Also keeps the fixed-point fillPoly coordinates inside int32.
_MAX_PROJ_MULT = 200.0
# Same sanity bound the quad path applies to an emitted centre (gate_lines._SANE_CENTRE_MULT),
# duplicated here so the A/B compares like with like: a gate centre may legitimately sit off-screen
# -- that is the whole point -- but a degenerate fit produces a "centre" tens of image-widths away
# (measured on real close-range frames: -35378 px, 55 widths out) and without this bound that
# garbage counts as coverage.
_SANE_CENTRE_MULT = 3.0
# Physical plausibility of the fitted range. The course is ~30 m of gates; the navigator caps vision
# at ~30-32 m. Anything outside this is a collapsed fit, not a distant gate.
_MIN_RANGE_M, _MAX_RANGE_M = 0.30, 80.0
# A mask smaller than this is a speck -- the same floor the line and quad paths use, so coverage
# numbers are comparable across arms.
_MIN_MASK_AREA_PX = 200.0


# ==================================================================================================
# the 3-D model
# ==================================================================================================
@dataclass(frozen=True)
class GateModel:
    """The gate as explicit 3-D vertices in the GATE frame (X right, Y DOWN, Z downrange).

    Mirrors ``blender_gen.bpy_scene._gate_frame_geometry`` exactly -- a square annulus prism,
    SYMMETRIC about Z=0, so the inner-opening corners sit at (+-0.75, +-0.75, 0+-d/2) and the
    detector's canonical keypoints are the mid-plane inner square. Fitting the same solid the
    renderer draws is the point: the seg labels are (or will be) traced off that render, so label
    and inference speak about the same object.

    ``depth = 0`` collapses this to the two FLAT concentric squares, which is exactly the shape the
    currently-shipped seg labels encode (they are derived from keypoint labels through
    ``seg_labels.seg_rows_from_corners``, which projects two flat quads). That is not a hypothetical
    ablation -- it is the matched model for today's weights, so it is a first-class option here.
    """

    inner_m: float = GATE_INNER_SIZE_M
    outer_m: float = GATE_OUTER_SIZE_M
    depth_m: float = GATE_DEPTH_M

    # ALL 16 vertices in ONE (16,3) array: outer front 0..3, outer back 4..7, inner front 8..11,
    # inner back 12..15, each LL,LR,UR,UL with y DOWN. One array because one matrix product per
    # scored pose is measurably cheaper than three -- at ~18 us of numpy call overhead each and a
    # couple of hundred evaluations per gate, splitting them cost 4 ms per fit.
    verts: np.ndarray = field(init=False, repr=False)
    outer_verts: np.ndarray = field(init=False, repr=False)   # (8,3) view, for callers/tests
    inner_front: np.ndarray = field(init=False, repr=False)   # (4,3) opening rim nearest the camera
    inner_back: np.ndarray = field(init=False, repr=False)    # (4,3) opening rim furthest away
    inner_mid: np.ndarray = field(init=False, repr=False)     # (4,3) the keypoint square at z=0

    def __post_init__(self):
        oh, ih, d = self.outer_m / 2.0, self.inner_m / 2.0, self.depth_m / 2.0
        # corner (x, y) order LL, LR, UR, UL with y DOWN -- identical to _gate_frame_geometry and to
        # gate_pose.gate_object_points, so a pose fitted here means the same thing everywhere else.
        sq = lambda h: [(-h, h), (h, h), (h, -h), (-h, -h)]                        # noqa: E731
        v = np.array([(x, y, -d) for x, y in sq(oh)] + [(x, y, d) for x, y in sq(oh)]
                     + [(x, y, -d) for x, y in sq(ih)] + [(x, y, d) for x, y in sq(ih)], float)
        object.__setattr__(self, "verts", v)
        object.__setattr__(self, "outer_verts", v[0:8])
        object.__setattr__(self, "inner_front", v[8:12])
        object.__setattr__(self, "inner_back", v[12:16])
        object.__setattr__(self, "inner_mid", np.array([(x, y, 0.0) for x, y in sq(ih)], float))


DEFAULT_MODEL = GateModel()
FLAT_MODEL = GateModel(depth_m=0.0)


def _project(pts3: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray):
    """(N,3) gate-frame points -> (N,2) pixels, or None if the pose is unprojectable.

    Returns None (rather than inf/NaN pixels) when any vertex is at or behind the optical plane or
    lands absurdly far out. Both are the same failure -- the perspective divide blowing up -- and a
    caller that scored them would be scoring numerical noise as if it were a silhouette.

    ONE matmul, not two: (p R^T + t) K^T == p (K R)^T + (K t), and K's last row is [0,0,1] so the
    homogeneous w is already the camera-frame Z used for the cheirality test. This runs a couple of
    hundred times per gate, so the saved matmul is worth the algebra."""
    h = pts3 @ (K @ R).T + (K @ t)
    if h[:, 2].min() <= _MIN_VERTEX_Z_M:
        return None
    uv = h[:, :2] / h[:, 2:3]
    if not np.isfinite(uv).all() or np.abs(uv).max() > _MAX_PROJ_MULT * float(K[0, 2] * 2.0):
        return None
    return uv


def _convex_hull(pts: np.ndarray):
    h = cv2.convexHull(np.ascontiguousarray(pts, np.float32).reshape(-1, 1, 2))
    return None if h is None or len(h) < 3 else h.reshape(-1, 2).astype(np.float64)


# The 12 edges of the outer box: front quad, back quad, then the four connecting verticals. Needed
# ONLY for near-plane clipping (below) -- the unclipped silhouette needs no topology at all, because
# the projection of a convex solid is the convex hull of its projected vertices.
_BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def _clip_near_solid(cam: np.ndarray, edges, znear: float):
    """Vertices of a convex solid CLIPPED to z >= znear, in camera coordinates. (n,3) or None.

    WHY THIS EXISTS -- it is worth 7 of 54 gates on the real val split. A 2.72 m gate at 0.6 m range
    (the drone about to fly through it, which is exactly when the centre is off-screen and the other
    solvers are weakest) has outer corners BESIDE and BEHIND the camera. Rejecting the whole pose
    because one of 16 vertices is behind the optical plane threw away every one of those frames --
    and they are the frames this whole path exists to serve. A graphics pipeline would never do
    that; it clips at the near plane and draws what remains.

    For a CONVEX solid the clipped solid's vertex set is exactly {vertices already in front} plus
    {intersections of edges that cross the plane}, so its projected silhouette is still just the
    convex hull of those projections -- no hidden-surface work."""
    z = cam[:, 2]
    keep = [cam[i] for i in range(len(cam)) if z[i] >= znear]
    for a, b in edges:
        za, zb = z[a], z[b]
        if (za >= znear) == (zb >= znear):
            continue
        s = (znear - za) / (zb - za)
        keep.append(cam[a] + (cam[b] - cam[a]) * s)
    return None if len(keep) < 3 else np.asarray(keep, float)


def _clip_near_poly(cam: np.ndarray, znear: float):
    """Sutherland-Hodgman clip of a planar 3-D polygon to z >= znear. (n,3), possibly empty."""
    out = []
    n = len(cam)
    for i in range(n):
        cur, prv = cam[i], cam[i - 1]
        ci, pi = cur[2] >= znear, prv[2] >= znear
        if ci:
            if not pi:
                out.append(prv + (cur - prv) * ((znear - prv[2]) / (cur[2] - prv[2])))
            out.append(cur)
        elif pi:
            out.append(prv + (cur - prv) * ((znear - prv[2]) / (cur[2] - prv[2])))
    return np.asarray(out, float).reshape(-1, 3)


def _project_cam(cam: np.ndarray, K: np.ndarray):
    """Camera-frame points (already known to be in front) -> pixels, or None if they blow up."""
    if len(cam) == 0:
        return None
    h = cam @ K.T
    uv = h[:, :2] / h[:, 2:3]
    if not np.isfinite(uv).all() or np.abs(uv).max() > _MAX_PROJ_MULT * float(K[0, 2] * 2.0):
        return None
    return uv


def _clip_convex(subject: np.ndarray, clipper: np.ndarray):
    """Intersection of two CONVEX polygons. (n,2) float array, possibly empty.

    Used for the see-through opening: a ray passes through the gate only if it clears BOTH the front
    rim and the back rim, so the visible hole is the intersection of the two projected inner
    squares. With depth 0 the two coincide and this is the identity, which is why the flat model
    falls out of the same code path.

    ``cv2.intersectConvexConvex`` rather than a Python Sutherland-Hodgman: measured 15 us vs 84 us,
    and this runs once per scored pose. It is orientation-agnostic (verified in
    tests/test_model_fit.py against an independent Sutherland-Hodgman implementation)."""
    a = np.ascontiguousarray(subject, np.float32).reshape(-1, 1, 2)
    b = np.ascontiguousarray(clipper, np.float32).reshape(-1, 1, 2)
    try:
        _, p = cv2.intersectConvexConvex(a, b, handleNested=True)
    except cv2.error:
        return np.empty((0, 2))
    if p is None or len(p) < 3:
        return np.empty((0, 2))
    return np.asarray(p, float).reshape(-1, 2)


def project_silhouette(R: np.ndarray, t: np.ndarray, K: np.ndarray = None,
                       model: GateModel = DEFAULT_MODEL):
    """The gate's TRUE projected silhouette at pose (R, t). Returns (frame_poly, opening_poly).

    ``frame_poly``   -- the outer boundary of everything the gate covers, i.e. the amodal
                        ``gate_frame`` class. The outer surface of the prism is CONVEX, so its
                        projection is exactly the convex hull of its 8 projected corners -- no
                        hidden-surface reasoning needed, and it automatically grows beyond the flat
                        outer square when the camera starts to see the outer side walls.
    ``opening_poly`` -- the see-through ``gate_opening`` class: the intersection of the front and
                        back inner rims (above). Empty when an oblique view closes the hole, which
                        is the correct answer, not a failure.

    Returns None only when the gate is ENTIRELY behind the camera, or when the projection blows up.
    A gate that merely STRADDLES the optical plane -- routine inside ~1.5 m, and precisely when the
    centre is off-screen -- is near-plane clipped and drawn, not discarded."""
    K = CAMERA_INTRINSICS_K if K is None else K
    v = model.verts
    h = v @ (K @ R).T + (K @ t)                  # ONE projection for all 16 vertices; h[:,2] == z
    if h[:, 2].min() > _MIN_VERTEX_Z_M:          # FAST PATH: nothing near the optical plane
        uv = h[:, :2] / h[:, 2:3]
        if not np.isfinite(uv).all() or np.abs(uv).max() > _MAX_PROJ_MULT * float(K[0, 2] * 2.0):
            return None
        frame_poly = _convex_hull(uv[0:8])
        if frame_poly is None:
            return None
        opening_poly = uv[8:12] if model.depth_m == 0.0 else _clip_convex(uv[8:12], uv[12:16])
        return frame_poly, opening_poly

    # SLOW PATH: part of the gate is at/behind the camera. Clip in 3-D, then project.
    cam = v @ R.T + t
    box = _clip_near_solid(cam[0:8], _BOX_EDGES, _MIN_VERTEX_Z_M)
    if box is None:
        return None                              # gate entirely behind the camera
    uv_box = _project_cam(box, K)
    if uv_box is None:
        return None
    frame_poly = _convex_hull(uv_box)
    if frame_poly is None:
        return None
    if model.depth_m == 0.0:
        fr = _project_cam(_clip_near_poly(cam[8:12], _MIN_VERTEX_Z_M), K)
        opening_poly = np.empty((0, 2)) if fr is None else fr
    else:
        fr = _project_cam(_clip_near_poly(cam[8:12], _MIN_VERTEX_Z_M), K)
        bk = _project_cam(_clip_near_poly(cam[12:16], _MIN_VERTEX_Z_M), K)
        # Both rims clipped by the SAME plane, so intersecting their projections is still the
        # see-through region -- restricted to the part of it that is in front of the camera.
        opening_poly = (np.empty((0, 2)) if fr is None or bk is None
                        else _clip_convex(fr, bk))
    return frame_poly, opening_poly


# ==================================================================================================
# rasterised IoU
# ==================================================================================================
_FILL_SHIFT = 3          # fillPoly fixed-point bits: the polygon edge is placed to 1/8 raster pixel


def _fill(poly: np.ndarray, scale: float, buf: np.ndarray) -> int:
    """Zero ``buf``, rasterise ``poly`` (full-resolution pixels) into it at 1/scale, return its area.

    Sub-pixel via fillPoly's fixed-point ``shift``. WHY IT MATTERS: without it the polygon edge
    snaps to whole raster pixels, so at scale 4 the score is CONSTANT over a 4-px pose translation
    and the search stalls on a plateau with the centre several pixels off. With shift the boundary
    moves in 1/8-pixel steps and the score keeps a usable staircase to descend. Coordinates are
    clamped to a range that cannot overflow the int32 fixed-point representation; a polygon fully
    outside the raster simply fills nothing, which is the correct score contribution (no
    intersection) rather than an error."""
    buf[:] = 0
    if poly is None or len(poly) < 3:
        return 0
    p = np.asarray(poly, float) / scale
    if not np.isfinite(p).all():
        return 0
    lim = 1 << 24
    p = np.clip(p * (1 << _FILL_SHIFT), -lim, lim).astype(np.int32)
    cv2.fillPoly(buf, [p.reshape(-1, 1, 2)], 1, cv2.LINE_8, _FILL_SHIFT)
    return cv2.countNonZero(buf)


def _iou(a: np.ndarray, b: np.ndarray, n_a: int, n_b: int, tmp: np.ndarray) -> float:
    """IoU of two binary rasters, given their precomputed populations.

    |A u B| = |A| + |B| - |A n B|, so only ONE array pass is needed instead of the two an explicit
    union would cost, and |B| (the predicted mask) is a constant that is counted once per gate
    rather than once per scored pose. cv2 rather than numpy: measured 8.8 us vs 27.9 us at 160x90,
    on a function called ~250 times per gate."""
    if n_a == 0 or n_b == 0:
        return 0.0
    cv2.bitwise_and(a, b, dst=tmp)
    inter = cv2.countNonZero(tmp)
    if inter == 0:
        return 0.0
    return inter / float(n_a + n_b - inter)


class _Target:
    """The predicted masks, pre-reduced to the scoring raster ONCE per gate.

    ``INTER_AREA`` then a 0.5 threshold is a majority vote over the scale x scale block, not
    nearest-neighbour sampling: at scale 4 a nearest-neighbour reduction drops or grows thin cropped
    slivers by whole raster pixels depending on phase, which is a systematic, pose-independent bias
    in the score."""

    def __init__(self, frame_mask, opening_mask, image_wh, scale: float):
        w, h = image_wh
        self.scale = float(scale)
        self.rw, self.rh = max(1, int(round(w / scale))), max(1, int(round(h / scale)))
        self.frame = self._reduce(frame_mask)
        self.opening = self._reduce(opening_mask)
        self.n_frame = 0 if self.frame is None else cv2.countNonZero(self.frame)
        self.n_opening = 0 if self.opening is None else cv2.countNonZero(self.opening)
        self.buf = np.zeros((self.rh, self.rw), np.uint8)
        self.tmp = np.zeros((self.rh, self.rw), np.uint8)

    def _reduce(self, m):
        if m is None:
            return None
        m = np.ascontiguousarray(np.asarray(m).astype(np.uint8))
        if m.shape != (self.rh, self.rw):
            # 255 not 1: INTER_AREA averages, and averaging a 0/1 mask then thresholding at 0.5
            # loses to integer truncation on some builds. Scale up, average, threshold at 127.
            m = cv2.resize(m * 255, (self.rw, self.rh), interpolation=cv2.INTER_AREA)
            m = (m >= 128).astype(np.uint8)
        return np.ascontiguousarray(m)


def score_pose(R, t, target: _Target, K, model: GateModel,
               w_frame: float = 1.0, w_open: float = 1.0):
    """Overlap score of the model at (R, t) against the predicted masks. -1.0 = unprojectable.

    ``IoU(frame) + IoU(opening)``, weight-normalised. The two carry different information and the
    weights are the one real tuning knob here (measured in scripts/eval_centre_ab.py: see the
    ``--weights`` sweep in the report). The opening is the cleaner target -- smaller, no amodal
    completion asked of the model, no neighbouring gate's frame to bleed into -- while the frame
    carries the 2.72 m scale and therefore most of the RANGE information.

    The raster covers the IMAGE ONLY, so the projected model is clipped to the frame exactly as the
    predicted mask is. That is deliberate: scoring the off-image part of the silhouette against a
    mask that structurally cannot contain it would penalise every correct cropped-gate pose."""
    got = project_silhouette(R, t, K, model)
    if got is None:
        return -1.0
    frame_poly, opening_poly = got
    total = 0.0
    wsum = 0.0
    if target.frame is not None and w_frame > 0.0:
        n = _fill(frame_poly, target.scale, target.buf)
        total += w_frame * _iou(target.buf, target.frame, n, target.n_frame, target.tmp)
        wsum += w_frame
    if target.opening is not None and w_open > 0.0:
        n = _fill(opening_poly, target.scale, target.buf)
        total += w_open * _iou(target.buf, target.opening, n, target.n_opening, target.tmp)
        wsum += w_open
    return total / wsum if wsum > 0.0 else -1.0


# ==================================================================================================
# pose parameterisation + search
# ==================================================================================================
# Six parameters, chosen so ONE UNIT OF EACH MOVES THE SILHOUETTE BY A COMPARABLE AMOUNT ON SCREEN
# -- raw (tx, ty, tz, rvec) does not (tz is metres, rvec is radians, and at 20 m a 0.1 m lateral
# shift is 1.6 px while at 2 m it is 16 px). Relative to an initial pose (R0, t0):
#   p[0], p[1]  du, dv    -- move the PROJECTED CENTRE by this many pixels (exactly)
#   p[2]        log s     -- scale the range: tz = tz0 * exp(s). Apparent size is 1/range, so this
#                            is the natural log-scale of the thing the image actually measures.
#   p[3..5]     rvec      -- rotate the gate ABOUT ITS OWN CENTRE (R = R0 @ Rodrigues(p)), so tilt
#                            and position stay decoupled and the centre does not drift when the
#                            optimiser explores orientation.
_STEP0 = np.array([3.0, 3.0, 0.06, 0.06, 0.06, 0.06])

# TRANSLATION PRE-SEARCH. Measured on synthetic CROPPED gates (ideal masks, so the objective is
# beyond suspicion): the hull-quad init sits 54-89 px from the true centre, because a clipped quad
# is the visible trapezoid rather than the gate. The 6-D search then stalled at IoU 0.89 while the
# TRUE pose scored 0.96 -- the objective was right and the search simply could not walk that far
# inside the budget. Nearly all of that init error is TRANSLATION, so two cheap parameters are
# searched first, with a step scaled to the gate's apparent size.
#
# ⚠ AND IT MUST BE BEST-OF-SWEEP, NOT OPPORTUNISTIC. Simply enlarging the 6-D search's step was
# tried first and made things WORSE (scores 0.44-0.97 vs 0.89-0.92, i.e. some fits collapsed): with
# a large step, "accept the first direction that improves at all" takes a big marginal move in the
# wrong dimension and then cannot recover. Evaluating all four translation directions and taking
# the best costs 4 evaluations per move and does not have that failure mode.
_PRE_SPAN_FRAC = 0.10
_PRE_PX_MIN, _PRE_PX_MAX = 3.0, 60.0
_PRE_BUDGET_FRAC = 0.35      # share of the evaluation budget the pre-search may spend


def _pose_from_params(p, R0, t0, K):
    """(du, dv, log s, rvec) relative to (R0, t0) -> (R, t). See the block comment above."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tz0 = float(t0[2])
    if tz0 <= 1e-6:
        return None
    u0, v0 = fx * t0[0] / tz0 + cx, fy * t0[1] / tz0 + cy
    tz = tz0 * float(np.exp(np.clip(p[2], -4.0, 4.0)))
    u, v = u0 + p[0], v0 + p[1]
    t = np.array([(u - cx) / fx * tz, (v - cy) / fy * tz, tz])
    rv = np.asarray(p[3:6], float)
    n = float(np.linalg.norm(rv))
    R = R0 if n < 1e-12 else R0 @ cv2.Rodrigues(rv.reshape(3, 1))[0]
    return R, t


def _apparent_span_px(R0, t0, K, model: GateModel) -> float:
    """Diagonal extent of the projected inner square -- the natural unit for a translation step.

    A gate 400 px across and one 40 px across need very different steps, and the quantity that
    scales is the init's likely ERROR, which comes from mis-fitting the gate's own outline."""
    uv = _project(model.inner_mid, R0, t0, K)
    if uv is None:
        return 0.0
    return float(np.linalg.norm(uv.max(axis=0) - uv.min(axis=0)))


def _translation_presearch(score_fn, span_px: float, budget: int):
    """Best-of-sweep compass over (du, dv) only. Returns (best_duv, best_score, n_evals).

    Two dimensions, four probes per move, so a move costs 4 evaluations and can never be a
    marginal jump in the wrong dimension (see the note at _PRE_SPAN_FRAC)."""
    p = np.zeros(2)
    s = float(np.clip(_PRE_SPAN_FRAC * span_px, _PRE_PX_MIN, _PRE_PX_MAX))
    best = score_fn(p)
    n = 1
    while n + 4 <= budget and s >= 1.0:
        cand = None
        for d in ((s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)):
            q = p + d
            v = score_fn(q)
            n += 1
            if v > best + 1e-7 and (cand is None or v > cand[0]):
                cand = (v, q)
        if cand is None:
            s *= 0.5
        else:
            best, p = cand
    return p, best, n


def _compass_search(score_fn, p0, step0, budget: int):
    """Deterministic pattern search. Returns (best_p, best_score, n_evals).

    WHY NOT A GRADIENT METHOD, OR NELDER-MEAD. The objective is a rasterised IoU: piecewise
    CONSTANT, with a step every time the silhouette boundary crosses a pixel centre. It has no
    gradient, and a simplex method contracts onto the flat regions between steps and reports
    convergence in the wrong place. A compass search only ever asks "is this point better?", which
    is exactly the question a piecewise-constant objective can answer, and its worst case is bounded
    by the budget rather than by a convergence criterion that may never fire.

    OPPORTUNISTIC + EXPANDING: accept the first improving direction rather than sweeping all twelve
    (halves the average cost) and grow that coordinate's step on success, so a bad initial scale is
    corrected in a few evaluations instead of being ground down one step at a time."""
    p = np.asarray(p0, float).copy()
    step = np.asarray(step0, float).copy()
    best = score_fn(p)
    n = 1
    floor = step0 * 0.02
    while n < budget and bool((step > floor).any()):
        moved = False
        for i in range(len(p)):
            if n >= budget:
                break
            for sgn in (1.0, -1.0):
                if n >= budget:
                    break
                q = p.copy()
                q[i] += sgn * step[i]
                s = score_fn(q)
                n += 1
                if s > best + 1e-7:
                    p, best, moved = q, s, True
                    step[i] = min(step[i] * 1.6, step0[i] * 8.0)
                    break
        if not moved:
            step *= 0.5
    return p, best, n


# ==================================================================================================
# initialisation
# ==================================================================================================
def _quad_signed_area(q) -> float:
    x, y = np.asarray(q, float)[:, 0], np.asarray(q, float)[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ippe_candidates(quad_px, size_m, K):
    """Both IPPE_SQUARE poses for a 4-corner quad of a known square. [(R, t), ...] (up to 2).

    The hull quad's vertex order is cyclic and its ORIENTATION depends on which way cv2.convexHull
    walked the contour, so it is normalised to the canonical order's signed-area sign first.
    Everything else is a symmetry of the gate: a cyclic shift is a 90-degree rotation about the gate
    Z axis, and the gate model is invariant under that (concentric squares, symmetric extrusion), so
    it changes the pose we report but NOT the silhouette we fit or the centre we emit."""
    q = np.asarray(quad_px, float)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        return []
    # gate_object_points' canonical LL,LR,UR,UL order has NEGATIVE signed area in image coordinates
    # (y down). A quad wound the other way would hand IPPE a mirrored correspondence.
    if _quad_signed_area(q) > 0.0:
        q = q[::-1].copy()
    h = size_m / 2.0
    obj = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    try:
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj.reshape(-1, 1, 3), np.ascontiguousarray(q).reshape(-1, 1, 2),
            K, np.zeros((4, 1)), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return []
    out = []
    for i in range(int(n)):
        R = cv2.Rodrigues(np.asarray(rvecs[i], float))[0]
        t = np.asarray(tvecs[i], float).reshape(3)
        if t[2] > _MIN_VERTEX_Z_M:
            out.append((R, t))
    return out


def _pose_from_homography(H, K):
    """Gate-plane -> image homography (plane coordinates in METRES) -> (R, t), or None.

    Standard planar decomposition: with G = K^-1 H, the first two columns are the scaled first two
    rotation columns and the third is the scaled translation. R is re-orthonormalised by SVD because
    a homography fitted to noisy lines is not exactly a rotation.

    WHY IT IS HERE. ``gate_lines.homography_from_lines`` solves exactly the configuration where the
    hull-quad init is worst -- a gate cropped to three edges, where the quad is the visible trapezoid
    and IPPE on it starts 54-89 px from the truth. The line solver can still solve it because it has
    a SECOND concentric square with a known metric ratio. Using it as an extra STARTING POINT (never
    as the answer) is the cheapest way to give the area fit a good basin on those frames."""
    try:
        G = np.linalg.inv(K) @ np.asarray(H, float)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(G).all():
        return None
    n1, n2 = np.linalg.norm(G[:, 0]), np.linalg.norm(G[:, 1])
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    lam = 2.0 / (n1 + n2)
    r1, r2, t = G[:, 0] * lam, G[:, 1] * lam, G[:, 2] * lam
    if t[2] < 0:                       # gate must be in front; H is only defined up to sign
        r1, r2, t = -r1, -r2, -t
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    try:
        u, _, vt = np.linalg.svd(R)
    except np.linalg.LinAlgError:
        return None
    R = u @ vt
    if np.linalg.det(R) < 0:
        R = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    if not (np.isfinite(R).all() and np.isfinite(t).all()) or not (_MIN_RANGE_M < np.linalg.norm(t) < _MAX_RANGE_M):
        return None
    return R, t


def _moment_init(mask, size_m, K):
    """Frontal pose from a mask's centroid + area alone -- the fallback when no quad can be fitted.

    Deliberately crude: it only has to land inside the optimiser's basin. Area gives range through
    the same weak-perspective relation as :func:`range_from_apparent_size` at zero tilt, and the
    centroid gives the bearing. Exists because the quad path returns None on a non-convex or
    collinear hull reduction, and an arm that cannot start where the quad cannot start could never
    beat it on coverage."""
    m = np.asarray(mask).astype(np.uint8)
    a = float(np.count_nonzero(m))
    if a < _MIN_MASK_AREA_PX:
        return None
    ys, xs = np.nonzero(m)
    u, v = float(xs.mean()), float(ys.mean())
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = float(np.sqrt(fx * fy * size_m * size_m / a))
    if not (_MIN_RANGE_M < d < _MAX_RANGE_M):
        return None
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    return np.eye(3), d * ray / float(np.linalg.norm(ray))


# ==================================================================================================
# public result + entry point
# ==================================================================================================
@dataclass
class GateFit:
    """One fitted gate. See the module docstring for which fields survive the 2-fold ambiguity."""

    centre_px: np.ndarray          # AMBIGUITY-SAFE. The projected gate centre; may be off-image.
    R_cam_gate: np.ndarray         # ambiguity-AFFECTED (this is the orientation, the ambiguous part)
    t_cam_gate: np.ndarray         # direction is safe, MAGNITUDE is not -- prefer range_m below
    range_m: float                 # AMBIGUITY-SAFE range from apparent SIZE (see range_from_...)
    t_norm_m: float                # ||t||: the same quantity taken the ambiguity-UNSAFE way
    bearing_unit: np.ndarray       # AMBIGUITY-SAFE unit ray through centre_px
    rel_position_m: np.ndarray     # AMBIGUITY-SAFE = range_m * bearing_unit  <- emit THIS
    score: float
    iou_frame: float
    iou_opening: float
    n_evals: int
    init_kind: str                 # "quad-opening" | "quad-frame" | "moment"
    ambiguity_margin: float        # best score - other IPPE branch's score (0.0 if only one)
    range_from_size_ok: bool = True   # False => range_m fell back to ||t|| and is NOT ambiguity-safe
    frame_poly: np.ndarray = None      # fitted silhouette, for overlay/debug
    opening_poly: np.ndarray = None


def range_from_apparent_size(R, t, K, size_m: float = GATE_INNER_SIZE_M) -> float:
    """Range to the gate centre from its KNOWN METRIC SIZE vs its APPARENT size. Ambiguity-safe.

    ``d = sqrt(fx * fy * S^2 * |cos theta| / A_px)`` -- the weak-perspective relation between a
    planar patch's metric area and its projected area, with the foreshortening ``|cos theta|``
    (theta = angle between the gate normal and the line of sight to its centre) taken from the
    fitted rotation.

    WHY THIS IS AMBIGUITY-SAFE AND ``||t||`` IS NOT. The IPPE 2-fold ambiguity flips the SIGN of the
    tilt, not its magnitude, so ``|cos theta|`` is identical on both branches; and ``A_px`` is read
    off the image, which by definition both branches reproduce. Every input is therefore
    branch-invariant. ``||t||`` is not: it is the one component of the translation the two branches
    can disagree about (their DIRECTIONS must agree, because the gate origin projects to the same
    pixel either way -- which is also why the bearing is safe).

    APPROXIMATE, and honestly so: weak perspective treats the patch as a single depth, which a 1.5 m
    square at 2 m is not. Measured against ``||t||`` and against task2 ground-truth range in the
    report accompanying this module."""
    A = abs(_quad_signed_area(_project_square(R, t, K, size_m)))
    if A <= 1e-6:
        return float("nan")
    n_cam = R @ np.array([0.0, 0.0, 1.0])            # gate normal in camera coordinates
    los = np.asarray(t, float)
    ln = float(np.linalg.norm(los))
    if ln < 1e-9:
        return float("nan")
    cos_theta = abs(float(np.dot(n_cam, los / ln)))
    return float(np.sqrt(K[0, 0] * K[1, 1] * size_m * size_m * cos_theta / A))


def _project_square(R, t, K, size_m):
    h = size_m / 2.0
    sq = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    uv = _project(sq, R, t, K)
    return np.zeros((4, 2)) if uv is None else uv


def _centre_px(R, t, K):
    z = float(t[2])
    if z <= 1e-9:
        return None
    return np.array([K[0, 0] * t[0] / z + K[0, 2], K[1, 1] * t[1] / z + K[1, 2]])


def _centre_is_sane(c, image_wh) -> bool:
    w_px, h_px = image_wh
    return not (abs(c[0]) > _SANE_CENTRE_MULT * w_px
                or abs(c[1] - h_px / 2.0) > _SANE_CENTRE_MULT * h_px)


def fit_gate_model(frame_mask, opening_mask=None, image_wh=(640, 360), *,
                   K: np.ndarray = None, model: GateModel = DEFAULT_MODEL,
                   scales=(2.0,), budget: int = 120,
                   w_frame: float = 1.0, w_open: float = 1.0,
                   min_score: float = 0.30, presearch: bool = False,
                   line_init: bool = False) -> GateFit | None:
    """Fit the 3-D gate model to ONE gate's predicted masks. Returns a :class:`GateFit` or None.

    ``scales`` is a coarse-to-fine raster schedule (image downsample factors); ``budget`` caps the
    TOTAL evaluations across all stages, so the worst-case wall time is bounded -- this has to fit a
    ~7 Hz flight loop that already shares its GPU.

    THE DEFAULTS ARE MEASURED, not assumed (40 synthetic poses, 1.6-20 m, clean and ragged masks):

        scales      budget   median centre err   wall
        (4,)        120        2.65 px           29 ms
        (4, 2)      220        1.06 px           56 ms
        (2,)        120        0.90 px           30 ms      <- default
        (2,)         60        1.08 px           21 ms

    A single FINE raster beats coarse-to-fine at equal cost. That is not what you would guess: a
    scale-2 evaluation costs only 15% more than a scale-4 one (123 us vs 107 us -- both are
    dominated by per-call overhead, not by pixels), while a scale-4 raster physically cannot resolve
    the last 2 px of centre, so the coarse stage spends evaluations it can never cash in. Coarse-to-
    fine would pay off if the basin were wide, but the hull-quad init already lands inside it.

    ``min_score`` rejects a fit that never found the gate. Without it a wandering optimiser returns
    a confident-looking pose with ~0 overlap, which is coverage on paper and garbage in fact -- the
    exact failure the line solver's p90 of 894 px is made of."""
    K = CAMERA_INTRINSICS_K if K is None else np.asarray(K, float)
    if frame_mask is None and opening_mask is None:
        return None
    fm = None if frame_mask is None else np.asarray(frame_mask).astype(np.uint8)
    om = None if opening_mask is None else np.asarray(opening_mask).astype(np.uint8)
    if fm is not None and np.count_nonzero(fm) < _MIN_MASK_AREA_PX:
        fm = None
    if om is not None and np.count_nonzero(om) < _MIN_MASK_AREA_PX:
        om = None
    if fm is None and om is None:
        return None

    # ---- initialisation: the hull quad, which is what makes this a LOCAL refinement -------------
    # The quad path already fits four straight lines to the whole mask boundary; running IPPE on it
    # puts us inside the right basin, so the search never has to explore 6-D pose space globally.
    inits = []
    for kind, mask, size_m in (("quad-opening", om, model.inner_m),
                               ("quad-frame", fm, model.outer_m)):
        if mask is None:
            continue
        got = quad_from_mask_ex(mask)
        if got is None:
            continue
        for R0, t0 in _ippe_candidates(got[0], size_m, K):
            inits.append((kind, R0, t0))
        if inits:
            break                      # prefer the opening: smaller, cleaner, no amodal completion
    if line_init:
        # OFF BY DEFAULT, and that is a MEASUREMENT, not an oversight. The line solver's plane as an
        # extra starting point (see _pose_from_homography) was scored on the 54 hand-labelled real
        # gates: median 19.7 px / p90 136.7 px WITH it against 19.1 / 141.4 WITHOUT -- a wash, for
        # ~3 ms per gate and a dependency on the other arm. Kept as a switch so the measurement is
        # reproducible; do not turn it on without re-measuring on real frames.
        try:
            from racer.vision.gate_lines import centre_from_seg_masks
            got = centre_from_seg_masks(fm, om, image_wh=image_wh)
        except Exception:
            got = None
        if got is not None:
            pose = _pose_from_homography(got[1], K)
            if pose is not None:
                inits.append(("line-H", pose[0], pose[1]))
    if not inits:
        mask, size_m = (om, model.inner_m) if om is not None else (fm, model.outer_m)
        mi = _moment_init(mask, size_m, K)
        if mi is None:
            return None
        inits.append(("moment", mi[0], mi[1]))

    target_coarse = _Target(fm, om, image_wh, scales[0])
    n_evals = 0

    # Score BOTH IPPE branches. This is NOT multi-start machinery for the centre (the centre is the
    # same on both branches by construction) -- it is the 2-fold ORIENTATION ambiguity, and the
    # margin between the two silhouette scores is the direct measurement of whether the gate's
    # 0.26 m depth can break an ambiguity that no planar method can. Costs two evaluations.
    scored = []
    for kind, R0, t0 in inits:
        s = score_pose(R0, t0, target_coarse, K, model, w_frame, w_open)
        n_evals += 1
        scored.append((s, kind, R0, t0))
    scored.sort(key=lambda r: -r[0])
    best_s, init_kind, R0, t0 = scored[0]
    ambiguity_margin = float(best_s - scored[1][0]) if len(scored) > 1 else 0.0

    # ---- coarse-to-fine refinement --------------------------------------------------------------
    R_best, t_best = R0, t0
    if presearch:
        # OFF BY DEFAULT -- it was MEASURED and it lost. See the note at _PRE_SPAN_FRAC; kept as a
        # switch so the measurement can be reproduced rather than taken on trust.
        def fn2(duv, _t=target_coarse, _R=R0, _t0=t0):
            pose = _pose_from_params(np.array([duv[0], duv[1], 0.0, 0.0, 0.0, 0.0]), _R, _t0, K)
            return -1.0 if pose is None else score_pose(pose[0], pose[1], _t, K, model,
                                                        w_frame, w_open)

        span = _apparent_span_px(R0, t0, K, model)
        duv, s0, used = _translation_presearch(fn2, span, int(budget * _PRE_BUDGET_FRAC))
        n_evals += used
        if s0 > best_s:
            best_s = s0
            pose = _pose_from_params(np.array([duv[0], duv[1], 0.0, 0.0, 0.0, 0.0]), R0, t0, K)
            if pose is not None:
                R_best, t_best = pose

    # ---- full 6-DoF refinement ------------------------------------------------------------------
    p = np.zeros(6)
    step = _STEP0.copy()
    target = target_coarse
    for si, scale in enumerate(scales):
        target = target_coarse if si == 0 else _Target(fm, om, image_wh, scale)

        def fn(q, _t=target, _R=R_best, _t0=t_best):
            pose = _pose_from_params(q, _R, _t0, K)
            return -1.0 if pose is None else score_pose(pose[0], pose[1], _t, K, model,
                                                        w_frame, w_open)

        left = budget - n_evals
        if left <= 2:
            break
        p, best_s, used = _compass_search(fn, np.zeros(6), step, left)
        n_evals += used
        pose = _pose_from_params(p, R_best, t_best, K)
        if pose is None:
            return None
        R_best, t_best = pose
        step = step * 0.35              # the fine stage starts where the coarse stage stopped caring

    if best_s < min_score:
        return None
    c = _centre_px(R_best, t_best, K)
    if c is None or not np.isfinite(c).all() or not _centre_is_sane(c, image_wh):
        return None
    rng_t = float(np.linalg.norm(t_best))
    if not (_MIN_RANGE_M < rng_t < _MAX_RANGE_M):
        return None

    # per-class IoU for diagnostics. Reuses the LAST stage's target -- rebuilding it costs 1.4 ms,
    # which would be the single most expensive line in the whole fit.
    final = target
    got = project_silhouette(R_best, t_best, K, model)
    iou_f = iou_o = float("nan")
    frame_poly = opening_poly = None
    if got is not None:
        frame_poly, opening_poly = got
        if final.frame is not None:
            n = _fill(frame_poly, final.scale, final.buf)
            iou_f = _iou(final.buf, final.frame, n, final.n_frame, final.tmp)
        if final.opening is not None:
            n = _fill(opening_poly, final.scale, final.buf)
            iou_o = _iou(final.buf, final.opening, n, final.n_opening, final.tmp)

    rng = range_from_apparent_size(R_best, t_best, K, model.inner_m)
    ray = np.array([(c[0] - K[0, 2]) / K[0, 0], (c[1] - K[1, 2]) / K[1, 1], 1.0])
    ray = ray / float(np.linalg.norm(ray))
    # The size-based range needs the WHOLE inner square to project (weak perspective over a patch
    # that is partly behind the camera is meaningless). Inside ~1 m it degenerates, and there we
    # fall back to ||t|| -- which is NOT ambiguity-safe, so say so in the result rather than hand
    # back a number that silently changed meaning.
    range_ok = bool(np.isfinite(rng) and _MIN_RANGE_M < rng < _MAX_RANGE_M)
    if not range_ok:
        rng = rng_t
    return GateFit(
        centre_px=c, R_cam_gate=R_best, t_cam_gate=t_best,
        range_m=float(rng), t_norm_m=rng_t, bearing_unit=ray, rel_position_m=float(rng) * ray,
        score=float(best_s), iou_frame=float(iou_f), iou_opening=float(iou_o),
        n_evals=int(n_evals), init_kind=init_kind, ambiguity_margin=ambiguity_margin,
        range_from_size_ok=range_ok, frame_poly=frame_poly, opening_poly=opening_poly)


def centre_from_masks_model(frame_mask, opening_mask=None, image_wh=(640, 360), **kw):
    """A/B-shaped wrapper: (centre_px, GateFit) or None -- mirrors ``centre_from_masks_quad``."""
    fit = fit_gate_model(frame_mask, opening_mask, image_wh, **kw)
    return None if fit is None else (fit.centre_px, fit)
