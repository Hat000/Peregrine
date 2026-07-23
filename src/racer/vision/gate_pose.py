"""Gate pose from detected corners via planar PnP (IPPE) with a 3-corner P3P fallback.

Turns a :class:`GateObservation` (3 or 4 inner-square corners in pixels) into a
:class:`GatePose` (6-DOF gate pose in the camera optical frame), using the known
1500 mm inner square and the camera intrinsics. Sim-independent and unit-testable
via :func:`project_gate_corners` (the synthetic projector).

Gate frame (chosen to match OpenCV's ``SOLVEPNP_IPPE_SQUARE`` native convention, so no
permutation is needed): origin at the gate centre, X=right, Y=DOWN, Z=downrange (the
direction of travel through the gate, pointing away from the approaching drone). A gate
viewed head-on therefore has ``R_cam_gate ~= I`` and ``t_cam_gate ~= [0, 0, range]``.
(NB: IPPE needs the object +Z normal to point away from the camera; a Y-up / normal-
toward-camera frame makes IPPE fail to recover the pose — verified the hard way.)

Canonical corner order — the detector MUST emit the 4 inner-square corners in exactly
this order (OpenCV's required IPPE_SQUARE object-point order); with h = size / 2:

    idx   gate (x, y)    image position, head-on gate
     0    (-h, +h)       lower-left
     1    (+h, +h)       lower-right
     2    (+h, -h)       upper-right
     3    (-h, -h)       upper-left

``solvePnP`` returns the gate pose expressed in the camera optical frame:

    p_cam = R_cam_gate @ p_gate + t_cam_gate

Two-fold ambiguity. IPPE_SQUARE (4 corners) returns the planar 2-fold ambiguity. We pick
the lower-reprojection-error solution UNLESS the geometry is genuinely ambiguous AND a
``prior`` is supplied. "Genuinely ambiguous" = ``ambiguity_ratio`` (= err_second/err_first)
below :data:`PRIOR_DISAMBIG_MAX_RATIO`: when one solution fits far better (high ratio, e.g.
a clearly off-axis gate) we trust that reprojection and IGNORE the prior, so a stale prior
can never force the geometrically-wrong "flipped" pose; only near-frontal gates (ratio ~1)
fall back to the prior to break the tie.

Three-corner fallback. When a closing gate clips out of frame we may see only 3 corners.
IPPE_SQUARE needs exactly 4, so we drop to ``cv2.solveP3P`` over the 3 matching object
points (selected by ``corner_ids``). P3P returns up to 4 discrete poses that ALL reproject
the 3 points exactly, so it cannot self-disambiguate; we lean on the ``prior`` (near-exact)
and fall back to a cheirality + most-frontal heuristic (~1-2 deg uncertain) when none is
given. Such fixes carry ``n_corners=3`` so the estimator can trust them less.

Spec: VADR-TS-002 sec 3.7 (gate), sec 3.8 (intrinsics, no distortion).
"""
from __future__ import annotations

import cv2
import numpy as np

from racer.contracts import GateObservation, GatePose
from racer.frames import CAMERA_INTRINSICS_K

GATE_INNER_SIZE_M = 1.5   # spec sec 3.7: inner square 1500 mm
# spec sec 3.7: OUTER structural square 2720 mm, CONCENTRIC + COPLANAR with the inner opening (the
# 8-keypoint training labels project both squares at the gate mid-plane z=0 — blender_gen/contract.py
# + geometry.py keep this in lockstep). Outer corners are ~1.81x the inner square: bigger apparent
# size + red-frame-on-dark-background contrast makes them the better-localised keypoints at range.
GATE_OUTER_SIZE_M = 2.72
_NO_DISTORTION = np.zeros((4, 1), dtype=np.float64)
_GATE_NORMAL = np.array([0.0, 0.0, 1.0])  # gate +Z (downrange) in the gate frame

# Use the temporal prior to break the IPPE 2-fold tie only when the two solutions fit
# nearly as well (ratio below this); above it, one solution is clearly better -> trust
# its reprojection and ignore the prior. Tunable at first sim contact. [review 2B]
PRIOR_DISAMBIG_MAX_RATIO = 5.0

# Additive sub-pixel floor for the IPPE ambiguity ratio. Without it, a frontal gate in
# near-zero noise drives the best reprojection error toward 0, so err2/err1 explodes (or hit
# the old `inf` sentinel) and mislabelled the MOST-ambiguous case as unambiguous -- bypassing
# the prior exactly when it is needed. With the floor, both-near-zero -> ratio ~1 (ambiguous,
# trust the prior); a genuinely better single solution (err2 >> err1) still yields a large
# ratio (unambiguous, trust reprojection). [red-team 2026-05-30]
AMBIGUITY_EPS_PX = 0.1

# Robust weighted-PnP refinement. Each corner is whitened by sigma_i = WEIGHTED_SIGMA_PX /
# clip(confidence_i, CONF_FLOOR, 1): a less-confident corner gets a looser sigma -> lower weight
# (a confidence ~0 can't blow up its weight). On top, a REDESCENDING Tukey biweight on the
# reprojection residual drives a confident-but-mislocalised corner (the tail failure) to ZERO
# weight -- fully rejecting it rather than merely down-weighting (as a convex Huber would). The
# knee is ANNEALED from lenient to strict across iterations (graduated non-convexity) so the
# estimator converges stably from the IPPE init. Tunable at first sim contact.
WEIGHTED_SIGMA_PX = 1.5
TUKEY_C_HI_PX = 40.0   # robust knee at iter 0 (lenient: every corner participates)
TUKEY_C_LO_PX = 8.0    # annealed to here (a corner reprojecting beyond this -> zero weight)
CONF_FLOOR = 0.1


def gate_object_points(inner_size_m: float = GATE_INNER_SIZE_M) -> np.ndarray:
    """The 4 inner-square corners in the gate's own frame, IPPE_SQUARE order. Shape (4,3)."""
    s = inner_size_m / 2.0
    return np.array(
        [
            [-s,  s, 0.0],   # 0  (-x,+y) lower-left  (gate Y is down)
            [ s,  s, 0.0],   # 1  (+x,+y) lower-right
            [ s, -s, 0.0],   # 2  (+x,-y) upper-right
            [-s, -s, 0.0],   # 3  (-x,-y) upper-left
        ],
        dtype=np.float64,
    )


def project_gate_corners(
    R_cam_gate: np.ndarray,
    t_cam_gate: np.ndarray,
    inner_size_m: float = GATE_INNER_SIZE_M,
    camera_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Project a gate at pose (R_cam_gate, t_cam_gate) to pixel corners. Inverse of PnP.

    Reusable synthetic projector: feeds unit tests, the auto-label pipeline, and ROI
    projection. Requires all corners in front of the camera (optical Z > 0).
    """
    K = CAMERA_INTRINSICS_K if camera_matrix is None else camera_matrix
    obj = gate_object_points(inner_size_m)
    cam = (R_cam_gate @ obj.T).T + np.asarray(t_cam_gate, dtype=np.float64)  # (4,3)
    if np.any(cam[:, 2] <= 0):
        raise ValueError("gate corner(s) at or behind the camera (optical Z <= 0)")
    uvw = (K @ cam.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def _rotation_geodesic(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) of the relative rotation between two rotation matrices."""
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _reproj_rms(obj_pts: np.ndarray, img_pts: np.ndarray, R: np.ndarray, t: np.ndarray,
                K: np.ndarray) -> float:
    """RMS pixel reprojection error of (R, t) over the given object/image correspondences.

    Guards the perspective divide: a point on/behind the optical plane (Z<=0) cannot
    reproject, so report +inf instead of dividing by ~0 and leaking a NaN into the error.
    Callers already select among cheirality-valid poses, so this is defence-in-depth. [red-team]
    """
    cam = (R @ obj_pts.T).T + t
    if np.any(cam[:, 2] <= 1e-9):
        return float("inf")
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return float(np.sqrt(np.mean(np.sum((uv - img_pts) ** 2, axis=1))))


def _solve(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray):
    """Run IPPE_SQUARE PnP; return list of (R, t, reproj_err) candidates (best first)."""
    img = np.ascontiguousarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)
    obj = np.ascontiguousarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img, K, _NO_DISTORTION, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
    except cv2.error:
        return []
    if not n:
        return []
    err_vals = np.asarray(errs, dtype=np.float64).ravel()
    out = []
    for i in range(n):
        R = cv2.Rodrigues(np.asarray(rvecs[i], dtype=np.float64))[0]
        t = np.asarray(tvecs[i], dtype=np.float64).reshape(3)
        out.append((R, t, float(err_vals[i])))
    return out


def _solve_p3p(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray):
    """Run 3-point P3P; return list of (R, t) candidate poses (up to 4, no error ranking)."""
    img = np.ascontiguousarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)
    obj = np.ascontiguousarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
    try:
        n, rvecs, tvecs = cv2.solveP3P(obj, img, K, _NO_DISTORTION, flags=cv2.SOLVEPNP_P3P)
    except cv2.error:
        return []
    out = []
    for i in range(int(n)):
        R = cv2.Rodrigues(np.asarray(rvecs[i], dtype=np.float64))[0]
        t = np.asarray(tvecs[i], dtype=np.float64).reshape(3)
        out.append((R, t))
    return out


def _ordered_corners(obs: GateObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (corners, ids, conf) sorted by canonical corner index, so rows match object points.
    ``conf`` is the per-corner confidence in the same order, or None if the obs carries none."""
    corners = np.ascontiguousarray(obs.corners_px, dtype=np.float64)
    n = corners.shape[0]
    ids = np.arange(n) if obs.corner_ids is None else np.asarray(obs.corner_ids).astype(int)
    order = np.argsort(ids, kind="stable")
    conf = None if obs.corner_confidence is None else np.asarray(obs.corner_confidence, dtype=np.float64)[order]
    return corners[order], ids[order], conf


_PLANE_INNER_32 = np.array([[-0.75, 0.75], [0.75, 0.75], [0.75, -0.75], [-0.75, -0.75]],
                           dtype=np.float32)          # gate-plane inner corners, LL,LR,UR,UL (y DOWN)
_PLANE_OUTER_32 = np.array([[-1.36, 1.36], [1.36, 1.36], [1.36, -1.36], [-1.36, -1.36]],
                           dtype=np.float32)          # gate-plane outer corners (GATE_OUTER_SIZE_M/2)


def inner_from_outer_homography(outer_px: np.ndarray) -> np.ndarray | None:
    """Inner-square pixel corners implied by the 4 OUTER corners via the exact gate-plane homography.

    Inner and outer squares are concentric + coplanar (spec 3.7), so ONE homography maps the gate
    plane to the image; the 4 outer correspondences determine it fully, and the inner corners are its
    image of the inner plane square (float32 exact to ~1e-4 px). Returns (4,2) float64, or None for a
    degenerate outer quad (near-collinear / sub-pixel — where the exactly-determined H explodes)."""
    o = np.ascontiguousarray(outer_px, dtype=np.float32)
    if o.shape != (4, 2) or not np.isfinite(o).all():
        return None
    x, y = o[:, 0], o[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))   # shoelace
    if area < 50.0:
        return None
    try:
        H = cv2.getPerspectiveTransform(_PLANE_OUTER_32, o)
        inner = cv2.perspectiveTransform(_PLANE_INNER_32.reshape(-1, 1, 2), H).reshape(4, 2)
    except cv2.error:
        return None
    inner = np.asarray(inner, dtype=np.float64)
    return inner if np.isfinite(inner).all() else None


# Soft consistency weighting scale + hard drop floor (see _valid_outer). w = exp(-d_norm^2) with
# d_norm = median disagreement / max(2.5 px, 5% span): ~1.0 for a corroborating outer square,
# 0.37 at the nominal tolerance, and effectively 0 for a contradicting one (a decoy at ~4x the
# tolerance -> w ~ 1e-7, dropped by the floor). SOFT (not a hard accept/reject) on purpose: a
# binary gate on a noisy statistic TOGGLES across consecutive frames, flip-flopping the estimator
# between two slightly different poses — itself a jitter source (measured on the 2026-07-05 real
# flight-clip bench: hard gate raised p90 range-step +34%). The continuous fade keeps frame-to-
# frame behaviour smooth while preserving the decoy safety via the hard floor.
_OUTER_W_FLOOR = 0.05


# Gate-plane coordinates of all 8 keypoints, in model order: inner LL,LR,UR,UL then outer LL,LR,UR,UL.
# Both squares are concentric + coplanar (spec 3.7), so ONE homography maps this plane to the image
# and ANY 4 of these 8 points in general position determine it.
_PLANE_ALL_8 = np.vstack([_PLANE_INNER_32, _PLANE_OUTER_32]).astype(np.float64)

# Below this |w| (relative to the median |w| of the fitted points) an inner corner is at/over the
# gate plane's VANISHING LINE: the projective divide explodes and the corner wraps to the opposite
# side of the image. Same failure the gate labeler hit deriving outer-from-inner; guarded identically.
_VANISH_W_FRAC = 0.08
_SANE_SPAN_MULT = 4.0     # a derived corner further out than this many image widths is "at infinity"

# DEGENERACY, exactly. A homography needs 4 fitted points with NO THREE COLLINEAR. Every one of the
# 8 keypoints sits on one of the gate's two diagonals -- LL/UR corners (inner 0,2 + outer 4,6) on
# one, LR/UL (inner 1,3 + outer 5,7) on the other -- and those are the ONLY collinear sets in the
# geometry (the square edges hold just 2 keypoints each, since inner and outer edges are not
# coincident). So "no 3 collinear" reduces to: at least 2 usable points on EACH diagonal. This is
# checked on the KNOWN gate geometry rather than the measured pixels, so it is noise-free -- a test
# on image points cannot distinguish a truly degenerate set from a foreshortened valid one.
_DIAG_A = np.array([0, 2, 4, 6])     # inner LL, inner UR, outer LL, outer UR
_DIAG_B = np.array([1, 3, 5, 7])     # inner LR, inner UL, outer LR, outer UL

# Minimum apparent size of the DERIVED inner square. The recovered corners are extrapolated, so
# keypoint noise scales as noise/span: on a square only a few pixels across the fit is numerically
# meaningless. Measured over 998 real frames the split is total — derived span <40 px produced a
# 68 m median range (p99 2579 m, i.e. nonsense) in 97% of cases, while >=40 px produced 1.8-16 m in
# 100% of cases, which is the close cropped-gate regime this rescue exists to serve. Those tiny
# detections are NOT cropped gates; they are distant ones where two keypoints happened to clear the
# confidence threshold. 40 px corresponds to roughly the navigator's 30-32 m vision range cap, so
# the floor discards only fixes that are rejected downstream anyway. [2026-07-21]
_MIN_DERIVED_SPAN_PX = 40.0


def gate_plane_homography(keypoints_px: np.ndarray, usable: np.ndarray) -> np.ndarray | None:
    """The gate-plane -> image homography implied by ANY >=4 usable keypoints, or None.

    The inner (1.5 m) and outer (2.72 m) squares are concentric and COPLANAR, so every keypoint --
    inner or outer -- samples the SAME plane-to-image map. Rejects only what makes the FIT itself
    meaningless: fewer than 4 usable points, a non-finite coordinate, or a structurally degenerate
    subset (all 8 keypoints lie on one of two diagonals, so "no 3 collinear" reduces to >=2 usable
    per diagonal -- see ``_DIAG_A``).

    Extracted so callers that need the PLANE, not just the inner square, share one implementation:
    ``inner_from_partial_keypoints`` (the deploy rescue) projects the inner square through it and
    adds its own vanishing-line / span guards, while the offline segmentation-label builder projects
    BOTH squares. Guards on the PROJECTION belong to the caller, since what counts as a usable
    result differs (a flight fix must be sane; a training polygon is clipped to the image anyway).
    """
    kp = np.asarray(keypoints_px, dtype=np.float64)
    m = np.asarray(usable, dtype=bool)
    if kp.shape != (8, 2) or m.shape != (8,) or int(m.sum()) < 4:
        return None
    if not (m[_DIAG_A].sum() >= 2 and m[_DIAG_B].sum() >= 2):
        return None                                   # degenerate: see _DIAG_A
    src, dst = _PLANE_ALL_8[m], kp[m]
    if not np.isfinite(dst).all():
        return None
    try:
        H, _ = cv2.findHomography(src, dst, method=0)
    except cv2.error:
        return None
    if H is None or not np.isfinite(H).all():
        return None
    return H


def project_gate_squares(H: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(inner_4x2, outer_4x2) image corners from a gate-plane homography, in canonical LL,LR,UR,UL
    order. Returns None if any corner is at or past the plane's vanishing line (w -> 0, where the
    corner wraps to the far side of the image and its pixel is meaningless). Corners may legitimately
    be OFF-FRAME -- that is the point; the caller clips."""
    out = []
    for plane in (_PLANE_INNER_32, _PLANE_OUTER_32):
        uvw = (H @ np.hstack([plane.astype(np.float64), np.ones((4, 1))]).T).T
        w = uvw[:, 2]
        if np.any(np.abs(w) < 1e-9) or (w.min() < 0.0 < w.max()):
            return None
        pts = uvw[:, :2] / w[:, None]
        if not np.isfinite(pts).all():
            return None
        out.append(pts)
    return out[0], out[1]


def inner_from_partial_keypoints(
    keypoints_px: np.ndarray,
    usable: np.ndarray,
    img_w: int = 640,
) -> np.ndarray | None:
    """The 4 inner corners implied by ANY >=4 usable keypoints, via the gate-plane homography.

    Generalises :func:`inner_from_outer_homography` (the all-outer special case): because the inner
    (1.5 m) and outer (2.72 m) squares are concentric and COPLANAR, every keypoint is a sample of the
    SAME plane-to-image homography, so a cropped gate showing e.g. two inner corners plus their two
    outer corners still determines the full gate — including the corners that are off-frame.

    ``keypoints_px`` is (8,2) in model order (inner LL,LR,UR,UL then outer LL,LR,UR,UL); ``usable``
    is a boolean (8,) mask of keypoints trusted enough to fit. Returns (4,2) inner corners in
    canonical IPPE order, or ``None`` when the fit is degenerate (fewer than 4 usable, three of them
    collinear — see ``_DIAG_A`` — or the gate plane is edge-on at its vanishing line).
    """
    kp = np.asarray(keypoints_px, dtype=np.float64)
    m = np.asarray(usable, dtype=bool)
    H = gate_plane_homography(kp, m)
    if H is None:
        return None
    src = _PLANE_ALL_8[m]
    uvw = (H @ np.hstack([_PLANE_INNER_32.astype(np.float64), np.ones((4, 1))]).T).T
    w = uvw[:, 2]
    w_ref = float(np.median(np.abs((H @ np.hstack([src, np.ones((len(src), 1))]).T).T[:, 2])))
    if w_ref < 1e-12 or np.any(np.abs(w) < _VANISH_W_FRAC * w_ref) or (w.min() < 0.0 < w.max()):
        return None                                   # at/over the vanishing line -> corners wrap
    inner = uvw[:, :2] / w[:, None]
    if not np.isfinite(inner).all() or np.abs(inner).max() > _SANE_SPAN_MULT * img_w:
        return None
    if float(np.linalg.norm(inner.max(axis=0) - inner.min(axis=0))) < _MIN_DERIVED_SPAN_PX:
        return None                                   # too small to be a meaningful fit
    return inner


def _valid_outer(obs: GateObservation, corners: np.ndarray, ids: np.ndarray,
                 ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The obs's OUTER-square corners as (px (4,2), consistency-WEIGHTED conf (4,)) — or (None, None).

    Outer corners are optional metadata (8-keypoint models only; see contracts.GateObservation),
    stored in canonical LL,LR,UR,UL order matching ``gate_object_points(GATE_OUTER_SIZE_M)``.

    CONSISTENCY WEIGHTING (the fusion's safety): the outer square only carries weight in the fit to
    the degree it geometrically CORROBORATES the measured inner corners — the inner corners implied
    by the outer square through the exact plane homography are compared to the measured ones, and
    the outer confidences are scaled by ``w = exp(-(median_disagreement / tol)^2)`` with
    ``tol = max(2.5 px, 5% of the inner apparent span)``. A 4-vs-4 inner/outer contradiction is
    unresolvable by robust weighting alone (no majority), so an outer set that tells a different
    story — decoy branding, a degraded outer head, mis-associated structure — fades to w~0 and is
    hard-DROPPED below ``_OUTER_W_FLOOR`` (the pose falls back to the proven inner-only fit,
    bit-exactly). Rescue observations (derived inners) score w~1 by construction. [outer-fusion
    2026-07-05; soft weighting after the hard-gate toggling measurement, same date]"""
    o = obs.outer_corners_px
    if o is None:
        return None, None
    o = np.ascontiguousarray(o, dtype=np.float64)
    if o.shape != (4, 2) or not np.isfinite(o).all():
        return None, None
    derived = inner_from_outer_homography(o)
    if derived is None:
        return None, None
    d = np.linalg.norm(derived[ids] - corners, axis=1)
    span = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))
    tol = max(2.5, 0.05 * span)
    w = float(np.exp(-((float(np.median(d)) / tol) ** 2)))
    if w < _OUTER_W_FLOOR:
        return None, None
    conf = (np.ones(4) if obs.outer_corner_confidence is None
            else np.asarray(obs.outer_corner_confidence, dtype=np.float64))
    return o, w * conf


def _estimate_ippe(obj: np.ndarray, img: np.ndarray, K: np.ndarray, prior: GatePose | None):
    """Full 4-corner pose via IPPE_SQUARE. Returns (R, t, reproj, ambiguity_ratio) or None."""
    cands = _solve(obj, img, K)
    if not cands:
        return None
    ambiguity_ratio: float | None = None
    if len(cands) >= 2:
        e_sorted = sorted(c[2] for c in cands)
        # Additive floor so both-near-zero errors (a frontal gate in low/zero noise) give
        # ratio ~1 (ambiguous -> use the prior) instead of the old inf sentinel that bypassed
        # the prior exactly when it is needed most. [red-team 2026-05-30]
        ambiguity_ratio = (e_sorted[1] + AMBIGUITY_EPS_PX) / (e_sorted[0] + AMBIGUITY_EPS_PX)
    # Trust the prior to break the tie only when the geometry is genuinely ambiguous;
    # otherwise the lowest-reprojection solution wins (so a stale prior can't force the
    # flipped, high-error pose). [review 2B]
    use_prior = (
        prior is not None
        and ambiguity_ratio is not None
        and ambiguity_ratio < PRIOR_DISAMBIG_MAX_RATIO
    )
    if use_prior:
        idx = min(range(len(cands)), key=lambda i: _rotation_geodesic(cands[i][0], prior.R_cam_gate))
    else:
        idx = min(range(len(cands)), key=lambda i: cands[i][2])
    R, t, reproj = cands[idx]
    return R, t, reproj, ambiguity_ratio


def _estimate_p3p(obj: np.ndarray, img: np.ndarray, K: np.ndarray, prior: GatePose | None):
    """3-corner pose via P3P. Returns (R, t, reproj, None) or None. [review 2C]

    P3P's solutions all reproject the 3 points, so they cannot be ranked by error. We keep
    only cheirality-valid (gate in front) poses and pick by the prior (near-exact) or, with
    no prior, the most head-on gate (its +Z pointing into the scene) -- a ~1-2 deg-uncertain
    last resort; callers at gate transit should supply a temporal/map prior. If NO solution is
    cheirality-valid we return None (PnP failed) rather than accepting a behind-camera pose --
    let the caller coast on IMU through the transit. [red-team 2026-05-30]
    """
    cands = _solve_p3p(obj, img, K)
    if not cands:
        return None
    in_front = [(R, t) for (R, t) in cands if np.all(((R @ obj.T).T + t)[:, 2] > 0)]
    if not in_front:
        return None
    if prior is not None:
        R, t = min(in_front, key=lambda c: _rotation_geodesic(c[0], prior.R_cam_gate))
    else:
        R, t = max(in_front, key=lambda c: float((c[0] @ _GATE_NORMAL)[2]))
    return R, t, _reproj_rms(obj, img, R, t, K), None


def _refine_pose(obj: np.ndarray, img: np.ndarray, K: np.ndarray, R0: np.ndarray, t0: np.ndarray,
                 conf: np.ndarray, sigma_px: float = WEIGHTED_SIGMA_PX, iters: int = 10):
    """Robust, confidence-weighted Gauss-Newton refinement of (R0, t0) over all corners.

    Minimises a per-corner-whitened reprojection error: corner i is whitened by
    ``sigma_i = sigma_px / clip(conf_i)`` (less confident -> looser -> lower weight) and reweighted
    each iteration by an annealed Tukey biweight on its residual -- a confident-but-wrong corner
    redescends to ZERO weight (fully rejected, not merely down-weighted). Initialised from the
    analytic IPPE/P3P pose so it refines locally without crossing the planar ambiguity. Returns
    ``(R, t, cov)`` with ``cov`` the (6,6) parameter covariance in ``[t(3), rvec(3)]`` layout from
    the effective (confidence x robust) weights (Fisher information), or ``cov=None``.
    """
    obj_c = np.ascontiguousarray(obj, dtype=np.float64).reshape(-1, 1, 3)
    img = np.ascontiguousarray(img, dtype=np.float64).reshape(-1, 2)
    rvec = cv2.Rodrigues(np.ascontiguousarray(R0, dtype=np.float64))[0].reshape(3, 1)
    tvec = np.asarray(t0, dtype=np.float64).reshape(3, 1)
    w_conf = 1.0 / (sigma_px / np.clip(np.asarray(conf, dtype=np.float64), CONF_FLOOR, 1.0)) ** 2
    robust = np.ones(obj_c.shape[0])

    for it in range(iters):
        c = TUKEY_C_HI_PX + (TUKEY_C_LO_PX - TUKEY_C_HI_PX) * (it / max(iters - 1, 1))  # anneal hi->lo
        proj, jac = cv2.projectPoints(obj_c, rvec, tvec, K, _NO_DISTORTION)
        resid = proj.reshape(-1, 2) - img                                   # (N, 2)
        u = np.linalg.norm(resid, axis=1) / c                               # normalised residual
        robust = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0)                 # Tukey biweight (redescending)
        if int((robust > 0).sum()) < 3:                                     # too few inliers to solve
            break
        w2 = np.repeat(w_conf * robust, 2)                                  # (2N,) per residual coord
        J = np.asarray(jac, dtype=np.float64)[:, 0:6]                       # d[u,v] / d[rvec, tvec]
        JtW = J.T * w2
        try:
            dp = np.linalg.solve(JtW @ J, JtW @ resid.reshape(-1))
        except np.linalg.LinAlgError:
            break
        rvec = rvec - dp[0:3].reshape(3, 1)
        tvec = tvec - dp[3:6].reshape(3, 1)
        if float(np.linalg.norm(dp)) < 1e-9:
            break

    R, t = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
    cov = None
    try:                                                                    # Fisher info -> covariance
        _, jac = cv2.projectPoints(obj_c, rvec, tvec, K, _NO_DISTORTION)
        J = np.asarray(jac, dtype=np.float64)[:, 0:6]
        cov_rt = np.linalg.inv((J.T * np.repeat(w_conf * robust, 2)) @ J)   # effective inliers, [rvec, tvec]
        swap = np.zeros((6, 6))
        swap[0:3, 3:6] = swap[3:6, 0:3] = np.eye(3)
        cov = swap @ cov_rt @ swap.T                                        # -> [t, rvec]
    except np.linalg.LinAlgError:
        cov = None
    return R, t, cov


def _refine_ok(R: np.ndarray, t: np.ndarray, R0: np.ndarray, t0: np.ndarray, obj: np.ndarray) -> bool:
    """Accept a refinement only if it stayed valid (gate in front, finite) and didn't wander far
    from the PnP init -- a diverged GN step is discarded in favour of the analytic IPPE/P3P pose."""
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(R))):
        return False
    if np.any(((R @ obj.T).T + t)[:, 2] <= 0):
        return False
    if _rotation_geodesic(R, R0) > 0.5:
        return False
    return bool(np.linalg.norm(t - t0) <= 0.5 * (np.linalg.norm(t0) + 1e-9))


def estimate_gate_pose(
    obs: GateObservation,
    camera_matrix: np.ndarray | None = None,
    inner_size_m: float = GATE_INNER_SIZE_M,
    prior: GatePose | None = None,
    compute_covariance: bool = False,
    corner_sigma_px: float = 1.5,
    n_samples: int = 24,
    rng: np.random.Generator | None = None,
    weighted_refine: bool = True,
    use_outer: bool = True,
) -> GatePose | None:
    """Estimate the gate pose from an observation. Returns ``None`` if PnP fails.

    With 4 corners, uses IPPE_SQUARE; with 3 (a clipped gate at transit), falls back to
    P3P over the matching corners. ``prior`` disambiguates the planar 2-fold ambiguity
    (IPPE, only when the geometry is near-frontal) and the P3P solutions (always, since
    they are otherwise indistinguishable).

    OUTER-corner fusion (2026-07-05, ``use_outer``): when the obs carries the 4 OUTER-square
    corners (8-keypoint models; concentric + coplanar 2.72 m square, see GATE_OUTER_SIZE_M),
    they join the SAME single pose fit:
      - the robust refinement runs over ALL available points (up to 8) — the statistically
        correct inner+outer fusion (confidence-whitened + Tukey), halving keypoint-noise-driven
        pose jitter vs inner-only and anchoring the fit on the high-contrast outer corners;
      - 3 inner + 4 confident outer: the OUTER square (a full IPPE_SQUARE problem) replaces the
        weakly-constrained P3P init, and the fix reports ``n_corners=4`` (a 7-point fit is
        better-constrained than plain inner-4, so the localization P3P inflation must not fire);
      - inner-4 IPPE failure falls back to an outer-square IPPE init.
    The emitted :class:`GatePose` contract is UNCHANGED (same fields, same meaning — centre
    translation + rotation of the SAME gate frame); ``reproj_error_px`` stays the INNER-corner
    RMS so its diagnostic scale is comparable across 4- and 8-keypoint models.

    With ``weighted_refine`` (default; requires ``obs.corner_confidence``), the
    IPPE pose is refined by a robust confidence-weighted Gauss-Newton over all corners: each
    corner is weighted by its confidence and Huber-reweighted by its residual, so a weak or
    mislocalised corner is downweighted instead of trusted equally -- and the per-corner weights
    give the pose covariance directly. With ``compute_covariance``, that analytic
    covariance over ``[t(3), rvec(3)]`` is returned (falling back to Monte-Carlo corner perturbation
    -- ``n_samples`` x, std ``corner_sigma_px`` -- when no confidences are present); plain 3-corner
    P3P fixes return ``None`` covariance and ``n_corners=3`` so callers inflate their measurement noise.
    """
    if not np.isfinite(obs.corners_px).all():
        return None
    # M+1 CENTRE-BEARING observations (2026-07-23) legitimately carry <3 inner corners -- the
    # regressed centre survives corner cropping, the pose does not. No pose is recoverable from
    # fewer than 3 points, so decline here rather than let P3P be handed a degenerate set; the
    # caller reads the 3-D emit off ``centre_px`` via racer.vision.centre_emit instead.
    if obs.corners_px.shape[0] < 3:
        return None
    K = CAMERA_INTRINSICS_K if camera_matrix is None else camera_matrix
    corners, ids, conf = _ordered_corners(obs)
    obj_full = gate_object_points(inner_size_m)
    n = corners.shape[0]
    outer_px, outer_conf = _valid_outer(obs, corners, ids) if use_outer else (None, None)
    # Outer corners join the INIT only when all 4 are confidently localised (the square solver
    # needs a trustworthy full square); the REFINEMENT below still weights each one individually.
    outer_init_ok = outer_px is not None and outer_conf is not None and bool((outer_conf >= 0.5).all())
    obj_outer = gate_object_points(GATE_OUTER_SIZE_M)

    if n == 4:
        result = _estimate_ippe(obj_full, corners, K, prior)
        if result is None and outer_init_ok:
            result = _estimate_ippe(obj_outer, outer_px, K, prior)   # outer-square fallback init
        n_corners, obj = 4, obj_full
    elif n == 3:
        obj = obj_full[ids]
        if outer_init_ok:
            # OUTER-ASSISTED 3-inner path: the outer square is a FULL IPPE_SQUARE problem — a
            # strictly better-constrained init than 3-point P3P (which cannot self-disambiguate).
            # The 3 inner corners join the joint refinement; report n_corners=4 (a 7-point fit),
            # so localization's x9 P3P covariance inflation does not fire on a strong fix.
            result = _estimate_ippe(obj_outer, outer_px, K, prior)
            n_corners = 4 if result is not None else 3
            if result is None:
                result = _estimate_p3p(obj, corners, K, prior)
        else:
            result = _estimate_p3p(obj, corners, K, prior)
            n_corners = 3
    else:
        return None
    if result is None:
        return None
    R, t, reproj, ambiguity_ratio = result
    reproj = _reproj_rms(obj, corners, R, t, K)   # INNER-corner RMS (diagnostic scale invariant)

    # A RESCUED observation (corners reconstructed through the gate-plane homography from a cropped
    # gate, contracts.GateObservation.derived_corners) is solved as a full square — that is the
    # better-conditioned fit — but REPORTED as n_corners=3, the established "trust this less" signal.
    # Measured: rescued centres carry ~3x the error of a measured 4-corner fix, which is the same
    # order as a P3P fix, so the existing x9 covariance inflation is approximately right. Reporting
    # 4 here would hand the estimator and the RL policy a cropped-gate fix disguised as a full one.
    if getattr(obs, "derived_corners", False):
        n_corners = 3

    # Robust confidence-weighted refinement over ALL available corners (inner + outer): downweights
    # a weak/wrong corner instead of trusting it equally (or hard-dropping it), and yields the pose
    # covariance. Only when the detector supplied per-corner confidences; guarded against a diverged
    # GN step. This joint fit IS the inner/outer fusion — one pose explains both squares.
    analytic_cov = None
    if weighted_refine and conf is not None and n_corners == 4:
        obj_r, img_r, conf_r = obj, corners, conf
        if outer_px is not None and outer_conf is not None:
            obj_r = np.vstack([obj_r, obj_outer])
            img_r = np.vstack([img_r, outer_px])
            conf_r = np.concatenate([conf_r, outer_conf])
        R_r, t_r, cov_r = _refine_pose(obj_r, img_r, K, R, t, conf_r, corner_sigma_px)
        if _refine_ok(R_r, t_r, R, t, obj_r):
            R, t, analytic_cov = R_r, t_r, cov_r
            reproj = _reproj_rms(obj, corners, R, t, K)

    covariance = None
    if compute_covariance and n_corners == 4:
        covariance = analytic_cov if analytic_cov is not None else _corner_perturbation_covariance(
            obj_full if n == 4 else obj, corners, K, R, corner_sigma_px, n_samples, rng
        )

    return GatePose(
        frame_id=obs.frame_id,
        sim_time_ns=obs.sim_time_ns,
        R_cam_gate=R,
        t_cam_gate=t,
        reproj_error_px=reproj,
        gate_id=obs.gate_id,
        covariance=covariance,
        ambiguity_ratio=ambiguity_ratio,
        n_corners=n_corners,
    )


def _corner_perturbation_covariance(
    obj: np.ndarray,
    corners_px: np.ndarray,
    K: np.ndarray,
    R_ref: np.ndarray,
    sigma_px: float,
    n_samples: int,
    rng: np.random.Generator | None,
) -> np.ndarray | None:
    """Monte-Carlo measurement covariance: jitter corners, re-solve, sample-cov of [t, rvec].

    Layout: ``[tx, ty, tz, rx, ry, rz]`` where (rx,ry,rz) is the rotation vector. The
    consistent IPPE branch (closest rotation to ``R_ref``) is taken each sample.
    """
    gen = np.random.default_rng() if rng is None else rng
    samples = []
    for _ in range(n_samples):
        noisy = corners_px + gen.normal(0.0, sigma_px, size=corners_px.shape)
        cands = _solve(obj, noisy, K)
        if not cands:
            continue
        R, t, _ = min(cands, key=lambda c: _rotation_geodesic(c[0], R_ref))
        rvec = cv2.Rodrigues(R)[0].reshape(3)
        samples.append(np.concatenate([t, rvec]))
    if len(samples) < 6:
        return None
    return np.cov(np.asarray(samples), rowvar=False)
