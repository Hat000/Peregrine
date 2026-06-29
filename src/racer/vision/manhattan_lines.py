"""Shared line / vanishing-point primitives for the Manhattan-world vision cues.

VQ2 (build 1.0.3379, "Now You See Me, Now You Don't") blocks ATTITUDE / LOCAL_POSITION_NED /
ODOMETRY on the wire and the IMU carries accel + gyro ONLY (no magnetometer, no barometer). So
the drone's absolute YAW (heading) and Z (height) are inertially unobservable and MUST be pinned
by vision (see ``docs/reactivation-2026-06-27/magfree-vision-yaw-scope.md``). The map-free anchor
is the warehouse's **Manhattan-world structure**: a regular rectilinear floor grid, a parallel
ceiling truss, vertical pillars and bright blue lane-lines, all axis-aligned to the warehouse
frame (recon frames 02/04/05/07). That structure is present even with no gate in view.

This module holds the low-level pieces both :mod:`racer.vision.heading_vp` (absolute yaw from
vanishing points) and :mod:`racer.vision.floor_height` (camera height from the floor plane) share:

* line-segment extraction (LSD with a Hough fallback) tuned for the bright-on-dark warehouse,
* homogeneous line / vanishing-point algebra,
* a RANSAC vanishing-point solver over segment directions.

It is PURE GEOMETRY / IMAGE MATH — no estimator state, no wiring. The recon line counts are
ample even in the darkest / motion-blurred frames (LSD finds ~300-600 segments at mean-gray 11-43),
so VP extraction is well-fed; the binding failure mode is motion blur smearing thin lines, gated
downstream by the inlier count / quality measures the consumers expose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # OpenCV is already a hard dep of gate_pose / red_glow_detector.
    import cv2
except Exception as exc:  # pragma: no cover - cv2 is installed in the test venv
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = exc


# --- Line-segment extraction -----------------------------------------------------------------

def extract_line_segments(
    image_bgr: np.ndarray,
    min_length_px: float = 25.0,
) -> np.ndarray:
    """Detect straight line segments in a BGR frame -> (N,4) array of ``[x0,y0,x1,y1]``.

    Uses OpenCV's :class:`LineSegmentDetector` (sub-pixel, no thresholds to tune) when it is
    available, falling back to ``Canny`` + ``HoughLinesP`` otherwise (some OpenCV builds ship
    LSD disabled for the old patent). Segments shorter than ``min_length_px`` are dropped — short
    segments are dominated by texture/noise and add no vanishing-point leverage. Returns an empty
    ``(0,4)`` array on a None/blank frame so callers can gate on ``len(...)``.
    """
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(f"manhattan_lines requires OpenCV (cv2): {_CV2_IMPORT_ERROR}")
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be (H,W,3) BGR uint8")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    segs: np.ndarray | None = None
    if hasattr(cv2, "createLineSegmentDetector"):
        try:
            lsd = cv2.createLineSegmentDetector()
            detected = lsd.detect(gray)[0]
            if detected is not None:
                segs = detected.reshape(-1, 4)
        except cv2.error:  # pragma: no cover - LSD disabled in some builds
            segs = None
    if segs is None:
        edges = cv2.Canny(gray, 30, 90)
        hl = cv2.HoughLinesP(
            edges, 1, np.pi / 180.0, threshold=40,
            minLineLength=int(min_length_px), maxLineGap=8,
        )
        segs = np.zeros((0, 4)) if hl is None else hl.reshape(-1, 4).astype(np.float64)

    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    if len(segs) == 0:
        return segs
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[lengths >= float(min_length_px)]


# --- Line / vanishing-point algebra ----------------------------------------------------------

def segment_homog_lines(segs: np.ndarray) -> np.ndarray:
    """(N,4) segments -> (N,3) homogeneous image lines through each segment's endpoints."""
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    p1 = np.column_stack([segs[:, 0], segs[:, 1], np.ones(len(segs))])
    p2 = np.column_stack([segs[:, 2], segs[:, 3], np.ones(len(segs))])
    return np.cross(p1, p2)


def segment_midpoints(segs: np.ndarray) -> np.ndarray:
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    return np.column_stack([(segs[:, 0] + segs[:, 2]) * 0.5, (segs[:, 1] + segs[:, 3]) * 0.5])


def segment_angles(segs: np.ndarray) -> np.ndarray:
    """Orientation (rad, in [-pi/2, pi/2)) of each segment in image coords (x right, y down)."""
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    a = np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])
    return (a + np.pi / 2.0) % np.pi - np.pi / 2.0


def _vp_consistency_residual(vp_px: np.ndarray, mids: np.ndarray, angs: np.ndarray) -> np.ndarray:
    """Per-segment angular residual (rad) between a segment and the line from its midpoint to the
    candidate vanishing point. A segment that passes through ``vp_px`` has residual 0. The wrap to
    [0, pi/2] makes it direction-agnostic (a line and its reversal are the same).

    Accepts either a single VP ``(2,)`` -> ``(N,)`` residuals, or a batch of VPs ``(M,2)`` ->
    ``(M,N)`` residuals (one row per VP). The batched form lets a RANSAC score every hypothesis in
    a single vectorized pass instead of a Python loop. The scalar computation is byte-identical to
    the per-VP path for any given VP (broadcasting only changes the iteration, not the arithmetic).
    """
    vp_px = np.asarray(vp_px, dtype=np.float64)
    if vp_px.ndim == 1:
        d = vp_px[None, :] - mids
        a_to_vp = np.arctan2(d[:, 1], d[:, 0])
        return np.abs(((a_to_vp - angs + np.pi / 2.0) % np.pi) - np.pi / 2.0)
    # Batched: vp_px (M,2), mids (N,2), angs (N,) -> (M,N).
    dx = vp_px[:, 0][:, None] - mids[:, 0][None, :]
    dy = vp_px[:, 1][:, None] - mids[:, 1][None, :]
    a_to_vp = np.arctan2(dy, dx)
    return np.abs(((a_to_vp - angs[None, :] + np.pi / 2.0) % np.pi) - np.pi / 2.0)


@dataclass(frozen=True)
class VanishingPoint:
    """One vanishing point recovered from a segment cluster.

    point_px      : (2,) image-plane vanishing point (may be far outside the frame; near-infinite
                    for a near-fronto-parallel pencil — see ``at_infinity``).
    inlier_mask   : (N,) bool over the input segments supporting this VP.
    inlier_count  : convenience int.
    rms_resid_rad : RMS angular residual of the inliers (lower = tighter pencil).
    at_infinity   : True when the supporting pencil is so parallel the VP is effectively at
                    infinity (the direction is then carried by ``direction_px``).
    direction_px  : (2,) unit image direction of the pencil (robust even when ``point_px`` blows up).
    """

    point_px: np.ndarray
    inlier_mask: np.ndarray
    inlier_count: int
    rms_resid_rad: float
    at_infinity: bool
    direction_px: np.ndarray


def fit_vanishing_point(
    segs: np.ndarray,
    inlier_thresh_deg: float = 1.5,
    iters: int = 2000,
    min_segments: int = 8,
    seed: int = 0,
) -> VanishingPoint | None:
    """RANSAC the dominant vanishing point of a set of segments.

    Samples pairs of segments, intersects their homogeneous lines for a VP hypothesis, and scores
    by the count of segments whose midpoint-to-VP direction matches their own orientation within
    ``inlier_thresh_deg``. Returns the best-supported VP (refined by a least-squares re-fit over
    its inliers) or ``None`` when there is not enough structure. Deterministic given ``seed``.
    """
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    n = len(segs)
    if n < int(min_segments):
        return None
    lines = segment_homog_lines(segs)
    mids = segment_midpoints(segs)
    angs = segment_angles(segs)
    thr = np.deg2rad(inlier_thresh_deg)
    iters = int(iters)
    rng = np.random.default_rng(seed)

    # --- Draw the SAME hypothesis pairs the original per-iteration loop would have ----------------
    # Replaying ``rng.choice(n, 2, replace=False)`` once per iteration into a pre-allocated array
    # consumes the RNG stream in the exact same order, so ``pairs[k]`` == the ``(i, j)`` the old
    # ``for _ in range(iters)`` loop produced at iteration ``k`` (bit-for-bit, given ``seed``). Only
    # the SCORING is vectorized below — the hypotheses are identical, so the chosen VP is identical.
    pairs = np.empty((iters, 2), dtype=np.intp)
    for k in range(iters):
        pairs[k] = rng.choice(n, 2, replace=False)
    li = lines[pairs[:, 0]]  # (M,3)
    lj = lines[pairs[:, 1]]  # (M,3)

    # --- Vectorized homogeneous intersection via EXPLICIT cross-product components ----------------
    # cross(a, b) = (a_y b_z - a_z b_y, a_z b_x - a_x b_z, a_x b_y - a_y b_x). The old loop called
    # ``np.cross`` on tiny 3-vectors 2000x/VP — numpy's np.cross is dominated by
    # moveaxis/normalize_axis_tuple overhead for tiny arrays, so this microcall-in-a-Python-loop was
    # the bulk of the 448 ms/tick. The explicit batched scalar form is mathematically identical and
    # collapses all 2000 hypotheses into a handful of array ops.
    ax, ay, az = li[:, 0], li[:, 1], li[:, 2]
    bx, by, bz = lj[:, 0], lj[:, 1], lj[:, 2]
    vh0 = ay * bz - az * by
    vh1 = az * bx - ax * bz
    vh2 = ax * by - ay * bx  # (M,)

    # Parallel-in-image hypotheses (|w| < 1e-9) were ``continue``d (never scored) in the original.
    valid = np.abs(vh2) >= 1e-9
    safe_w = np.where(valid, vh2, 1.0)  # avoid div-by-zero; masked out below
    vp_batch = np.column_stack([vh0 / safe_w, vh1 / safe_w])  # (M,2)

    # --- Score every hypothesis in one pass: (M,N) residual matrix -> per-hypothesis inlier count --
    resid = _vp_consistency_residual(vp_batch, mids, angs)  # (M,N)
    masks = resid < thr
    scores = masks.sum(axis=1).astype(np.int64)  # (M,)
    scores[~valid] = -1  # parallel hypotheses can never win (matches the original's skip)

    if iters == 0 or not valid.any():
        return None
    # ``np.argmax`` returns the FIRST maximal index -> identical tie-break to the original loop's
    # ``if score > best_score`` (strict >, keeps the earliest-iteration hypothesis on ties).
    best_iter = int(np.argmax(scores))
    best_score = int(scores[best_iter])
    best_mask = masks[best_iter]

    if best_score < 0 or best_score < max(int(min_segments), 4):
        return None

    return _refine_vp(segs, lines, mids, angs, best_mask, thr)


def _refine_vp(
    segs: np.ndarray,
    lines: np.ndarray,
    mids: np.ndarray,
    angs: np.ndarray,
    mask: np.ndarray,
    thr: float,
) -> VanishingPoint | None:
    """Least-squares VP over the inlier lines (Hartley-normalised), then re-score residuals.

    The VP is the point minimising the algebraic distance to all inlier lines: the right-null
    vector of the stacked inlier-line matrix. Robust to the VP being at/near infinity (we keep the
    full homogeneous solution and only de-homogenise when finite)."""
    inl_lines = lines[mask]
    if len(inl_lines) < 2:
        return None
    # Normalise each line so its (a,b) part is unit -> well-conditioned SVD.
    norms = np.linalg.norm(inl_lines[:, :2], axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    A = inl_lines / norms
    _, _, vh = np.linalg.svd(A)
    vp_h = vh[-1]

    if abs(vp_h[2]) < 1e-9 * np.linalg.norm(vp_h[:2]):
        at_infinity = True
        direction = vp_h[:2] / (np.linalg.norm(vp_h[:2]) + 1e-12)
        # A finite stand-in far along the direction lets the residual re-score behave.
        point = direction * 1e6
    else:
        at_infinity = False
        point = vp_h[:2] / vp_h[2]
        # direction = mean inlier orientation (sign-free): use the dominant segment direction
        ca, sa = np.cos(2 * angs[mask]).mean(), np.sin(2 * angs[mask]).mean()
        mean_ang = 0.5 * np.arctan2(sa, ca)
        direction = np.array([np.cos(mean_ang), np.sin(mean_ang)])

    resid = _vp_consistency_residual(point, mids, angs)
    final_mask = resid < thr
    if int(final_mask.sum()) < 2:
        final_mask = mask
    rms = float(np.sqrt(np.mean(resid[final_mask] ** 2))) if final_mask.any() else float("inf")
    return VanishingPoint(
        point_px=point,
        inlier_mask=final_mask,
        inlier_count=int(final_mask.sum()),
        rms_resid_rad=rms,
        at_infinity=at_infinity,
        direction_px=direction,
    )


def fit_multiple_vanishing_points(
    segs: np.ndarray,
    n_vps: int = 3,
    inlier_thresh_deg: float = 1.5,
    iters: int = 2000,
    min_inliers: int = 8,
    seed: int = 0,
) -> list[VanishingPoint]:
    """Greedily peel off up to ``n_vps`` vanishing points (strongest first), removing inliers
    between rounds. Manhattan scenes have 2 horizontal + 1 vertical dominant VPs."""
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    remaining = np.ones(len(segs), dtype=bool)
    out: list[VanishingPoint] = []
    for k in range(int(n_vps)):
        idx = np.flatnonzero(remaining)
        if len(idx) < max(int(min_inliers), 8):
            break
        vp = fit_vanishing_point(
            segs[idx], inlier_thresh_deg=inlier_thresh_deg, iters=iters,
            min_segments=max(int(min_inliers), 8), seed=seed + k,
        )
        if vp is None or vp.inlier_count < int(min_inliers):
            break
        # Map the sub-problem inlier mask back to the global index space.
        global_mask = np.zeros(len(segs), dtype=bool)
        global_mask[idx[vp.inlier_mask]] = True
        out.append(
            VanishingPoint(
                point_px=vp.point_px,
                inlier_mask=global_mask,
                inlier_count=int(global_mask.sum()),
                rms_resid_rad=vp.rms_resid_rad,
                at_infinity=vp.at_infinity,
                direction_px=vp.direction_px,
            )
        )
        remaining &= ~global_mask
    return out
