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

GATE_INNER_SIZE_M = 1.5  # spec sec 3.7: inner square 1500 mm
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
) -> GatePose | None:
    """Estimate the gate pose from an observation. Returns ``None`` if PnP fails.

    With 4 corners, uses IPPE_SQUARE; with 3 (a clipped gate at transit), falls back to
    P3P over the matching corners. ``prior`` disambiguates the planar 2-fold ambiguity
    (IPPE, only when the geometry is near-frontal) and the P3P solutions (always, since
    they are otherwise indistinguishable).

    With ``weighted_refine`` (default; 4-corner, requires ``obs.corner_confidence``), the
    IPPE pose is refined by a robust confidence-weighted Gauss-Newton over all corners: each
    corner is weighted by its confidence and Huber-reweighted by its residual, so a weak or
    mislocalised corner is downweighted instead of trusted equally -- and the per-corner weights
    give the pose covariance directly. With ``compute_covariance`` (4-corner only), that analytic
    covariance over ``[t(3), rvec(3)]`` is returned (falling back to Monte-Carlo corner perturbation
    -- ``n_samples`` x, std ``corner_sigma_px`` -- when no confidences are present); 3-corner fixes
    return ``None`` covariance and ``n_corners=3`` so callers inflate their measurement noise.
    """
    if not np.isfinite(obs.corners_px).all():
        return None
    K = CAMERA_INTRINSICS_K if camera_matrix is None else camera_matrix
    corners, ids, conf = _ordered_corners(obs)
    obj_full = gate_object_points(inner_size_m)
    n = corners.shape[0]

    if n == 4:
        result = _estimate_ippe(obj_full, corners, K, prior)
        n_corners, obj = 4, obj_full
    elif n == 3:
        obj = obj_full[ids]
        result = _estimate_p3p(obj, corners, K, prior)
        n_corners = 3
    else:
        return None
    if result is None:
        return None
    R, t, reproj, ambiguity_ratio = result

    # Robust confidence-weighted refinement over all 4 corners: downweights a weak/wrong corner
    # instead of trusting it equally (or hard-dropping it), and yields the pose covariance. Only
    # when the detector supplied per-corner confidences; guarded against a diverged GN step.
    analytic_cov = None
    if weighted_refine and conf is not None and n_corners == 4:
        R_r, t_r, cov_r = _refine_pose(obj_full, corners, K, R, t, conf, corner_sigma_px)
        if _refine_ok(R_r, t_r, R, t, obj_full):
            R, t, analytic_cov = R_r, t_r, cov_r
            reproj = _reproj_rms(obj_full, corners, R, t, K)

    covariance = None
    if compute_covariance and n_corners == 4:
        covariance = analytic_cov if analytic_cov is not None else _corner_perturbation_covariance(
            obj_full, corners, K, R, corner_sigma_px, n_samples, rng
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
