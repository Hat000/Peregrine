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
    """RMS pixel reprojection error of (R, t) over the given object/image correspondences."""
    cam = (R @ obj_pts.T).T + t
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


def _ordered_corners(obs: GateObservation) -> tuple[np.ndarray, np.ndarray]:
    """Return (corners, ids) sorted by canonical corner index, so rows match object points."""
    corners = np.ascontiguousarray(obs.corners_px, dtype=np.float64)
    n = corners.shape[0]
    ids = np.arange(n) if obs.corner_ids is None else np.asarray(obs.corner_ids).astype(int)
    order = np.argsort(ids, kind="stable")
    return corners[order], ids[order]


def _estimate_ippe(obj: np.ndarray, img: np.ndarray, K: np.ndarray, prior: GatePose | None):
    """Full 4-corner pose via IPPE_SQUARE. Returns (R, t, reproj, ambiguity_ratio) or None."""
    cands = _solve(obj, img, K)
    if not cands:
        return None
    ambiguity_ratio: float | None = None
    if len(cands) >= 2:
        e_sorted = sorted(c[2] for c in cands)
        ambiguity_ratio = e_sorted[1] / e_sorted[0] if e_sorted[0] > 1e-9 else float("inf")
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
    last resort; callers at gate transit should supply a temporal/map prior.
    """
    cands = _solve_p3p(obj, img, K)
    if not cands:
        return None
    in_front = [(R, t) for (R, t) in cands if np.all(((R @ obj.T).T + t)[:, 2] > 0)]
    valid = in_front or cands
    if prior is not None:
        R, t = min(valid, key=lambda c: _rotation_geodesic(c[0], prior.R_cam_gate))
    else:
        R, t = max(valid, key=lambda c: float((c[0] @ _GATE_NORMAL)[2]))
    return R, t, _reproj_rms(obj, img, R, t, K), None


def estimate_gate_pose(
    obs: GateObservation,
    camera_matrix: np.ndarray | None = None,
    inner_size_m: float = GATE_INNER_SIZE_M,
    prior: GatePose | None = None,
    compute_covariance: bool = False,
    corner_sigma_px: float = 1.5,
    n_samples: int = 24,
    rng: np.random.Generator | None = None,
) -> GatePose | None:
    """Estimate the gate pose from an observation. Returns ``None`` if PnP fails.

    With 4 corners, uses IPPE_SQUARE; with 3 (a clipped gate at transit), falls back to
    P3P over the matching corners. ``prior`` disambiguates the planar 2-fold ambiguity
    (IPPE, only when the geometry is near-frontal) and the P3P solutions (always, since
    they are otherwise indistinguishable). With ``compute_covariance`` (4-corner only), a
    (6,6) measurement covariance over ``[t(3), rvec(3)]`` is estimated by perturbing the
    corners (``n_samples`` x, std ``corner_sigma_px``); 3-corner fixes return ``None``
    covariance and ``n_corners=3`` so callers inflate their measurement noise.
    """
    if not np.isfinite(obs.corners_px).all():
        return None
    K = CAMERA_INTRINSICS_K if camera_matrix is None else camera_matrix
    corners, ids = _ordered_corners(obs)
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

    covariance = None
    if compute_covariance and n_corners == 4:
        covariance = _corner_perturbation_covariance(
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
