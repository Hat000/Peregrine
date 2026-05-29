"""Gate pose from detected corners via planar PnP (IPPE).

Turns a :class:`GateObservation` (4 inner-square corners in pixels) into a
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

IPPE returns the planar 2-fold ambiguity (two solutions). We pick the lower-
reprojection-error one, or — when a ``prior`` pose is supplied — the one whose
rotation is closest to it (temporal disambiguation). ``ambiguity_ratio`` =
err_second / err_first reports how separable the two were (>>1 = unambiguous).

Spec: VADR-TS-002 sec 3.7 (gate), sec 3.8 (intrinsics, no distortion).
"""
from __future__ import annotations

import cv2
import numpy as np

from racer.contracts import GateObservation, GatePose
from racer.frames import CAMERA_INTRINSICS_K

GATE_INNER_SIZE_M = 1.5  # spec sec 3.7: inner square 1500 mm
_NO_DISTORTION = np.zeros((4, 1), dtype=np.float64)


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

    If ``prior`` is given, the ambiguous solution closest to ``prior.R_cam_gate`` is
    chosen (temporal disambiguation); otherwise the lower-reprojection-error one.
    With ``compute_covariance``, a (6,6) measurement covariance over ``[t(3), rvec(3)]``
    is estimated by perturbing the corners (``n_samples`` x, std ``corner_sigma_px``).
    """
    if not np.isfinite(obs.corners_px).all():
        return None
    K = CAMERA_INTRINSICS_K if camera_matrix is None else camera_matrix
    obj = gate_object_points(inner_size_m)
    cands = _solve(obj, obs.corners_px, K)
    if not cands:
        return None

    if prior is None:
        idx = min(range(len(cands)), key=lambda i: cands[i][2])
    else:
        idx = min(range(len(cands)), key=lambda i: _rotation_geodesic(cands[i][0], prior.R_cam_gate))
    R, t, reproj = cands[idx]

    ambiguity_ratio: float | None = None
    if len(cands) >= 2:
        e_sorted = sorted(c[2] for c in cands)
        ambiguity_ratio = e_sorted[1] / e_sorted[0] if e_sorted[0] > 1e-9 else float("inf")

    covariance = None
    if compute_covariance:
        covariance = _corner_perturbation_covariance(
            obj, obs.corners_px, K, R, corner_sigma_px, n_samples, rng
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
