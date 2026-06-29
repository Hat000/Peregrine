"""Absolute heading (yaw) from Manhattan-world vanishing points — a MAP-FREE yaw source for VQ2.

THE PROBLEM. VQ2 confirms the wire carries accel + gyro only (no magnetometer, no barometer) and
blocks ATTITUDE/LOCAL_POSITION_NED/ODOMETRY. Gravity (accel) observes roll/pitch, but **nothing
inertial observes yaw**: the ESKF accel update is rank-2 (``H = -skew(g_hat)`` has a zero yaw
column), so yaw is a free integrator on gyro-z bias and drifts unbounded. The magnetometer that
would have pinned it is absent. The drift-free yaw reference must come from vision.

THE CUE. The warehouse is a Manhattan world: a regular floor grid, a parallel ceiling truss,
vertical pillars and bright blue lane-lines, all axis-aligned to the warehouse frame (recon frames
02/04/05/07; LSD finds ~300-600 segments even in the darkest/blurred frames). The two orthogonal
**horizontal** vanishing directions fix the camera's absolute yaw in the warehouse frame. This
function would do the magnetometer's job — drift-free, no gate, no map.

WHAT THIS MODULE DOES. ``estimate_heading(frame, roll, pitch)`` extracts line segments, fits the
dominant vanishing points, selects the most-horizontal one (a horizontal VP encodes a world
heading; the vertical VP / gravity axis is already pinned by accel and is rejected here), and
converts it — using the +20deg camera mount and the gravity-known roll/pitch — into an absolute
warehouse yaw. It returns a :class:`HeadingEstimate` with the heading, a quality measure and the
supporting line count, or ``None`` when there is not enough structure.

THE 90deg LATTICE AMBIGUITY (read this).  A Manhattan grid's horizontal VP is only defined **modulo
90deg**: swapping the warehouse "x" and "y" axes is indistinguishable from a single horizontal VP.
We therefore return the heading **wrapped into [-45deg, +45deg)** (``heading_mod90_rad``) and DO NOT
silently pick a 90deg branch. Disambiguation — adding the correct multiple of 90deg — is the
CONSUMER's job at the wiring step, via gyro-yaw continuity (never let the reported absolute heading
jump 90deg between frames) and/or a gate-bearing prior. This module exposes the raw mod-90 datum and
the candidate branch headings; it does not commit to one. (See the scope doc §1.2 / §5.)

OUTPUT CONTRACT (for the later ESKF pseudo-measurement step; this module does NOT wire anything):
    HeadingEstimate(
        heading_mod90_rad : float   # absolute warehouse yaw wrapped to [-pi/4, +pi/4)
        quality           : float   # in [0,1]; gate the ESKF update on this
        n_support         : int     # supporting (inlier) line count
        vp_px             : (2,)     # the horizontal VP used (image pixels)
        horizontality     : float   # |world-z| of the recovered direction (0 = perfectly level)
        branch_headings_rad : (4,)   # the four 90deg-branch absolute headings (consumer disambiguates)
    )

SIGN CONVENTION (FD-checked against ``frames.py``). Yaw is the aerospace 3-2-1 yaw about world-down
(NED), matching ``R_world_from_body(roll, pitch, yaw)``. A horizontal world direction ``d_world``
projects to VP pixel ``K @ R_camera_from_body() @ R_world_from_body(r,p,y).T @ d_world``; we invert
that. yaw=0 means the camera's optical-axis horizontal projection looks down the warehouse "x" axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from racer.frames import CAMERA_INTRINSICS_K, R_camera_from_body
from racer.vision.manhattan_lines import (
    VanishingPoint,
    extract_line_segments,
    fit_multiple_vanishing_points,
)

_K_INV = np.linalg.inv(CAMERA_INTRINSICS_K)


@dataclass(frozen=True)
class HeadingEstimate:
    """Absolute warehouse heading from a horizontal vanishing point. See module docstring for the
    full contract and the 90deg-lattice caveat (``heading_mod90_rad`` is mod 90deg by construction;
    the consumer disambiguates the branch)."""

    heading_mod90_rad: float
    quality: float
    n_support: int
    vp_px: np.ndarray
    horizontality: float
    branch_headings_rad: np.ndarray


def yaw_from_vp_pixel(
    u: float, v: float, roll: float, pitch: float
) -> tuple[float, float]:
    """Convert a vanishing-point pixel into the warehouse yaw of the horizontal world direction it
    encodes, given gravity-known ``roll``/``pitch``. Returns ``(yaw_rad, world_z)`` where
    ``world_z`` is the vertical component of the recovered world direction — near 0 for a true
    horizontal VP, large for the vertical/pillar VP (used to reject non-horizontal VPs).

    Derivation (FD-checked): a pixel back-projects to a camera ray ``d_cam = K^-1 [u,v,1]``; the
    body ray is ``d_body = R_camera_from_body().T @ d_cam`` (mount-aware, independent of yaw). With
    the gravity-known roll/pitch, ``w = Ry(pitch) Rx(roll) @ d_body`` is the world direction up to
    the yaw rotation about world-down. ``d_world = Rz(yaw) @ w``; choosing ``yaw = -atan2(w_y, w_x)``
    aligns this direction with the warehouse x-axis (heading 0). ``world_z = w_z`` is yaw-invariant.
    """
    d_cam = _K_INV @ np.array([float(u), float(v), 1.0])
    n = np.linalg.norm(d_cam)
    if n < 1e-12:
        return 0.0, 1.0
    d_cam /= n
    d_body = R_camera_from_body().T @ d_cam
    # Undo roll then pitch (the gravity-observed tilt) to land in a yaw-only-rotated world frame.
    w = Rotation.from_euler("YX", [float(pitch), float(roll)]).as_matrix() @ d_body
    yaw = -np.arctan2(w[1], w[0])
    return float(yaw), float(w[2])


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _wrap_mod90(a: float) -> float:
    """Wrap an angle into [-pi/4, +pi/4) — the mod-90deg lattice cell of a Manhattan VP."""
    return (a + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0


def estimate_heading(
    frame_bgr: np.ndarray,
    roll: float = 0.0,
    pitch: float = 0.0,
    *,
    min_length_px: float = 25.0,
    inlier_thresh_deg: float = 1.5,
    max_horizontality: float = 0.30,
    min_support: int = 12,
    ransac_iters: int = 2000,
    seed: int = 0,
) -> HeadingEstimate | None:
    """Estimate the absolute warehouse heading (mod 90deg) from one BGR frame.

    Parameters
    ----------
    frame_bgr : (H,W,3) uint8 BGR image.
    roll, pitch : gravity-known body attitude (rad). With no ATTITUDE on the VQ2 wire these come
        from the ESKF accel tilt solution; default 0 treats the camera as level (fine for the
        synthetic level cases and a reasonable prior when tilt is small).
    max_horizontality : reject a VP whose recovered world direction has ``|world_z|`` above this
        (i.e. it is the vertical/pillar VP or a steeply-tilted truss, not a heading cue).
    min_support : minimum inlier line count for a usable heading.

    Returns a :class:`HeadingEstimate` or ``None`` (too few lines / no horizontal VP / blur).
    The reported ``heading_mod90_rad`` is mod 90deg; the consumer disambiguates the branch
    (gyro continuity / gate-bearing prior) — this function NEVER silently commits to one.
    """
    segs = extract_line_segments(frame_bgr, min_length_px=min_length_px)
    if len(segs) < int(min_support):
        return None
    vps = fit_multiple_vanishing_points(
        segs, n_vps=3, inlier_thresh_deg=inlier_thresh_deg,
        iters=ransac_iters, min_inliers=max(int(min_support) // 2, 6), seed=seed,
    )
    if not vps:
        return None

    # Score every candidate VP: prefer horizontal (small |world_z|), well-supported, tight pencils.
    best: tuple[float, VanishingPoint, float, float] | None = None
    for vp in vps:
        yaw, world_z = yaw_from_vp_pixel(vp.point_px[0], vp.point_px[1], roll, pitch)
        horiz = abs(world_z)
        if horiz > float(max_horizontality):
            continue  # the vertical/pillar VP (or a steep beam) — not a heading cue
        if vp.inlier_count < int(min_support):
            continue
        # Higher is better: support, horizontality, tightness.
        tight = 1.0 / (1.0 + vp.rms_resid_rad / np.deg2rad(inlier_thresh_deg))
        score = vp.inlier_count * (1.0 - horiz / float(max_horizontality)) * tight
        if best is None or score > best[0]:
            best = (score, vp, yaw, horiz)

    if best is None:
        return None
    _, vp, yaw, horiz = best

    heading_mod90 = _wrap_mod90(yaw)
    branches = np.array([_wrap_pi(heading_mod90 + k * np.pi / 2.0) for k in range(4)])

    # Quality in [0,1]: saturates with support, penalised by poor horizontality and loose pencil.
    support_q = float(np.clip(vp.inlier_count / 60.0, 0.0, 1.0))
    horiz_q = float(np.clip(1.0 - horiz / float(max_horizontality), 0.0, 1.0))
    tight_q = float(1.0 / (1.0 + vp.rms_resid_rad / np.deg2rad(inlier_thresh_deg)))
    quality = float(np.clip(support_q * horiz_q * tight_q, 0.0, 1.0))

    return HeadingEstimate(
        heading_mod90_rad=float(heading_mod90),
        quality=quality,
        n_support=int(vp.inlier_count),
        vp_px=vp.point_px.copy(),
        horizontality=float(horiz),
        branch_headings_rad=branches,
    )
