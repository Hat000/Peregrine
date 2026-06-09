"""Robust gate data-association + post-PnP fix-consistency verification.

Replaces the naive nearest-predicted-centre association (a fixed 150 px gate on the centre
distance) that produced the measured 46% raw catastrophic world-fix tail on the course
(handoff/perception-char-2026-06-08): on a receding collinear course every gate's predicted
centre clusters near the image centre, so a NEAR detection happily matched a map gate
40-160 m downrange (87/312 fixes), and close-range partial/garbage detections matched the
current gate with a solved depth 1.2-13x the true range (56/312). Both failure modes share
one signature -- the detection's apparent GEOMETRY (scale, and the solved PnP depth)
disagrees wildly with what the map + the state prior predict -- while genuinely good fixes
agree in depth to ~3% of range typically (p90 11%). So we associate and verify on that
geometry, in the spirit of the directive:
overfit GEOMETRY, randomize APPEARANCE -- the map and the given attitude are trusted hard;
nothing here depends on what the gates look like.

Three layers (all pure geometry, shared by the live ``Navigator`` and the offline
``characterize_perception`` harness so the measurement stays faithful to the loop):

1. :func:`predict_gates_in_camera` -- project every map gate through the state prior
   (KF position + given attitude) into the camera: predicted pose, centre pixel AND the
   four predicted corner pixels (the apparent shape).
2. :func:`associate` -- match a detection to the candidate gate whose predicted corners
   agree with the detected corners under a scale-NORMALISED score: centre offset is
   measured in units of the predicted apparent size (a fixed pixel gate is meaningless
   across a 2-160 m candidate range) and the apparent-size ratio is hard-gated. A far
   background gate predicts a tiny shape, so a near detection can no longer match it,
   and vice versa; a detection that matches NO gate's predicted geometry (the close-range
   partial-structure junk) associates to nothing and produces no fix.
3. :func:`range_consistent` -- after PnP, the solved depth must agree with the predicted
   range to the associated gate (the known 1.5 m gate size makes PnP depth metric, so a
   big disagreement means the solver locked onto the wrong-scale structure or a degenerate
   flip, regardless of how well it reprojects). Tolerance is relative with an absolute
   floor, so a VQ2 state prior that is off by ~1 m never starves the filter.

The temporal/motion term lives in the PRIOR itself: the navigator re-predicts every frame
from the KF state (IMU-propagated between fixes), so association + the IPPE/P3P flip
disambiguation always see a fresh motion-consistent prediction instead of a sticky last
pose estimate. The Mahalanobis chi2 innovation gate stays downstream as the final backstop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from racer.contracts import Gate, GateObservation
from racer.frames import CAMERA_INTRINSICS_K, R_camera_from_body
from racer.vision.gate_pose import project_gate_corners

# -- tunables (sized from the measured course data; see module docstring) -------------------
# Hard gate on detected/predicted apparent-size ratio. Good fixes measured ~1.0 +- pixel
# noise; the wrong-gate tail sat at 2-7x and the close-range junk at ~0.4-0.5x.
ASSOC_MAX_SIZE_RATIO = 1.6
# Hard gate on the centre offset in units of the predicted apparent size (RMS corner radius).
# A ~1 m lateral prior error at 10 m is ~0.7 units; the collinear background-gate confusions
# were many units out once normalised.
ASSOC_MAX_CENTER_UNITS = 2.5
# Score weight of the log size-ratio term against the normalised corner distance.
ASSOC_SIZE_WEIGHT = 2.0
# Post-PnP depth sanity: |solved range - predicted range| <= max(abs floor, rel * predicted).
# Sized on the course data (handoff/shadowpc-assoc-flipfix-2026-06-09/reltol_sizing.py):
# good-fix |depth err|/range p90 = 0.11, and 0.15 costs ZERO sub-metre fixes while cutting
# the residual catastrophic leak 5 -> 2 of 189 (1.1%); 0.25 left the long-range depth-noise
# tail through, 0.10 starts eating the good-fix noise band. The absolute floor keeps a
# close-range fix alive under a ~1 m VQ2 prior error.
RANGE_REL_TOL = 0.15
RANGE_ABS_TOL_M = 1.0


@dataclass(frozen=True)
class PredictedGate:
    """One map gate projected through the state prior into the camera."""

    gate_id: int
    R_cam_gate: np.ndarray      # (3,3) predicted gate rotation in the camera frame
    t_cam_gate: np.ndarray      # (3,) predicted gate origin in the camera frame
    center_px: np.ndarray       # (2,) predicted centre pixel
    corners_px: np.ndarray | None  # (4,2) predicted corner pixels, canonical order;
                                   # None if any corner is at/behind the camera plane

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.t_cam_gate))


def predict_gates_in_camera(
    gates: list[Gate],
    drone_pos: np.ndarray,
    R_world_body: np.ndarray,
    camera_matrix: np.ndarray | None = None,
) -> dict[int, PredictedGate]:
    """Project every in-front map gate through the state prior. gate_id -> PredictedGate.

    The camera shares the body origin (spec 3.8). A gate whose origin is behind the camera
    is dropped; a gate whose origin is in front but with a corner at/behind the camera
    plane (a gate being transited) keeps its centre but carries ``corners_px=None`` -- it
    cannot be shape-checked, so :func:`associate` will not match to it.
    """
    K = np.asarray(CAMERA_INTRINSICS_K if camera_matrix is None else camera_matrix, float)
    R_world_camera = np.asarray(R_world_body, float) @ R_camera_from_body().T
    R_camera_world = R_world_camera.T
    out: dict[int, PredictedGate] = {}
    for gate in gates:
        t_pred = R_camera_world @ (gate.position_ned - np.asarray(drone_pos, float))
        if t_pred[2] <= 0:                                   # behind the camera
            continue
        uv = K @ t_pred
        center_px = uv[:2] / uv[2]
        R_pred = R_camera_world @ gate.R_world_gate
        try:
            corners = project_gate_corners(R_pred, t_pred, gate.inner_size_m, K)
        except ValueError:                                   # corner at/behind the camera
            corners = None
        out[gate.gate_id] = PredictedGate(gate.gate_id, R_pred, t_pred, center_px, corners)
    return out


def apparent_size_px(corners_px: np.ndarray) -> float:
    """Apparent scale of a corner set: RMS corner distance from its own centroid."""
    c = np.asarray(corners_px, float)
    return float(np.sqrt(np.mean(np.sum((c - c.mean(axis=0)) ** 2, axis=1))))


def association_score(
    obs_corners_px: np.ndarray,
    pred_corners_px: np.ndarray,
    max_size_ratio: float = ASSOC_MAX_SIZE_RATIO,
    max_center_units: float = ASSOC_MAX_CENTER_UNITS,
    size_weight: float = ASSOC_SIZE_WEIGHT,
) -> float | None:
    """Geometric agreement between a detection and ONE predicted gate (lower = better).

    ``pred_corners_px`` must already be subset to the same canonical corners the
    observation carries (rows correspond). Returns None when the pair fails a hard gate:
    apparent-size ratio outside [1/max_size_ratio, max_size_ratio], or centre offset
    beyond ``max_center_units`` predicted-size units. Otherwise the score is the
    size-normalised centre distance plus ``size_weight * |log(size ratio)|``.
    """
    obs = np.asarray(obs_corners_px, float)
    pred = np.asarray(pred_corners_px, float)
    s_pred = apparent_size_px(pred)
    s_obs = apparent_size_px(obs)
    if s_pred <= 1e-9 or s_obs <= 1e-9:
        return None
    ratio = s_obs / s_pred
    if not (1.0 / max_size_ratio <= ratio <= max_size_ratio):
        return None
    d_center = float(np.linalg.norm(obs.mean(axis=0) - pred.mean(axis=0))) / s_pred
    if d_center > max_center_units:
        return None
    return d_center + size_weight * abs(float(np.log(ratio)))


def associate_scored(
    obs: GateObservation,
    predicted: dict[int, PredictedGate],
    max_size_ratio: float = ASSOC_MAX_SIZE_RATIO,
    max_center_units: float = ASSOC_MAX_CENTER_UNITS,
) -> tuple[int, float] | None:
    """Match a detection to the map gate whose predicted SHAPE agrees best.

    Returns ``(gate_id, score)`` (lower score = better agreement) or None when no gate's
    predicted geometry is compatible. A 3-corner observation is compared against the same
    3 canonical predicted corners (``corner_ids``), so the scale measure stays
    like-for-like. Gates without predictable corners (mid-transit) are skipped -- the P3P
    transit coast owns that regime.
    """
    corners = np.asarray(obs.corners_px, float)
    ids = None if obs.corner_ids is None else np.asarray(obs.corner_ids, int)
    best: tuple[int, float] | None = None
    for gate_id, pg in predicted.items():
        if pg.corners_px is None:
            continue
        pred = pg.corners_px if ids is None else pg.corners_px[ids]
        score = association_score(corners, pred, max_size_ratio, max_center_units)
        if score is not None and (best is None or score < best[1]):
            best = (gate_id, score)
    return best


def associate(
    obs: GateObservation,
    predicted: dict[int, PredictedGate],
    max_size_ratio: float = ASSOC_MAX_SIZE_RATIO,
    max_center_units: float = ASSOC_MAX_CENTER_UNITS,
) -> int | None:
    """:func:`associate_scored`, id only (the navigator's call shape)."""
    best = associate_scored(obs, predicted, max_size_ratio, max_center_units)
    return None if best is None else best[0]


def range_consistent(
    solved_range_m: float,
    predicted_range_m: float,
    rel_tol: float = RANGE_REL_TOL,
    abs_tol_m: float = RANGE_ABS_TOL_M,
) -> bool:
    """Known-gate-size depth sanity: does the PnP depth agree with the predicted range?

    The 1.5 m inner square makes the PnP depth metric, so a solved range far from the
    map+prior prediction means the solver locked onto the wrong structure / a degenerate
    flip -- reject the fix before it reaches the filter. Good fixes agree to ~1 m.
    """
    return abs(float(solved_range_m) - float(predicted_range_m)) <= max(
        abs_tol_m, rel_tol * float(predicted_range_m)
    )
