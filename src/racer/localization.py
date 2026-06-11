"""Localization glue: a camera-relative gate PnP + the gate's known world pose ->
a drone world-position measurement for the Kalman filter.

The camera shares the body origin (spec 3.8) and is tilted per ``racer.frames``. With
the drone attitude TRUSTED (from telemetry) we know camera->world; the gate's world
position is known from the map; the PnP gives the gate origin in the camera frame.
Therefore:

    R_world_camera = R_world_body @ R_camera_from_body().T
    p_drone_world  = gate.position_ned - R_world_camera @ gatepose.t_cam_gate

The drone POSITION depends only on the PnP translation (and the trusted attitude), not
on the gate's estimated rotation. The fix covariance carries three measured terms: the
rotated PnP translation block (inflated, ``PNP_FIX_COV_INFLATION``), the attitude
lever-arm term (``frames.ATTITUDE_NOISE_STD_RAD``), and the constant-systematics floor
(``FIX_COV_FLOOR_STD``):

    Cov(p_drone) = K * R_wc @ Cov(t_cam_gate) @ R_wc.T
                   + sigma_theta^2 (|L|^2 I - L L^T) + sigma_floor^2 I

A camera lever-arm offset (the official sim says same origin; the Elodin rig offsets the
camera) and multi-gate PnP against all visible corners remain future refinements.
"""
from __future__ import annotations

import numpy as np

from racer.contracts import Gate, GatePose
from racer.frames import ATTITUDE_NOISE_STD_RAD, R_camera_from_body
from racer.state_estimator import LinearKF


# Analytic-PnP covariance inflation [perception-char 2026-06-08]. The 4-corner world-fix covariance
# is propagated from the confidence-weighted-refine Fisher information, which models ONLY the per-
# corner pixel noise (``WEIGHTED_SIGMA_PX``); it understates the real fix error (sub-pixel detector
# bias, heavy-tailed corner localisation, the 1.5 m gate-size model mismatch). Measured consequence:
# the navigator's chi2_0.999 = 16.27 innovation gate dropped ~20% of GOOD fixes because their analytic
# cov was ~3.5x too tight (good-fix maha ~ f*chi2(3), with f ~ 3.5). This scalar inflates the analytic
# PnP translation cov so the gate keeps catching wrong-gate / depth-flip fixes (maha in the hundreds-
# to-thousands) while passing healthy ones. Conservative default; sweep + confirm on per-gate course
# bundles against the <=~2% catastrophic-leak ceiling. Does NOT touch the attitude lever-arm term
# (physically calibrated) nor the no-covariance fallback. See handoff/perception-char-2026-06-08.
PNP_FIX_COV_INFLATION = 2.0   # variance multiplier on the analytic 4-corner PnP world-fix covariance

# Isotropic 1-sigma FLOOR on the world-fix covariance [vision-pkg2 2026-06-10]. The measured fix
# error carries range-INDEPENDENT systematics no scaled term can cover: a ~+0.3 m-high constant
# vertical offset (map opening-centre vs true / camera-height), per-gate lateral offsets
# (+-0.1..0.2 m), and a close-range depth bias (~+0.5 m under 8 m). At short range the attitude
# lever-arm term vanishes (|L| -> 0) and the analytic PnP cov is mm-tight, so those constants
# made the chi2_0.999 gate reject 35% of GOOD gate-0 close fixes (maha in the hundreds). Joint
# MLE with the attitude term on the canonical course recording: floor 0.407 m (shipped 0.40),
# attitude 1.37 deg -- good-fix over-rejection 13%->0% (<1 m) / 17%->1-2% (<3 m) at unchanged
# catastrophic leak. See handoff/shadowpc-vision-pkg2-2026-06-10.
FIX_COV_FLOOR_STD = 0.40      # m, added in quadrature to every vision world-fix covariance


def gate_pose_to_world_position(
    gate_pose: GatePose,
    gate: Gate,
    R_world_body: np.ndarray,
    default_position_std: float = 0.3,
    attitude_noise_std: float = ATTITUDE_NOISE_STD_RAD,
    pnp_cov_inflation: float = PNP_FIX_COV_INFLATION,
    fix_cov_floor_std: float = FIX_COV_FLOOR_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Drone world position (NED) + 3x3 covariance from one gate sighting.

    ``R_world_body`` is the trusted attitude (e.g. ``frames.R_world_from_body(roll,
    pitch, yaw)``). If ``gate_pose.covariance`` is None, falls back to an isotropic
    ``default_position_std`` for the PnP term. When present, the analytic PnP covariance is scaled by
    ``pnp_cov_inflation`` (it is optimistic -- see ``PNP_FIX_COV_INFLATION``) before propagation.

    The fix is ``p = gate_pos - R_world_camera @ t_cam_gate``, so its error has THREE measured
    components [vision-pkg2 2026-06-10]:
    the PnP translation (``gate_pose.covariance``, pixel noise); attitude error rotating the
    lever arm ``L = R_world_camera @ t_cam_gate`` -- dominant at range (a 1.4-deg error at 20 m
    is ~0.5 m, far above the PnP cm-noise), added as the induced covariance of a small random
    rotation of L, ``sigma_theta^2 (|L|^2 I - L L^T)`` (PSD; the same skew-projection the predict
    step uses), keyed off ``attitude_noise_std`` (measured chain value, see
    ``frames.ATTITUDE_NOISE_STD_RAD``); and the range-independent constant systematics
    (map-centre/per-gate/close-range-depth offsets), dominant at SHORT range where the other two
    terms vanish, covered by the isotropic ``fix_cov_floor_std`` floor (``FIX_COV_FLOOR_STD``).
    [red-team 2026-05-30; measured + floor added vision-pkg2 2026-06-10]
    """
    R_world_camera = np.asarray(R_world_body, dtype=np.float64) @ R_camera_from_body().T
    lever = R_world_camera @ np.asarray(gate_pose.t_cam_gate, dtype=np.float64)  # gate rel. drone, world NED
    position_ned = gate.position_ned - lever
    if gate_pose.covariance is not None:
        sigma_tt = np.asarray(gate_pose.covariance, dtype=np.float64)[:3, :3]
        # Inflate the (optimistic) analytic PnP translation cov before propagating -- see
        # ``PNP_FIX_COV_INFLATION``. Only the analytic branch; the fallback below is already loose.
        cov = pnp_cov_inflation * (R_world_camera @ sigma_tt @ R_world_camera.T)
    else:
        cov = (default_position_std**2) * np.eye(3)
    if attitude_noise_std > 0.0:
        cov = cov + attitude_noise_std**2 * (float(lever @ lever) * np.eye(3) - np.outer(lever, lever))
    if fix_cov_floor_std > 0.0:
        cov = cov + (fix_cov_floor_std**2) * np.eye(3)
    return position_ned, cov


# Gate-transit coast policy [red-team 2026-05-30, Tier B]. A 3-corner P3P fix (a gate
# clipping out of frame at transit; GatePose.n_corners < 4) is geometrically weaker and can't
# self-disambiguate, so we DON'T let it yank the estimate: inflate its measurement covariance
# so the KF leans on the IMU prediction (a "soft coast") instead of snapping to a maybe-wrong
# pose. The HARD coast (drop vision entirely within X m of a gate) and an innovation /
# Mahalanobis gate that rejects wrong-gate "teleport" fixes (master-plan NEG-3; needs the
# mapper + live range-to-gate) live in the navigator loop and are deferred to that wiring.
P3P_FIX_COV_INFLATION = 9.0   # variance multiplier (=3x std) for a 3-corner fix; tune at first contact


def apply_gate_pose_update(
    kf: LinearKF,
    gate_pose: GatePose,
    gate: Gate,
    R_world_body: np.ndarray,
    default_position_std: float = 0.3,
    attitude_noise_std: float = ATTITUDE_NOISE_STD_RAD,
    pnp_cov_inflation: float = PNP_FIX_COV_INFLATION,
    fix_cov_floor_std: float = FIX_COV_FLOOR_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a gate sighting to a world-position fix and apply it to the KF.

    ``attitude_noise_std`` adds the lever-arm attitude-uncertainty term to the fix covariance
    (see ``gate_pose_to_world_position``) so distant fixes are trusted appropriately less, and
    ``fix_cov_floor_std`` the constant-systematics floor so close fixes are not over-trusted.
    ``pnp_cov_inflation`` scales the (optimistic) analytic 4-corner PnP cov so the innovation gate
    keeps a healthy fix instead of over-rejecting it. A weak 3-corner (P3P) fix has its covariance
    inflated by ``P3P_FIX_COV_INFLATION`` on top, so it nudges rather than snaps the estimate at gate
    transit (the soft gate-transit coast).
    """
    position_ned, cov = gate_pose_to_world_position(
        gate_pose, gate, R_world_body, default_position_std, attitude_noise_std,
        pnp_cov_inflation, fix_cov_floor_std
    )
    if gate_pose.n_corners < 4:
        cov = cov * P3P_FIX_COV_INFLATION
    kf.update_position(position_ned, cov)
    return position_ned, cov
