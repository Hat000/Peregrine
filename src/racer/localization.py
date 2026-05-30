"""Localization glue: a camera-relative gate PnP + the gate's known world pose ->
a drone world-position measurement for the Kalman filter.

The camera shares the body origin (spec 3.8) and is tilted per ``racer.frames``. With
the drone attitude TRUSTED (from telemetry) we know camera->world; the gate's world
position is known from the map; the PnP gives the gate origin in the camera frame.
Therefore:

    R_world_camera = R_world_body @ R_camera_from_body().T
    p_drone_world  = gate.position_ned - R_world_camera @ gatepose.t_cam_gate

The drone POSITION depends only on the PnP translation (and the trusted attitude), not
on the gate's estimated rotation, so we propagate only the translation covariance:

    Cov(p_drone) = R_world_camera @ Cov(t_cam_gate) @ R_world_camera.T

Trusted attitude is treated as exact here (consistent with the linear-KF design); an
attitude-uncertainty term, a camera lever-arm (the official sim says same origin; the
Elodin rig offsets the camera), and multi-gate PnP against all visible corners are
future refinements.
"""
from __future__ import annotations

import numpy as np

from racer.contracts import Gate, GatePose
from racer.frames import R_camera_from_body
from racer.state_estimator import LinearKF


def gate_pose_to_world_position(
    gate_pose: GatePose,
    gate: Gate,
    R_world_body: np.ndarray,
    default_position_std: float = 0.3,
    attitude_noise_std: float = np.deg2rad(1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Drone world position (NED) + 3x3 covariance from one gate sighting.

    ``R_world_body`` is the trusted attitude (e.g. ``frames.R_world_from_body(roll,
    pitch, yaw)``). If ``gate_pose.covariance`` is None, falls back to an isotropic
    ``default_position_std`` for the PnP term.

    The fix is ``p = gate_pos - R_world_camera @ t_cam_gate``, so its error has TWO sources:
    the PnP translation (``gate_pose.covariance``, pixel noise) AND attitude error rotating the
    lever arm ``L = R_world_camera @ t_cam_gate``. The second dominates at range -- a 1-deg
    attitude error at 20 m is ~0.35 m, far larger than the PnP cm-noise -- so omitting it makes
    the KF wildly over-trust distant fixes and snap (notably the z-axis) against the baro. We add
    the induced covariance of a small random rotation of L, ``sigma_theta^2 (|L|^2 I - L L^T)``
    (PSD; the same skew-projection the predict step uses), keyed off ``attitude_noise_std`` (the
    given-attitude 1-sigma, tunable at first contact). [red-team 2026-05-30]
    """
    R_world_camera = np.asarray(R_world_body, dtype=np.float64) @ R_camera_from_body().T
    lever = R_world_camera @ np.asarray(gate_pose.t_cam_gate, dtype=np.float64)  # gate rel. drone, world NED
    position_ned = gate.position_ned - lever
    if gate_pose.covariance is not None:
        sigma_tt = np.asarray(gate_pose.covariance, dtype=np.float64)[:3, :3]
        cov = R_world_camera @ sigma_tt @ R_world_camera.T
    else:
        cov = (default_position_std**2) * np.eye(3)
    if attitude_noise_std > 0.0:
        cov = cov + attitude_noise_std**2 * (float(lever @ lever) * np.eye(3) - np.outer(lever, lever))
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
    attitude_noise_std: float = np.deg2rad(1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a gate sighting to a world-position fix and apply it to the KF.

    ``attitude_noise_std`` adds the lever-arm attitude-uncertainty term to the fix covariance
    (see ``gate_pose_to_world_position``) so distant fixes are trusted appropriately less. A weak
    3-corner (P3P) fix has its covariance inflated by ``P3P_FIX_COV_INFLATION`` on top, so it
    nudges rather than snaps the estimate at gate transit (the soft gate-transit coast).
    """
    position_ned, cov = gate_pose_to_world_position(
        gate_pose, gate, R_world_body, default_position_std, attitude_noise_std
    )
    if gate_pose.n_corners < 4:
        cov = cov * P3P_FIX_COV_INFLATION
    kf.update_position(position_ned, cov)
    return position_ned, cov
