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
) -> tuple[np.ndarray, np.ndarray]:
    """Drone world position (NED) + 3x3 covariance from one gate sighting.

    ``R_world_body`` is the trusted attitude (e.g. ``frames.R_world_from_body(roll,
    pitch, yaw)``). If ``gate_pose.covariance`` is None, falls back to an isotropic
    ``default_position_std``.
    """
    R_world_camera = np.asarray(R_world_body, dtype=np.float64) @ R_camera_from_body().T
    position_ned = gate.position_ned - R_world_camera @ gate_pose.t_cam_gate
    if gate_pose.covariance is not None:
        sigma_tt = np.asarray(gate_pose.covariance, dtype=np.float64)[:3, :3]
        cov = R_world_camera @ sigma_tt @ R_world_camera.T
    else:
        cov = (default_position_std**2) * np.eye(3)
    return position_ned, cov


def apply_gate_pose_update(
    kf: LinearKF,
    gate_pose: GatePose,
    gate: Gate,
    R_world_body: np.ndarray,
    default_position_std: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a gate sighting to a world-position fix and apply it to the KF."""
    position_ned, cov = gate_pose_to_world_position(
        gate_pose, gate, R_world_body, default_position_std
    )
    kf.update_position(position_ned, cov)
    return position_ned, cov
