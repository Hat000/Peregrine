"""Coordinate frame transformations for the AI Grand Prix sim.

Frames in use (all right-handed):
- World NED  (MAV_FRAME_LOCAL_NED): origin at arming point, X north, Y east, Z down.
- Body  FRD  (MAV_FRAME_BODY_NED):  origin at vehicle, X forward, Y right, Z down.
- Camera optical (OpenCV convention): origin at vehicle, X right, Y down, Z forward.
  Tilted +20 deg about body Y (camera pitched UP). Spec VADR-TS-002 sec 3.8.

Convention: ``R_a_from_b`` takes a vector expressed in frame B and returns it
expressed in frame A, i.e. ``v_a = R_a_from_b @ v_b``.
All angles in radians.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

CAMERA_PITCH_RAD = np.deg2rad(20.0)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360

CAMERA_INTRINSICS_K = np.array(
    [[320.0,   0.0, 320.0],
     [  0.0, 320.0, 180.0],
     [  0.0,   0.0,   1.0]],
    dtype=np.float64,
)

# Axis swap: tilted body FRD (X-fwd, Y-right, Z-down) -> camera optical (X-right, Y-down, Z-fwd).
_R_CAMERA_FROM_TILTED_BODY = np.array(
    [[0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def R_world_from_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Aerospace 3-2-1 intrinsic (yaw-pitch-roll) NED Euler angles -> rotation matrix."""
    return Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()


def R_camera_from_body() -> np.ndarray:
    # Passive rotation by +20 deg about body Y (frame rotated, vector representation changes oppositely).
    R_tilted_from_body = Rotation.from_euler("Y", -CAMERA_PITCH_RAD).as_matrix()
    return _R_CAMERA_FROM_TILTED_BODY @ R_tilted_from_body


def world_point_in_body(
    p_world: np.ndarray,
    R_wb: np.ndarray,
    body_origin_in_world: np.ndarray,
) -> np.ndarray:
    return R_wb.T @ (p_world - body_origin_in_world)


def world_point_in_camera(
    p_world: np.ndarray,
    R_wb: np.ndarray,
    body_origin_in_world: np.ndarray,
) -> np.ndarray:
    return R_camera_from_body() @ world_point_in_body(p_world, R_wb, body_origin_in_world)


def project_camera_point(p_camera: np.ndarray) -> tuple[float, float] | None:
    if p_camera[2] <= 0:
        return None
    uvw = CAMERA_INTRINSICS_K @ p_camera
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])
