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

# Measured 1-sigma of the chain's attitude-equivalent error (the given attitude as exercised
# end-to-end by the vision chain: odo quat decode -> camera mount -> PnP world fix), replacing
# the np.deg2rad(1.0) first-contact GUESS that lived independently in localization, navigator
# and state_estimator. Derived by 2-term MLE on the canonical 6/6 course recording
# (20260607_194615_course_60s, 165 offered fixes): residual lever-arm angular spread after the
# constant-floor term (FIX_COV_FLOOR_STD) absorbs the range-independent systematics. The raw
# per-frame angular spread (~1.3-1.9 deg/axis) contains flight-specific systematics (roll/
# trajectory-correlated wander, per-gate offsets) that do NOT transfer across runs (cross-
# validated against the 2026-06-05 task2 bundle), so they are covered as noise here rather
# than calibrated. MLE 1.37 deg, profile flat 1.25-1.5; shipped 1.4.
# [vision-pkg2 2026-06-10, handoff/shadowpc-vision-pkg2-2026-06-10]
ATTITUDE_NOISE_STD_RAD = float(np.deg2rad(1.4))

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


def euler_from_quat_wxyz(q_wxyz) -> tuple[float, float, float]:
    """Body(FRD)->world(NED) quaternion (w, x, y, z; scalar-FIRST MAVLink order) -> aerospace
    3-2-1 Euler ``(roll, pitch, yaw)`` in radians. Exact inverse of ``R_world_from_body``'s
    'ZYX' convention, so the controller's command (``_euler_to_wxyz``) and the feedback share
    ONE convention -- the property that keeps a PD attitude loop sign-consistent.

    This is the canonical orientation extraction: the sim's ATTITUDE Euler has an inverted
    pitch sign (confirmed at first contact against the accel-gravity vector + the FPV view),
    so orientation is taken from the ODOMETRY quaternion and decoded here instead. A degenerate
    (near-zero-norm) quaternion returns zeros rather than raising."""
    w, x, y, z = (float(v) for v in q_wxyz)
    if (w * w + x * x + y * y + z * z) < 1e-12:
        return 0.0, 0.0, 0.0
    yaw, pitch, roll = Rotation.from_quat([x, y, z, w]).as_euler("ZYX")
    return float(roll), float(pitch), float(yaw)


def body_rate_from_quats(q_prev_wxyz, q_cur_wxyz, dt: float) -> np.ndarray:
    """Body-frame angular velocity (rad/s, FRD) from two consecutive body->world attitude
    quaternions (w,x,y,z) and the time between them.

    ``R_cur = R_prev @ exp(omega_body * dt)`` => ``omega_body = rotvec(R_prev^T R_cur) / dt``.
    Same body-frame convention as the controller's attitude-error rotvec, so it is a drop-in,
    SIGN-CORRECT body rate. Needed because the sim's ODOMETRY angular_rate (rollspeed/pitchspeed/
    yawspeed) is sign-inverted vs the true attitude derivative on at least pitch (measured
    2026-06-03), which turns a ``-kd*rate`` damping term into anti-damping. Differencing the
    trusted quaternion sidesteps that entirely. Returns zeros for a non-positive dt or a
    degenerate (near-zero-norm) quaternion."""
    if dt <= 0.0:
        return np.zeros(3)
    qp = np.asarray(q_prev_wxyz, dtype=np.float64)
    qc = np.asarray(q_cur_wxyz, dtype=np.float64)
    # Guard degenerate quaternions -- scipy's from_quat RAISES on a zero-norm quat. An
    # uninitialised orientation (all-zero before the first ODOMETRY) would otherwise crash the
    # caller mid-loop; match euler_from_quat_wxyz and treat it as no rotation. [review 2026-06-04]
    if float(qp @ qp) < 1e-12 or float(qc @ qc) < 1e-12:
        return np.zeros(3)
    R_prev = Rotation.from_quat([qp[1], qp[2], qp[3], qp[0]])
    R_cur = Rotation.from_quat([qc[1], qc[2], qc[3], qc[0]])
    return (R_prev.inv() * R_cur).as_rotvec() / dt


def world_vec_from_body_quat(v_body, q_wxyz) -> np.ndarray:
    """Rotate a body-frame (FRD) vector into world (NED) via a body->world attitude quaternion
    (w, x, y, z; scalar-FIRST MAVLink order): ``v_world = R(q) @ v_body``.

    Needed for the sim's ODOMETRY twist: ``vx/vy/vz`` (and the angular rates) are reported in
    ``child_frame_id`` = MAV_FRAME_BODY_NED (8, body z-down; confirmed from the wire), NOT the
    world ``frame_id`` (=MAV_FRAME_LOCAL_NED). The raw twist must be rotated here
    to match LOCAL_POSITION_NED's world velocity before either is stored as ``velocity_ned`` --
    otherwise a yawed+moving drone records a frame-mixed velocity (at yaw=-180deg the body vx is
    SIGN-FLIPPED vs world), which is the corruption that fed the KF + controller damping + planner.
    A degenerate (near-zero-norm) quaternion returns the input unchanged rather than raising
    (matches ``euler_from_quat_wxyz`` / ``body_rate_from_quats``)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    v = np.asarray(v_body, dtype=np.float64)
    if float(q @ q) < 1e-12:
        return v.copy()
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return R @ v


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


def horizontal_fov_deg() -> float:
    """Horizontal field of view (deg) from the intrinsics. fx=320, W=640 -> 90 deg.
    NB the spec labels '90 deg VFoV' but that is actually the HORIZONTAL FoV."""
    return float(np.rad2deg(2.0 * np.arctan(IMAGE_WIDTH / (2.0 * CAMERA_INTRINSICS_K[0, 0]))))


def vertical_fov_deg() -> float:
    """Vertical field of view (deg). fy=320, H=360 -> ~58.7 deg (NOT 90)."""
    return float(np.rad2deg(2.0 * np.arctan(IMAGE_HEIGHT / (2.0 * CAMERA_INTRINSICS_K[1, 1]))))


def camera_elevation_band_deg() -> tuple[float, float]:
    """Body-frame elevation range the camera sees (deg): the optical axis is +20 deg (up),
    spanning +/- VFoV/2 about it. ~(-9.4, +49.4) -> the camera 'looks UP'; low gates and
    climbs can fall out of the bottom of the frame (a planner/visibility consideration)."""
    half = vertical_fov_deg() / 2.0
    axis = float(np.rad2deg(CAMERA_PITCH_RAD))
    return (axis - half, axis + half)
