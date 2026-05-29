"""Data contracts — the frozen interfaces between modules.

These dataclasses are the *stable* boundaries of the autonomy stack. Modules depend
on these types, not on each other, so a VQ1->VQ2 component swap stays local. Treat
every instance as an immutable value snapshot (the dataclasses are ``frozen``); never
mutate the numpy arrays inside one.

Conventions (read once, never guess — sign/frame errors are the #1 risk here):
- World frame: ``MAV_FRAME_LOCAL_NED``. X=north, Y=east, Z=DOWN. Origin = arming point.
- Body frame: FRD. X=forward, Y=right, Z=down.
- Camera frame (PnP): OpenCV optical. X=right, Y=down, Z=forward. The body->camera
  rotation (incl. the +20 deg up-tilt) lives in ``racer.frames``.
- Units: metres, m/s, m/s^2, radians, rad/s. Pixels for image coords. Thrust in [0, 1].
- Quaternions are ``(w, x, y, z)`` scalar-FIRST (MAVLink convention).
  NB: ``scipy ... Rotation`` uses ``(x, y, z, w)`` scalar-LAST — convert deliberately.
- Time: ``sim_time_ns`` is the master timeline (simulator epoch, nanoseconds); drive all
  loops + latency comp off it. ``recv_monotonic_ns`` (``time.monotonic_ns``) is the local
  arrival time, for measuring end-to-end latency.
  TODO(clock): telemetry stamps ``sim_time_ns`` best-effort from the source message's own
  clock (ATTITUDE.time_boot_ms vs HIGHRES_IMU.time_usec may use different epochs). Reconcile
  via TIMESYNC once ``msg_audit`` characterises the offsets — see master-plan hole #10.

Equality: array-bearing contracts use ``eq=False`` (identity equality) to dodge numpy's
ambiguous-truth pitfalls; compare fields explicitly if you ever need value equality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


def _vec(n: int):
    """A zero-vector default for a dataclass field (float64, length n)."""
    return field(default_factory=lambda: np.zeros(n, dtype=np.float64))


# ---------------------------------------------------------------------------
# SENSE
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Frame:
    """One reassembled camera image (perception input). Produced by ``jpeg_receiver``."""

    frame_id: int
    sim_time_ns: int
    image_bgr: np.ndarray            # (360, 640, 3) uint8, BGR (OpenCV order)
    recv_monotonic_ns: int = 0       # local arrival time, for latency profiling


@dataclass(frozen=True, eq=False)
class DroneState:
    """Latest sim telemetry as an immutable snapshot. Produced by ``mavlink_client``.

    Orientation + angular rate come from ATTITUDE; accel/mag/baro from HIGHRES_IMU;
    ``armed`` from HEARTBEAT. ``position_ned`` / ``velocity_ned`` are populated ONLY if the
    sim emits a bearing message (e.g. LOCAL_POSITION_NED) — normally ``None`` (derive them).
    Fields may originate from different messages at slightly different sim-times;
    ``sim_time_ns`` is the most recent update (see the clock TODO in the module docstring).
    """

    sim_time_ns: int = 0
    recv_monotonic_ns: int = 0

    # Orientation — GIVEN by the sim (NED Euler, radians). The estimator trusts these.
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    angular_rate_body: np.ndarray = _vec(3)   # (rollspeed, pitchspeed, yawspeed) rad/s, FRD

    # HIGHRES_IMU
    accel_body: np.ndarray = _vec(3)          # m/s^2, FRD; specific force (incl. gravity reaction)
    mag_body: np.ndarray | None = None        # FRD; None if unpopulated -> no free mag yaw
    baro_pressure_hpa: float | None = None    # absolute pressure; None if unpopulated -> no free baro z

    # Provided pose/vel — the big hedge. Normally None; if the sim emits them the
    # localisation problem collapses, so msg_audit must check and we capture eagerly.
    position_ned: np.ndarray | None = None
    velocity_ned: np.ndarray | None = None

    armed: bool = False
    status_flags: int = 0


# ---------------------------------------------------------------------------
# PERCEPTION
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class GateObservation:
    """A single detected gate in one frame, BEFORE PnP. Produced by the detector.

    ``corners_px`` are the 4 inner-square corners in the canonical order expected by
    ``gate_pose`` (the same order as the 3D model points). 8-corner (inner+outer) is a
    later upgrade and would extend this contract, not replace it.
    """

    frame_id: int
    sim_time_ns: int
    corners_px: np.ndarray                       # (4, 2) float, pixel coords, ordered
    corner_confidence: np.ndarray | None = None  # (4,) per-keypoint confidence, or None
    score: float = 1.0                           # object detection confidence
    bbox_xywh: np.ndarray | None = None          # (4,) optional, for ROI / debug
    gate_id: int | None = None                   # filled by data-association; None from raw detector

    def __post_init__(self) -> None:
        assert self.corners_px.shape == (4, 2), f"corners_px must be (4,2), got {self.corners_px.shape}"


@dataclass(frozen=True, eq=False)
class GatePose:
    """6-DOF gate pose from PnP (camera-relative, map-agnostic). Produced by ``gate_pose``.

    Direction: ``p_cam = R_cam_gate @ p_gate + t_cam_gate`` (a point in the gate's own
    frame -> camera optical frame). The drone/camera position in the gate frame is the
    ``cam_in_gate`` property. The mapper/estimator combine this with the gate's known
    world pose (via ``gate_id``) to localise the drone.
    """

    frame_id: int
    sim_time_ns: int
    R_cam_gate: np.ndarray                        # (3, 3)
    t_cam_gate: np.ndarray                        # (3,) metres, gate origin in camera frame
    reproj_error_px: float
    gate_id: int | None = None
    covariance: np.ndarray | None = None         # (6,6) pose cov [t(3), rot(3)]; from corner sampling
    ambiguity_ratio: float | None = None         # IPPE 2-fold: err2/err1 (>>1 = unambiguous); None if n/a

    def __post_init__(self) -> None:
        assert self.R_cam_gate.shape == (3, 3), f"R_cam_gate must be (3,3), got {self.R_cam_gate.shape}"
        assert self.t_cam_gate.shape == (3,), f"t_cam_gate must be (3,), got {self.t_cam_gate.shape}"

    @property
    def range_m(self) -> float:
        """Distance from camera to gate centre."""
        return float(np.linalg.norm(self.t_cam_gate))

    @property
    def cam_in_gate(self) -> np.ndarray:
        """Camera (drone) position expressed in the gate's own frame."""
        return -self.R_cam_gate.T @ self.t_cam_gate


# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Gate:
    """A mapped gate in the world (NED). Produced by the mapper; consumed by the
    localizer (anchors a PnP fix to the world) and the planner.

    Orientation = the gate frame (X=right, Y=down, Z=downrange/through-direction, the
    same convention as ``gate_pose``) expressed in world NED, as a rotation matrix.
    This contract may evolve once we learn how the sim communicates gate
    positions/order (risk R4 — likely "rough" data per the FAQ).
    """

    gate_id: int
    position_ned: np.ndarray       # (3,) gate centre in world NED
    R_world_gate: np.ndarray       # (3,3) gate frame -> world NED
    inner_size_m: float = 1.5      # spec 3.7 inner square (m)

    def __post_init__(self) -> None:
        assert self.position_ned.shape == (3,), f"position_ned must be (3,), got {self.position_ned.shape}"
        assert self.R_world_gate.shape == (3, 3), f"R_world_gate must be (3,3), got {self.R_world_gate.shape}"

    @property
    def normal_ned(self) -> np.ndarray:
        """Through-direction (gate +Z, downrange) in world NED."""
        return self.R_world_gate[:, 2].copy()


# ---------------------------------------------------------------------------
# ESTIMATION
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class NavState:
    """Fused state estimate — the navigator's output, consumed by planner + controller.

    ``position_ned`` / ``velocity_ned`` are derived (vision PnP + IMU + baro). Attitude is
    carried as Euler (consistent with telemetry + ``racer.frames``); the controller converts
    to a quaternion via ``frames`` when building an attitude target. ``time_since_vision_
    update_s`` drives the "slow down / widen margins when coasting blind" policy (the caller
    owns the threshold).
    """

    sim_time_ns: int
    position_ned: np.ndarray = _vec(3)
    velocity_ned: np.ndarray = _vec(3)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    angular_rate_body: np.ndarray = _vec(3)
    pos_vel_covariance: np.ndarray | None = None       # (6,6) [pos(3), vel(3)]
    time_since_vision_update_s: float = float("inf")    # how long since the last vision fix


# ---------------------------------------------------------------------------
# ACT
# ---------------------------------------------------------------------------
class ControlMode(IntEnum):
    """Which setpoint a command carries; selects the MAVLink message + type_mask."""

    POSITION = 0   # SET_POSITION_TARGET_LOCAL_NED: position (+ optional vel/accel/yaw feedforward)
    VELOCITY = 1   # SET_POSITION_TARGET_LOCAL_NED: velocity (+ optional accel/yaw)
    ATTITUDE = 2   # SET_ATTITUDE_TARGET: attitude quaternion + collective thrust (angle mode)
    BODY_RATE = 3  # SET_ATTITUDE_TARGET: body rates + collective thrust (acro / CTBR)


@dataclass(frozen=True, eq=False)
class ControlCommand:
    """Controller output. ``mavlink_client.send_command`` translates it to the right
    message + type_mask. Set only the fields relevant to ``mode``; leave the rest ``None``.
    All world quantities NED, body quantities FRD, angles radians, thrust normalised [0, 1].
    """

    mode: ControlMode
    sim_time_ns: int = 0

    # POSITION / VELOCITY (world NED). Provide position OR velocity as primary per mode;
    # the remaining non-None terms ride along as feedforward.
    position_ned: np.ndarray | None = None
    velocity_ned: np.ndarray | None = None
    accel_ned: np.ndarray | None = None
    yaw: float | None = None
    yaw_rate: float | None = None

    # ATTITUDE (angle mode)
    attitude_quat_wxyz: np.ndarray | None = None
    # BODY_RATE (acro / CTBR)
    body_rate: np.ndarray | None = None          # (3,) rad/s, FRD
    # shared by ATTITUDE + BODY_RATE
    thrust: float | None = None                  # normalised [0, 1]
