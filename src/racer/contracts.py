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
  Clock isolation: ``sim_time_ns`` is driven SOLELY by HIGHRES_IMU.time_usec (one epoch,
  high rate); ATTITUDE.time_boot_ms is a different epoch and is NOT written here, so the
  timeline stays monotonic (writing both made it oscillate — see mavlink_client review 2A).
  Orientation thus carries the most-recent IMU stamp (sub-IMU-period stale). TODO(clock):
  reconcile the epochs via TIMESYNC once ``msg_audit`` characterises the offsets — master
  plan hole #10. Until then, ``recv_monotonic_ns`` is the safe clock for cross-stream dt.

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
    jpeg_bytes: bytes | None = None  # raw JPEG exactly as received; set by jpeg_receiver,
                                     # consumed by the recorder for bit-exact replay. None
                                     # when a Frame is synthesised rather than received.


@dataclass(frozen=True, eq=False)
class DroneState:
    """Latest sim telemetry as an immutable snapshot. Produced by ``mavlink_client``.

    Orientation + angular rate come from ODOMETRY (the sim's ATTITUDE Euler has an inverted
    pitch sign, confirmed at first contact); accel/mag/baro from HIGHRES_IMU;
    ``armed`` from HEARTBEAT. ``position_ned`` / ``velocity_ned`` are populated ONLY if the
    sim emits a bearing message (e.g. LOCAL_POSITION_NED) — normally ``None`` (derive them).
    Fields may originate from different messages at slightly different sim-times;
    ``sim_time_ns`` is the most recent update (see the clock TODO in the module docstring).
    """

    sim_time_ns: int = 0
    recv_monotonic_ns: int = 0

    # Orientation — from the ODOMETRY quaternion (body FRD -> world NED, scalar-first w,x,y,z).
    # This is the canonical source; roll/pitch/yaw below are DERIVED from it via
    # frames.euler_from_quat_wxyz. The sim's ATTITUDE Euler is NOT used (its pitch sign is
    # inverted vs the accel-gravity vector + FPV view — first-contact 2026-06-02).
    orientation_ned_wxyz: np.ndarray | None = None
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    angular_rate_body: np.ndarray = _vec(3)   # (rollspeed, pitchspeed, yawspeed) rad/s FRD, from ODOMETRY

    # HIGHRES_IMU
    accel_body: np.ndarray = _vec(3)          # m/s^2, FRD; specific force (incl. gravity reaction)
    gyro_body: np.ndarray | None = None       # rad/s, FRD; RAW HIGHRES_IMU gyro (xgyro/ygyro/zgyro).
                                              # None until the first HIGHRES_IMU message. THE non-blocked
                                              # gyro source for the VQ2 AHRS (case-C self-localizing); raw
                                              # TRUE FRD, NOT sign-conjugated. ``angular_rate_body`` above
                                              # stays the ODOMETRY-derived (blocked-in-VQ2) rate.
    gyro_body_raw: np.ndarray | None = None   # rad/s, FRD; the RAW parsed HIGHRES_IMU gyro BEFORE the
                                              # ``gyro_sign`` per-axis correction (instrumentation only,
                                              # logged for the A14 yaw-steer-sign probe). gyro_body is the
                                              # post-sign value the AHRS uses; this is the pre-sign stash and
                                              # is NEVER consumed by control/estimate. None until first IMU.
    mag_body: np.ndarray | None = None        # FRD; None if unpopulated -> no free mag yaw
    baro_pressure_hpa: float | None = None    # absolute pressure; None if unpopulated -> no free baro z

    # Provided pose/vel — the big hedge. Normally None; if the sim emits them the
    # localisation problem collapses, so msg_audit must check and we capture eagerly.
    position_ned: np.ndarray | None = None
    velocity_ned: np.ndarray | None = None

    armed: bool = False
    status_flags: int = 0

    # The current TARGET gate index from RACE_STATUS.active_gate_index (ENCAPSULATED_DATA, 4 Hz; the
    # gate-ordering signal on the VQ2 wire). None until a RACE_STATUS arrives. Threaded onto the snapshot
    # (additive, like gyro_body) so the case-C Navigator's gate-bearing yaw lock (use_gate_bearing_yaw)
    # knows which mapped gate is the active target. Indexes the ORDERED gate list; consumers map it to a
    # gate_id. Unused by the VQ1 / case-A path -> default None is byte-identical.
    active_gate_index: int | None = None

    # Bumped by the sim (ODOMETRY.reset_counter) when its epoch restarts; a change means the
    # estimator should reinitialise rather than integrate across the discontinuity.
    reset_counter: int = 0

    # PER-FIELD ODOMETRY arrival stamp (``time.monotonic_ns``), set ONLY by the ODOMETRY
    # branch. ``recv_monotonic_ns`` above is shared across HIGHRES_IMU / LOCAL_POSITION_NED /
    # ODOMETRY, so it cannot tell you when the *attitude* + *body rate* (ODOMETRY-only fields)
    # last refreshed: a selective ODOMETRY drop while LPN/IMU keep arriving leaves the quat +
    # angular rate frozen yet ``recv_monotonic_ns`` reads fresh. Consumers that fly open-loop
    # on the attitude (the RL deploy loop) must gate on THIS age, not the shared one. 0 until
    # the first ODOMETRY (autonomy-readiness audit D1/D2, 2026-06-13).
    odo_recv_ns: int = 0


# ---------------------------------------------------------------------------
# PERCEPTION
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class GateObservation:
    """A single detected gate in one frame, BEFORE PnP. Produced by the detector.

    ``corners_px`` are inner-square corners in pixels. Normally all 4 (canonical order,
    matching the 3D model points), but as the drone closes on a gate the 20-deg up-tilt +
    ~59-deg VFoV push the lower corners out of frame, so we keep flying on whatever is
    still visible: 3 corners are enough for a (P3P) pose. When fewer than 4 are given,
    ``corner_ids`` MUST say which canonical corners they are (0=LL, 1=LR, 2=UR, 3=UL) so
    the PnP can pick the matching object points — order alone is ambiguous. With all 4,
    ``corner_ids`` may be omitted (assumed canonical [0,1,2,3]).

    OUTER-corner augmentation (2026-07-05): an 8-keypoint model also localises the 4 OUTER
    frame corners (the 2.72 m square, concentric + coplanar with the inner opening — see
    blender_gen/contract.py). When present they ride along as ``outer_corners_px`` /
    ``outer_corner_confidence`` (canonical LL,LR,UR,UL order, ALWAYS all 4 rows) and
    ``gate_pose`` fuses them into the same 6-DOF pose. ``corners_px`` REMAINS the inner
    square — association, ensemble dedup and every downstream consumer are unchanged; the
    outer fields are optional metadata a 4-keypoint model simply never fills (None).

    ``derived_corners`` (2026-07-21) marks a RESCUED observation: the gate cropped out of frame, so
    ``corners_px`` was RECONSTRUCTED through the gate-plane homography from whatever keypoints were
    still visible, rather than measured. Two consequences are enforced below, because a rescue that
    looked like an ordinary 4-corner fix would misinform every consumer downstream:
      - ``inner_area_px`` / ``visible_area_ratio`` return None. Those describe the APPARENT size and
        foreshortening of the opening; on a partly off-frame square the reconstructed quad is not an
        apparent area, and the RL contract already masks on None for a cropped gate.
      - ``gate_pose`` reports ``n_corners=3``, the existing "trust this less" signal (it fires the
        localization P3P covariance inflation and the RL policy's own low-trust branch). Measured
        justification: rescued centres are ~3x noisier than measured 4-corner ones (0.060 m vs
        0.008-0.022 m on ablation; 2.4-4.9x on real flight temporal continuity, motion-matched).
    """

    frame_id: int
    sim_time_ns: int
    corners_px: np.ndarray                       # (N, 2) float pixel coords, N in {3, 4}
    corner_ids: np.ndarray | None = None         # (N,) canonical corner index 0..3 per row; None => [0..N-1]
    corner_confidence: np.ndarray | None = None  # (N,) per-keypoint confidence, or None
    score: float = 1.0                           # object detection confidence
    bbox_xywh: np.ndarray | None = None          # (4,) optional, for ROI / debug
    gate_id: int | None = None                   # filled by data-association; None from raw detector
    outer_corners_px: np.ndarray | None = None   # (4,2) OUTER-square corners px (8-kpt models), or None
    outer_corner_confidence: np.ndarray | None = None  # (4,) per-outer-keypoint confidence, or None
    derived_corners: bool = False                # corners_px RECONSTRUCTED, not measured (see below)
    # --- M+1 DIRECTLY-REGRESSED CENTRE (5-kpt models, 2026-07-23) ---------------------------------
    # The 5th keypoint of the M+1 model: the gate-opening centre regressed by its OWN head slot, not
    # derived from the corners. That independence is the entire point -- a keypoint head emits each
    # slot separately, so the centre survives corner cropping and only fails when the CENTRE itself
    # leaves frame. Measured on real hand labels: 55.7% of gate views have the centre in frame while
    # >=1 inner corner is cropped -- the population an 8-kpt corner-derived centre is structurally
    # blind to (M coverage 52.8% vs M+1 99%, and 100% on cropped gates).
    # These are METADATA in exactly the sense outer_corners_px is: corners_px REMAINS the inner
    # square and every existing consumer is unchanged. A 4/8-kpt model simply leaves them None.
    centre_px: np.ndarray | None = None          # (2,) regressed gate-opening centre, pixels
    centre_confidence: float | None = None       # scalar keypoint confidence for that centre

    def __post_init__(self) -> None:
        c = self.corners_px
        assert c.ndim == 2 and c.shape[1] == 2, f"corners_px must be (N,2), got {c.shape}"
        n = c.shape[0]
        # A CENTRE-BEARING observation (M+1, 2026-07-23) is valid with FEWER than 3 corners -- that is
        # the whole point of the directly-regressed centre, which emits from its own head slot and so
        # survives corner cropping. 55.7% of real gate views have the centre in frame with >=1 inner
        # corner cropped; the old floor of 3 discarded exactly that population. Corners still cap at 4
        # and still must be identified. Without a centre the floor stands: a corner-only observation
        # with <3 corners has neither a pose nor a bearing and is unusable.
        # NOTE for PnP consumers: n < 3 means NO pose is recoverable -- gate_pose must skip these and
        # read the emit off centre_px instead (racer.vision.centre_emit).
        if self.centre_px is None:
            assert 3 <= n <= 4, f"need 3 or 4 corners without a centre, got {n}"
        else:
            assert n <= 4, f"at most 4 inner corners, got {n}"
        if self.corner_ids is None:
            assert n in (0, 4), "corner_ids is required when 1-3 corners are given"
        else:
            ids = np.asarray(self.corner_ids)
            assert ids.shape == (n,), f"corner_ids must be ({n},), got {ids.shape}"
            uniq = {int(i) for i in ids}
            assert uniq <= {0, 1, 2, 3} and len(uniq) == n, "corner_ids must be distinct indices in 0..3"
        if self.corner_confidence is not None:
            assert self.corner_confidence.shape == (n,), \
                f"corner_confidence must be ({n},), got {self.corner_confidence.shape}"
        if self.outer_corners_px is not None:
            o = self.outer_corners_px
            assert o.shape == (4, 2), f"outer_corners_px must be (4,2) when present, got {o.shape}"
        if self.outer_corner_confidence is not None:
            assert self.outer_corners_px is not None, "outer confidence requires outer_corners_px"
            assert self.outer_corner_confidence.shape == (4,), \
                f"outer_corner_confidence must be (4,), got {self.outer_corner_confidence.shape}"
        if self.centre_px is not None:
            c2 = np.asarray(self.centre_px)
            assert c2.shape == (2,), f"centre_px must be (2,) when present, got {c2.shape}"
        if self.centre_confidence is not None:
            assert self.centre_px is not None, "centre_confidence requires centre_px"

    # --- egocentric alignment byproduct (RL deploy obs contract, 2026-07-06) --------------------
    # DERIVED from the inner-4 corners (no PnP, no flip), so a property not a field. None when <4
    # corners (a partial/P3P view can't define the quad area).
    @property
    def inner_area_px(self) -> float | None:
        """Apparent inner-opening quad area (px^2, shoelace), or None if <4 corners -- or if the
        corners were RECONSTRUCTED (``derived_corners``), where the quad is not an apparent area.
        The raw byproduct RL emits alongside the ratio (cheap insurance for deriving other cues)."""
        if self.derived_corners:
            return None
        c = np.asarray(self.corners_px, dtype=float)
        if c.shape != (4, 2):
            return None
        x, y = c[:, 0], c[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    @property
    def visible_area_ratio(self) -> float | None:
        """Range-FREE foreshortening ratio = inner-quad area / max-edge^2, ~[0,1] (1 = head-on,
        smaller = sharper approach; ~|cos(approach angle)|). None if <4 corners. The CANONICAL
        egocentric alignment field the policy consumes (RL 2026-07-06): no range/PnP coupling,
        robust to the corner noise that wrecked the PnP normal. Measured sigma ~0.05 -> a COARSE
        >=20-25 deg misalignment cue (see rl-egocentric-obs-contract / foreshorten_analysis.py)."""
        area = self.inner_area_px
        if area is None:
            return None
        c = np.asarray(self.corners_px, dtype=float)
        e2 = max(float((c[i] - c[(i + 1) % 4]) @ (c[i] - c[(i + 1) % 4])) for i in range(4))
        return float(area / e2) if e2 > 1e-9 else None


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
    n_corners: int = 4                            # corners used: 4 => IPPE_SQUARE; 3 => P3P (less constrained, trust less)
    # M+1 centre-emit provenance (2026-07-23). "corners" = range from the identified-corner pairs
    # (measured 2.4% rel inside the 30 m cap); "bbox" = the corner-free fallback, which conflates
    # range with tilt and reads FAR on a cropped gate -- the DANGEROUS direction. Logged per
    # decision so a flight can be audited for which one it flew; a bbox range that misleads the
    # policy would otherwise be invisible on the wire. None on the PnP path.
    range_src: str | None = None
    n_pairs: int = 0                              # corner pairs that voted on the range (0 for bbox/PnP)

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
    # Calibrated gate-frame position 1-sigma (C2 estimator chain, BLUEPRINT §1.6): the KF position
    # covariance projected into the last-fix gate plane. Feeds the FUTURE inc8 confidence channel
    # (obs[17:18] = clip(sigma_ref/sigma_hat)); BUILT now, UNCONSUMED by the inc7 17-dim obs. ``inf``
    # until the first accepted gate-relative fix (no gate frame -> no projection). [C2-ESTIMATOR-CHAIN]
    nav_inplane_sigma: float = float("inf")             # m, in-plane (gate-plane) 1-sigma
    nav_along_sigma: float = float("inf")               # m, along-track (gate-normal) 1-sigma


# ---------------------------------------------------------------------------
# PLAN  (the THINK -> ACT seam)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Setpoint:
    """A reference state for the controller — the planner's output (THINK -> ACT seam).

    The differential-flatness reference a trajectory generator emits: a desired world-NED
    position and/or velocity (+ optional acceleration feedforward) and a desired heading.
    Provide whatever the planner computes and leave the rest ``None``; the controller uses
    what its mode needs (POSITION leans on the sim stabilizer; the attitude path runs a PD
    law on the position/velocity error plus the accel feedforward). This frozen seam lets
    the planner swap (reactive LOS -> min-snap -> MPCC/RL) without touching the controller.
    """

    sim_time_ns: int = 0
    position_ned: np.ndarray | None = None
    velocity_ned: np.ndarray | None = None
    accel_ned: np.ndarray | None = None
    yaw: float | None = None
    yaw_rate: float | None = None
    # Launch ramp [0, 1]: a transient authority scale the Mission ramps up at the takeoff->RUN
    # handoff. The decoupled CTBR controller multiplies its commanded horizontal acceleration
    # (hence the desired TILT) by this, so the attitude target grows smoothly from level instead
    # of STEPPING to the ~45 deg cruise lean in one tick -- a step saturates the body-rate clamp
    # and tumbles (sim build 1.0.3364, the start-transient knife-edge). ``None`` = full authority
    # (1.0); only the RUN-entry window sets it < 1. Time-based, so it is tick-phase/rate invariant.
    launch_ramp: float | None = None


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
