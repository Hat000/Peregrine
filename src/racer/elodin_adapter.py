"""Adapter between the Elodin practice rig and our data contracts.

Elodin (github.com/elodin-sys/ai-grand-prix, Apache-2.0) is our dev/validation
surrogate until the official MAVLink sim ships. It is NOT MAVLink: a contestant
implements ``autopilot(update: SensorUpdate) -> RCCommand``, called every physics
tick. This module converts Elodin's SensorUpdate -> our contracts and our
ControlCommand -> Elodin RC channels, so the SAME autonomy stack runs on Elodin
(dev) and on the official MAVLink sim (via ``mavlink_client``).

It deliberately does NOT import ``elodin`` (which has no Windows/3.13 wheel and is
heavy): everything here operates on plain numpy arrays + our contracts, so it is
unit-testable in our normal venv. The thin glue that imports Elodin's
``SensorUpdate``/``RCCommand`` and runs the loop lives in a separate solver module
(on the Linux/WSL machine that actually runs the rig).

Frame conventions (from Elodin ARCHITECTURE.md) — the trap-rich part:
- Elodin world = ENU (X=East, Y=North, Z=Up); body = FLU (Forward-Left-Up).
- Our contracts = NED world (X=North, Y=East, Z=Down), FRD body (Fwd, Right, Down).
- Elodin quaternion is scalar-LAST ``[qx, qy, qz, qw]`` (== scipy convention).
- Elodin ``baro`` is ALTITUDE in metres (not pressure); ``mag`` is the body-frame
  direction of ENU-North (normalised).

DISCIPLINE: Elodin hands us ground-truth world pose/velocity; the OFFICIAL sim does
NOT. By default we do NOT put GT into DroneState (position_ned/velocity_ned stay
None, like the official sim) so the stack is forced to solve the real, vision-
derived localisation problem. GT is returned separately for the eval harness to
score the estimator against. (Attitude IS taken from Elodin's quaternion — that's
legitimate: the official sim hands us attitude too, at similar quality.)

CONTROL is PROVISIONAL: Elodin/Betaflight take RC sticks (PWM). Betaflight ANGLE
mode maps roll/pitch sticks -> desired angle, yaw stick -> yaw RATE, throttle ->
throttle. So ATTITUDE maps (modulo yaw, which needs a rate from the controller);
POSITION/VELOCITY are unavailable (no position loop). The stick<->angle gains and
SIGNS must be calibrated against the running rig before trusting closed-loop flight.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from racer.atmosphere import altitude_to_pressure_hpa
from racer.contracts import ControlCommand, ControlMode, DroneState, Frame

# ENU world -> NED world: [E, N, U] -> [N, E, -U]   (proper rotation, det +1)
R_NED_FROM_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# FLU body -> FRD body: [F, L, U] -> [F, -R, -U]   (its own inverse)
R_FRD_FROM_FLU = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


@dataclass(frozen=True)
class GroundTruth:
    """Elodin's true world state (for eval only — the official sim won't give this)."""

    position_ned: np.ndarray
    velocity_ned: np.ndarray
    quat_wxyz: np.ndarray  # body FRD -> world NED


def _quat_wxyz(rot: Rotation) -> np.ndarray:
    x, y, z, w = rot.as_quat()  # scipy is scalar-last
    return np.array([w, x, y, z])


def attitude_ned_frd_from_elodin_quat(quat_xyzw: np.ndarray) -> Rotation:
    """Elodin orientation quat (FLU body -> ENU world, scalar-last) -> our FRD->NED rotation."""
    r_enu_flu = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    r_ned_frd = R_NED_FROM_ENU @ r_enu_flu @ R_FRD_FROM_FLU
    return Rotation.from_matrix(r_ned_frd)


def sensorupdate_to_state(
    *,
    t: float,
    world_pos: np.ndarray,   # [qx, qy, qz, qw, x, y, z] (ENU, scalar-last quat)
    world_vel: np.ndarray,   # [wx, wy, wz, vx, vy, vz] (ENU)
    gyro: np.ndarray,        # body FLU rad/s
    accel: np.ndarray,       # body FLU m/s^2 (specific force)
    mag: np.ndarray | None = None,   # body FLU, normalised
    baro: float | None = None,       # altitude (m, ENU up)
    sim_time_ns: int | None = None,
    expose_ground_truth: bool = False,
) -> tuple[DroneState, GroundTruth]:
    """Convert an Elodin SensorUpdate's fields into our DroneState + the GT for eval."""
    world_pos = np.asarray(world_pos, dtype=np.float64)
    world_vel = np.asarray(world_vel, dtype=np.float64)

    att = attitude_ned_frd_from_elodin_quat(world_pos[:4])
    yaw, pitch, roll = att.as_euler("ZYX")  # matches racer.frames convention

    pos_ned = R_NED_FROM_ENU @ world_pos[4:7]
    vel_ned = R_NED_FROM_ENU @ world_vel[3:6]
    gt = GroundTruth(position_ned=pos_ned, velocity_ned=vel_ned, quat_wxyz=_quat_wxyz(att))

    state = DroneState(
        sim_time_ns=int(t * 1e9) if sim_time_ns is None else sim_time_ns,
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
        angular_rate_body=R_FRD_FROM_FLU @ np.asarray(gyro, dtype=np.float64),
        accel_body=R_FRD_FROM_FLU @ np.asarray(accel, dtype=np.float64),
        mag_body=(R_FRD_FROM_FLU @ np.asarray(mag, dtype=np.float64)) if mag is not None else None,
        baro_pressure_hpa=(altitude_to_pressure_hpa(float(baro)) if baro is not None else None),
        # GT hidden by default so the stack solves the real localisation problem.
        position_ned=(pos_ned if expose_ground_truth else None),
        velocity_ned=(vel_ned if expose_ground_truth else None),
        armed=True,
    )
    return state, gt


def frame_from_rgba(frame_rgba: np.ndarray, frame_id: int, sim_time_ns: int) -> Frame:
    """Elodin RGBA (H, W, 4) uint8 -> our BGR Frame (cv2 channel order)."""
    rgba = np.asarray(frame_rgba)
    bgr = np.ascontiguousarray(rgba[:, :, [2, 1, 0]])  # drop alpha, swap R<->B
    return Frame(frame_id=frame_id, sim_time_ns=sim_time_ns, image_bgr=bgr)


# --- control direction (PROVISIONAL — calibrate gains + signs on the running rig) ---
@dataclass(frozen=True)
class RcChannels:
    """RC output mirroring Elodin's RCCommand (PWM us). The thin solver glue copies
    these into an Elodin ``RCCommand``."""

    throttle: int = 1000
    roll: int = 1500
    pitch: int = 1500
    yaw: int = 1500
    arm: int = 1000
    aux2: int = 1500
    aux3: int = 1500
    aux4: int = 1500


@dataclass(frozen=True)
class RCParams:
    """Stick mapping. ALL VALUES ARE PLACEHOLDERS — calibrate against the rig."""

    center: int = 1500
    half_range: int = 500
    max_angle_rad: float = np.deg2rad(45.0)
    max_yaw_rate_rad_s: float = np.deg2rad(200.0)
    throttle_min: int = 1000
    throttle_max: int = 2000
    arm_pwm: int = 1800
    disarm_pwm: int = 1000
    roll_sign: int = 1
    pitch_sign: int = 1   # api.py says pitch<1500=forward; verify on the rig
    yaw_sign: int = 1


def _stick(value: float, full_scale: float, sign: int, p: RCParams) -> int:
    frac = float(np.clip(sign * value / full_scale, -1.0, 1.0))
    return int(round(p.center + frac * p.half_range))


def attitude_command_to_rc(
    roll_rad: float,
    pitch_rad: float,
    yaw_rate_rad_s: float,
    thrust: float,
    params: RCParams | None = None,
    armed: bool = True,
) -> RcChannels:
    """Betaflight ANGLE-mode command (roll/pitch ANGLE, yaw RATE, throttle) -> RC PWM."""
    p = params or RCParams()
    throttle = int(round(np.clip(
        p.throttle_min + thrust * (p.throttle_max - p.throttle_min), 1000, 2000)))
    return RcChannels(
        throttle=throttle,
        roll=_stick(roll_rad, p.max_angle_rad, p.roll_sign, p),
        pitch=_stick(pitch_rad, p.max_angle_rad, p.pitch_sign, p),
        yaw=_stick(yaw_rate_rad_s, p.max_yaw_rate_rad_s, p.yaw_sign, p),
        arm=p.arm_pwm if armed else p.disarm_pwm,
    )


def controlcommand_to_rc(
    cmd: ControlCommand, params: RCParams | None = None, armed: bool = True
) -> RcChannels:
    """Map our ControlCommand to RC channels for the Betaflight rig (PROVISIONAL)."""
    p = params or RCParams()
    if cmd.mode == ControlMode.ATTITUDE:
        assert cmd.attitude_quat_wxyz is not None and cmd.thrust is not None
        w, x, y, z = cmd.attitude_quat_wxyz
        yaw, pitch, roll = Rotation.from_quat([x, y, z, w]).as_euler("ZYX")
        # ANGLE-mode yaw is a RATE, not the target yaw angle; a yaw-hold needs the
        # controller to emit a yaw rate. Provisional: 0 until the controller exists.
        return attitude_command_to_rc(roll, pitch, 0.0, cmd.thrust, p, armed)
    if cmd.mode == ControlMode.BODY_RATE:
        assert cmd.body_rate is not None and cmd.thrust is not None
        r = cmd.body_rate
        # ACRO-style rate->stick. NB the rig's default config is ANGLE mode; switch
        # Betaflight to ACRO to use this. Provisional gains.
        return RcChannels(
            throttle=int(round(np.clip(
                p.throttle_min + cmd.thrust * (p.throttle_max - p.throttle_min), 1000, 2000))),
            roll=_stick(float(r[0]), p.max_yaw_rate_rad_s, p.roll_sign, p),
            pitch=_stick(float(r[1]), p.max_yaw_rate_rad_s, p.pitch_sign, p),
            yaw=_stick(float(r[2]), p.max_yaw_rate_rad_s, p.yaw_sign, p),
            arm=p.arm_pwm if armed else p.disarm_pwm,
        )
    raise NotImplementedError(
        f"{cmd.mode.name} control is unavailable on the Betaflight rig (no position "
        "loop). Use position/velocity setpoints only against the official MAVLink sim."
    )
