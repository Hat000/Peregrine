"""Controller — the ACT link. Turns a :class:`Setpoint` (planner output) + the current
:class:`NavState` into a :class:`ControlCommand` for ``mavlink_client.send_command``.

Two realizations behind one interface (``Controller.command``), selected by ``mode``:

- **POSITION / VELOCITY (the VQ1 floor).** Pass the setpoint straight through as a
  SET_POSITION_TARGET; the sim's built-in Stabilized Controller closes the loop. Almost no
  hand-rolled control — this is the "lean on the stabilizer" baseline and the position-
  easy-mode test. If the sim honors position targets, this alone flies the course.

- **ATTITUDE (the speed-ward upgrade).** A geometric (Lee/Mellinger-style) law: a PD on the
  position/velocity error plus the setpoint's acceleration feedforward gives a desired world
  acceleration; that maps to a desired thrust direction (-> attitude quaternion at the
  requested heading) and a collective thrust. Body-rate (true CTBR) mode is the next step.

Frames: world NED (z down, g = +9.80665 on z); body FRD (x fwd, y right, z down); thrust acts
along body -z (up). Quaternion out is (w,x,y,z) = R_world_body, the MAVLink attitude convention.

CAUTION — the throttle scale is NOT yet calibrated. ``_accel_to_attitude`` maps required
specific force to a normalized [0,1] throttle with a placeholder linear curve anchored at
``hover_thrust``. The real throttle->thrust mapping (TWR, curve, lag) is the keystone
first-contact measurement (``innerloop_step``, risk R2). The attitude DIRECTION is correct
and tested; only the thrust MAGNITUDE awaits system-ID.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from racer.contracts import ControlCommand, ControlMode, NavState, Setpoint
from racer.frames import R_world_from_body

_G = 9.80665
_GRAVITY_NED = np.array([0.0, 0.0, _G])     # NED: gravity points +z (down)
_WORLD_UP = np.array([0.0, 0.0, -1.0])      # NED up
_IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])


def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale ``v`` down so its norm is at most ``max_norm`` (direction preserved)."""
    n = float(np.linalg.norm(v))
    if n > max_norm > 0.0:
        return v * (max_norm / n)
    return v


@dataclass
class Controller:
    """Setpoint + NavState -> ControlCommand. See module docstring for the two modes."""

    mode: ControlMode = ControlMode.POSITION
    # Attitude-path tracking gains (the sim stabilizer owns this loop in POSITION/VELOCITY).
    kp_pos: float = 1.5            # 1/s^2, position-error -> accel
    kd_vel: float = 2.0           # 1/s,   velocity-error -> accel
    hover_thrust: float = 0.5      # normalized throttle that holds a hover (innerloop_step: ~0.489)
    max_tilt_rad: float = np.deg2rad(45.0)
    # Measured throttle map (innerloop_step): up-accel ~ slope*(thrust - hover), so
    # thrust = hover + (|f| - g)/slope. More accurate than the |f|/g placeholder, which
    # over-thrusts ~30% on climbs/maneuvers. None => fall back to the placeholder.
    thrust_slope_mps2: float | None = None
    # Safety bounds for the attitude path (the position/velocity easy-mode runs away on this
    # sim, so the geometric attitude law is the floor and must be tamed for far gate carrots):
    # cap the demanded acceleration (=> bounds tilt) and clamp the position-error fed to kp_pos
    # (=> bounded pure-pursuit, so a 24 m-away carrot doesn't saturate to 45 deg). None => off.
    max_accel_mps2: float | None = None
    max_pos_error_m: float | None = None
    # CTBR (BODY_RATE) inner loop -- the control path for ACRO. This sim stays in ACRO: the
    # attitude-quat setpoint does NOT switch it to ANGLE, and position/velocity run away (first
    # contact 2026-06-02). A proportional attitude->body-rate law: omega = kp_att *
    # axis-angle(R_cur^T R_des), which the sim's fast (~48 ms) rate loop tracks. Same desired
    # attitude + thrust as the ATTITUDE path; only the final actuation differs.
    kp_att: float = 4.0              # 1/s, attitude-error -> commanded body-rate
    max_body_rate_rps: float = 2.0   # rad/s, clamp on the commanded body rate (safety)
    kd_att: float = 0.0              # rate damping: omega -= kd_att * current body rate. This sim's
                                     # ACRO rate loop is very underdamped (measured rates overshoot the
                                     # command ~2.7x -> a pure-P attitude loop tumbles), so subtract the
                                     # measured body rate to curb the overshoot. 0 = off (legacy).

    def command(self, nav: NavState, setpoint: Setpoint) -> ControlCommand:
        """Compute the control command for the current state + reference."""
        if self.mode in (ControlMode.POSITION, ControlMode.VELOCITY):
            return self._setpoint_passthrough(setpoint)
        if self.mode == ControlMode.ATTITUDE:
            return self._attitude_command(nav, setpoint)
        if self.mode == ControlMode.BODY_RATE:
            return self._body_rate_command(nav, setpoint)
        raise ValueError(f"unknown control mode: {self.mode!r}")

    # -- floor: lean on the sim's stabilized controller ---------------------
    def _setpoint_passthrough(self, sp: Setpoint) -> ControlCommand:
        """Forward the setpoint as a SET_POSITION_TARGET; mavlink_client builds the type_mask
        from which fields are non-None (position primary, velocity/accel/yaw ride along)."""
        return ControlCommand(
            mode=self.mode,
            sim_time_ns=sp.sim_time_ns,
            position_ned=sp.position_ned,
            velocity_ned=sp.velocity_ned,
            accel_ned=sp.accel_ned,
            yaw=sp.yaw,
            yaw_rate=sp.yaw_rate,
        )

    # -- upgrade: geometric attitude control --------------------------------
    def _desired_acceleration(self, nav: NavState, sp: Setpoint) -> np.ndarray:
        """World-NED desired acceleration: accel feedforward + PD on position/velocity error."""
        a = np.zeros(3)
        if sp.accel_ned is not None:
            a = a + np.asarray(sp.accel_ned, dtype=np.float64)
        if sp.position_ned is not None:
            err = np.asarray(sp.position_ned, dtype=np.float64) - nav.position_ned
            if self.max_pos_error_m is not None:               # bounded pure-pursuit: a far gate
                err = _clip_norm(err, self.max_pos_error_m)     # carrot can't saturate the tilt
            a = a + self.kp_pos * err
        if sp.velocity_ned is not None:
            a = a + self.kd_vel * (np.asarray(sp.velocity_ned, dtype=np.float64) - nav.velocity_ned)
        elif sp.position_ned is not None:
            a = a - self.kd_vel * np.asarray(nav.velocity_ned, dtype=np.float64)  # damp when only position is given
        if self.max_accel_mps2 is not None:                     # cap the maneuver accel -> bounds tilt
            a = _clip_norm(a, self.max_accel_mps2)
        return a

    def _attitude_command(self, nav: NavState, sp: Setpoint) -> ControlCommand:
        a_des = self._desired_acceleration(nav, sp)
        yaw = sp.yaw if sp.yaw is not None else nav.yaw
        q_wxyz, thrust = self._accel_to_attitude(a_des, yaw)
        return ControlCommand(
            mode=ControlMode.ATTITUDE,
            sim_time_ns=sp.sim_time_ns,
            attitude_quat_wxyz=q_wxyz,
            thrust=thrust,
        )

    # -- CTBR: body-rate + thrust (the ACRO control path) -------------------
    def _body_rate_command(self, nav: NavState, sp: Setpoint) -> ControlCommand:
        """Geometric attitude->body-rate law for ACRO (collective-thrust + body-rate).

        Same desired attitude + collective thrust as the ATTITUDE path, but instead of sending
        the attitude (which this sim ignores -- it stays in ACRO), command the BODY RATE that
        rotates the current attitude toward the desired one. With R_cur, R_des = world<-body
        rotations, the error rotation in the body frame is ``R_e = R_cur^T R_des``; its axis-angle
        ``rotvec(R_e)`` is the small-rotation that aligns them, so ``omega = kp_att * rotvec(R_e)``
        drives the error to zero (the sim's fast rate loop tracks omega). Thrust rides along the
        current body -z; a_des is bounded (max_accel_mps2) so the attitude error -- hence the
        transient thrust-mispointing -- stays small.
        """
        a_des = self._desired_acceleration(nav, sp)
        yaw = sp.yaw if sp.yaw is not None else nav.yaw
        q_des_wxyz, thrust = self._accel_to_attitude(a_des, yaw)
        R_des = Rotation.from_quat(
            [q_des_wxyz[1], q_des_wxyz[2], q_des_wxyz[3], q_des_wxyz[0]]
        ).as_matrix()
        R_cur = R_world_from_body(nav.roll, nav.pitch, nav.yaw)
        rotvec = Rotation.from_matrix(R_cur.T @ R_des).as_rotvec()   # body-frame axis-angle error
        omega = self.kp_att * rotvec
        if self.kd_att > 0.0:                                        # damp the underdamped rate loop
            omega = omega - self.kd_att * np.asarray(nav.angular_rate_body, dtype=np.float64)
        omega = _clip_norm(omega, self.max_body_rate_rps)
        return ControlCommand(
            mode=ControlMode.BODY_RATE,
            sim_time_ns=sp.sim_time_ns,
            body_rate=omega,
            thrust=thrust,
        )

    def _accel_to_attitude(self, a_des_world: np.ndarray, yaw: float) -> tuple[np.ndarray, float]:
        """Desired world accel + heading -> (attitude quaternion wxyz, normalized thrust)."""
        # Specific force the rotors must produce, in world NED. Hover (a_des=0) -> -g (points up).
        f_world = np.asarray(a_des_world, dtype=np.float64) - _GRAVITY_NED
        f_mag = float(np.linalg.norm(f_world))
        if f_mag < 1e-6:
            return _IDENTITY_QUAT_WXYZ.copy(), 0.0  # free-fall reference: level, no thrust
        thrust_dir = f_world / f_mag                # world unit vector the thrust points along (up-ish)

        # Clamp the tilt away from vertical so the attitude never demands a near-horizontal
        # (or inverted) thrust the airframe can't hold.
        cos_tilt = float(np.clip(thrust_dir @ _WORLD_UP, -1.0, 1.0))
        if np.arccos(cos_tilt) > self.max_tilt_rad:
            axis = np.cross(_WORLD_UP, thrust_dir)
            n = np.linalg.norm(axis)
            if n > 1e-9:
                axis = axis / n
            else:
                # thrust_dir is (anti)parallel to vertical: past the tilt limit only when
                # pointing straight DOWN (a pathological dive), where no tilt axis is defined.
                # The old `if n > 1e-9` guard silently SKIPPED the clamp here, leaving an
                # inverted thrust command. Pick an arbitrary horizontal axis so we still clamp
                # to the limit and recover toward upright. [red-team 2026-05-30]
                axis = np.array([1.0, 0.0, 0.0])
            thrust_dir = Rotation.from_rotvec(axis * self.max_tilt_rad).apply(_WORLD_UP)

        # Build R_world_body (FRD) from the desired body-down axis + heading.
        b3 = -thrust_dir                                   # body z (down) in world; hover -> [0,0,1]
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])    # desired forward heading in the world plane
        b2 = np.cross(b3, x_c)
        n2 = float(np.linalg.norm(b2))
        if n2 < 1e-6:
            # b3 is (anti)parallel to the heading vector -- only reachable at ~horizontal
            # thrust, which the tilt clamp normally prevents. Use the heading's left-normal as
            # body-y to keep a defined, heading-consistent frame instead of dividing by ~0 and
            # emitting NaNs into the attitude quaternion. [red-team 2026-05-30]
            b2 = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
            n2 = float(np.linalg.norm(b2))
        b2 = b2 / n2                                       # body y (right)
        b1 = np.cross(b2, b3)                              # body x (forward)
        R_wb = np.column_stack([b1, b2, b3])
        q_xyzw = Rotation.from_matrix(R_wb).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        # Throttle from the required specific force. When the tilt was clamped, size the
        # collective so its VERTICAL (world-up) projection still meets the vertical specific
        # force the setpoint needs. Otherwise a large horizontal demand keeps the full,
        # un-clamped magnitude at a clamped 45-deg angle, so the surplus vertical component
        # (|f| * cos 45) rockets the drone skyward on every hard turn or brake. Altitude
        # priority: keep exactly what holds us up, sacrifice the horizontal we can't reach.
        # The formula is an identity when the tilt is NOT clamped (cos_tilt = |f_up|/f_mag),
        # so the un-clamped path is unchanged. [red-team 2026-05-30]
        f_up = float(-f_world[2])                    # required upward specific force (world up +)
        cos_tilt = float(thrust_dir @ _WORLD_UP)     # cos(actual tilt) after any clamp
        if f_up > 1e-9 and cos_tilt > 1e-6:
            f_mag = f_up / cos_tilt
        if self.thrust_slope_mps2 is not None and self.thrust_slope_mps2 > 0.0:
            # Measured affine throttle map (innerloop_step): |f| = g at hover, slope m/s^2 per
            # unit thrust => thrust = hover + (|f| - g)/slope. Passes through (g, hover) like the
            # placeholder but uses the real slope (the placeholder's |f|/g over-thrusts ~30%).
            thrust = self.hover_thrust + (f_mag - _G) / self.thrust_slope_mps2
        else:
            # PLACEHOLDER throttle scale (linear, anchored so |f|=g -> hover) until system-ID.
            thrust = self.hover_thrust * f_mag / _G
        return q_wxyz, float(np.clip(thrust, 0.0, 1.0))
