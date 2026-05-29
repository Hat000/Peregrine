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

_G = 9.80665
_GRAVITY_NED = np.array([0.0, 0.0, _G])     # NED: gravity points +z (down)
_WORLD_UP = np.array([0.0, 0.0, -1.0])      # NED up
_IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass
class Controller:
    """Setpoint + NavState -> ControlCommand. See module docstring for the two modes."""

    mode: ControlMode = ControlMode.POSITION
    # Attitude-path tracking gains (the sim stabilizer owns this loop in POSITION/VELOCITY).
    kp_pos: float = 1.5            # 1/s^2, position-error -> accel
    kd_vel: float = 2.0           # 1/s,   velocity-error -> accel
    # Plant/thrust model for the attitude path — PLACEHOLDER, system-ID at first contact (R2).
    hover_thrust: float = 0.5      # normalized throttle that holds a hover
    max_tilt_rad: float = np.deg2rad(45.0)

    def command(self, nav: NavState, setpoint: Setpoint) -> ControlCommand:
        """Compute the control command for the current state + reference."""
        if self.mode in (ControlMode.POSITION, ControlMode.VELOCITY):
            return self._setpoint_passthrough(setpoint)
        if self.mode == ControlMode.ATTITUDE:
            return self._attitude_command(nav, setpoint)
        # BODY_RATE (true CTBR) needs the inner-loop rate model — the documented speed upgrade.
        raise NotImplementedError(f"controller mode {self.mode!r} not implemented yet (CTBR is future work)")

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
            a = a + self.kp_pos * (np.asarray(sp.position_ned, dtype=np.float64) - nav.position_ned)
        if sp.velocity_ned is not None:
            a = a + self.kd_vel * (np.asarray(sp.velocity_ned, dtype=np.float64) - nav.velocity_ned)
        elif sp.position_ned is not None:
            a = a - self.kd_vel * np.asarray(nav.velocity_ned, dtype=np.float64)  # damp when only position is given
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
                thrust_dir = Rotation.from_rotvec((axis / n) * self.max_tilt_rad).apply(_WORLD_UP)

        # Build R_world_body (FRD) from the desired body-down axis + heading.
        b3 = -thrust_dir                                   # body z (down) in world; hover -> [0,0,1]
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])    # desired forward heading in the world plane
        b2 = np.cross(b3, x_c)
        b2 /= np.linalg.norm(b2)                           # body y (right)
        b1 = np.cross(b2, b3)                              # body x (forward)
        R_wb = np.column_stack([b1, b2, b3])
        q_xyzw = Rotation.from_matrix(R_wb).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        # PLACEHOLDER throttle: proportional to required specific force, anchored so |f|=g -> hover.
        # Calibrate the real throttle->thrust curve via innerloop_step at first contact (R2).
        thrust = float(np.clip(self.hover_thrust * f_mag / _G, 0.0, 1.0))
        return q_wxyz, thrust
