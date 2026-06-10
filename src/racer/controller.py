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

from dataclasses import dataclass, field

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


def level_hold_body_rate(
    roll: float, pitch: float, yaw: float, yaw_hold: float,
    angular_rate_body: np.ndarray, *,
    kp: float, kd: float, body_rate_sign: np.ndarray, max_rate: float,
    ff_gain: float = 1.0,
) -> np.ndarray:
    """Body-rate command (sim actuation convention) that drives the attitude toward LEVEL at
    ``yaw_hold`` with rate damping. The same rotvec error + sign mapping the CTBR controller
    uses, specialised to a level target and a FIXED collective thrust (no position/thrust
    coupling) — so it can't pump altitude the way the full position loop does.

    Used by the open-loop rate-STEP sysid probe as the 're-level' between steps: zero rate
    does NOT re-level (body-rate is a rate, not an attitude, command), so a closed level-hold is
    needed to undo each step's attitude excursion and to cancel the −17.8° resting tilt that
    would otherwise drift the drone forward at ~g·tan(17.8°). The damping uses the trusted
    ODOMETRY rate BEFORE the sign map (the convention the stable damped run validated).

    ``ff_gain`` divides the commanded rate by the MEASURED inner-loop rate scaling (~2.7×): the
    sim amplifies a commanded body rate ~2.7×, so a naive kp on the attitude error overshoots and
    oscillates (the first live hover-sweep). Dividing by the gain makes the realised rate equal
    (kp·err − kd·rate) on a unity plant, so kp/kd tune as a normal attitude loop. ff_gain=1 keeps
    the legacy (un-fed-forward) behaviour."""
    R_cur = R_world_from_body(roll, pitch, yaw)
    R_des = R_world_from_body(0.0, 0.0, yaw_hold)
    rotvec = Rotation.from_matrix(R_cur.T @ R_des).as_rotvec()    # trusted-FRD attitude error
    omega = (kp * rotvec - kd * np.asarray(angular_rate_body, dtype=np.float64)) / max(ff_gain, 1e-6)
    omega = _clip_norm(omega, max_rate)
    return omega * np.asarray(body_rate_sign, dtype=np.float64)   # -> sim actuation convention


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
    # Per-axis sign of the body-rate COMMAND the sim expects, in (roll, pitch, yaw). The pure
    # geometric law above is computed in trusted FRD; this maps it to the sim's actuation
    # convention. MEASURED at first contact (2026-06-02, offline command-vs-response replay):
    # this sim INVERTS roll + yaw body-rate commands (pitch is correct) -- a +roll/+yaw command
    # rotates the drone the other way, so a pure-P loop spirals. The default is the identity
    # (sim-agnostic, the pure law that the tests check); fly_vq1 passes the measured [-1,1,-1].
    body_rate_sign: np.ndarray = field(default_factory=lambda: np.ones(3))

    # -- DECOUPLED CTBR (the plant-matched controller, system-ID'd 2026-06-03) ----------------
    # The sim is ACRO-only for control (any setpoint type either runs away or the keepalive forces
    # ACRO); CTBR (body-rate + EXPLICIT thrust) is the only clean full-3-axis path. Set
    # ``decoupled=True`` to fly the matched law: vertical = a velocity-damped altitude hold around
    # ``hover_thrust`` (the sim's auto-thrust is broken, so WE own thrust); horizontal = a PD on the
    # xy error -> a tilt DIRECTION (reusing the tested _accel_to_attitude geometry, vertical zeroed);
    # attitude->rate = the rotvec law with the MEASURED feedforward + sign-correct damping.
    decoupled: bool = False
    # Steady inner-loop rate gain (MEASURED: roll 2.73, pitch 2.68, yaw 2.38 -> ~2.6). The sim
    # amplifies a commanded body rate ~2.6x at steady state, so divide the commanded rate by it to
    # get a unity-gain attitude loop (a naive kp overshoots + tumbles). 1.0 = no feedforward.
    ff_gain: float = 1.0
    # Sign of the ODOMETRY angular_rate vs the TRUE attitude derivative, per axis (MEASURED:
    # [+1,-1,+1] -- only PITCH's rate is inverted, same quirk as the ATTITUDE Euler). Multiply the
    # measured rate by this BEFORE the -kd_att damping, else pitch damping is anti-damping (tumble).
    odo_rate_sign: np.ndarray = field(default_factory=lambda: np.ones(3))
    # Sign of the ODOMETRY-quaternion-derived ATTITUDE (roll,pitch,yaw) vs the TRUE physical angle.
    # The lateral loop ran away (+y drift with +roll, which physically needs -roll -> positive
    # feedback, measured gate0_center2 2026-06-04): the decoded ROLL is inverted. Multiply the euler
    # used to build R_cur by this so the attitude error is physical. Pitch reads true (forward
    # flight works) => [-1, +1, +1] for the live sim. Pair with the matching roll flips in
    # body_rate_sign (+1) and odo_rate_sign (-1) so command + rate damping are consistent.
    odo_att_sign: np.ndarray = field(default_factory=lambda: np.ones(3))
    # Altitude-hold (vertical thrust) channel: thrust = hover + kp_alt*(z - z_target) +
    # kd_alt*(vz - vz_target), clamped. NED z+ = down, so a SINK (z>target / vz>0) -> more thrust.
    kp_alt: float = 0.0
    kd_alt: float = 0.0
    alt_thrust_lo: float = 0.0
    alt_thrust_hi: float = 1.0
    # Fly this many metres ABOVE the setpoint altitude (NED: z_target -= offset, since z+ = down).
    # The gate map's position may not be the opening CENTRE, and any residual alt droop sits the
    # drone low -- visually confirmed clipping the BOTTOM bar of gate 0 (2026-06-04). A positive
    # offset raises the whole vertical target for margin / to calibrate against where the gate
    # opening actually is.
    alt_offset_m: float = 0.0
    # Tilt-compensate the collective: divide the alt-hold thrust by cos(roll)*cos(pitch) (the
    # world-vertical fraction of body thrust, R[2,2]) so a forward LEAN doesn't silently sag
    # altitude. Without it, pitching to fly forward drops the vertical thrust component, the soft
    # alt PD over-corrects, and the drone balloons UP into the gate (measured: 2 m climb threading
    # gate 0). Floor at 0.5 (=60 deg) so a near-horizontal attitude can't blow the throttle up.
    tilt_comp: bool = False
    # Horizontal: when set, use VELOCITY-TARGETING (cap SPEED, not the position gain). A clamped
    # position error + unbounded velocity damping (the old PD) lets the position pull dominate, so
    # the drone overshoots cruise AND under-corrects laterally (it flew 5 m/s and missed the gate
    # ~1 m sideways). Instead: desired_vel = clip(kp_pos*err, max_speed); a_h = kd_vel*(des_vel -
    # vel) -- strong position tracking at a bounded speed. None => legacy PD. [teammate red-team]
    max_speed: float | None = None

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
        if self.decoupled:
            return self._decoupled_body_rate(nav, sp)
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
        omega = omega * np.asarray(self.body_rate_sign, dtype=np.float64)   # -> sim actuation convention
        return ControlCommand(
            mode=ControlMode.BODY_RATE,
            sim_time_ns=sp.sim_time_ns,
            body_rate=omega,
            thrust=thrust,
        )

    def _decoupled_body_rate(self, nav: NavState, sp: Setpoint) -> ControlCommand:
        """Plant-matched CTBR (system-ID'd 2026-06-03). Decoupled cascade:

        VERTICAL  velocity-damped altitude hold around ``hover_thrust`` -- WE own thrust because the
                  sim's auto-thrust (velocity/position modes) climbs away. NED z+ down.
        HORIZONTAL PD on the xy position/velocity error -> a desired horizontal acceleration ->
                  a tilt DIRECTION via the tested ``_accel_to_attitude`` geometry (vertical zeroed,
                  so the tilt is purely for translation; the alt-hold owns the collective).
        ATTITUDE  omega = (kp_att*rotvec(R_cur^T R_des) - kd_att*corrected_rate)/ff_gain, clamped,
                  then * body_rate_sign. ``ff_gain`` undoes the ~2.6x inner-loop amplification;
                  ``odo_rate_sign`` fixes the inverted-pitch ODOMETRY rate so damping is real."""
        pos = np.asarray(nav.position_ned, dtype=np.float64)
        vel = np.asarray(nav.velocity_ned, dtype=np.float64)
        # -- vertical: altitude hold -> collective thrust --
        z_t = float(sp.position_ned[2]) if sp.position_ned is not None else float(pos[2])
        z_t = z_t - self.alt_offset_m                  # fly above the gate line (NED z+ = down)
        vz_t = float(sp.velocity_ned[2]) if sp.velocity_ned is not None else 0.0
        thrust = self.hover_thrust + self.kp_alt * (pos[2] - z_t) + self.kd_alt * (vel[2] - vz_t)
        if self.tilt_comp:                             # undo the vertical-thrust loss from leaning
            cos_tilt = float(np.cos(nav.roll) * np.cos(nav.pitch))   # = R[2,2], world-up fraction
            thrust = thrust / max(cos_tilt, 0.5)       # floor at 60 deg so it can't blow up
        thrust = float(np.clip(thrust, self.alt_thrust_lo, self.alt_thrust_hi))
        # -- horizontal: xy error -> desired horizontal accel (vertical zeroed) --
        a_h = np.zeros(3)
        if sp.accel_ned is not None:
            a_h = a_h + np.asarray(sp.accel_ned, dtype=np.float64)
        if self.max_speed is not None and sp.position_ned is not None:
            # velocity-targeting: a capped desired velocity toward the target, then damp to it
            err = np.asarray(sp.position_ned, dtype=np.float64) - pos
            err[2] = 0.0
            if sp.yaw is not None:
                # DECOUPLE along-track (cap SPEED) from cross-track (correct POSITION at full gain).
                # _clip_norm preserves the error DIRECTION, which the huge along-track component
                # dominates -> the cross-track velocity target shrinks to noise and a small lateral
                # disturbance wins (measured: +y drift to +12 m aiming 1.8 m off-axis, gate0_center1).
                # Split on the gate axis (sp.yaw): cap only the forward target, keep cross-track full.
                ad = np.array([np.cos(sp.yaw), np.sin(sp.yaw), 0.0])     # along-track (gate-axis) dir
                along = float(np.clip(self.kp_pos * float(err @ ad), -self.max_speed, self.max_speed))
                cross = _clip_norm(self.kp_pos * (err - float(err @ ad) * ad), self.max_speed)
                des_vel = along * ad + cross
            else:
                des_vel = _clip_norm(self.kp_pos * err, self.max_speed)
            vh = np.array([vel[0], vel[1], 0.0])
            a_h = a_h + self.kd_vel * (des_vel - vh)
        else:                                          # legacy PD (kept for compatibility)
            if sp.position_ned is not None:
                err = np.asarray(sp.position_ned, dtype=np.float64) - pos
                err[2] = 0.0
                if self.max_pos_error_m is not None:
                    err = _clip_norm(err, self.max_pos_error_m)
                a_h = a_h + self.kp_pos * err
            if sp.velocity_ned is not None:
                a_h = a_h + self.kd_vel * (np.asarray(sp.velocity_ned, dtype=np.float64) - vel)
            elif sp.position_ned is not None:
                a_h = a_h - self.kd_vel * vel
        a_h[2] = 0.0                                  # the alt-hold owns vertical
        if self.max_accel_mps2 is not None:
            a_h = _clip_norm(a_h, self.max_accel_mps2)
        if sp.launch_ramp is not None:                # takeoff->RUN launch ramp (Mission-driven)
            # Ramp the commanded horizontal accel (hence the desired TILT) up from zero so the
            # attitude target can't STEP to the ~45 deg cruise lean in one tick. A step there makes
            # the attitude error -> kp_att*rotvec saturate the body-rate clamp, and the underdamped
            # rate loop overshoots into a tumble (sim 1.0.3364 start-transient: roll cmd hit the
            # 8 rad/s clamp at t+1.22 s). At ramp=0 a_h=0 -> level hover attitude = the current
            # attitude post-takeoff -> ~zero rate command, regardless of tick phase. Vertical
            # (alt-hold) authority is untouched, so the climb is never starved. [build-1.0.3364 fix]
            a_h = a_h * float(np.clip(sp.launch_ramp, 0.0, 1.0))
        # -- attitude: tilt direction from a_h, then rotvec -> body rate (matched) --
        yaw = sp.yaw if sp.yaw is not None else nav.yaw
        q_des_wxyz, _ = self._accel_to_attitude(a_h, yaw)            # direction only (ignore its thrust)
        R_des = Rotation.from_quat(
            [q_des_wxyz[1], q_des_wxyz[2], q_des_wxyz[3], q_des_wxyz[0]]
        ).as_matrix()
        asign = np.asarray(self.odo_att_sign, dtype=np.float64)
        R_cur = R_world_from_body(nav.roll * asign[0], nav.pitch * asign[1], nav.yaw * asign[2])
        rotvec = Rotation.from_matrix(R_cur.T @ R_des).as_rotvec()
        rate = np.asarray(nav.angular_rate_body, dtype=np.float64) * np.asarray(self.odo_rate_sign, dtype=np.float64)
        omega = (self.kp_att * rotvec - self.kd_att * rate) / max(self.ff_gain, 1e-6)
        omega = _clip_norm(omega, self.max_body_rate_rps)
        omega = omega * np.asarray(self.body_rate_sign, dtype=np.float64)
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
        elif f_up <= 0.0:
            # Dive demanded FASTER than gravity (a_des down > g) while the tilt clamp holds us
            # upright: an upright quad cannot thrust downward, so the most it can do is CUT the
            # throttle and free-fall at g. Without this the old code skipped the rescale and kept
            # the full |f_world| magnitude at an upright attitude -> a positive thrust that rockets
            # the drone SKYWARD on every hard brake/dive (a runaway positive-feedback loop into the
            # ceiling). [2026-06-03 teammate red-team; reproduced live]
            f_mag = 0.0
        if self.thrust_slope_mps2 is not None and self.thrust_slope_mps2 > 0.0:
            # Measured affine throttle map (innerloop_step): |f| = g at hover, slope m/s^2 per
            # unit thrust => thrust = hover + (|f| - g)/slope. Passes through (g, hover) like the
            # placeholder but uses the real slope (the placeholder's |f|/g over-thrusts ~30%).
            thrust = self.hover_thrust + (f_mag - _G) / self.thrust_slope_mps2
        else:
            # PLACEHOLDER throttle scale (linear, anchored so |f|=g -> hover) until system-ID.
            thrust = self.hover_thrust * f_mag / _G
        return q_wxyz, float(np.clip(thrust, 0.0, 1.0))
