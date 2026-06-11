"""CtbrPlant — an offline forward-simulator of the sim's CTBR (body-rate + collective) plant.

It consumes a :class:`ControlCommand` (BODY_RATE: ``body_rate`` + ``thrust``) and integrates a
rigid-body quadrotor model, producing a :class:`DroneState` *identical in shape* to what
``mavlink_client`` emits. So the SAME :class:`~racer.navigator.Navigator` +
:class:`~racer.controller.Controller` run against it — a drop-in for the live sim — letting CTBR
gains/signs be tuned OFFLINE, deterministically, in milliseconds per step, instead of live under a
race countdown. The whole point: end the live whack-a-mole (roadmap step B; see
``project_ctbr_control_sysid.md`` and ``MEMORY.md``).

Conventions (match :mod:`racer.contracts` + :mod:`racer.controller`): world NED (X north, Y east,
Z DOWN; g = +9.80665 on Z), body FRD (X fwd, Y right, Z down), thrust acts along body **-Z** (up).
Orientation quaternion is (w, x, y, z) = R_world_body (MAVLink scalar-first).

This is a CANONICAL plant by default: a +roll-rate command rolls right-wing-down and the drone
accelerates +Y (right); thrust = ``hover_thrust`` nets zero vertical acceleration (it hovers);
``rate_gain``/``rate_sign`` are identity. The *real* sim adds measured quirks on top (steady inner-
loop rate gain ~2.6x, roll/yaw rate-sign inversions — see the sysid memo); set those on
:class:`CtbrPlantConfig` for a SIM-FAITHFUL twin whose tuning transfers to the real sim. The inner
rate loop is a first-order lag (``rate_tau_s``) toward ``rate_gain * rate_sign * commanded_rate``;
optionally the steady gain is the measured STATIC amplitude-dependent "super-rate" map
``rate_gain / (1 - super_rate_s * min(|cmd|, pi)/pi)`` with a per-axis slew limit ``alpha_max_rps2``
(characterize-sweep 2026-06-10 -- the flat-2.5 gain under-predicts authority by up to 42% at full
stick; both new params default OFF for exact legacy behavior).

NB this models the CLEAN CTBR rotor physics — it deliberately does NOT model the sim's broken
vertical *velocity-setpoint* auto-thrust (the "altitude balloon" / World-A runaway); in CTBR we own
thrust, so that quirk is a separate, still-uncharacterised concern (do not assume the twin captures
it).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from racer.contracts import ControlCommand, DroneState
from racer.frames import euler_from_quat_wxyz

_G = 9.80665


def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > max_norm > 0.0:
        return v * (max_norm / n)
    return v


def _wxyz_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Aerospace 3-2-1 (ZYX) Euler -> body->world quaternion (w,x,y,z). Inverse of
    ``frames.euler_from_quat_wxyz`` -- used to re-emit a sign-flipped (telemetry-convention) attitude."""
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


@dataclass
class CtbrPlantConfig:
    """Plant parameters. Defaults are a canonical unity-gain quadrotor; set ``rate_gain`` /
    ``rate_sign`` to the measured sim values for a faithful twin (sysid: roll 2.73, pitch 2.68,
    yaw 2.38; roll/yaw command-sign inverted)."""

    hover_thrust: float = 0.26          # collective that nets ZERO vertical accel (measured ~0.26)
    g: float = _G
    # Inner rate loop: realized body rate is a first-order lag (time constant ``rate_tau_s``)
    # toward ``rate_gain * rate_sign * commanded_rate``.
    rate_tau_s: float = 0.05
    rate_gain: np.ndarray = field(default_factory=lambda: np.ones(3))   # steady realized/commanded
    rate_sign: np.ndarray = field(default_factory=lambda: np.ones(3))   # sim sign on the command
    # STATIC amplitude-dependent gain map ("super-rate"; characterize-sweep 2026-06-10): the sim's
    # inner-loop DC gain GROWS with command amplitude, per axis:
    #     g(|c|) = rate_gain / (1 - super_rate_s * min(|c|, pi) / pi)
    # measured s ~= 0.30 roll/pitch (sustained gains 2.50 -> 3.50 over |cmd| 0.3 -> 3.14); yaw uses
    # the same form (s_yaw ~= s_roll) -- the measured level-attitude ~7.4 rad/s yaw plateau is a
    # KNOWN UNMODELED caveat (maneuver-dependent; racing yaw cmds are small; own pass if it matters).
    # None -> the legacy FLAT rate_gain (bit-identical pre-map behavior). Scalar or per-axis (3,).
    super_rate_s: float | np.ndarray | None = None
    # Per-axis slew limit on the REALIZED rate (rad/s^2): each step's omega increment is clamped to
    # +-alpha_max_rps2 * dt (measured saturated rise ~250-280 roll/pitch, ~78 yaw). None -> unlimited
    # (legacy). Scalar or per-axis (3,).
    alpha_max_rps2: float | np.ndarray | None = None
    linear_drag: float = 0.0            # optional world-frame linear drag (1/s); 0 = ideal
    # Sim-to-real LATENCY the ideal twin lacked -- why kp_alt=4 looked stable offline but relay-
    # oscillated live (VERIFY rung 1, 2026-06-06): a transport delay on the sense->command->act loop
    # + a first-order lag on the realized collective (motor spin-up). With the controller's thrust
    # clips, a too-stiff alt PD + this delay = a relay limit cycle. Both 0 = instantaneous (canonical);
    # fit to the live limit-cycle period.
    cmd_latency_s: float = 0.0          # transport delay applied to the whole command (s)
    thrust_tau_s: float = 0.0           # first-order lag on the realized collective (s)
    # Sanity NORM clamp on the realized body rate. With the super-rate map ON the real DC ceiling is
    # g(pi)*pi ~= 11.2 rad/s per axis (~18.5 worst-case 3-axis norm) -- keep this >= ~11.5 rad/s per
    # axis or it re-introduces the exact flat-2.5 ceiling artifact the map removes.
    max_omega_rps: float = 25.0
    # MEASUREMENT-report signs on the EMITTED ODOMETRY state, vs the TRUE physical attitude/rate the
    # plant integrates (the PHYSICS always uses the true frame -> correct thrust direction). The real
    # sim's telemetry inverts the ODOMETRY-quaternion ROLL and the angular_rate on ROLL+PITCH (the
    # Gate-0 saga + sysid extract), so a faithful twin emits these inversions and the controller's
    # ``odo_att_sign`` / ``odo_rate_sign`` undo them EXACTLY as live -- making the measured live signs
    # transfer. Both default [1,1,1] = canonical (emit the true frame). These touch only ``state()``,
    # never the physics. (att report = euler signs applied to the emitted roll/pitch/yaw + quaternion.)
    odo_att_report_sign: np.ndarray = field(default_factory=lambda: np.ones(3))
    odo_rate_report_sign: np.ndarray = field(default_factory=lambda: np.ones(3))


class CtbrPlant:
    """Stateful forward-simulator. Call :meth:`step` with a command + dt, read :meth:`state`."""

    def __init__(
        self,
        config: CtbrPlantConfig | None = None,
        *,
        position_ned=None,
        velocity_ned=None,
        q_wxyz=None,
        t0_ns: int = 0,
    ) -> None:
        self.cfg = config or CtbrPlantConfig()
        self.pos = np.zeros(3) if position_ned is None else np.asarray(position_ned, float).copy()
        self.vel = np.zeros(3) if velocity_ned is None else np.asarray(velocity_ned, float).copy()
        q = np.array([1.0, 0.0, 0.0, 0.0]) if q_wxyz is None else np.asarray(q_wxyz, float).copy()
        self.q = q / max(float(np.linalg.norm(q)), 1e-12)
        self.omega = np.zeros(3)                                   # body rate FRD (rad/s)
        self.accel_body = np.array([0.0, 0.0, -self.cfg.g])        # specific force, body (hover)
        self.t_ns = int(t0_ns)
        self._cmd_buf: list[ControlCommand] = []                   # transport-delay buffer (cmd_latency_s)
        self._thrust = float(self.cfg.hover_thrust)                # realized collective (thrust_tau_s lag)

    @staticmethod
    def _from_quat(q_wxyz: np.ndarray) -> Rotation:
        return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

    @staticmethod
    def _to_wxyz(r: Rotation) -> np.ndarray:
        x, y, z, w = r.as_quat()
        return np.array([w, x, y, z], dtype=np.float64)

    def step(self, cmd: ControlCommand, dt: float) -> None:
        """Advance the plant by ``dt`` seconds under one body-rate + thrust command."""
        if dt <= 0.0:
            return
        cfg = self.cfg
        # --- sense->command->act TRANSPORT DELAY: apply the command from cmd_latency_s ago (the
        # dynamic the ideal twin lacked; with the thrust clips it turns a too-stiff alt PD into a
        # relay limit cycle). 0 -> instantaneous (canonical). ---
        self._cmd_buf.append(cmd)
        nlag = int(round(cfg.cmd_latency_s / dt)) if cfg.cmd_latency_s > 0.0 else 0
        cmd = self._cmd_buf[-(nlag + 1)] if len(self._cmd_buf) >= nlag + 1 else self._cmd_buf[0]
        if len(self._cmd_buf) > nlag + 2:
            self._cmd_buf.pop(0)
        # --- inner rate loop: first-order lag toward the sim's realized steady rate. The steady
        # gain is FLAT rate_gain (legacy) or the measured STATIC amplitude-dependent map when
        # super_rate_s is set (see CtbrPlantConfig); the increment is slew-limited when
        # alpha_max_rps2 is set. Both off -> bit-identical to the pre-map update. ---
        cmd_rate = np.zeros(3) if cmd.body_rate is None else np.asarray(cmd.body_rate, dtype=np.float64)
        gain = np.asarray(cfg.rate_gain)
        if cfg.super_rate_s is not None:
            s = np.asarray(cfg.super_rate_s, dtype=np.float64)
            gain = gain / (1.0 - s * np.minimum(np.abs(cmd_rate), np.pi) / np.pi)
        target = gain * np.asarray(cfg.rate_sign) * cmd_rate
        alpha = 1.0 - np.exp(-dt / max(cfg.rate_tau_s, 1e-9))
        domega = alpha * (target - self.omega)
        if cfg.alpha_max_rps2 is not None:
            lim = np.asarray(cfg.alpha_max_rps2, dtype=np.float64) * dt
            domega = np.clip(domega, -lim, lim)
        self.omega = _clip_norm(self.omega + domega, cfg.max_omega_rps)
        # --- attitude: integrate the body-frame rate (right-multiply: omega is in body) ---
        R_cur = self._from_quat(self.q)
        R_new = R_cur * Rotation.from_rotvec(self.omega * dt)
        self.q = self._to_wxyz(R_new)
        # --- thrust -> body-up specific force -> world accel + gravity (+ optional drag) ---
        thrust_cmd = 0.0 if cmd.thrust is None else float(cmd.thrust)
        if cfg.thrust_tau_s > 0.0:                                 # actuator spin-up lag (0 -> instant)
            self._thrust += (1.0 - np.exp(-dt / cfg.thrust_tau_s)) * (thrust_cmd - self._thrust)
        else:
            self._thrust = thrust_cmd
        a_up = cfg.g * (self._thrust / cfg.hover_thrust)           # thrust=hover -> g (balances)
        f_world = R_new.as_matrix() @ np.array([0.0, 0.0, -a_up])  # body -Z (up) in world NED
        f_world = f_world - cfg.linear_drag * self.vel             # specific force incl. drag
        accel = f_world + np.array([0.0, 0.0, cfg.g])              # + gravity (NED +Z down)
        # semi-implicit Euler (update velocity first, then position with the new velocity)
        self.vel = self.vel + accel * dt
        self.pos = self.pos + self.vel * dt
        self.accel_body = R_new.as_matrix().T @ f_world           # what an accelerometer reads
        self.t_ns += int(round(dt * 1e9))

    def state(self) -> DroneState:
        """Current plant state as a :class:`DroneState` (world pos/vel; ODOMETRY-style attitude)."""
        roll, pitch, yaw = euler_from_quat_wxyz(self.q)               # TRUE physical attitude
        q_out = self.q.copy()
        asign = np.asarray(self.cfg.odo_att_report_sign)
        if not np.allclose(asign, 1.0):                              # emit the telemetry convention
            roll, pitch, yaw = roll * asign[0], pitch * asign[1], yaw * asign[2]
            q_out = _wxyz_from_euler(roll, pitch, yaw)
        return DroneState(
            sim_time_ns=self.t_ns,
            position_ned=self.pos.copy(),
            velocity_ned=self.vel.copy(),
            orientation_ned_wxyz=q_out,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            angular_rate_body=self.omega * np.asarray(self.cfg.odo_rate_report_sign),  # ODOMETRY convention
            accel_body=self.accel_body.copy(),
            armed=True,
        )
