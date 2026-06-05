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
rate loop is a first-order lag (``rate_tau_s``) toward ``rate_gain * rate_sign * commanded_rate``.

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
    linear_drag: float = 0.0            # optional world-frame linear drag (1/s); 0 = ideal
    max_omega_rps: float = 25.0         # sanity clamp on realized body rate


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
        # --- inner rate loop: first-order lag toward the sim's realized steady rate ---
        cmd_rate = np.zeros(3) if cmd.body_rate is None else np.asarray(cmd.body_rate, dtype=np.float64)
        target = np.asarray(cfg.rate_gain) * np.asarray(cfg.rate_sign) * cmd_rate
        alpha = 1.0 - np.exp(-dt / max(cfg.rate_tau_s, 1e-9))
        self.omega = _clip_norm(self.omega + alpha * (target - self.omega), cfg.max_omega_rps)
        # --- attitude: integrate the body-frame rate (right-multiply: omega is in body) ---
        R_cur = self._from_quat(self.q)
        R_new = R_cur * Rotation.from_rotvec(self.omega * dt)
        self.q = self._to_wxyz(R_new)
        # --- thrust -> body-up specific force -> world accel + gravity (+ optional drag) ---
        thrust = 0.0 if cmd.thrust is None else float(cmd.thrust)
        a_up = cfg.g * (thrust / cfg.hover_thrust)                 # thrust=hover -> g (balances)
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
        roll, pitch, yaw = euler_from_quat_wxyz(self.q)
        return DroneState(
            sim_time_ns=self.t_ns,
            position_ned=self.pos.copy(),
            velocity_ned=self.vel.copy(),
            orientation_ned_wxyz=self.q.copy(),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            angular_rate_body=self.omega.copy(),
            accel_body=self.accel_body.copy(),
            armed=True,
        )
