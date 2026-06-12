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
stick; both new params default OFF for exact legacy behavior). The twin-falsify campaign
(2026-06-11) additionally measured the AERO: drag is body-frame QUADRATIC and direction-dependent
(``quad_drag_c2``; the legacy linear 0.2111/s under-brakes 2.2x at 9 m/s) and the collective->accel
map is CONVEX (``coll_map_thr``/``coll_map_accel`` knot table; full stick is ~2.1x the linear
model). All three aero params default OFF (None) for exact legacy behavior.

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


def _quad_c2_table(c2) -> np.ndarray:
    """Normalise a ``quad_drag_c2`` spec (scalar | (3,) | (3, 2)) to the (3, 2) per-axis, per-sign
    coefficient table: ``[:, 0]`` applies where ``v_body[axis] >= 0``, ``[:, 1]`` where ``< 0``."""
    t = np.asarray(c2, dtype=np.float64)
    if t.ndim == 0:
        return np.full((3, 2), float(t))
    if t.shape == (3,):
        return np.stack([t, t], axis=-1)
    if t.shape == (3, 2):
        return t
    raise ValueError(f"quad_drag_c2 must be a scalar, (3,) or (3, 2); got shape {t.shape}")


def _mixer_r_fit(rate_gain, rate_sign, super_rate_s, hover, idle, kappa_err, zeta_yaw) -> np.ndarray:
    """Authority normalisation (3,) at the S14 slew-fit condition (hover collective, single-axis
    sustained pi command, zero rate): the realized/demanded differential ratio there. alpha_max
    was measured WITH the mixer already throttling at that point, so the slew limit scales by
    Q = r/r_fit -- keeping the fit point exact -- rather than by the raw r. Duplicated
    operation-for-operation in ``racer.rl_plant`` (parity-pinned; twin is ground truth).
    ``rate_sign`` only flips the sign of the demand -- |delta/d| is sign-invariant -- but is
    threaded through so the arithmetic matches the in-step computation bit-for-bit."""
    gain = np.asarray(rate_gain, dtype=np.float64)
    if super_rate_s is not None:
        s = np.asarray(super_rate_s, dtype=np.float64)
        gain = gain / (1.0 - s * np.minimum(np.pi, np.pi) / np.pi)
    target = gain * np.asarray(rate_sign, dtype=np.float64) * np.pi   # one full-stick axis at a time
    d = kappa_err * target
    eta = zeta_yaw / (zeta_yaw + np.maximum(hover, 0.0))
    d = np.concatenate([d[..., 0:2], d[..., 2:3] * eta])
    u_hi = np.clip(hover + np.abs(d), idle, 1.0)
    u_lo = np.clip(hover - np.abs(d), idle, 1.0)
    delta = (u_hi - u_lo) * 0.5
    return delta / np.abs(d)


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
    # MEASURED AERO (twin-falsify 2026-06-11, ``handoff/shadowpc-twin-falsify-2026-06-10/
    # WRITEUP.md`` Sections 2-3 + 7). Both default OFF (None) = exact legacy behavior.
    # Body-frame QUADRATIC drag, direction-dependent: f_b = -c2 (.) |v_b| (.) v_b (elementwise),
    # with a per-axis, per-SIGN coefficient table (3, 2): [axis][0] applies where v_b[axis] >= 0,
    # [axis][1] where < 0. Measured (1/m): x 0.042 nose-first / 0.058 tail-first; y 0.055 both;
    # z 0.054 descend (+z, FRD) / 0.076 climb (-z). Scalar = isotropic (pooled 0.052); (3,) =
    # per-axis sign-symmetric. ADDS to ``linear_drag`` (the measured-aero config zeroes
    # linear_drag; a small d1 + quad is the WRITEUP's "mixed" form). The legacy linear 0.2111/s
    # over-brakes at low speed and under-brakes 2.2x at 9 m/s -- exactly the race-speed band.
    quad_drag_c2: float | np.ndarray | None = None
    # MEASURED collective->accel map (same campaign): the sim's thrust curve is CONVEX -- full
    # stick is 78.3 m/s^2 ~= 2.1x the linear g*thr/hover model, sub-linear below hover. When BOTH
    # arrays are set, a_up = np.interp(realised_collective, coll_map_thr, coll_map_accel) REPLACES
    # the linear map (np.interp clamps to the end knots outside the range -- collective beyond the
    # last knot saturates, faithful to the [0,1] stick). Setting exactly one raises. None -> OFF.
    coll_map_thr: np.ndarray | None = None     # (K,) increasing collective knots [0..1]
    coll_map_accel: np.ndarray | None = None   # (K,) body-up specific accel at the knots (m/s^2)
    # MOTOR-MIXER coupling (live-deploy diag 2026-06-11, ``handoff/shadowpc-live-deploy-diag-
    # 2026-06-11/WRITEUP.md`` Sections 2 + 8; fit in ``handoff/laptop-s17-mixer-inc6-2026-06-11/
    # fit_mixer.py``). The sim's per-motor commands are collective +- rate-PID differentials,
    # clipped to [idle, 1] -- so (low collective x high rate demand) generates UNCOMMANDED LIFT
    # via the clipped low pair (measured: thr 0 + yaw 3.14 -> motors [.08,.73,.73,.08], a_up
    # 9.4 m/s^2 ~= hover at commanded ZERO), and (collective ~1) leaves no up-headroom -- rate
    # authority degrades during thrust pulses. Model (all four set together; None -> exact
    # legacy, thrust and rates independent):
    #   d_ax  = kappa_err*(target_ax - omega_ax) + kappa_hold*omega_ax;  d_yaw *= zeta/(zeta+c)
    #   u_i   = clip(c + S_i . d, idle, 1)  (X-quad signs);  c_eff = mean(u) -> the thrust map
    #   Q_ax  = (delta_ax/d_ax)/r_fit_ax scales the alpha_max slew limit (delta = realized
    #           differential (S^T u)/4; r_fit = the same ratio at the S14 slew-fit condition --
    #           hover collective, single-axis pi -- because alpha_max was measured WITH the
    #           mixer throttling there; requires alpha_max_rps2 set).
    mixer_idle: float | None = None         # motor idle clip floor (collective units), ~0.05
    mixer_kappa_err: float | None = None    # differential per rad/s of rate ERROR, ~0.073
    mixer_kappa_hold: float | None = None   # differential per rad/s of HELD rate, ~0.046
    mixer_zeta_yaw: float | None = None     # yaw effectiveness d_yaw *= zeta/(zeta+c), ~0.34
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
        # --- realised collective FIRST (moved ahead of the rate loop for the mixer, which needs
        # it; the lag is independent of the rate/attitude blocks, so the legacy floats are
        # unchanged -- pinned by tests/test_measured_aero's inline-legacy bit-identity test) ---
        thrust_cmd = 0.0 if cmd.thrust is None else float(cmd.thrust)
        if cfg.thrust_tau_s > 0.0:                                 # actuator spin-up lag (0 -> instant)
            self._thrust += (1.0 - np.exp(-dt / cfg.thrust_tau_s)) * (thrust_cmd - self._thrust)
        else:
            self._thrust = thrust_cmd
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
        c_eff = None                                               # mixer-OFF: thrust map sees _thrust
        if cfg.mixer_idle is not None:
            # --- MOTOR MIXER (live-deploy diag 2026-06-11; see CtbrPlantConfig). Per-motor
            # commands = collective +- the rate-loop differential demand, clipped to [idle, 1]:
            # the clipped MEAN re-enters the thrust map (parasitic lift at the bottom rail,
            # thrust sag at the top), the clipped DIFFERENTIAL throttles the slew authority
            # (Q = r/r_fit; r_fit-normalised because alpha_max was fit mixer-throttled). ---
            if (cfg.mixer_kappa_err is None or cfg.mixer_kappa_hold is None
                    or cfg.mixer_zeta_yaw is None):
                raise ValueError("mixer_idle/kappa_err/kappa_hold/zeta_yaw must be set together")
            if cfg.alpha_max_rps2 is None:
                raise ValueError("the mixer authority model scales the measured slew limits: "
                                 "set alpha_max_rps2 when the mixer is on")
            if not (0.0 <= cfg.mixer_idle < 1.0 and cfg.mixer_kappa_err > 0.0
                    and cfg.mixer_kappa_hold >= 0.0 and cfg.mixer_zeta_yaw > 0.0):
                # mirror rl_plant's range check: kappa_err == 0 or zeta == 0 makes the r_fit
                # normalisation 0/0 -> NaN, surfacing as an unrelated scipy quaternion error
                raise ValueError("mixer params out of range: need 0 <= idle < 1, kappa_err > 0, "
                                 "kappa_hold >= 0, zeta_yaw > 0")
            c = float(self._thrust)
            e = target - self.omega
            d = cfg.mixer_kappa_err * e + cfg.mixer_kappa_hold * self.omega
            eta = cfg.mixer_zeta_yaw / (cfg.mixer_zeta_yaw + np.maximum(c, 0.0))
            d = np.concatenate([d[..., 0:2], d[..., 2:3] * eta])
            u0 = np.clip(c + (d[..., 0] + d[..., 1] + d[..., 2]), cfg.mixer_idle, 1.0)
            u1 = np.clip(c + (-d[..., 0] + d[..., 1] - d[..., 2]), cfg.mixer_idle, 1.0)
            u2 = np.clip(c + (d[..., 0] - d[..., 1] - d[..., 2]), cfg.mixer_idle, 1.0)
            u3 = np.clip(c + (-d[..., 0] - d[..., 1] + d[..., 2]), cfg.mixer_idle, 1.0)
            c_eff = (u0 + u1 + u2 + u3) * 0.25
            delta = np.stack([(u0 - u1 + u2 - u3) * 0.25,
                              (u0 + u1 - u2 - u3) * 0.25,
                              (u0 - u1 - u2 + u3) * 0.25], axis=-1)
            d_safe = np.where(np.abs(d) > 1e-9, d, 1.0)
            r = np.where(np.abs(d) > 1e-9, np.clip(delta / d_safe, 0.0, 1.0), 1.0)
            r_fit = _mixer_r_fit(cfg.rate_gain, cfg.rate_sign, cfg.super_rate_s,
                                 cfg.hover_thrust, cfg.mixer_idle, cfg.mixer_kappa_err,
                                 cfg.mixer_zeta_yaw)
            lim = np.asarray(cfg.alpha_max_rps2, dtype=np.float64) * (r / r_fit) * dt
            domega = np.clip(domega, -lim, lim)
        elif cfg.alpha_max_rps2 is not None:
            lim = np.asarray(cfg.alpha_max_rps2, dtype=np.float64) * dt
            domega = np.clip(domega, -lim, lim)
        self.omega = _clip_norm(self.omega + domega, cfg.max_omega_rps)
        # --- attitude: integrate the body-frame rate (right-multiply: omega is in body) ---
        R_cur = self._from_quat(self.q)
        R_new = R_cur * Rotation.from_rotvec(self.omega * dt)
        self.q = self._to_wxyz(R_new)
        # --- thrust -> body-up specific force -> world accel + gravity (+ optional drag). The
        # thrust map consumes the realised collective, or the mixer's clipped motor MEAN when
        # the mixer is on (the parasitic-lift / thrust-sag channel). ---
        coll = self._thrust if c_eff is None else c_eff
        if cfg.coll_map_thr is not None or cfg.coll_map_accel is not None:
            if cfg.coll_map_thr is None or cfg.coll_map_accel is None:
                raise ValueError("coll_map_thr and coll_map_accel must be set together")
            # measured CONVEX collective map (twin-falsify 2026-06-11) on the REALISED collective
            a_up = float(np.interp(coll, np.asarray(cfg.coll_map_thr, dtype=np.float64),
                                   np.asarray(cfg.coll_map_accel, dtype=np.float64)))
        else:
            a_up = cfg.g * (coll / cfg.hover_thrust)               # thrust=hover -> g (balances)
        R_m = R_new.as_matrix()
        f_world = R_m @ np.array([0.0, 0.0, -a_up])                # body -Z (up) in world NED
        f_world = f_world - cfg.linear_drag * self.vel             # specific force incl. drag
        if cfg.quad_drag_c2 is not None:
            # measured body-frame direction-dependent QUADRATIC drag (uses the OLD velocity,
            # like the linear term): per-axis coefficient picked by the sign of v_body
            c2 = _quad_c2_table(cfg.quad_drag_c2)
            v_b = R_m.T @ self.vel
            c = np.where(v_b >= 0.0, c2[:, 0], c2[:, 1])
            f_world = f_world + R_m @ (-(c * np.abs(v_b) * v_b))
        accel = f_world + np.array([0.0, 0.0, cfg.g])              # + gravity (NED +Z down)
        # semi-implicit Euler (update velocity first, then position with the new velocity)
        self.vel = self.vel + accel * dt
        self.pos = self.pos + self.vel * dt
        self.accel_body = R_m.T @ f_world                          # what an accelerometer reads
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
