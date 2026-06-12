"""rl_plant -- framework-agnostic, telemetry-free CTBR quadrotor dynamics for RL training.

This is the PHYSICS of :class:`racer.twin.CtbrPlant` (the system-ID'd offline twin) re-expressed
as a clean, pure-:mod:`numpy`, batch-friendly, pure-function ``step``. It is the dynamics we inject
into a massively-parallel RL substrate (DiffAero/Crazyflow): same plant the model-based stack was
tuned against, so a policy trained here transfers. Parity-tested cell-for-cell against ``twin.py``
in ``tests/test_rl_plant_parity.py`` -- **twin.py is ground truth**; if they disagree, fix this file.

Why a separate module (not reuse twin.py):
  * **Telemetry-free.** twin.py's ``state()`` re-emits the sim's ODOMETRY *reporting* inversions
    (``odo_att_report_sign`` / ``odo_rate_report_sign``) so the live controller's compensating signs
    transfer. Those are MEASUREMENT artifacts, NOT physics -- an RL plant must never see them. Here
    there is no ``state()``, no DroneState, no report signs: only the true physical state evolves.
  * **Batch + portable.** twin.py builds a per-call ``scipy.spatial.transform.Rotation`` (one object,
    one env). This module operates on ``(..., 3)`` / ``(..., 4)`` arrays with broadcasting and its own
    quaternion helpers -- no scipy, no ``racer.frames`` import -- so the same code vectorises over
    ``n_envs`` and ports to torch by a near-mechanical ``np.`` -> ``torch.`` substitution (op names:
    ``concatenate``->``cat``, ``np.cross``->``torch.linalg.cross``, etc.). Keep all batch dims in the
    leading ``...``; never assume a fixed batch rank.

DISCRETE UPDATE (semi-implicit Euler, fixed ``dt``), per :func:`step` -- identical to
``CtbrPlant.step`` with the latency params defaulted off:

    # 1. inner rate loop -- first-order lag toward the sim's realised steady rate. The steady gain
    #    is FLAT rate_gain (legacy, super_rate_s=None) or the measured STATIC amplitude-dependent
    #    "super-rate" map (characterize-sweep 2026-06-10): gain grows with command amplitude.
    gain    = rate_gain / (1 - super_rate_s * min(|cmd_rate|, pi) / pi)   # per axis; flat if s=None
    target  = gain * rate_sign * cmd_rate                           # (sim amplifies ~2.5x at small cmd,
                                                                    #  ~3.5x at full stick; inverts yaw cmd)
    domega  = (1 - exp(-dt / rate_tau_s)) * (target - omega)
    domega  <- clip(domega, +-alpha_max_rps2 * dt)                  # per-axis slew limit (None -> off)
    omega   <- clip_to_norm(omega + domega, max_omega_rps)          # sanity clamp on |omega|
    # 2. attitude -- integrate the BODY rate (right-multiply by the body-frame increment)
    q       <- normalize( q (x) exp(omega * dt) )                   # R_world_body, Hamilton product
    # 3. realised collective -- optional first-order actuator lag (thrust_tau_s = 0 -> instant)
    thrust  <- thrust + (1 - exp(-dt / thrust_tau_s)) * (cmd_thrust - thrust)
    # 4. translation -- thrust along body -Z (up), drag, gravity. The thrust map is the linear
    #    g*thr/hover (legacy) or the measured CONVEX knot table when coll_map_* is set; drag is
    #    world-frame linear (legacy) plus the measured body-frame direction-dependent QUADRATIC
    #    term when quad_drag_c2 is set (twin-falsify 2026-06-11; all three default OFF).
    a_up    = interp(thrust, coll_map_thr, coll_map_accel)          # or g * thrust / hover_thrust
    f_world = R(q) @ [0, 0, -a_up] - linear_drag * vel             # specific force incl. drag (OLD vel)
    v_b     = R(q).T @ vel                                          # quad drag: body-frame, per-axis
    f_world += R(q) @ (-c2[sign(v_b)] * |v_b| * v_b)                #   coefficient picked by sign(v_b)
    accel   = f_world + [0, 0, g]                                   # + gravity (NED +Z down)
    vel     <- vel + accel * dt                                     # update velocity first ...
    pos     <- pos + vel  * dt                                      # ... then position with the NEW velocity

STATE -- :class:`PlantState`, all pure physics (NO accel_body, NO telemetry):
    pos     (..., 3)  world NED position (m)        [X north, Y east, Z DOWN]
    vel     (..., 3)  world NED velocity (m/s)
    quat    (..., 4)  attitude, wxyz scalar-first, R_world_body (body FRD -> world NED)
    omega   (..., 3)  body-frame angular rate (rad/s, FRD)
    thrust  (...,)    realised normalised collective [0, 1] (carried for the thrust_tau_s lag)
    act_buf (..., K, 4) | None  transport-delay ring buffer; None unless transport_delay_steps > 0

ACTION -- ``(..., 4)`` CTBR command ``[wx, wy, wz, collective]``: body-rate setpoint (rad/s, FRD) +
    normalised collective [0, 1]. Same semantics as ``ControlCommand(BODY_RATE)`` (body_rate, thrust).

PARAMS -- :class:`PlantParams`. Defaults ARE the validated sim-faithful PHYSICS
    (``twin.faithful_config`` with the telemetry report-signs dropped): hover 0.2656, rate_tau 0.019 s,
    rate_gain [2.501, 2.504, 2.231], rate_sign [+1, +1, -1] (TRAINED-WORLD convention -- the live
    sim's TRUE command sign is [+1,+1,+1], FRAME-AUDIT 2026-06-12; see the field comment),
    linear_drag 0.2111 /s, g 9.80665. The MEASURED super-rate map/slew
    (``super_rate_s=SUPER_RATE_S_MEASURED``, ``alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED``) default to
    None/OFF for exact backward compatibility -- turn them ON for sim-faithful saturated authority
    (the flat 2.5 gain under-predicts full-stick authority by up to 42%). The MEASURED aero
    (``quad_drag_c2=QUAD_DRAG_C2_MEASURED`` + ``coll_map_thr/accel=COLL_MAP_*_MEASURED`` with
    ``linear_drag=0.0``; twin-falsify 2026-06-11) likewise defaults to None/OFF -- turn it ON for
    sim-faithful braking (legacy under-brakes 2.2x at 9 m/s) and climb authority (legacy
    under-predicts full-stick thrust 2.1x).

SIGN CONVENTIONS (PHYSICS ONLY):
    world NED (Z down, g = +9.80665 on +Z), body FRD (X fwd, Y right, Z down), thrust acts along body
    -Z (up). Quaternion wxyz, R_world_body, Hamilton convention (matches scipy ``Rotation`` and
    ``racer.frames``). A +roll-rate rolls right-wing-down and accelerates +Y; a -pitch-rate pitches
    nose-down and accelerates +X; thrust == hover nets zero vertical accel.
    *** NEVER add ``odo_att_sign`` / ``odo_rate_sign`` / ``body_rate_sign`` here -- those undo the sim's
    TELEMETRY reporting artifacts in the live controller; the dynamics are telemetry-free by design. ***

LATENCY MODEL (both default OFF -- the validated config has them off):
    * transport delay: integer ``transport_delay_steps`` (= round(cmd_latency_s / dt)); the command
      applied is the one issued that many steps ago, via the ``act_buf`` ring buffer carried in state.
    * actuator lag: ``thrust_tau_s`` first-order lag on the realised collective (via ``thrust`` state).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "PlantParams",
    "PlantState",
    "step",
    "rollout",
    "hover_action",
    "quat_multiply",
    "rotvec_to_quat",
    "quat_rotate",
    "quat_rotate_inverse",
    "quat_conjugate",
    "quat_normalize",
    "SUPER_RATE_S_MEASURED",
    "ALPHA_MAX_RPS2_MEASURED",
    "QUAD_DRAG_C2_MEASURED",
    "QUAD_DRAG_C2_POOLED",
    "COLL_MAP_THR_MEASURED",
    "COLL_MAP_ACCEL_MEASURED",
    "LAPSE_SPEED_MEASURED",
    "LAPSE_FACTOR_MEASURED",
    "MIXER_IDLE_MEASURED",
    "MIXER_KAPPA_ERR_MEASURED",
    "MIXER_KAPPA_HOLD_MEASURED",
    "MIXER_ZETA_YAW_MEASURED",
    "mixer_r_fit",
]

_G = 9.80665
_BODY_UP = np.array([0.0, 0.0, -1.0])   # thrust direction in body FRD (up = -Z)

# Measured super-rate map nominals (characterize-sweep 2026-06-10, WRITEUP Section 3; 15-run
# controlled magnitude sweep on the live sim). s is good to ~+-0.02; slew measured 250-280 rad/s^2
# roll/pitch, ~78 yaw (level). These are the DR centers and the values faithful map-ON configs use.
SUPER_RATE_S_MEASURED = 0.30
ALPHA_MAX_RPS2_MEASURED = np.array([260.0, 260.0, 80.0])

# Measured AERO nominals (twin-falsify campaign 2026-06-10/11, handoff/shadowpc-twin-falsify-
# 2026-06-10/WRITEUP.md Sections 2-3 + 7; 40+ recordings, predictions committed pre-flight).
# Body-frame quadratic drag coefficients (1/m): rows = body FRD axis, column 0 applies where
# v_body[axis] >= 0, column 1 where < 0. x: 0.042 nose-first / 0.058 tail-first (WRITEUP per-
# family coast fits); y: 0.055 symmetric; z: 0.0539 descend (+z) / 0.0756 climb (-z) (joint
# vertical fit, vert_fit_coef.npy full precision -- the WRITEUP rounds these to 0.054/0.076).
QUAD_DRAG_C2_MEASURED = np.array([[0.042, 0.058],
                                  [0.055, 0.055],
                                  [0.0539309301924107, 0.0756168595765378]])
QUAD_DRAG_C2_POOLED = 0.052          # isotropic pooled coast fit -- the WRITEUP Section 7 DR-band center
# Convex collective->accel knot table (WRITEUP Section 3 joint fit, 4812 samples; the hover knot
# recovers g to 2.3% -- kept as measured, NOT snapped to g). Knots 0.0/0.10 are not fit-grade
# (drag colinearity); per Section 7 they are replaced by the linear extrapolation of the
# 0.15-0.20 segment, FLOORED AT 0: the raw extrapolation (-5.60 / -0.45 m/s^2) is worse-than-
# free-fall, contradicting the measured c000 near-free-fall (Section 8). The floor encodes the
# motor idle deadband instead. np.interp clamps beyond 1.0 (the [0,1] stick saturates there).
COLL_MAP_THR_MEASURED = np.array([0.0, 0.10, 0.15, 0.20, 0.2656, 0.32, 0.40, 0.45,
                                  0.55, 0.60, 0.80, 1.0])
COLL_MAP_ACCEL_MEASURED = np.array([0.0, 0.0,
                                    2.1279523370741864, 4.7051594638528362,
                                    9.5804078429104393, 13.576760297067407,
                                    21.708896781382972, 26.488309151664271,
                                    38.748577917265877, 42.360958058019655,
                                    58.431876299624356, 78.282838504684648])

# 🚩 VOIDED BY FRAME-AUDIT 2026-06-12 (handoff/laptop-frame-audit-2026-06-12): the S18 "thrust
# lapse vs airspeed" was an ARTIFACT of the mirrored attitude reading. The S18 fit projected the
# measured specific force onto b3 from the AS-IS ODOMETRY quat, whose East component is sign-
# flipped (R_y(pi) telemetry conjugation); a ~20-deg lateral bank component during the early
# standing-start climb yields exactly the apparent 0.74-0.88 deficit, "recovering" with speed
# because the trajectory straightens. Re-fit with the TRUE attitude on the same 17 runs + smooth
# ticks: ratio 1.00 at 3-6 m/s, 1.04-1.10 above (no lapse structure; the >1 tail is drag/map
# attribution, <=10%). DO NOT enable these knots in any plant config or train with dr_lapse on
# this curve. The mechanism below is kept (harmless, defaults OFF, parameter-generic tests);
# the MEASURED constants are retained only so historical analyses reproduce.
LAPSE_SPEED_MEASURED = np.array([0.0, 4.0, 8.0, 12.0, 15.0])
LAPSE_FACTOR_MEASURED = np.array([1.0, 0.78, 0.80, 0.92, 1.0])

# Measured MOTOR-MIXER coupling nominals (live-deploy diag 2026-06-11, handoff/shadowpc-live-
# deploy-diag-2026-06-11/WRITEUP.md Section 2; fit in handoff/laptop-s17-mixer-inc6-2026-06-11/
# fit_mixer.py from the mixer_probe recording aggregates). kappa_err fits 0.0730/0.0730 on two
# independent probes (yaw rail + roll/pitch rail); kappa_hold is the single settled-spin point
# (roll/pitch unmeasured -- symmetric + DR); zeta_yaw from the hover-collective mean-preservation
# point (the max-motor reading tensions it ~11% -- wide DR band); idle from the all-zero probe.
MIXER_IDLE_MEASURED = 0.05
MIXER_KAPPA_ERR_MEASURED = 0.073
MIXER_KAPPA_HOLD_MEASURED = 0.046
MIXER_ZETA_YAW_MEASURED = 0.34


# --------------------------------------------------------------------------- quaternion helpers
# wxyz (scalar-first) unit quaternions, R_world_body, Hamilton convention -- matches scipy
# ``Rotation`` (asserted in the parity test) and ``racer.frames``. All operate on (..., 4) / (..., 3)
# arrays with broadcasting; no scipy, no per-env objects, torch-portable.
def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a (x) b`` of wxyz quaternions; rotation composition ``R(a) @ R(b)``.
    Broadcasts over leading batch dims. Shapes ``(..., 4) x (..., 4) -> (..., 4)``."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector (axis * angle, rad) -> wxyz unit quaternion ``exp(rotvec)``.
    ``q = (cos(theta/2), sin(theta/2) * axis)`` with ``theta = |rotvec|``. Numerically robust at
    ``theta -> 0`` via ``sinc`` (no divide-by-|rotvec|). Shapes ``(..., 3) -> (..., 4)``."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)        # (..., 1)
    half = 0.5 * theta
    w = np.cos(half)                                             # (..., 1)
    # xyz = sin(half) * axis = sin(half)/theta * rotvec = 0.5 * sinc(half/pi) * rotvec  (sinc(0)=1)
    scale = 0.5 * np.sinc(half / np.pi)                          # (..., 1) == sin(half)/theta
    xyz = scale * rotvec                                          # (..., 3)
    return np.concatenate([w, xyz], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse, for a unit quaternion) of a wxyz quaternion. Shapes ``(..., 4) -> (..., 4)``."""
    return np.concatenate([q[..., 0:1], -q[..., 1:4]], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate body vector ``v`` into world: ``R(q) @ v`` with ``R = R_world_body``.
    ``v' = v + 2 w (u x v) + 2 u x (u x v)``, ``u = xyz``. Broadcasts; ``v`` may be ``(3,)`` or
    ``(..., 3)``. Shapes ``(..., 4), (..., 3) -> (..., 3)``."""
    v = np.asarray(v, dtype=np.float64)
    w = q[..., 0:1]                                              # (..., 1)
    u = q[..., 1:4]                                              # (..., 3)
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector ``v`` into body: ``R(q).T @ v``. Shapes ``(..., 4), (..., 3) -> (..., 3)``."""
    return quat_rotate(quat_conjugate(q), v)


def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return ``q`` scaled to unit norm along the last axis. Shapes ``(..., 4) -> (..., 4)``."""
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, eps)


def _clip_to_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale ``v`` down (per batch element) so ``|v| <= max_norm``; no-op where already inside or
    ``max_norm <= 0``. Batched form of ``twin._clip_norm``. Shapes ``(..., 3) -> (..., 3)``."""
    if max_norm <= 0.0:
        return v
    n = np.linalg.norm(v, axis=-1, keepdims=True)               # (..., 1)
    scale = np.where(n > max_norm, max_norm / np.maximum(n, 1e-12), 1.0)
    return v * scale


def mixer_r_fit(rate_gain, rate_sign, super_rate_s, hover, idle, kappa_err, zeta_yaw) -> np.ndarray:
    """Mixer authority normalisation (3,) at the S14 slew-fit condition (hover collective,
    single-axis sustained pi command, zero rate): the realized/demanded differential ratio
    there. ``alpha_max_rps2`` was measured WITH the mixer already throttling at that point, so
    the in-step slew limit scales by Q = r/r_fit (fit point exact) rather than by the raw r.
    Operation-for-operation duplicate of ``racer.twin._mixer_r_fit`` (parity-pinned; twin is
    ground truth)."""
    gain = np.asarray(rate_gain, dtype=np.float64)
    if super_rate_s is not None:
        s = np.asarray(super_rate_s, dtype=np.float64)
        gain = gain / (1.0 - s * np.minimum(np.pi, np.pi) / np.pi)
    target = gain * np.asarray(rate_sign, dtype=np.float64) * np.pi
    d = kappa_err * target
    eta = zeta_yaw / (zeta_yaw + np.maximum(hover, 0.0))
    d = np.concatenate([d[..., 0:2], d[..., 2:3] * eta])
    u_hi = np.clip(hover + np.abs(d), idle, 1.0)
    u_lo = np.clip(hover - np.abs(d), idle, 1.0)
    delta = (u_hi - u_lo) * 0.5
    return delta / np.abs(d)


def _interp1d(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Batched piecewise-linear interpolation with end-knot clamping -- ``np.interp`` semantics
    (bit-identical for finite inputs: same segment selection, same ``y0 + slope*(x - x0)``
    arithmetic, exact ``fp`` values at the knots and beyond the ends). Written out explicitly so
    the torch backend can mirror it operation-for-operation (``np.interp`` has no torch analog).
    Shapes ``(...,), (K,), (K,) -> (...,)``."""
    idx = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, xp.shape[0] - 2)
    x0 = xp[idx]
    y0 = fp[idx]
    slope = (fp[idx + 1] - y0) / (xp[idx + 1] - x0)
    y = y0 + slope * (x - x0)
    y = np.where(x <= xp[0], fp[0], y)
    y = np.where(x >= xp[-1], fp[-1], y)
    return y


# --------------------------------------------------------------------------- params
@dataclass
class PlantParams:
    """PHYSICS parameters (telemetry-free). Defaults = the validated sim-faithful plant
    (``twin.faithful_config`` minus the telemetry report-signs)."""

    hover_thrust: float = 0.2656            # collective at zero net vertical accel
    g: float = _G
    rate_tau_s: float = 0.0190             # inner rate-loop first-order time constant (s)
    # realised body rate is a first-order lag toward ``gain * rate_sign * cmd_rate`` where gain is
    # flat rate_gain (legacy) or the super-rate map when super_rate_s is set (see module docstring)
    rate_gain: np.ndarray = field(default_factory=lambda: np.array([2.501, 2.504, 2.231]))
    # TRAINED-WORLD convention, NOT live physics: the live sim's TRUE command->rate sign is
    # [+1,+1,+1] (FRAME-AUDIT 2026-06-12; the historical "yaw inverted" was the telemetry
    # conjugation read back as physics). Every shipped checkpoint trained against [+1,+1,-1]
    # through the [1,-1,-1] FLU->FRD adapter; fly_rl's wire map [+1,-1,+1] preserves that
    # composition exactly. DO NOT change this default -- it anchors the trained semantics
    # (and twin parity); deploy/eval emulation carries the live sign separately.
    rate_sign: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, -1.0]))
    # STATIC amplitude-dependent gain map strength s (scalar or (3,)); None -> flat legacy gain.
    # Measured SUPER_RATE_S_MEASURED=0.30 roll/pitch; yaw same form (the level-attitude ~7.4 rad/s
    # yaw plateau is a KNOWN UNMODELED caveat -- racing yaw cmds are small; own pass if it matters).
    super_rate_s: float | np.ndarray | None = None
    # per-axis slew limit on the realised rate (rad/s^2; scalar or (3,)); None -> unlimited.
    # Measured ALPHA_MAX_RPS2_MEASURED=[260, 260, 80].
    alpha_max_rps2: float | np.ndarray | None = None
    linear_drag: float = 0.2111            # world-frame linear drag (1/s)
    # Measured body-frame direction-dependent QUADRATIC drag (twin-falsify 2026-06-11): scalar |
    # (3,) | (3,2) per-axis-per-sign table, normalised to (3,2) in __post_init__ ([:, 0] where
    # v_body >= 0, [:, 1] where < 0). ADDS to linear_drag (the measured config zeroes linear_drag).
    # None -> OFF (exact legacy). Nominal = QUAD_DRAG_C2_MEASURED.
    quad_drag_c2: float | np.ndarray | None = None
    # Measured CONVEX collective->accel knot table (same campaign): when both are set,
    # a_up = interp(thrust, coll_map_thr, coll_map_accel) REPLACES g*thrust/hover_thrust
    # (clamped to the end knots outside the range). Set together or not at all (validated).
    # None -> OFF (exact legacy). Nominals = COLL_MAP_THR_MEASURED / COLL_MAP_ACCEL_MEASURED.
    coll_map_thr: np.ndarray | None = None
    coll_map_accel: np.ndarray | None = None
    # Measured THRUST LAPSE vs airspeed (S18 2026-06-12): a_up *= interp(|vel|, lapse_speed,
    # lapse_factor). The collective map is fit at ~0 airspeed and over-predicts thrust ~22% in the
    # 4-12 m/s band; this multiplicative factor (a function of OLD world speed |vel|, like drag)
    # corrects it. Set together (validated: matching 1-D, K >= 2, strictly-increasing speed knots).
    # None -> OFF (exact legacy). Nominals = LAPSE_SPEED_MEASURED / LAPSE_FACTOR_MEASURED.
    lapse_speed: np.ndarray | None = None
    lapse_factor: np.ndarray | None = None
    # Measured MOTOR-MIXER coupling (live-deploy diag 2026-06-11): per-motor commands =
    # collective +- the rate-loop differential demand d_ax = kappa_err*(target - omega) +
    # kappa_hold*omega (yaw scaled by zeta/(zeta + collective)), clipped to [idle, 1] -- the
    # clipped MEAN re-enters the thrust map (parasitic lift at the bottom rail, sag at the top),
    # the clipped DIFFERENTIAL scales the slew limit by Q = r/r_fit (see :func:`mixer_r_fit`).
    # All four set together or none; requires alpha_max_rps2. None -> OFF (exact legacy:
    # thrust and rates independent -- the corner the live transfer failures came from).
    # Nominals = MIXER_*_MEASURED.
    mixer_idle: float | None = None
    mixer_kappa_err: float | None = None
    mixer_kappa_hold: float | None = None
    mixer_zeta_yaw: float | None = None
    thrust_tau_s: float = 0.0              # actuator (collective) first-order lag (s); 0 -> instant
    transport_delay_steps: int = 0         # command transport delay in integer steps; 0 -> OFF
    # Sanity NORM clamp on |omega|. With the map ON the DC ceiling is g(pi)*pi ~= 11.2 rad/s per
    # axis (~18.5 worst-case 3-axis norm) -- keep >= ~11.5 or the flat-gain ceiling artifact returns.
    max_omega_rps: float = 25.0

    def __post_init__(self) -> None:
        self.rate_gain = np.asarray(self.rate_gain, dtype=np.float64)
        self.rate_sign = np.asarray(self.rate_sign, dtype=np.float64)
        if self.super_rate_s is not None:
            self.super_rate_s = np.asarray(self.super_rate_s, dtype=np.float64)
        if self.alpha_max_rps2 is not None:
            self.alpha_max_rps2 = np.asarray(self.alpha_max_rps2, dtype=np.float64)
        if self.quad_drag_c2 is not None:
            t = np.asarray(self.quad_drag_c2, dtype=np.float64)
            if t.ndim == 0:
                t = np.full((3, 2), float(t))
            elif t.shape == (3,):
                t = np.stack([t, t], axis=-1)
            elif t.shape != (3, 2):
                raise ValueError(f"quad_drag_c2 must be a scalar, (3,) or (3, 2); got shape {t.shape}")
            self.quad_drag_c2 = t
        if (self.coll_map_thr is None) != (self.coll_map_accel is None):
            raise ValueError("coll_map_thr and coll_map_accel must be set together")
        if self.coll_map_thr is not None:
            thr = np.asarray(self.coll_map_thr, dtype=np.float64)
            acc = np.asarray(self.coll_map_accel, dtype=np.float64)
            if thr.ndim != 1 or thr.shape != acc.shape or thr.shape[0] < 2:
                raise ValueError("coll_map_thr/coll_map_accel must be matching 1-D arrays, K >= 2")
            if not np.all(np.diff(thr) > 0.0):
                raise ValueError("coll_map_thr knots must be strictly increasing")
            self.coll_map_thr = thr
            self.coll_map_accel = acc
        if (self.lapse_speed is None) != (self.lapse_factor is None):
            raise ValueError("lapse_speed and lapse_factor must be set together")
        if self.lapse_speed is not None:
            ls = np.asarray(self.lapse_speed, dtype=np.float64)
            lf = np.asarray(self.lapse_factor, dtype=np.float64)
            if ls.ndim != 1 or ls.shape != lf.shape or ls.shape[0] < 2:
                raise ValueError("lapse_speed/lapse_factor must be matching 1-D arrays, K >= 2")
            if not np.all(np.diff(ls) > 0.0):
                raise ValueError("lapse_speed knots must be strictly increasing")
            self.lapse_speed = ls
            self.lapse_factor = lf
        mix = (self.mixer_idle, self.mixer_kappa_err, self.mixer_kappa_hold, self.mixer_zeta_yaw)
        if any(m is not None for m in mix) and not all(m is not None for m in mix):
            raise ValueError("mixer_idle/mixer_kappa_err/mixer_kappa_hold/mixer_zeta_yaw "
                             "must be set together")
        self._mixer_r_fit = None
        if self.mixer_idle is not None:
            if self.alpha_max_rps2 is None:
                raise ValueError("the mixer authority model scales the measured slew limits: "
                                 "set alpha_max_rps2 when the mixer is on")
            if not (0.0 <= self.mixer_idle < 1.0 and self.mixer_kappa_err > 0.0
                    and self.mixer_kappa_hold >= 0.0 and self.mixer_zeta_yaw > 0.0):
                raise ValueError("mixer params out of range: need 0 <= idle < 1, kappa_err > 0, "
                                 "kappa_hold >= 0, zeta_yaw > 0")
            self._mixer_r_fit = mixer_r_fit(
                self.rate_gain, self.rate_sign, self.super_rate_s, self.hover_thrust,
                self.mixer_idle, self.mixer_kappa_err, self.mixer_zeta_yaw)


# --------------------------------------------------------------------------- state
@dataclass(frozen=True)
class PlantState:
    """Pure physical state. Arrays carry a leading batch shape ``...`` (``()`` for a single env)."""

    pos: np.ndarray            # (..., 3) world NED
    vel: np.ndarray            # (..., 3) world NED
    quat: np.ndarray           # (..., 4) wxyz, R_world_body
    omega: np.ndarray          # (..., 3) body FRD
    thrust: np.ndarray         # (...,)   realised collective
    act_buf: np.ndarray | None = None   # (..., K, 4) transport-delay ring buffer; None if K == 0

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.pos.shape[:-1]

    @classmethod
    def hover(cls, batch_shape: tuple[int, ...] = (), params: PlantParams | None = None) -> "PlantState":
        """A level hover state (identity attitude, zero rate/velocity, ``thrust = hover_thrust``).
        Seeds the transport-delay buffer with the hover command when ``transport_delay_steps > 0``."""
        params = params or PlantParams()
        bs = tuple(batch_shape)
        quat = np.zeros(bs + (4,))
        quat[..., 0] = 1.0
        thrust = np.full(bs, params.hover_thrust, dtype=np.float64)
        act_buf = None
        k = int(params.transport_delay_steps)
        if k > 0:
            hov = hover_action(params, bs)                       # (..., 4)
            act_buf = np.broadcast_to(hov[..., None, :], bs + (k, 4)).copy()
        return cls(
            pos=np.zeros(bs + (3,)),
            vel=np.zeros(bs + (3,)),
            quat=quat,
            omega=np.zeros(bs + (3,)),
            thrust=thrust,
            act_buf=act_buf,
        )


def hover_action(params: PlantParams | None = None, batch_shape: tuple[int, ...] = ()) -> np.ndarray:
    """The CTBR command that holds a level hover: zero body rate + ``hover_thrust``. Shape ``(..., 4)``."""
    params = params or PlantParams()
    a = np.zeros(tuple(batch_shape) + (4,), dtype=np.float64)
    a[..., 3] = params.hover_thrust
    return a


# --------------------------------------------------------------------------- step
def step(state: PlantState, action: np.ndarray, dt: float, params: PlantParams) -> PlantState:
    """Advance the plant one step of ``dt`` seconds under one CTBR ``action`` ``[wx, wy, wz, collective]``.

    Pure function: returns a new :class:`PlantState`, does not mutate ``state``. ``action`` broadcasts
    against ``state``'s batch shape. ``dt`` is a scalar. Mirrors :meth:`racer.twin.CtbrPlant.step`
    (PHYSICS only). ``dt <= 0`` returns ``state`` unchanged."""
    if dt <= 0.0:
        return state
    action = np.asarray(action, dtype=np.float64)

    # --- transport delay: apply the command from transport_delay_steps ago (ring buffer in state) ---
    k = int(params.transport_delay_steps)
    if k > 0:
        buf = state.act_buf
        if buf is None:                                          # cold buffer -> seed with this action
            buf = np.broadcast_to(action[..., None, :], action.shape[:-1] + (k, 4)).copy()
        applied = buf[..., 0, :]                                 # (..., 4) oldest queued command
        new_buf = np.concatenate([buf[..., 1:, :], action[..., None, :]], axis=-2)
    else:
        applied = action
        new_buf = None
    cmd_rate = applied[..., :3]                                  # (..., 3) rad/s
    cmd_thrust = applied[..., 3]                                 # (...,)

    # --- 1. realised collective FIRST (moved ahead of the rate loop for the mixer, which needs
    # it; independent of the rate/attitude blocks, so legacy floats are unchanged): optional
    # first-order actuator lag (thrust_tau_s = 0 -> instant) ---
    if params.thrust_tau_s > 0.0:
        beta = 1.0 - np.exp(-dt / params.thrust_tau_s)
        thrust = state.thrust + beta * (cmd_thrust - state.thrust)
    else:
        thrust = np.broadcast_to(cmd_thrust, state.thrust.shape).astype(np.float64, copy=True)

    # --- 2. inner rate loop: first-order lag toward the sim's realised steady rate (flat gain or
    # the super-rate map), optional per-axis slew limit, then norm-clamp. Mirrors twin.py
    # operation-for-operation (bit-identical floats); all new params None -> exact legacy update. ---
    gain = params.rate_gain
    if params.super_rate_s is not None:
        gain = gain / (1.0 - params.super_rate_s * np.minimum(np.abs(cmd_rate), np.pi) / np.pi)
    target = gain * params.rate_sign * cmd_rate                  # (..., 3)
    alpha = 1.0 - np.exp(-dt / max(params.rate_tau_s, 1e-9))
    domega = alpha * (target - state.omega)
    c_eff = None                                                 # mixer-OFF: thrust map sees thrust
    if params.mixer_idle is not None:
        # --- MOTOR MIXER (live-deploy diag 2026-06-11; see PlantParams + twin.py). Mirrors
        # twin operation-for-operation: clipped motor MEAN -> the thrust map (parasitic lift /
        # sag), clipped DIFFERENTIAL -> Q = r/r_fit scaling of the slew limit. ---
        e = target - state.omega
        d = params.mixer_kappa_err * e + params.mixer_kappa_hold * state.omega
        eta = params.mixer_zeta_yaw / (params.mixer_zeta_yaw + np.maximum(thrust, 0.0))
        d = np.concatenate([d[..., 0:2], d[..., 2:3] * eta[..., None]], axis=-1)
        u0 = np.clip(thrust + (d[..., 0] + d[..., 1] + d[..., 2]), params.mixer_idle, 1.0)
        u1 = np.clip(thrust + (-d[..., 0] + d[..., 1] - d[..., 2]), params.mixer_idle, 1.0)
        u2 = np.clip(thrust + (d[..., 0] - d[..., 1] - d[..., 2]), params.mixer_idle, 1.0)
        u3 = np.clip(thrust + (-d[..., 0] - d[..., 1] + d[..., 2]), params.mixer_idle, 1.0)
        c_eff = (u0 + u1 + u2 + u3) * 0.25
        delta = np.stack([(u0 - u1 + u2 - u3) * 0.25,
                          (u0 + u1 - u2 - u3) * 0.25,
                          (u0 - u1 - u2 + u3) * 0.25], axis=-1)
        d_safe = np.where(np.abs(d) > 1e-9, d, 1.0)
        r = np.where(np.abs(d) > 1e-9, np.clip(delta / d_safe, 0.0, 1.0), 1.0)
        lim = params.alpha_max_rps2 * (r / params._mixer_r_fit) * dt
        domega = np.clip(domega, -lim, lim)
    elif params.alpha_max_rps2 is not None:
        lim = params.alpha_max_rps2 * dt
        domega = np.clip(domega, -lim, lim)
    omega = _clip_to_norm(state.omega + domega, params.max_omega_rps)

    # --- 3. attitude: integrate the BODY-frame rate (right-multiply by the body-frame increment) ---
    quat = quat_normalize(quat_multiply(state.quat, rotvec_to_quat(omega * dt)))

    # --- 4. translation: thrust along body -Z (up) -> world, drag, gravity (NED +Z down).
    # Thrust map: linear g*thr/hover (legacy) or the measured convex knot table, consuming the
    # realised collective or the mixer's clipped motor MEAN; drag: world linear (legacy) + the
    # measured body-frame sign-split quadratic term (twin-falsify 2026-06-11). All aero/mixer
    # params None -> bit-identical legacy floats. ---
    coll = thrust if c_eff is None else c_eff
    if params.coll_map_thr is not None:
        a_up = _interp1d(coll, params.coll_map_thr, params.coll_map_accel)     # (...,)
    else:
        a_up = params.g * coll / params.hover_thrust             # (...,)
    if params.lapse_speed is not None:                           # S18 airspeed thrust lapse
        speed = np.sqrt(np.sum(state.vel * state.vel, axis=-1))  # (...,) OLD world speed |vel|
        a_up = a_up * _interp1d(speed, params.lapse_speed, params.lapse_factor)
    f_world = a_up[..., None] * quat_rotate(quat, _BODY_UP)      # (..., 3) body -Z (up) in world NED
    f_world = f_world - params.linear_drag * state.vel           # specific force incl. drag (OLD vel)
    if params.quad_drag_c2 is not None:
        v_b = quat_rotate_inverse(quat, state.vel)               # (..., 3) OLD velocity, body frame
        c = np.where(v_b >= 0.0, params.quad_drag_c2[..., 0], params.quad_drag_c2[..., 1])
        f_world = f_world + quat_rotate(quat, -(c * np.abs(v_b) * v_b))
    accel = f_world + np.array([0.0, 0.0, params.g])             # + gravity
    vel = state.vel + accel * dt                                 # semi-implicit: velocity first ...
    pos = state.pos + vel * dt                                   # ... then position with the NEW velocity

    return PlantState(pos=pos, vel=vel, quat=quat, omega=omega, thrust=thrust, act_buf=new_buf)


def rollout(state: PlantState, actions: np.ndarray, dt: float, params: PlantParams) -> PlantState:
    """Convenience: apply a sequence of ``actions`` (shape ``(T, ..., 4)``, leading time axis) and
    return the final state. A thin loop over :func:`step` (the RL env owns its own loop in practice)."""
    actions = np.asarray(actions, dtype=np.float64)
    for t in range(actions.shape[0]):
        state = step(state, actions[t], dt, params)
    return state
