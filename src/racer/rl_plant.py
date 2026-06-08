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

    # 1. inner rate loop -- first-order lag toward the sim's realised steady rate
    target  = rate_gain * rate_sign * cmd_rate                      # (sim amplifies ~2.5x; inverts yaw cmd)
    omega   <- omega + (1 - exp(-dt / rate_tau_s)) * (target - omega)
    omega   <- clip_to_norm(omega, max_omega_rps)                   # sanity clamp on |omega|
    # 2. attitude -- integrate the BODY rate (right-multiply by the body-frame increment)
    q       <- normalize( q (x) exp(omega * dt) )                   # R_world_body, Hamilton product
    # 3. realised collective -- optional first-order actuator lag (thrust_tau_s = 0 -> instant)
    thrust  <- thrust + (1 - exp(-dt / thrust_tau_s)) * (cmd_thrust - thrust)
    # 4. translation -- thrust along body -Z (up), world-frame linear drag, gravity
    a_up    = g * thrust / hover_thrust                             # thrust == hover -> a_up == g (balances)
    f_world = R(q) @ [0, 0, -a_up] - linear_drag * vel             # specific force incl. drag (OLD vel)
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
    rate_gain [2.501, 2.504, 2.231], rate_sign [+1, +1, -1] (sim inverts ONLY the yaw command,
    physically), linear_drag 0.2111 /s, g 9.80665.

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
]

_G = 9.80665
_BODY_UP = np.array([0.0, 0.0, -1.0])   # thrust direction in body FRD (up = -Z)


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


# --------------------------------------------------------------------------- params
@dataclass
class PlantParams:
    """PHYSICS parameters (telemetry-free). Defaults = the validated sim-faithful plant
    (``twin.faithful_config`` minus the telemetry report-signs)."""

    hover_thrust: float = 0.2656            # collective at zero net vertical accel
    g: float = _G
    rate_tau_s: float = 0.0190             # inner rate-loop first-order time constant (s)
    # realised body rate is a first-order lag toward ``rate_gain * rate_sign * cmd_rate``
    rate_gain: np.ndarray = field(default_factory=lambda: np.array([2.501, 2.504, 2.231]))
    rate_sign: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, -1.0]))  # PHYSICS: yaw cmd inverted
    linear_drag: float = 0.2111            # world-frame linear drag (1/s)
    thrust_tau_s: float = 0.0              # actuator (collective) first-order lag (s); 0 -> instant
    transport_delay_steps: int = 0         # command transport delay in integer steps; 0 -> OFF
    max_omega_rps: float = 25.0            # sanity clamp on |omega|

    def __post_init__(self) -> None:
        self.rate_gain = np.asarray(self.rate_gain, dtype=np.float64)
        self.rate_sign = np.asarray(self.rate_sign, dtype=np.float64)


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

    # --- 1. inner rate loop: first-order lag toward the sim's realised steady rate, then norm-clamp ---
    target = params.rate_gain * params.rate_sign * cmd_rate      # (..., 3)
    alpha = 1.0 - np.exp(-dt / max(params.rate_tau_s, 1e-9))
    omega = _clip_to_norm(state.omega + alpha * (target - state.omega), params.max_omega_rps)

    # --- 2. attitude: integrate the BODY-frame rate (right-multiply by the body-frame increment) ---
    quat = quat_normalize(quat_multiply(state.quat, rotvec_to_quat(omega * dt)))

    # --- 3. realised collective: optional first-order actuator lag (thrust_tau_s = 0 -> instant) ---
    if params.thrust_tau_s > 0.0:
        beta = 1.0 - np.exp(-dt / params.thrust_tau_s)
        thrust = state.thrust + beta * (cmd_thrust - state.thrust)
    else:
        thrust = np.broadcast_to(cmd_thrust, state.thrust.shape).astype(np.float64, copy=True)

    # --- 4. translation: thrust along body -Z (up) -> world, world-frame drag, gravity (NED +Z down) ---
    a_up = params.g * thrust / params.hover_thrust               # (...,)
    f_world = a_up[..., None] * quat_rotate(quat, _BODY_UP)      # (..., 3) body -Z (up) in world NED
    f_world = f_world - params.linear_drag * state.vel           # specific force incl. drag (OLD vel)
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
