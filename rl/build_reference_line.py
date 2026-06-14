"""build_reference_line.py -- rebuild the inc8 reference line Gamma on the MEASURED
corrected-aero plant (emits rl/reference_line_inc8.json).

WHY THIS EXISTS
---------------
The inc8 R1' reward is arc-length progress along a contact-safe reference line Gamma
(it replaces progress-to-gate-CENTER, which can pay a speed-pushed policy to clip the
frame). The shipped ``rl/reference_line_vq1.json`` is UNUSABLE for that: it was generated
by TOGT-Planner on a LINEAR-drag plant (``linear_drag 0.21``, ``omega_max [11,11,7]``,
``T/W 3.765`` -- no quad-drag, no super-rate amplitude dependence, no mixer). Its
``lap_time_s 4.55`` and its PATH GEOMETRY assume speeds past the real ~39 m/s v^2-drag
wall (drag-infeasible), and it carries a ~170 deg heading inversion. Rewarding a policy
to track an infeasible geometry rewards an unflyable line.

This rebuilds Gamma on the MEASURED plant (``src/racer/rl_plant.py``) so R1' rewards a
flyable, contact-safe path:

  * GEOMETRY: a smooth (natural cubic-spline, min-curvature) line through the 6 VQ1 gate
    CENTRES (canonical: ``fly_rl._GATE_POS_ZUP * _FLIP``, sourced by AST so there is a
    single source of truth -- never the json's pos_ned) plus the spawn and a short
    post-finish run-out. Interpolating through the centres => DEAD-CENTRE crossings
    (in-plane miss ~ 0) => the full contact-true pass band is free.

  * FEASIBILITY (the fix): a corrected-aero forward-backward TOPP (adapted from
    ``handoff/ultracode-planning-togt-s2-2026-06-13/proto_envelope_topp.py``) whose
    speed/accel envelope is set by the MEASURED constants: the CONVEX collective map
    (full-stick specific force 78.28 m/s^2 ~ 8 g, NOT the linear plant's 3.765 g), the
    v^2 QUAD-DRAG wall (~39 m/s), and -- the correction over the proto -- the EXACT
    body-tilt cone ``tilt(f_thrust) <= tilt_max``. The proto bounded only |f_thrust|
    (a thrust BALL centred at gravity); on this DESCENDING course that ball admits
    INVERTED thrust (the proto's own feas block reports required tilt 113-153 deg, i.e.
    the drone thrusting DOWNWARD to out-accelerate gravity on the descent). A quadrotor
    only thrusts body-up, so an honest line must keep ``tilt(f_thrust) <= tilt_max``.
    On level flight this cone reduces EXACTLY to the proto's ``a_lat = g*tan(tilt)``
    cornering cap; on slopes it additionally caps the forward-accel / braking tilt.
    Drag is carried inside the thrust budget (thrust must beat drag tangentially).

  * FINDING (data, not the brief's premise): the ~4.6-4.7 s "honest bound" is the
    FULL-ATTITUDE ('ball'/TOGT) regime, which on this descent needs SUSTAINED INVERTED
    thrust (descent median tilt ~100 deg) and yaw rate ~13 > the ~11 rad/s envelope --
    rate-INFEASIBLE. The honest UPRIGHT, rate/collective-feasible lap is ~7.9-8.5 s
    (measured quad-drag ~7x the linear plant at 30 m/s caps upright descent speed). The
    emitted Gamma is the margin-preserving upright line (cone 75 deg, ~8.45 s). The GEOMETRY
    is identical across all speed profiles, so the R1' arc-length progress reward is
    unaffected by the speed choice. A rate/collective/speed feasibility certificate is
    computed on the EMITTED samples (the actual product). See the handoff REPORT.md.

The body rates / attitude / collective are reconstructed by the standard differential-
flatness inverse (thrust along body-up = f_thrust direction; body-x = travel tangent),
matching ``proto_envelope_topp.twin_sanity`` and ``peregrine_racing`` conventions.

NOTE ON "T/W 3.765": that is the LINEAR plant's full-stick (g*1.0/hover = 9.81/0.2656 =
3.765 g) -- the defect. The MEASURED convex map delivers 78.28 m/s^2 = 7.98 g at full
stick (2.1x the linear plant; see rl_plant docstring). Feasibility here is the MEASURED
ceiling (thrust_norm <= 1.0 <=> |f_thrust| <= 78.28 m/s^2); provenance records the
measured T/W ~ 7.98, not 3.765.

RUN:
  PYTHONPATH=src .venv/Scripts/python.exe rl/build_reference_line.py --sweep
  PYTHONPATH=src .venv/Scripts/python.exe rl/build_reference_line.py --emit
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from racer.rl_plant import (  # noqa: E402
    COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED, QUAD_DRAG_C2_POOLED,
    SUPER_RATE_S_MEASURED, ALPHA_MAX_RPS2_MEASURED,
    MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED,
    MIXER_ZETA_YAW_MEASURED,
)

_G = 9.80665
_WORLD_UP = np.array([0.0, 0.0, -1.0])      # NED: up = -Z
_GRAV = np.array([0.0, 0.0, _G])            # NED: gravity pulls +Z (down)

# ---------------------------------------------------------------------------
# MEASURED corrected-aero envelope (rl_plant.py -- the REAL plant)
# ---------------------------------------------------------------------------
A_UP_MAX = float(COLL_MAP_ACCEL_MEASURED[-1])   # 78.2828 m/s^2 full-stick body-up (~7.98 g)
C2_POOLED = float(QUAD_DRAG_C2_POOLED)          # 0.052 1/m isotropic pooled coast drag
HOVER = 0.2656
TW_MEASURED = A_UP_MAX / _G                      # ~7.98 (NOT the linear plant's 3.765)

# ---------------------------------------------------------------------------
# Spec / validity constants
# ---------------------------------------------------------------------------
HALF_OPEN = 0.75            # gate validity half-opening, in-plane (m)
RADIUS_PRIMARY = 0.38       # inc7-trained worst-case contact halo (m); band = HALF_OPEN - r
GATE_YAW = np.pi            # all VQ1 gates face along-course (yaw = pi)
RUN_OUT_M = 8.0             # straight post-finish run-out past the last gate (m)
DEFAULT_TILT_MAX_DEG = 75.0  # upright body-tilt envelope: last cap with full thrust+rate margin
#   (>=76 deg rides the collective ceiling; the absolute upright feasible edge is ~78 deg / ~7.9 s)
V_FLOOR = 0.5               # standing-start speed floor (m/s) -- time-integration stability only


# ===========================================================================
# 0. CANONICAL gate centres -- AST-sourced from fly_rl (NEVER the json's pos_ned)
# ===========================================================================
def load_canonical_gates_ned(fly_rl_path: Path | None = None) -> np.ndarray:
    """Return the 6 VQ1 gate centres in world NED, sourced from the canonical constants
    ``_GATE_POS_ZUP`` and ``_FLIP`` in ``rl/fly_rl.py`` (NED = _GATE_POS_ZUP * _FLIP).

    Read via ``ast`` from the source text -- NOT imported -- because fly_rl pulls in torch
    + MAVLink at module load. This keeps a SINGLE source of truth (drift-proof) with zero
    heavy deps. Raises if either constant is missing/unparseable (never silently wrong)."""
    fly_rl_path = fly_rl_path or (_HERE / "fly_rl.py")
    tree = ast.parse(Path(fly_rl_path).read_text())
    found: dict[str, np.ndarray] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in ("_GATE_POS_ZUP", "_FLIP"):
                val = node.value
                # both are np.array(<literal list>, dtype=...) -> take the first positional arg
                if isinstance(val, ast.Call) and val.args:
                    found[tgt.id] = np.array(ast.literal_eval(val.args[0]), dtype=np.float64)
    if "_GATE_POS_ZUP" not in found or "_FLIP" not in found:
        raise RuntimeError(f"could not source _GATE_POS_ZUP / _FLIP from {fly_rl_path}")
    zup, flip = found["_GATE_POS_ZUP"], found["_FLIP"]
    if zup.shape != (6, 3) or flip.shape != (3,):
        raise RuntimeError(f"unexpected canonical shapes: ZUP {zup.shape}, FLIP {flip.shape}")
    return zup * flip            # NED = Z-up * [1, -1, -1]


# Spawn (simstart) in NED -- ~ground, level (z up ~ 0). Matches the proto's START_NED.
START_NED = np.array([0.0, 0.0, -0.02])


# ===========================================================================
# 1. GEOMETRIC line: natural cubic spline through [spawn, 6 gate centres] + run-out
# ===========================================================================
def build_line(gates_ned: np.ndarray, n_dense: int = 4000, run_out_m: float = RUN_OUT_M):
    """Min-curvature (natural cubic-spline) line through spawn + the 6 gate centres,
    chord-length parameterised, densely sampled in arc length, with a STRAIGHT run-out
    appended past the last gate (so the RL progress reward is defined past the finish
    without bending the final gate approach).

    Returns dense geometry arrays: r (M,3), t_hat (M,3), n_hat (M,3), kappa (M,),
    s (M,) arc length, plus the per-gate spline knot arc-length ``gate_s`` (6,)."""
    wp = np.vstack([START_NED, gates_ned])
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    u_wp = np.concatenate([[0.0], np.cumsum(seg)])
    cs = CubicSpline(u_wp, wp, bc_type="natural")
    u = np.linspace(0.0, u_wp[-1], n_dense)
    r = cs(u)
    d1 = cs(u, 1)
    ds_du = np.maximum(np.linalg.norm(d1, axis=1), 1e-12)
    t_hat = d1 / ds_du[:, None]
    s = np.concatenate([[0.0], np.cumsum((ds_du[:-1] + ds_du[1:]) / 2 * np.diff(u))])

    # straight run-out: continue along the final tangent for run_out_m
    if run_out_m > 0.0:
        t_end = t_hat[-1]
        n_extra = max(2, int(run_out_m / (s[-1] / n_dense)))
        ds_extra = np.linspace(s[-1] / n_dense, run_out_m, n_extra)
        r_extra = r[-1] + np.outer(ds_extra, t_end)
        r = np.vstack([r, r_extra])
        t_hat = np.vstack([t_hat, np.tile(t_end, (n_extra, 1))])
        s = np.concatenate([s, s[-1] + ds_extra])

    # curvature + principal normal from the dense samples (robust on the appended segment)
    dthat = np.gradient(t_hat, s, axis=0)
    kappa = np.linalg.norm(dthat, axis=1)
    n_hat = np.where(kappa[:, None] > 1e-9, dthat / np.maximum(kappa[:, None], 1e-12), 0.0)

    # per-gate spline knot arc length (u_wp[1:] are the 6 gates) -> nearest dense-s
    gate_s = np.interp(u_wp[1:], u, s[:n_dense])
    return dict(cs=cs, u_wp=u_wp, r=r, t_hat=t_hat, n_hat=n_hat, kappa=kappa, s=s,
                gate_s=gate_s, n_dense=n_dense)


# ===========================================================================
# 2. CORRECTED-AERO TOPP with the EXACT body-tilt cone (the fix over the proto)
# ===========================================================================
def _frenet_gravity(t_hat: np.ndarray, n_hat: np.ndarray):
    """Gravity components along the Frenet axes (tangent / principal-normal / binormal)."""
    b_hat = np.cross(t_hat, n_hat)
    bn = np.linalg.norm(b_hat, axis=1)
    b_hat = np.where(bn[:, None] > 1e-9, b_hat / np.maximum(bn[:, None], 1e-12), 0.0)
    g_t = t_hat @ _GRAV
    g_n = n_hat @ _GRAV
    g_b = b_hat @ _GRAV
    return g_t, g_n, g_b


def topp_corrected_aero(geom: dict, tilt_max_deg: float, model: str = "ball",
                        a_up_max: float = A_UP_MAX, c2: float = C2_POOLED,
                        v_floor: float = V_FLOOR, n_sweeps: int = 8):
    """Time-optimal speed profile on ``geom`` under the MEASURED corrected-aero envelope.

    At every sample the realised THRUST specific force is
        f_thrust(a_tan) = (a_tan + c2 v^2) * t_hat + (kappa v^2) * n_hat - grav
    (thrust must also overcome drag tangentially). ``|f_thrust| <= a_up_max`` (the convex
    collective ceiling) always binds. Two cornering/attitude models:

      "ball"  -- the proto / C++ TOGT corrected-aero model (cross-checked to the 4.714 s
                 refined lap). Cornering speed is capped by the STYLE envelope
                 a_lat = g*tan(tilt_max) AND the thrust-ball corner ceiling; the thrust
                 vector has FULL attitude freedom (standard quadrotor SO(3) model -- it may
                 tilt past 90 deg on the steep descent, same as TOGT). This is the model the
                 ~4.6-4.7 s honest bound comes from. DEFAULT.

      "cone"  -- a STRICTER upright variant: additionally require angle(f_thrust, world_up)
                 <= tilt_max (no inverted thrust). Physically the most conservative; on this
                 17-19 deg descent under the measured 2x quad-drag it caps the descent near
                 the knife-edge (~25-33 m/s) -> ~6-8 s lap. Kept for the honesty comparison.

    The v-ceiling is the max v at which holding the path (a_tan = 0) is feasible; the
    forward-backward sweep then uses the feasible a_tan extremes at each (i, v)."""
    t_hat = geom["t_hat"]
    n_hat = geom["n_hat"]
    kappa = geom["kappa"]
    s = geom["s"]
    N = len(s)
    seglen = np.diff(s)
    g_t, g_n, g_b = _frenet_gravity(t_hat, n_hat)
    tz = t_hat[:, 2]
    nz = n_hat[:, 2]
    k_min = np.cos(np.radians(tilt_max_deg))
    a_lat_max = 1e6 if tilt_max_deg >= 89.999 else _G * np.tan(np.radians(tilt_max_deg))

    # ----- top-speed cap from the v^2 drag wall (straight, a_tan = 0) -----
    a_fwd_straight = np.sqrt(np.maximum(a_up_max ** 2 - (g_n ** 2 + g_b ** 2), 0.0)) + g_t
    v_drag = np.sqrt(np.maximum(a_fwd_straight, 0.0) / c2)

    if model == "ball":
        # proto cornering ceilings: style (a_lat) + thrust-ball corner hold
        v_corner_style = np.where(kappa > 1e-6, np.sqrt(a_lat_max / np.maximum(kappa, 1e-12)), np.inf)
        rad_ball = a_up_max ** 2 - g_t ** 2 - g_b ** 2
        a_c_max_ball = np.where(rad_ball > 0.0, g_n + np.sqrt(np.maximum(rad_ball, 0.0)), 0.0)
        v_corner_ball = np.where(kappa > 1e-6,
                                 np.sqrt(np.maximum(a_c_max_ball, 0.0) / np.maximum(kappa, 1e-12)), np.inf)
        v_ceiling = np.minimum.reduce([v_corner_style, v_corner_ball, v_drag])
    else:  # "cone" -- vectorised bisection on the upright-feasible-hold predicate
        v_lo = np.zeros(N)
        v_hi = np.minimum(v_drag, 60.0).copy()
        for _ in range(48):
            v_mid = 0.5 * (v_lo + v_hi)
            C = kappa * v_mid * v_mid
            A = c2 * v_mid * v_mid
            D = a_up_max ** 2 - (C - g_n) ** 2 - g_b ** 2
            sqrtD = np.sqrt(np.maximum(D, 0.0))
            ball_ok = (D >= 0.0) & (A >= g_t - sqrtD) & (A <= g_t + sqrtD)
            fmag = np.sqrt(np.maximum(A * A + C * C + _G ** 2 - 2 * A * g_t - 2 * C * g_n, 1e-12))
            f_up = _G - A * tz - C * nz
            ok = ball_ok & (f_up > 0.0) & (f_up >= k_min * fmag - 1e-9)
            v_lo = np.where(ok, v_mid, v_lo)
            v_hi = np.where(ok, v_hi, v_mid)
        v_ceiling = v_lo

    def a_tan_extremes(i, v):
        """(a_brake_max>=0, a_fwd_max>=0): max feasible decel magnitude and accel at (i,v)."""
        C = kappa[i] * v * v
        drag = c2 * v * v
        D = a_up_max * a_up_max - (C - g_n[i]) ** 2 - g_b[i] ** 2
        if D <= 0.0:
            return 0.0, 0.0
        a_thrust_tan = np.sqrt(D)
        if model == "ball":
            a_fwd = max(0.0, g_t[i] + a_thrust_tan - drag)        # grav-assist + thrust - drag
            a_brake = max(0.0, a_thrust_tan + drag - g_t[i])      # thrust + drag - grav
            return a_brake, a_fwd
        # "cone": intersect the ball A-range with the upright tilt cone
        A = np.linspace(g_t[i] - a_thrust_tan, g_t[i] + a_thrust_tan, 129)
        f2 = A * A + C * C + _G ** 2 - 2 * A * g_t[i] - 2 * C * g_n[i]
        fmag = np.sqrt(np.maximum(f2, 1e-12))
        f_up = _G - A * tz[i] - C * nz[i]
        ok = (f_up > 0.0) & (f_up >= k_min * fmag - 1e-9)
        if not ok.any():
            return 0.0, 0.0
        Af = A[ok]
        return float(max(0.0, drag - (Af.min()))), float(max(0.0, Af.max() - drag))

    # ----- forward-backward sweeps (iterate to a fixed point) -----
    v = np.minimum(v_ceiling, v_drag)
    v = np.maximum(np.minimum(v, 1e6), v_floor)
    v[0] = min(v[0], v_floor)
    v_finish = float(min(v_ceiling[-1], 1e6))
    for _sweep in range(n_sweeps):
        v[0] = min(v[0], v_floor)
        for i in range(N - 1):                          # forward (accelerate)
            _, a_fwd = a_tan_extremes(i, v[i])
            v[i + 1] = min(v[i + 1], np.sqrt(max(v[i] ** 2 + 2.0 * a_fwd * seglen[i], 0.0)))
        v[-1] = min(v[-1], v_finish)
        for i in range(N - 2, -1, -1):                  # backward (brake)
            a_brake, _ = a_tan_extremes(i + 1, v[i + 1])
            v[i] = min(v[i], np.sqrt(max(v[i + 1] ** 2 + 2.0 * a_brake * seglen[i], 0.0)))

    # ----- realised per-segment tangential accel (clean; no np.gradient ringing) -----
    a_tan_real = np.zeros(N)
    a_tan_real[:-1] = (v[1:] ** 2 - v[:-1] ** 2) / (2.0 * np.maximum(seglen, 1e-9))
    a_tan_real[-1] = a_tan_real[-2]

    # ----- integrate time -----
    t = np.zeros(N)
    for i in range(N - 1):
        t[i + 1] = t[i] + seglen[i] / max((v[i] + v[i + 1]) / 2.0, 1e-6)

    return dict(v=v, t=t, v_ceiling=v_ceiling, v_drag=v_drag, seglen=seglen,
                a_tan_real=a_tan_real, model=model, tilt_max_deg=tilt_max_deg)


# ===========================================================================
# 3. DIFFERENTIAL-FLATNESS inverse: attitude (quat), body rates, collective
# ===========================================================================
def _quat_from_Rwb(Rwb: np.ndarray) -> np.ndarray:
    """R_world_body (3,3) -> wxyz unit quaternion (scalar-first), sign w >= 0."""
    m = Rwb
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        Sq = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * Sq
        x = (m[2, 1] - m[1, 2]) / Sq
        y = (m[0, 2] - m[2, 0]) / Sq
        z = (m[1, 0] - m[0, 1]) / Sq
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        Sq = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / Sq
        x = 0.25 * Sq
        y = (m[0, 1] + m[1, 0]) / Sq
        z = (m[0, 2] + m[2, 0]) / Sq
    elif m[1, 1] > m[2, 2]:
        Sq = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / Sq
        x = (m[0, 1] + m[1, 0]) / Sq
        y = 0.25 * Sq
        z = (m[1, 2] + m[2, 1]) / Sq
    else:
        Sq = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / Sq
        x = (m[0, 2] + m[2, 0]) / Sq
        y = (m[1, 2] + m[2, 1]) / Sq
        z = 0.25 * Sq
    q = np.array([w, x, y, z])
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _box_smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving-average (edge-replicated) -- finite-jerk smoothing of a 1-D signal."""
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")[: len(x)]


def flatness_inverse(pos, vel, acc, c2=C2_POOLED, dt=None):
    """Reconstruct (yaw, thrust_norm, quat_wxyz, omega_frd) along the trajectory by
    differential flatness (matches proto twin_sanity / peregrine conventions).

    f_thrust = acc - grav + c2*|vel|*vel  (thrust must supply accel, cancel gravity, beat
    drag). Body-up (= -body_z) points along f_thrust; body-x (forward) along the travel
    tangent projected off body-z; yaw = NED heading of the travel tangent. Body rates are
    finite-differenced from the attitude quaternions in the BODY frame."""
    n = len(pos)
    speed = np.linalg.norm(vel, axis=1)
    vhat = vel / np.maximum(speed[:, None], 1e-9)
    f = acc - _GRAV + c2 * speed[:, None] * vel            # required thrust specific force
    fmag = np.linalg.norm(f, axis=1)
    up = f / np.maximum(fmag[:, None], 1e-9)                # body-up direction (world)
    thrust_norm = np.interp(fmag, COLL_MAP_ACCEL_MEASURED, COLL_MAP_THR_MEASURED)
    thrust_norm = np.clip(thrust_norm, 0.0, 1.0)
    yaw = np.arctan2(vhat[:, 1], vhat[:, 0])               # NED heading of travel

    quat = np.zeros((n, 4))
    tilt_deg = np.degrees(np.arccos(np.clip(up @ _WORLD_UP, -1.0, 1.0)))
    for i in range(n):
        zb = -up[i]                                        # body +Z (down) = -body_up
        fwd = vhat[i]                                      # travel tangent
        xb = fwd - (fwd @ zb) * zb
        if np.linalg.norm(xb) < 1e-6:
            ref = np.array([1.0, 0.0, 0.0])
            xb = ref - (ref @ zb) * zb
        xb = xb / max(np.linalg.norm(xb), 1e-9)
        yb = np.cross(zb, xb)
        Rwb = np.column_stack([xb, yb, zb])
        quat[i] = _quat_from_Rwb(Rwb)

    # body-frame angular rate from quaternion finite differences: q[i+1] = q[i] (x) exp(w*dt)
    omega = np.zeros((n, 3))
    if dt is not None:
        for i in range(n - 1):
            qa, qb = quat[i], quat[i + 1]
            if qa @ qb < 0.0:
                qb = -qb
            dq = _qmul(_qconj(qa), qb)                     # body-frame increment
            ang = 2.0 * dq[1:4]                            # small-angle rotation vector
            omega[i] = ang / dt
        omega[-1] = omega[-2]
    return dict(yaw=yaw, thrust_norm=thrust_norm, quat=quat, omega=omega,
                fmag=fmag, tilt_deg=tilt_deg, speed=speed)


# ===========================================================================
# 4. Resample the TOPP plan onto a uniform TIME grid (the emitted samples)
# ===========================================================================
def resample_uniform_time(geom, prof, hz=100.0):
    """Resample the arc-length plan onto a uniform-time grid at ``hz``. Returns the emitted
    arrays (t, pos, vel, acc, yaw, thrust_norm, quat, omega) + flatness diagnostics.

    Acceleration is reconstructed ANALYTICALLY from the Frenet decomposition
    ``acc = a_tan * t_hat + kappa v^2 * n_hat`` (a_tan = the realised per-segment value),
    NOT ``np.gradient(vel)`` -- the latter rings at the speed-profile accel<->brake kinks
    and injected spurious 30-170 rad/s spikes into the flatness body rates."""
    s = geom["s"]
    r = geom["r"]
    t_hat = geom["t_hat"]
    n_hat = geom["n_hat"]
    kappa = geom["kappa"]
    t_plan = prof["t"]
    v_plan = prof["v"]
    a_tan_plan = prof["a_tan_real"]
    dt = 1.0 / hz
    T = float(t_plan[-1])
    n = int(np.floor(T / dt)) + 1
    tg = np.arange(n) * dt
    tg[-1] = min(tg[-1], T)
    # invert t_plan -> s(t) (t_plan strictly increasing since v >= v_floor > 0)
    s_of_t = np.interp(tg, t_plan, s)
    pos = np.vstack([np.interp(s_of_t, s, r[:, k]) for k in range(3)]).T
    that = np.vstack([np.interp(s_of_t, s, t_hat[:, k]) for k in range(3)]).T
    that = that / np.maximum(np.linalg.norm(that, axis=1, keepdims=True), 1e-9)
    nhat = np.vstack([np.interp(s_of_t, s, n_hat[:, k]) for k in range(3)]).T
    vmag = np.interp(s_of_t, s, v_plan)
    kap = np.interp(s_of_t, s, kappa)
    a_tan = np.interp(s_of_t, s, a_tan_plan)
    # finite-jerk smoothing of the tangential accel: the forward-backward TOPP is bang-bang
    # (C0 speed, discontinuous a_tan at accel<->brake switches), which injects one-sample
    # attitude-rate spikes. A real drone uses finite jerk; smooth a_tan over ~0.15 s.
    a_tan = _box_smooth(a_tan, max(1, int(round(0.15 * hz)) | 1))
    vel = vmag[:, None] * that
    acc = a_tan[:, None] * that + (kap * vmag * vmag)[:, None] * nhat
    flat = flatness_inverse(pos, vel, acc, dt=dt)
    return dict(t=tg, pos=pos, vel=vel, acc=acc, dt=dt, **flat)


# ===========================================================================
# 5. Gate crossings on the EMITTED polyline (the actual stored product)
# ===========================================================================
def gate_crossings(pos, t, gates_ned, gate_yaw=GATE_YAW):
    """Where the emitted polyline crosses each gate plane. VQ1 gates are level and face
    yaw=pi, so the gate plane normal is +-X (world) and the in-plane (aperture) miss is the
    Y-Z offset. Crossing found by interpolating to X = gate_X (course is monotone in X)."""
    out = []
    X = pos[:, 0]
    for gi, gc in enumerate(gates_ned):
        xg = gc[0]
        # bracket the crossing nearest the gate (course X decreases monotonically)
        sign = np.sign(X - xg)
        cross = np.where(np.diff(sign) != 0)[0]
        if len(cross) == 0:
            j = int(np.argmin(np.abs(X - xg)))
            seg = [max(0, j - 1), j]
        else:
            # pick the bracket whose midpoint is closest to the gate centre
            mids = 0.5 * (pos[cross] + pos[cross + 1])
            seg_i = int(cross[np.argmin(np.linalg.norm(mids - gc, axis=1))])
            seg = [seg_i, seg_i + 1]
        a, b = pos[seg[0]], pos[seg[1]]
        denom = (b[0] - a[0])
        frac = 0.0 if abs(denom) < 1e-12 else (xg - a[0]) / denom
        frac = float(np.clip(frac, 0.0, 1.0))
        p = a + frac * (b - a)
        tc = float(t[seg[0]] + frac * (t[seg[1]] - t[seg[0]]))
        # in-plane (Y-Z) miss vs the canonical gate centre
        miss_h = float(p[1] - gc[1])               # East / horizontal
        miss_v = float(p[2] - gc[2])               # Down / vertical
        miss = float(np.hypot(miss_h, miss_v))
        out.append(dict(gate_id=gi, t=tc, pos_ned=[float(x) for x in p],
                        miss_m=miss, miss_h_m=miss_h, miss_v_m=miss_v))
    return out


# ===========================================================================
# 6. Assemble + emit the v1-schema JSON
# ===========================================================================
def feasibility_summary(emit, gates_ned, warm_t=0.25):
    """Feasibility certificate computed on the EMITTED samples (the actual product)."""
    t = emit["t"]
    warm = t >= warm_t
    w = emit["omega"][warm]
    cr = gate_crossings(emit["pos"], t, gates_ned)
    return {
        "lap_time_s": round(float(cr[-1]["t"]), 4),
        "peak_speed_mps": round(float(emit["speed"].max()), 4),
        "peak_body_rate_axis_rps": {
            "roll": round(float(np.abs(w[:, 0]).max()), 4),
            "pitch": round(float(np.abs(w[:, 1]).max()), 4),
            "yaw": round(float(np.abs(w[:, 2]).max()), 4),
        },
        "peak_body_rate_norm_rps": round(float(np.linalg.norm(w, axis=1).max()), 4),
        "peak_collective_norm": round(float(emit["thrust_norm"].max()), 4),
        "peak_thrust_a_up_mps2": round(float(emit["fmag"][warm].max()), 4),
        "peak_tilt_deg": round(float(emit["tilt_deg"][warm].max()), 4),
        "frac_inverted_tilt_gt90": round(float(np.mean(emit["tilt_deg"][warm] > 90.0)), 4),
    }


def build_payload(geom, prof, emit, gates_ned, tilt_max_deg, model, ball_ref=None):
    crossings = gate_crossings(emit["pos"], emit["t"], gates_ned)
    lap_time_s = float(crossings[-1]["t"])             # time of the LAST gate crossing
    total_duration_s = float(emit["t"][-1])
    feas = feasibility_summary(emit, gates_ned)

    payload = {
        "schema": "peregrine.reference_line.v1",
        "frame": "world NED (X north, Y east, Z down), body FRD; yaw = NED heading (atan2 E,N)",
        "source": {
            "case": "corrected_aero_inc8_upright",
            "generator": ("rl/build_reference_line.py -- natural cubic-spline through canonical "
                          "VQ1 gate centres (fly_rl._GATE_POS_ZUP*_FLIP) + corrected-aero "
                          f"forward-backward TOPP, model='{model}' (upright body-tilt cone, no "
                          "inverted thrust); MEASURED rl_plant envelope, NOT the linear-drag plant"),
            "handoff": "p2-inc8-rl-refline-2026-06-14",
            "supersedes": "rl/reference_line_vq1.json (TOGT on linear-drag plant; drag-infeasible)",
            "plant": {
                "name": "rl_plant.py MEASURED corrected-aero",
                "thrust_to_weight": round(TW_MEASURED, 4),       # ~7.98 (full-stick 78.28 m/s^2)
                "full_stick_a_up_mps2": round(A_UP_MAX, 4),
                "hover_thrust": HOVER,
                "quad_drag_c2_pooled": C2_POOLED,
                "super_rate_s": float(SUPER_RATE_S_MEASURED),
                "alpha_max_rps2": [float(x) for x in ALPHA_MAX_RPS2_MEASURED],
                "mixer": {
                    "idle": MIXER_IDLE_MEASURED, "kappa_err": MIXER_KAPPA_ERR_MEASURED,
                    "kappa_hold": MIXER_KAPPA_HOLD_MEASURED, "zeta_yaw": MIXER_ZETA_YAW_MEASURED,
                },
                "v_drag_wall_level_mps": round(float(prof["v_drag"].min()), 3),
                "tilt_cone_deg": tilt_max_deg,
                "note": ("T/W 3.765 in reference_line_vq1.json is the LINEAR plant's full-stick "
                         "(g/hover); the measured convex map delivers 7.98 g. Feasibility ceiling "
                         "= thrust_norm<=1.0 <=> |f_thrust|<=78.28 m/s^2. The measured quad-drag "
                         "is ~7x the linear plant at 30 m/s, so the steep (16-19 deg) descent is "
                         "the binding region for UPRIGHT flight."),
            },
            "feasibility": feas,
            "finding": (
                "The brief's ~4.6-4.7 s 'honest lap bound' is the FULL-ATTITUDE (TOGT/proto ball) "
                "bound and is NOT achievable upright or within the rate envelope: it requires "
                "SUSTAINED INVERTED thrust on the descent (median tilt ~100 deg, ~50% of the "
                "course tilt>90 deg) and yaw body-rate ~13 rad/s > the ~11 rad/s super-rate "
                "envelope (yaw alpha ~108 > 80 rps^2). The measured quad-drag is ~7x the linear "
                "plant at 30 m/s, so the steep descent caps UPRIGHT speed near ~30 m/s. This "
                "emitted line is the fastest UPRIGHT profile with full thrust+rate MARGIN "
                "(tilt cone 75 deg, lap ~8.5 s, peak collective ~0.91, peak rate ~5.4 rad/s); "
                "the absolute upright feasible edge (collective-ceiling-limited, tilt ~78 deg) is "
                "~7.9 s. The geometry (gate-centred spline) is IDENTICAL across all these profiles "
                "-- only the attached speed/feasibility certificate differs -- so the R1' "
                "arc-length progress reward is unaffected by the speed choice. See the handoff "
                "report for the full ball-vs-cone data."
            ),
            "comparison_full_attitude_ball_bound": ball_ref,
        },
        "lap_time_s": lap_time_s,
        "total_duration_s": total_duration_s,
        "gate_crossings": crossings,
        "t": [float(x) for x in emit["t"]],
        "pos_ned": emit["pos"].tolist(),
        "vel_ned": emit["vel"].tolist(),
        "acc_ned": emit["acc"].tolist(),
        "yaw": [float(x) for x in emit["yaw"]],
        "quat_wxyz": emit["quat"].tolist(),
        "omega_frd": emit["omega"].tolist(),
        "thrust_norm": [float(x) for x in emit["thrust_norm"]],
    }
    return payload


def generate(tilt_max_deg=DEFAULT_TILT_MAX_DEG, model="cone", hz=100.0,
             n_dense=4000, run_out_m=RUN_OUT_M, with_ball_ref=True):
    gates = load_canonical_gates_ned()
    geom = build_line(gates, n_dense=n_dense, run_out_m=run_out_m)
    prof = topp_corrected_aero(geom, tilt_max_deg, model=model)
    emit = resample_uniform_time(geom, prof, hz=hz)
    ball_ref = None
    if with_ball_ref:
        bprof = topp_corrected_aero(geom, tilt_max_deg, model="ball")
        bemit = resample_uniform_time(geom, bprof, hz=hz)
        ball_ref = feasibility_summary(bemit, gates)
        ball_ref["note"] = ("full-attitude (TOGT/proto) bound -- RATE-INFEASIBLE (yaw>11) and "
                            "sustained inverted; recorded for comparison, NOT emitted as Gamma")
    payload = build_payload(geom, prof, emit, gates, tilt_max_deg, model, ball_ref)
    return payload, geom, prof, emit, gates


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilt-max", type=float, default=DEFAULT_TILT_MAX_DEG)
    ap.add_argument("--model", choices=["cone", "ball"], default="cone",
                    help="cone = upright (emitted); ball = full-attitude TOGT bound (comparison)")
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--sweep", action="store_true", help="print lap/feasibility vs tilt cap")
    ap.add_argument("--emit", action="store_true", help="write rl/reference_line_inc8.json")
    ap.add_argument("--out", type=str, default=str(_HERE / "reference_line_inc8.json"))
    args = ap.parse_args()

    gates = load_canonical_gates_ned()
    print(f"[CANON] gate centres NED (fly_rl._GATE_POS_ZUP*_FLIP):")
    for i, g in enumerate(gates):
        print(f"   G{i}: [{g[0]:9.4f} {g[1]:8.4f} {g[2]:8.4f}]")
    geom = build_line(gates, n_dense=4000, run_out_m=RUN_OUT_M)
    print(f"[BUDGET] full-stick a_up = {A_UP_MAX:.3f} m/s^2 ({TW_MEASURED:.2f} g); "
          f"c2_pooled = {C2_POOLED}; line length (+run-out) = {geom['s'][-1]:.2f} m; "
          f"max curvature = {geom['kappa'].max():.4f} (min radius {1/geom['kappa'].max():.1f} m)")

    if args.sweep:
        print(f"\n[SWEEP] lap & feasibility vs body-tilt cone (model='{args.model}')")
        for tc in [60.0, 65.0, 70.0, 75.0, 78.0, 80.0, 85.0, 90.0]:
            prof = topp_corrected_aero(geom, tc, model=args.model)
            emit = resample_uniform_time(geom, prof, hz=args.hz)
            cr = gate_crossings(emit["pos"], emit["t"], gates)
            warm = emit["t"] >= 0.25
            ax = np.abs(emit["omega"][warm])
            lap = cr[-1]["t"]
            misses = [c["miss_m"] for c in cr]
            print(f"  tilt<={tc:4.0f}  lap={lap:6.3f}s  v_max={emit['speed'].max():5.1f}  "
                  f"rate_axis(r/p/y)={ax[:,0].max():4.1f}/{ax[:,1].max():4.1f}/{ax[:,2].max():4.1f}  "
                  f"coll_pk={emit['thrust_norm'].max():.3f}  fmag_pk={emit['fmag'][warm].max():5.1f}"
                  f"  tilt_pk={emit['tilt_deg'][warm].max():5.1f}  inv%={100*np.mean(emit['tilt_deg'][warm]>90):.0f}"
                  f"  miss_max={max(misses):.4f}")

    payload, geom, prof, emit, gates = generate(tilt_max_deg=args.tilt_max, model=args.model, hz=args.hz)
    cr = payload["gate_crossings"]
    print(f"\n[BUILD model={args.model} tilt<={args.tilt_max:.0f}] lap_time_s={payload['lap_time_s']:.4f}  "
          f"total_duration_s={payload['total_duration_s']:.4f}  N={len(payload['t'])}")
    print(f"   feasibility: {json.dumps(payload['source']['feasibility'])}")
    print("   per-gate crossing miss (m): in-plane | h | v")
    for c in cr:
        print(f"     G{c['gate_id']}: miss={c['miss_m']:.4f}  h={c['miss_h_m']:+.4f}  "
              f"v={c['miss_v_m']:+.4f}  @t={c['t']:.3f}  pos={[round(x,3) for x in c['pos_ned']]}")

    if args.emit:
        Path(args.out).write_text(json.dumps(payload))
        print(f"\n[WROTE] {args.out}  ({Path(args.out).stat().st_size/1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
