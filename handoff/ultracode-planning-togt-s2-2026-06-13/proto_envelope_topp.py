"""proto_envelope_topp.py -- ENVELOPE-AWARE CORRECTED-AERO time-optimal prototype (C1).

Fengyou: this is the honest achievable-time-vs-tilt curve on the REAL (corrected-aero)
plant, NOT the falsified linear plant the shipped 4.55 s reference line was built on.

WHAT THIS IS
------------
A point-mass forward-backward TOPP (time-optimal path parameterization), structurally
identical to src/racer/speed_profile.py, but with a CORRECTED-AERO acceleration budget
derived from the measured plant constants in src/racer/rl_plant.py:

  * TANGENTIAL (longitudinal) accel budget from the CONVEX collective map
    (COLL_MAP_*_MEASURED): full-stick body-up specific force ~78.28 m/s^2 (~8 g). The
    point-mass must (a) cancel gravity, (b) supply centripetal, (c) supply tangential.
    Thrust is a single vector of bounded magnitude a_up<=a_up_max, so the available
    tangential accel couples to how much thrust is already spent on gravity + cornering:
        a_tan_thrust(v,kappa) = sqrt( a_up_max^2 - |a_perp|^2 )  - quad_drag_brake(v)
    where a_perp is the component of (gravity-cancel + centripetal) perpendicular to the
    path tangent. On the descending course gravity has a tangential component too (it
    HELPS on descents) -- we include the signed gravity-along-tangent term.

  * LATERAL accel cap = the STYLE ENVELOPE: a_lat_max = g*tan(tilt_cap). This is the
    SAME tilt that R4 in peregrine_racing.py penalizes (total tilt of body-z from
    world-z; a_lat = g*tan(tilt) for a level-ish turn). Cornering speed on a segment of
    curvature kappa is then capped by  kappa*v^2 <= a_lat_max  =>  v <= sqrt(a_lat_max/kappa).
    This is the whole point: cornering speed is COUPLED to the tilt envelope.

  * TOP-SPEED cap from the v^2 QUAD-DRAG wall: at top speed all available forward thrust
    balances drag. v_top solves  a_fwd_max(at that tilt) = c2 * v_top^2. With the convex
    map ceiling and pooled c2=0.052 this lands ~38-39 m/s (matches the corrected-aero
    TOGT max_speed 39.26 m/s).

ACCEL BUDGET MODEL (point mass, NED, Z down)
--------------------------------------------
Thrust acts body-up; |a_thrust| in [0, a_up_max]. Required perpendicular-to-path accel:
    a_perp_vec = (gravity-cancel needed to hold the path's vertical) + centripetal
We decompose into tangential (t_hat) and the remainder. For the speed profile we need the
scalar tangential accel limit a_tan_max(v, kappa, slope):
    grav_vec       = [0,0,+g]            (NED; gravity pulls +Z down)
    To not fall, thrust must cancel grav_vec's component off the desired accel. The drone's
    achievable accel set is { a_thrust + grav_vec : |a_thrust| <= a_up_max }, a ball of
    radius a_up_max centered at grav_vec (NED). The reachable accel is thus any vector
    within a_up_max of grav_vec. Project onto path frame:
      - centripetal demand: a_c = kappa*v^2 along the in-plane normal (magnitude).
      - The MAX tangential accel given we must also deliver a_c perpendicular:
            a_tan_thrust = sqrt(a_up_max^2 - (a_c - g_perp)^2)? -> we use the ball form:
        Reachable accel vector A must satisfy |A - grav_vec| <= a_up_max. Decompose A into
        t_hat (tangential) and n_hat (path normal, the centripetal direction). Demand on
        n_hat is a_c (centripetal, = kappa v^2). grav_vec projects g_t onto t_hat and g_n
        onto n_hat. Then the constraint is:
            (a_tan - g_t)^2 + (a_c - g_n)^2 + (g_b)^2 <= a_up_max^2
        where g_b is grav's component on the binormal (3rd path axis). Solve for a_tan:
            a_tan <= g_t + sqrt( a_up_max^2 - (a_c - g_n)^2 - g_b^2 )
        Then SUBTRACT body-frame quad drag braking (approx along -t_hat at speed v).
    The LATERAL ENVELOPE additionally caps a_c (hence v) via a_lat_max = g*tan(tilt_cap).

This is the standard "thrust ball minus gravity" TOPP budget, with the style-envelope
lateral cap layered on. It is a PLANNING BOUND (point mass; tilt-rate / body-rate dynamics
unmodeled -- TOGT shows rates barely bind, -0.04 s). Cross-checked against the C++ TOGT
corrected-aero refined lap (4.714 s) at the unconstrained (90 deg) end.

RUN:  PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-planning-togt-s2-2026-06-13/proto_envelope_topp.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from racer.rl_plant import (  # noqa: E402
    COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED, QUAD_DRAG_C2_POOLED,
    QUAD_DRAG_C2_MEASURED, PlantParams, PlantState, step as plant_step,
    ALPHA_MAX_RPS2_MEASURED, SUPER_RATE_S_MEASURED, MIXER_IDLE_MEASURED,
    MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED,
)

_G = 9.80665

# ---------------------------------------------------------------------------
# Track (world NED, Z DOWN). Gate centres from track_map (prompt-confirmed).
# ---------------------------------------------------------------------------
GATES_NED = np.array([
    [-23.30, -0.40, -0.03],   # G0
    [-46.89, -2.50,  5.07],   # G1
    [-74.59,  1.20, 13.67],   # G2
    [-111.49, -5.10, 24.57],  # G3
    [-135.49, -0.80, 25.36],  # G4
    [-159.19, -4.40, 25.97],  # G5
], dtype=np.float64)
START_NED = np.array([0.0, 0.0, -0.02])   # spawn (simstart), z up ~ ground

HALF_OPEN = 0.75          # validity half-opening (in-plane)
GATE_YAW = np.pi          # all gates face along-course (~pi)

# ---------------------------------------------------------------------------
# Corrected-aero plant constants (from rl_plant.py, the REAL envelope)
# ---------------------------------------------------------------------------
A_UP_MAX = float(COLL_MAP_ACCEL_MEASURED[-1])   # 78.2828 m/s^2 full-stick body-up
A_UP_HOVER = _G                                 # ~9.58 measured at hover knot; use g for the cancel
C2_POOLED = QUAD_DRAG_C2_POOLED                 # 0.052 1/m isotropic pooled drag


def a_up_of_thrust(thr: float) -> float:
    return float(np.interp(thr, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED))


# ===========================================================================
# 1. GEOMETRIC LINE through gate centres (cubic spline, chord-length param)
# ===========================================================================
def build_line(waypoints: np.ndarray, n_samples: int = 4000):
    """Natural cubic spline through waypoints (incl. start), chord-length parameterized.
    Returns dense samples + geometry: r (N,3), t_hat, kappa (curvature 1/m), s (arc len),
    and the in-plane normal at each gate plane for miss scoring later."""
    wp = np.asarray(waypoints, float)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    u_wp = np.concatenate([[0.0], np.cumsum(seg)])
    cs = CubicSpline(u_wp, wp, bc_type="natural")
    u = np.linspace(0.0, u_wp[-1], n_samples)
    r = cs(u)
    d1 = cs(u, 1)
    d2 = cs(u, 2)
    ds_du = np.maximum(np.linalg.norm(d1, axis=1), 1e-9)
    t_hat = d1 / ds_du[:, None]
    # curvature = |r' x r''| / |r'|^3
    kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / ds_du ** 3
    # arc length (trapezoid in u)
    s = np.concatenate([[0.0], np.cumsum((ds_du[:-1] + ds_du[1:]) / 2 * np.diff(u))])
    return dict(cs=cs, u=u, u_wp=u_wp, r=r, t_hat=t_hat, d1=d1, d2=d2,
                ds_du=ds_du, kappa=kappa, s=s)


# ===========================================================================
# 2. CORRECTED-AERO TOPP (forward-backward, thrust-ball minus gravity budget,
#    plus style-envelope lateral cap + v^2 drag-wall top-speed cap)
# ===========================================================================
def topp_corrected_aero(geom: dict, tilt_cap_deg: float,
                        a_up_max: float = A_UP_MAX, c2: float = C2_POOLED,
                        v_floor: float = 0.5):
    """Time-optimal speed profile on geom with the corrected-aero budget.

    tilt_cap_deg: style-envelope cap on TOTAL tilt -> a_lat_max = g*tan(tilt_cap).
                  90 deg => unconstrained (a_lat_max -> very large).
    Returns dict with v (N,), t (N,), lap_total_s, and the budget arrays.
    """
    r = geom["r"]
    t_hat = geom["t_hat"]
    kappa = geom["kappa"]
    s = geom["s"]
    N = len(r)
    seglen = np.diff(s)

    # In-plane (path-normal) unit vector: principal normal n_hat = (d/ds t_hat) normalized.
    dthat = np.gradient(t_hat, s, axis=0)
    n_norm = np.linalg.norm(dthat, axis=1)
    n_hat = np.where(n_norm[:, None] > 1e-9, dthat / np.maximum(n_norm[:, None], 1e-12),
                     0.0)
    # binormal
    b_hat = np.cross(t_hat, n_hat)
    bn = np.linalg.norm(b_hat, axis=1)
    b_hat = np.where(bn[:, None] > 1e-9, b_hat / np.maximum(bn[:, None], 1e-12), 0.0)

    grav = np.array([0.0, 0.0, _G])         # NED: +Z down
    g_t = t_hat @ grav                      # gravity component along tangent (helps on descents)
    g_n = n_hat @ grav                      # along principal normal
    g_b = b_hat @ grav                      # along binormal

    # ----- lateral (style) envelope -----
    if tilt_cap_deg >= 89.999:
        a_lat_max = 1e6                     # unconstrained
    else:
        a_lat_max = _G * np.tan(np.radians(tilt_cap_deg))

    # ----- cornering speed ceiling from the lateral (style) cap -----
    # centripetal a_c = kappa*v^2 must be <= a_lat_max  -> v <= sqrt(a_lat_max/kappa)
    v_corner_style = np.where(kappa > 1e-6, np.sqrt(a_lat_max / np.maximum(kappa, 1e-12)),
                              np.inf)

    # ----- cornering speed ceiling from the THRUST BALL (the quadrotor thrust-magnitude
    # constraint, INDEPENDENT of the style cap). Even with zero tangential accel the drone
    # must produce specific force f = (-g_t on t_hat) + (a_c - g_n on n_hat) + (-g_b on b_hat)
    # to merely HOLD the corner at speed v (a_c = kappa v^2). Feasible only if |f| <= a_up_max:
    #     g_t^2 + (a_c - g_n)^2 + g_b^2 <= a_up_max^2
    # Solve the largest a_c (hence v) that fits. (This is what was leaking >ceiling before.)
    rad_ball = a_up_max ** 2 - g_t ** 2 - g_b ** 2          # room left on the n_hat axis
    a_c_max_ball = np.where(rad_ball > 0.0, g_n + np.sqrt(np.maximum(rad_ball, 0.0)), 0.0)
    # a_c = kappa v^2 <= a_c_max_ball  ->  v <= sqrt(a_c_max_ball / kappa)
    v_corner_ball = np.where(kappa > 1e-6,
                             np.sqrt(np.maximum(a_c_max_ball, 0.0) / np.maximum(kappa, 1e-12)),
                             np.inf)
    v_corner = np.minimum(v_corner_style, v_corner_ball)

    # ----- top-speed cap from the v^2 drag wall -----
    # Straight-and-level-ish: forward thrust available after cancelling gravity must equal
    # drag c2*v^2. Max forward thrust component = sqrt(a_up_max^2 - g_perp^2) where on a
    # straight g_perp ~ g (mostly vertical). Solve a_fwd = c2 v^2.
    # Use the conservative straight-flight value (kappa->0): a_fwd_straight = sqrt(a_up_max^2 - (g_n^2+g_b^2)) + g_t
    a_fwd_straight = np.sqrt(np.maximum(a_up_max ** 2 - (g_n ** 2 + g_b ** 2), 0.0)) + g_t
    v_drag = np.sqrt(np.maximum(a_fwd_straight, 0.0) / c2)

    v_ceiling = np.minimum(v_corner, v_drag)
    v = np.minimum(v_ceiling, 1e6)
    v = np.maximum(v, v_floor)

    def a_tan_budget(i: int, vi: float, a_c: float, forward: bool) -> float:
        """Max |tangential accel| available at sample i, speed vi, with centripetal load a_c.
        Thrust ball of radius a_up_max centered at grav (NED): the achievable accel set is
        {a_thrust + grav : |a_thrust| <= a_up_max}. Spending (a_c - g_n) on n_hat and g_b on
        b_hat leaves tangential room:
              (a_tan - g_t)^2 <= a_up_max^2 - (a_c - g_n)^2 - g_b^2
        Quad-drag braking (c2 v^2, body ~ along -t_hat) hurts accel / helps braking.
        a_c is passed in so the caller can use the CONSERVATIVE (worst-of-segment) centripetal
        load -- the discrete sweep otherwise leaks the ball where kappa rises across a segment.
        """
        rad2 = a_up_max ** 2 - (a_c - g_n[i]) ** 2 - g_b[i] ** 2
        a_thrust_tan = np.sqrt(rad2) if rad2 > 0.0 else 0.0
        drag = c2 * vi * vi
        if forward:
            return max(0.0, g_t[i] + a_thrust_tan - drag)          # grav-assist + thrust - drag
        return max(0.0, a_thrust_tan + drag - g_t[i])              # brake: thrust + drag - grav

    # Forward-backward sweeps, ITERATED to a fixed point so the realized (a_tan, a_cen) pair
    # stays inside the thrust ball even where curvature rises across a segment (the discrete
    # single-pass sweep leaks the ball there; iterating with the worst-of-endpoints a_c closes it).
    v[0] = min(v[0], v_floor)                       # standing start at rest
    # Race finish = CROSS the last gate; speed is FREE (no braking wall). TOGT finishes at
    # ~38 m/s. Forcing a low finish speed would demand >ceiling deceleration in the last few
    # metres (a pure artifact). v_finish = the ceiling at the last sample.
    v_finish = float(min(v_ceiling[-1], 1e6))
    for _sweep in range(6):
        # forward (accelerate): centripetal load uses the END-of-segment speed (the one we're
        # integrating TO) so we never authorize a tangential accel that pushes a_cen out of the ball.
        v[0] = min(v[0], v_floor)
        for i in range(N - 1):
            a_c = kappa[i + 1] * v[i + 1] * v[i + 1]    # conservative: load at the target sample
            a = a_tan_budget(i, v[i], a_c, forward=True)
            v[i + 1] = min(v[i + 1], np.sqrt(max(v[i] ** 2 + 2.0 * a * seglen[i], 0.0)))
        # backward (brake): the braking deceleration over a segment must fit the ball at BOTH
        # endpoints. Use the WORSE (larger) centripetal load over the segment so corner-entry
        # braking (fast straight -> capped corner) does not exceed the thrust ball.
        v[-1] = min(v[-1], v_finish)
        for i in range(N - 2, -1, -1):
            a_c = max(kappa[i] * v[i] * v[i], kappa[i + 1] * v[i + 1] * v[i + 1])
            a = a_tan_budget(i + 1, v[i + 1], a_c, forward=False)
            v[i] = min(v[i], np.sqrt(max(v[i + 1] ** 2 + 2.0 * a * seglen[i], 0.0)))

    # Integrate time
    t = np.zeros(N)
    for i in range(N - 1):
        t[i + 1] = t[i] + seglen[i] / max((v[i] + v[i + 1]) / 2.0, 1e-6)

    # ---- specific-force feasibility check. Tangential accel taken from the INTEGRATOR-realized
    # segment values a_real = (v[i+1]^2 - v[i]^2)/(2 seglen) -- the exact basis of the lap time,
    # with no central-difference ringing (np.gradient overshoots at the braking-onset knees).
    # Required specific force |f| = |a_tan*t_hat + a_cen*n_hat - grav| must fit the thrust ball.
    a_tan_real = np.zeros(N)
    a_tan_real[:-1] = (v[1:] ** 2 - v[:-1] ** 2) / (2.0 * np.maximum(seglen, 1e-9))
    a_cen_real = kappa * v * v
    aworld = a_tan_real[:, None] * t_hat + a_cen_real[:, None] * n_hat
    f_native = aworld - grav
    f_mag_native = np.linalg.norm(f_native, axis=1)
    up_native = f_native / np.maximum(f_mag_native[:, None], 1e-9)
    tilt_native = np.degrees(np.arccos(np.clip(up_native @ np.array([0.0, 0.0, -1.0]), -1, 1)))
    # report excluding the first 1 m (standing-start ramp: at v~0 the accel direction swings
    # wildly and tilt is meaningless; the speed budget itself is enforced there).
    body = s > 1.0
    feas = dict(
        max_req_a_up=float(f_mag_native[body].max()),
        p99_req_a_up=float(np.percentile(f_mag_native[body], 99)),
        frac_over_ceiling=float(np.mean(f_mag_native[body] > a_up_max + 1e-6)),
        max_req_tilt_deg=float(tilt_native[body].max()),
        p99_req_tilt_deg=float(np.percentile(tilt_native[body], 99)),
    )

    return dict(v=v, t=t, lap_total_s=float(t[-1]), v_corner=v_corner, v_drag=v_drag,
                v_corner_style=v_corner_style, v_corner_ball=v_corner_ball,
                a_lat_max=a_lat_max, n_hat=n_hat, g_t=g_t, g_n=g_n, g_b=g_b,
                seglen=seglen, feas=feas)


# ===========================================================================
# 3. Gate miss scoring (in-plane Euclidean miss at each gate plane)
# ===========================================================================
def gate_crossings(geom: dict, prof: dict, gates_ned: np.ndarray):
    """For each gate, find where the line crosses the gate plane (normal ~ along-course,
    i.e. the X axis in NED here since gates face ~pi) and report in-plane Euclidean miss
    + crossing time. Gate plane normal taken from the line tangent nearest the gate."""
    r = geom["r"]
    t = prof["t"]
    crossings = []
    for gi, gc in enumerate(gates_ned):
        # nearest sample to gate centre
        d = np.linalg.norm(r - gc, axis=1)
        j = int(np.argmin(d))
        # plane normal: along-course (gate faces ~pi -> normal ~ +/-X). Use the line tangent.
        nrm = geom["t_hat"][j]
        nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
        # in-plane miss = component of (r[j]-gc) perpendicular to nrm
        rel = r[j] - gc
        perp = rel - (rel @ nrm) * nrm
        miss = float(np.linalg.norm(perp))
        crossings.append(dict(gate=gi, t=float(t[j]), miss_m=miss,
                              pos=r[j].tolist(), idx=j))
    return crossings


def contact_free(crossings, body_radius: float) -> tuple[bool, list]:
    """All gates clear if in-plane miss < (HALF_OPEN - body_radius)."""
    band = HALF_OPEN - body_radius
    flags = [c["miss_m"] < band for c in crossings]
    return all(flags), flags


# ===========================================================================
# 4. Optional twin validation (roll feedforward yaw+accel through CtbrPlant)
# ===========================================================================
def twin_sanity(geom: dict, prof: dict, dt: float = 1.0 / 100.0):
    """Roll the planned (pos,vel) timeline through the measured-aero rl_plant via a simple
    feedforward+PD tracker to sanity-check realizability. Returns max tracking error and
    realized lap-ish. This is a CRUDE realizability probe, NOT the deliverable time."""
    # resample the plan on a uniform time grid
    t = prof["t"]
    T = t[-1]
    n = int(T / dt) + 1
    tg = np.linspace(0, T, n)
    r = geom["r"]
    s = geom["s"]
    v = prof["v"]
    t_hat = geom["t_hat"]
    # plan pos/vel at uniform time via interp on cumulative time
    pos_plan = np.vstack([np.interp(tg, t, r[:, k]) for k in range(3)]).T
    vel_mag = np.interp(tg, t, v)
    that_g = np.vstack([np.interp(tg, t, t_hat[:, k]) for k in range(3)]).T
    that_g /= np.maximum(np.linalg.norm(that_g, axis=1, keepdims=True), 1e-9)
    vel_plan = vel_mag[:, None] * that_g
    acc_plan = np.gradient(vel_plan, dt, axis=0)

    params = PlantParams(
        rate_sign=np.array([1.0, 1.0, 1.0]),
        super_rate_s=SUPER_RATE_S_MEASURED,
        alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
        linear_drag=0.0,
        quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
        coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
        coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy(),
        mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
        mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED, mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED,
    )
    from scipy.spatial.transform import Rotation
    from racer.rl_plant import quat_multiply, quat_conjugate, quat_rotate
    st = PlantState(pos=pos_plan[0].copy(), vel=vel_plan[0].copy(),
                    quat=np.array([1.0, 0.0, 0.0, 0.0]), omega=np.zeros(3),
                    thrust=np.float64(0.2656))
    grav = np.array([0.0, 0.0, _G])           # NED gravity specific accel (+Z down)
    max_perp_err = 0.0
    # Geometric flatness inverse: desired specific force f_des (world NED) must be produced
    # by thrust along body-UP. thrust contributes a_up * R(q)@[0,0,-1]. So body-up axis
    # (= -body_z) must point along f_des/|f_des|, i.e. body_z = -f_des/|f_des|.
    for k in range(1, len(tg)):
        # f_des = a_des - grav  (the specific force; thrust must supply this).
        f_des = acc_plan[k] - grav
        # position + velocity PD correction (added to the specific-force demand)
        f_des = f_des + 8.0 * (pos_plan[k] - st.pos) + 5.0 * (vel_plan[k] - st.vel)
        f_mag = float(np.linalg.norm(f_des))
        up_des = f_des / max(f_mag, 1e-6)     # body-UP should point here
        zb = -up_des                          # body +Z (down)
        # yaw: project path tangent onto the plane perpendicular to zb to set body-x (fwd)
        xb = that_g[k] - (that_g[k] @ zb) * zb
        if np.linalg.norm(xb) < 1e-6:
            xb = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ zb) * zb
        xb /= max(np.linalg.norm(xb), 1e-9)
        yb = np.cross(zb, xb)
        Rwb = np.column_stack([xb, yb, zb])   # columns = body axes in world (R_world_body)
        qd = Rotation.from_matrix(Rwb).as_quat()             # xyzw
        q_des = np.array([qd[3], qd[0], qd[1], qd[2]])       # wxyz
        # attitude error -> body rate command (proportional)
        q_err = quat_multiply(quat_conjugate(st.quat), q_des)
        if q_err[0] < 0:
            q_err = -q_err
        ang_err = 2.0 * q_err[1:4]            # small-angle body-frame rotation vector
        rate_cmd = np.clip(20.0 * ang_err, -11.0, 11.0)
        # collective from desired body-up accel magnitude via inverse of convex map
        thr = float(np.interp(f_mag, COLL_MAP_ACCEL_MEASURED, COLL_MAP_THR_MEASURED))
        action = np.concatenate([rate_cmd, [thr]])
        st = plant_step(st, action, dt, params)
        # perpendicular tracking error
        rel = st.pos - pos_plan[k]
        th = that_g[k]
        perp = rel - (rel @ th) * th
        max_perp_err = max(max_perp_err, float(np.linalg.norm(perp)))
    return dict(max_perp_err_m=max_perp_err, realized_end_pos=st.pos.tolist(),
                plan_end_pos=pos_plan[-1].tolist(), n_steps=len(tg))


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> int:
    wp = np.vstack([START_NED, GATES_NED])
    geom = build_line(wp, n_samples=5000)

    # assert the line passes near gate centres
    line_pass = []
    for gi, gc in enumerate(GATES_NED):
        d = np.linalg.norm(geom["r"] - gc, axis=1)
        line_pass.append(float(d.min()))
    max_line_dev = max(line_pass)
    print(f"[LINE] max distance of spline to any gate centre = {max_line_dev:.4f} m "
          f"(per-gate: {[round(x,4) for x in line_pass]})")
    assert max_line_dev <= 0.5, f"line deviates {max_line_dev:.3f} m from a gate centre (>0.5)"

    print(f"[BUDGET] a_up_max(full-stick) = {A_UP_MAX:.3f} m/s^2  "
          f"({A_UP_MAX/_G:.2f} g);  c2_pooled = {C2_POOLED}")
    print(f"[BUDGET] line length = {geom['s'][-1]:.2f} m;  max curvature = "
          f"{geom['kappa'].max():.4f} 1/m (min radius {1.0/max(geom['kappa'].max(),1e-9):.2f} m)")

    tilt_caps = [60.0, 65.0, 75.0, 80.0, 90.0]
    radii = [0.38, 0.33, 0.28]

    rows = []
    sweep = []
    for tc in tilt_caps:
        prof = topp_corrected_aero(geom, tc)
        cr = gate_crossings(geom, prof, GATES_NED)
        misses = [c["miss_m"] for c in cr]
        cf38, fl38 = contact_free(cr, 0.38)
        cf33, fl33 = contact_free(cr, 0.33)
        cf28, fl28 = contact_free(cr, 0.28)
        a_lat = prof["a_lat_max"]
        vmax = float(prof["v"].max())
        rows.append(dict(tilt_deg=tc, lap_s=prof["lap_total_s"],
                         a_lat_max=a_lat, vmax=vmax,
                         misses=misses, cf38=cf38, cf33=cf33, cf28=cf28,
                         feas=prof["feas"]))
        # primary config = r=0.38 (inc7-trained). contact_free reported at r=0.38.
        sweep.append(dict(tilt_deg=tc, lap_time_s=round(prof["lap_total_s"], 4),
                          contact_free=bool(cf38)))
        fe = prof["feas"]
        print(f"\n[TILT {tc:.0f} deg]  a_lat_max={a_lat:.1f} m/s^2  "
              f"v_max={vmax:.1f} m/s  LAP={prof['lap_total_s']:.3f} s")
        print(f"    FEAS: max req a_up={fe['max_req_a_up']:.1f} "
              f"(ceiling {A_UP_MAX:.1f}; frac_over={fe['frac_over_ceiling']:.3f})  "
              f"max req tilt={fe['max_req_tilt_deg']:.1f} deg  p99={fe['p99_req_tilt_deg']:.1f} deg")
        print(f"    per-gate in-plane miss (m): {[round(m,3) for m in misses]}")
        print(f"    contact-free r=0.38 band<{HALF_OPEN-0.38:.2f}: {cf38}  flags={fl38}")
        print(f"    contact-free r=0.33 band<{HALF_OPEN-0.33:.2f}: {cf33}  flags={fl33}")
        print(f"    contact-free r=0.28 band<{HALF_OPEN-0.28:.2f}: {cf28}  flags={fl28}")

    # ----- cross-check vs TOGT corrected-aero refined lap 4.714 s -----
    lap90 = [r for r in rows if r["tilt_deg"] == 90.0][0]["lap_s"]
    lap60 = [r for r in rows if r["tilt_deg"] == 60.0][0]["lap_s"]
    print(f"\n[CROSS-CHECK] 90deg/unconstrained TOPP lap = {lap90:.3f} s vs "
          f"TOGT corrected-aero refined 4.714 s (TOGT init 4.123 s). "
          f"{'OK (within bound band)' if 3.8 <= lap90 <= 5.2 else 'CHECK -- out of band'}")
    print(f"[STYLE] 60deg style-respecting achievable lap = {lap60:.3f} s; "
          f"tax vs 90deg = {lap60-lap90:.2f} s")

    # ----- realizability: OPEN-LOOP feasibility is the primary realizability signal (the
    # 'feas' block per tilt: required specific force vs the 78.3 m/s^2 thrust ceiling). A
    # CLOSED-LOOP twin-track is offered as a secondary probe but a naive feedforward+PD tracker
    # CANNOT follow a 35+ m/s TOPP plan through the measured plant (67 ms latency, super-rate
    # attitude lag) -- that divergence is itself the 'a good tracker is REQUIRED' message
    # (cf. the existing geometric tracker needing k=1.85). It is NOT the deliverable time.
    twin = None
    try:
        prof60 = topp_corrected_aero(geom, 60.0)
        twin = twin_sanity(geom, prof60)
        print(f"\n[TWIN PROBE @60deg]  (SECONDARY; naive feedforward+PD tracker)  "
              f"max perpendicular tracking error = {twin['max_perp_err_m']:.1f} m over "
              f"{twin['n_steps']} steps.")
        print(f"    A naive point-tracker DIVERGES on this aggressive plan -- expected; the "
              f"OPEN-LOOP feasibility ('feas' above) is the realizability verdict, and a "
              f"capable tracker (RL / MPCC) is required to realize the bound. NOT the lap time.")
    except Exception as e:  # noqa: BLE001
        print(f"\n[TWIN PROBE] skipped ({type(e).__name__}: {e})")

    # ----- write artifacts -----
    out = dict(
        a_up_max=A_UP_MAX, c2_pooled=C2_POOLED, line_length_m=float(geom["s"][-1]),
        max_line_dev_m=max_line_dev, max_curvature=float(geom["kappa"].max()),
        tilt_sweep=sweep,
        rows=[dict(tilt_deg=r["tilt_deg"], lap_s=round(r["lap_s"], 4),
                   a_lat_max=round(r["a_lat_max"], 2), vmax=round(r["vmax"], 2),
                   misses=[round(m, 4) for m in r["misses"]],
                   cf38=r["cf38"], cf33=r["cf33"], cf28=r["cf28"],
                   feas={k: round(v, 3) for k, v in r["feas"].items()}) for r in rows],
        togt_corrected_aero_refined_s=4.7139, togt_corrected_aero_init_s=4.1227,
        twin_probe=twin,
    )
    (_HERE / "envelope_topp_result.json").write_text(json.dumps(out, indent=2))
    print(f"\n[WROTE] {_HERE / 'envelope_topp_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
