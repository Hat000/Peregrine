"""Phase-B prototype: MIN-SNAP geometry + corrected-aero TOPP lap-time estimate.

OFFLINE ANALYSIS ONLY. Pure numpy/scipy. No sim, no SLURM, no network.

Pieces:
  1. min-snap geometry: a smooth C2 path through the 6 gate centres (+ pad start).
     A clamped cubic (v=0 ends) is used as a faithful, slightly-conservative
     (higher-curvature) proxy for a true 7th-order min-snap QP line -- the curvature
     DISTRIBUTION is what bounds the TOPP, and min-snap only lowers it.
  2. corrected-aero TOPP (forward-backward): a thrust-BUDGET-COUPLED speed profile.
     The thrust vector simultaneously cancels gravity, fights quadratic drag, and
     supplies centripetal + tangential accel. Two caps: thrust magnitude ceiling
     (~78 m/s^2) AND a tilt-cone lateral cap (g*tan(tilt)).
  3. tracking realizability: feed the reference accel back through rl_plant.step
     (parity-identical to the twin) with a geometric thrust+attitude+rate controller
     and measure tracking error + gate misses -> the time the line ACTUALLY flies at.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "rl"))

from racer.rl_plant import (PlantParams, PlantState, step as plant_step,
                            SUPER_RATE_S_MEASURED, ALPHA_MAX_RPS2_MEASURED,
                            QUAD_DRAG_C2_MEASURED, QUAD_DRAG_C2_POOLED,
                            COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED,
                            MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED,
                            quat_rotate, quat_rotate_inverse, quat_multiply,
                            rotvec_to_quat, quat_normalize)

G = 9.80665
HOVER = 0.2656

GATES = np.array([
    [-23.30, -0.40, -0.03],
    [-46.89, -2.50,  5.07],
    [-74.59,  1.20, 13.67],
    [-111.49, -5.10, 24.57],
    [-135.49, -0.80, 25.36],
    [-159.19, -4.40, 25.97],
])
PAD = np.array([0.0, 0.0, 0.02])
FINISH_PLANE_X = GATES[-1, 0]

A_UP_MAX = float(COLL_MAP_ACCEL_MEASURED[-1])     # ~78.3 m/s^2 full stick
C2 = QUAD_DRAG_C2_POOLED                            # 0.052 /m


# --------------------------------------------------------------------------- geometry
def smooth_path(wp: np.ndarray, n: int = 4000):
    """Clamped cubic through wp; robust arc-length / tangent / curvature sampling.
    Curvature computed from finite differences of the unit tangent w.r.t. arc length
    (stable; avoids the cross/ds^3 blow-up at near-stationary spline samples)."""
    from scipy.interpolate import CubicSpline
    t_knot = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))])
    cs = CubicSpline(t_knot, wp, bc_type=((1, np.zeros(3)), (1, np.zeros(3))))
    u = np.linspace(0, t_knot[-1], n)
    r = cs(u)
    d1 = cs(u, 1)
    ds = np.maximum(np.linalg.norm(d1, axis=1), 1e-9)
    tang = d1 / ds[:, None]
    # arc length
    seg = np.linalg.norm(np.diff(r, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    # curvature = |d tang / d s| via central differences in arc length
    dt = np.gradient(tang, axis=0)
    dsg = np.gradient(s)
    kappa = np.linalg.norm(dt, axis=1) / np.maximum(dsg, 1e-9)
    kappa = np.clip(kappa, 0, 5.0)   # cap absurd endpoint artifacts (min radius 0.2 m)
    return r, tang, kappa, s


def margin_waypoints(gates: np.ndarray, lateral_margin: float = 0.0):
    """Optionally nudge gate waypoints toward the straightened chord by `lateral_margin`
    so the planned crossing sits up to `lateral_margin` off-centre (still <= 0.5 m).
    Here we keep crossings AT centre (margin 0) for the conservative validity story;
    pass a small value to model a centre-biased racing line."""
    return gates.copy()


# --------------------------------------------------------------------------- TOPP (coupled)
def coupled_v_ceiling(kappa, a_up_max, tilt_cap_deg):
    a_lat_tilt = G * np.tan(np.radians(tilt_cap_deg))
    a_lat_thrust = np.sqrt(max(a_up_max**2 - G**2, 0.0))
    a_lat = min(a_lat_thrust, a_lat_tilt)
    return np.where(kappa > 1e-6, np.sqrt(a_lat / np.maximum(kappa, 1e-12)), np.inf), a_lat


def a_tan_budget(v, kappa, climbing, a_up_max, tilt_cap_deg):
    drag = C2 * v * v
    g_tan = G * climbing                       # descending(+) assists forward
    centri = kappa * v * v
    rem = a_up_max**2 - G**2 - centri**2
    a_thrust = np.sqrt(rem) if rem > 0 else 0.0
    a_tilt = G * np.tan(np.radians(tilt_cap_deg))
    a_inplane = min(a_thrust, a_tilt)
    a_accel = max(0.0, a_inplane + g_tan - drag)
    a_decel = max(0.0, a_inplane - g_tan + drag)
    return a_accel, a_decel


def topp_corrected(r, tang, kappa, s, a_up_max=A_UP_MAX, v_top=39.0,
                   tilt_cap_deg=60.0, v_start=0.0, v_end=None):
    N = len(s)
    seglen = np.diff(s)
    climbing = tang[:, 2]
    vceil, a_lat = coupled_v_ceiling(kappa, a_up_max, tilt_cap_deg)
    v = np.minimum(v_top, vceil)
    v[0] = min(v[0], v_start)
    for i in range(N - 1):
        a_acc, _ = a_tan_budget(v[i], kappa[i], climbing[i], a_up_max, tilt_cap_deg)
        v[i + 1] = min(v[i + 1], np.sqrt(max(v[i]**2 + 2 * a_acc * seglen[i], 0.0)))
    if v_end is not None:
        v[-1] = min(v[-1], v_end)
    for i in range(N - 2, -1, -1):
        _, a_dec = a_tan_budget(v[i + 1], kappa[i + 1], climbing[i + 1], a_up_max, tilt_cap_deg)
        v[i] = min(v[i], np.sqrt(max(v[i + 1]**2 + 2 * a_dec * seglen[i], 0.0)))
    t = np.zeros(N)
    for i in range(N - 1):
        t[i + 1] = t[i] + seglen[i] / max((v[i] + v[i + 1]) / 2, 1e-6)
    return v, t, vceil, a_lat


def lap_time_to_finish(r, t):
    x = r[:, 0]
    for i in range(1, len(x)):
        if x[i - 1] > FINISH_PLANE_X >= x[i]:
            f = (x[i - 1] - FINISH_PLANE_X) / (x[i - 1] - x[i])
            return t[i - 1] + f * (t[i] - t[i - 1])
    return t[-1]


def gate_miss(r, gate_xyz):
    x = r[:, 0]; gx = gate_xyz[0]; best = np.inf
    for i in range(1, len(x)):
        if (x[i - 1] - gx) * (x[i] - gx) <= 0 and x[i - 1] != x[i]:
            f = (gx - x[i - 1]) / (x[i] - x[i - 1])
            p = r[i - 1] + f * (r[i] - r[i - 1])
            best = min(best, np.hypot(p[1] - gate_xyz[1], p[2] - gate_xyz[2]))
    return best


def per_gate_curvature(r, kappa, gates):
    """Curvature sampled at the nearest path point to each gate centre."""
    out = []
    for g in gates:
        i = int(np.argmin(np.linalg.norm(r - g, axis=1)))
        out.append(kappa[i])
    return out


# --------------------------------------------------------------------------- tracking realizability
def build_plant(mixer=True):
    aero = dict(rate_sign=np.array([1.0, 1.0, 1.0]),   # live true sign
                super_rate_s=SUPER_RATE_S_MEASURED,
                alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
                linear_drag=0.0,
                quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
                coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
                coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy())
    if mixer:
        return PlantParams(**aero, mixer_idle=MIXER_IDLE_MEASURED,
                           mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
                           mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED,
                           mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED)
    return PlantParams(**aero)


def accel_to_thrust_norm(a_up):
    """Invert the convex collective->accel map to get normalized collective for a body-up accel."""
    return float(np.interp(a_up, COLL_MAP_ACCEL_MEASURED, COLL_MAP_THR_MEASURED))


def track_line(r_ref, v_ref, t_ref, params, dt=1/30.0, latency_steps=2,
               kp_pos=8.0, kd_vel=4.0, kp_att=12.0):
    """Geometric tracker: at each control tick, compute desired world accel
    a_des = a_ff + kp_pos*(p_ref-p) + kd_vel*(v_ref-v); convert to a thrust vector
    (a_des - g_vec), get collective from |thrust| and desired attitude from its
    direction; command a body rate that rotates current b3 toward desired b3.
    Returns flown positions, gate misses, and lap time (gate-5 plane crossing).
    Includes an integer transport delay (latency_steps) and the measured plant.
    """
    # resample the reference at the control rate
    T = t_ref[-1]
    tg = np.arange(0, T, dt)
    pr = np.vstack([np.interp(tg, t_ref, r_ref[:, k]) for k in range(3)]).T
    vr = np.vstack([np.interp(tg, t_ref, v_ref[:, k]) for k in range(3)]).T
    ar = np.zeros_like(vr)
    ar[1:-1] = (vr[2:] - vr[:-2]) / (2 * dt)
    g_vec = np.array([0.0, 0.0, G])

    st = PlantState(pos=PAD.copy(), vel=np.zeros(3),
                    quat=np.array([1.0, 0.0, 0.0, 0.0]),  # level
                    omega=np.zeros(3), thrust=np.float64(HOVER))
    # transport-delay buffer of commands
    from collections import deque
    buf = deque([np.array([0, 0, 0, HOVER])] * latency_steps, maxlen=latency_steps)

    flown = [st.pos.copy()]
    for k in range(len(tg)):
        p, v = st.pos, st.vel
        a_des = ar[k] + kp_pos * (pr[k] - p) + kd_vel * (vr[k] - v)
        f_thrust = a_des - g_vec            # NED: world accel = f_thrust + g_vec - drag
        # (ignore drag feedforward here -> tracker compensates; conservative)
        mag = np.linalg.norm(f_thrust)
        b3_des = -f_thrust / max(mag, 1e-6)  # body-up (thrust) direction in world (NED, up=-z)
        a_up_des = mag
        coll = accel_to_thrust_norm(a_up_des)
        # current body-up in world
        b3_cur = quat_rotate(st.quat, np.array([0.0, 0.0, -1.0]))
        # rotation axis to align b3_cur -> b3_des (world), then express in body
        axis_w = np.cross(b3_cur, b3_des)
        sin_a = np.linalg.norm(axis_w)
        ang = np.arctan2(sin_a, np.clip(np.dot(b3_cur, b3_des), -1, 1))
        if sin_a > 1e-6:
            axis_w = axis_w / sin_a
        rate_w = kp_att * ang * axis_w
        rate_b = quat_rotate_inverse(st.quat, rate_w)
        rate_b = np.clip(rate_b, -11.0, 11.0)
        cmd = np.array([rate_b[0], rate_b[1], rate_b[2], coll * HOVER])
        buf.append(cmd)
        applied = buf[0]
        st = plant_step(st, applied, dt, params)
        flown.append(st.pos.copy())
        if st.pos[0] < FINISH_PLANE_X - 5:    # past finish
            break
    flown = np.array(flown)
    misses = [gate_miss(flown, g) for g in GATES]
    # lap time = gate5 plane crossing
    lap = None
    x = flown[:, 0]
    for i in range(1, len(x)):
        if x[i - 1] > FINISH_PLANE_X >= x[i]:
            f = (x[i - 1] - FINISH_PLANE_X) / (x[i - 1] - x[i])
            lap = (i - 1 + f) * dt
            break
    return flown, misses, lap


def run(tilt_cap_deg, label, do_track=False):
    wp = np.vstack([PAD, GATES])
    r, tang, kappa, s = smooth_path(wp, n=4000)
    v, t, vceil, a_lat = topp_corrected(r, tang, kappa, s, tilt_cap_deg=tilt_cap_deg,
                                        v_start=0.0, v_end=None)
    lap = lap_time_to_finish(r, t)
    misses = [gate_miss(r, g) for g in GATES]
    kg = per_gate_curvature(r, kappa, GATES)
    print(f"\n=== {label}  (tilt_cap={tilt_cap_deg} deg, lateral cap {a_lat:.1f} m/s^2) ===")
    print(f"  path length        : {s[-1]:.1f} m")
    print(f"  IDEAL lap (G5 plane): {lap:.3f} s")
    print(f"  v_max / v_mean     : {v.max():.1f} / {v.mean():.1f} m/s")
    print(f"  per-gate curvature : " + ", ".join(f"{k:.3f}" for k in kg) + " 1/m")
    print(f"  per-gate min-radius: " + ", ".join(f"{1/max(k,1e-6):.0f}" for k in kg) + " m")
    print(f"  gate misses        : " + ", ".join(f"{m:.2f}" for m in misses) + " m")
    if do_track:
        params = build_plant(mixer=True)
        flown, tmisses, tlap = track_line(r, v[:, None] * tang, t, params)
        print(f"  --- TRACKING (measured plant, 2-tick latency, geometric ctrl) ---")
        print(f"  tracked lap        : {tlap}")
        print(f"  tracked misses     : " + ", ".join(f"{m:.2f}" for m in tmisses) + " m")
        print(f"  max miss valid?    : {max(tmisses) < 0.75 - 0.38}")
    return lap, max(misses), v.max()


if __name__ == "__main__":
    print("MIN-SNAP + CORRECTED-AERO TOPP prototype")
    print(f"  a_up_max (full stick) = {A_UP_MAX:.1f} m/s^2 ({A_UP_MAX/G:.2f} g)")
    print(f"  pooled quad drag c2   = {C2} /m  -> v^2 wall ~ {np.sqrt(A_UP_MAX/C2):.1f} m/s")
    print(f"  thrust-lateral cap    = sqrt(a_up^2 - g^2) = {np.sqrt(A_UP_MAX**2-G**2):.1f} m/s^2")
    for tc in (60.0, 65.0, 75.0, 80.0):
        run(tc, "centre-stacked geometry", do_track=(tc in (60.0, 75.0)))
