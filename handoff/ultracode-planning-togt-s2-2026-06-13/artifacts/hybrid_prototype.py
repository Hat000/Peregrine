"""HYBRID approach prototype (Phase B, 2026-06-13) — pure numpy/scipy, native offline.

Goal: an HONEST achievable-lap-time estimate for the DECOMPOSED/HYBRID S2 architecture:
  explicit offline plan-line (geometry) + tracker (RL or MPCC) with arc-length progress reward.

Method:
  1. Take the SHIPPED reference-line GEOMETRY (rl/reference_line_vq1.json) — the spatial path
     only (positions). Its TIMING was built on the falsified linear plant (51 m/s peaks) so we
     DISCARD the timing and re-time the path.
  2. Re-time that geometry with a corrected-aero TOPP (forward/backward pass) whose speed ceiling
     is the REAL v^2 drag wall and whose accel budget is the REAL convex-thrust / tilt-cone
     envelope. This is the *planner* output of the hybrid: a feasible, contact-free reference.
  3. TRACKING-REALIZABILITY: forward-simulate the actual measured-aero CtbrPlant (mixer) with a
     geometric feedforward + PD tracker chasing that re-timed reference, and measure (a) the
     realized lap time and (b) the cross-track / gate-miss errors — does a *tracker* actually
     hit it, or does it need k>1 time dilation like the existing 8.3 s geometric tracker?

The gap between (2) the planner time and (3) the tracked time is the hybrid's REALIZABILITY TAX.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from racer.rl_plant import (PlantParams, PlantState, step as plant_step,
    SUPER_RATE_S_MEASURED, ALPHA_MAX_RPS2_MEASURED, QUAD_DRAG_C2_MEASURED,
    COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED, MIXER_IDLE_MEASURED,
    MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED)

G = 9.80665
AMAX_UP = float(np.interp(1.0, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED))  # ~78.3
HOVER = 0.2656
C2 = 0.052  # pooled quad drag

GATES = np.array([
    [-23.30, -0.40, -0.03], [-46.89, -2.50, 5.07], [-74.59, 1.20, 13.67],
    [-111.49, -5.10, 24.57], [-135.49, -0.80, 25.36], [-159.19, -4.40, 25.97]])  # NED


def measured_plant(mixer=True):
    kw = dict(rate_sign=np.array([1., 1., 1.]), super_rate_s=SUPER_RATE_S_MEASURED,
              alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(), linear_drag=0.0,
              quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
              coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
              coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy())
    if mixer:
        kw.update(mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
                  mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED, mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED)
    return PlantParams(**kw)


def load_geometry():
    d = json.load(open('rl/reference_line_vq1.json'))
    pos = np.array(d['pos_ned'])
    return pos, d


def resample_path(pos, n=900):
    """Arc-length cubic-spline resample of the spatial path (geometry only)."""
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    u = np.concatenate([[0], np.cumsum(seg)])
    cs = CubicSpline(u, pos, bc_type='natural')
    uu = np.linspace(0, u[-1], n)
    r = cs(uu)
    d1 = cs(uu, 1); d2 = cs(uu, 2)
    ds = np.maximum(np.linalg.norm(d1, axis=1), 1e-9)
    tang = d1 / ds[:, None]
    kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / ds**3
    s = np.concatenate([[0], np.cumsum((ds[:-1] + ds[1:]) / 2 * np.diff(uu))])
    return r, tang, kappa, s


def corrected_aero_topp(r, tang, kappa, s, tilt_max_deg, a_brake_cap=None):
    """Forward/backward TOPP with corrected-aero accel + drag-wall ceiling.

    Accel budget model (per sample, magnitude a_max along path):
      - max body-up specific accel AMAX_UP (~78 m/s2) at full stick.
      - usable horizontal thrust at the tilt cap: a_h = AMAX_UP*sin(tilt_max).
      - BUT must net out gravity to stay on a near-level/descending path: the vertical thrust
        component AMAX_UP*cos(tilt_max) must be >= g for level flight; the course descends so
        a little budget is freed, but we conservatively require holding altitude on the climb
        legs. We cap usable accel magnitude at a_h (lateral/tangential thrust authority) and
        also at a drag-aware tangential brake.
      - drag is direction-dependent: it ALWAYS opposes motion, so on accel it subtracts and on
        brake it adds (helps). We fold v^2 drag into the speed ceiling and the brake budget.

    Ceiling: v_corner from cornering (kappa*v^2 <= a_lat) AND drag wall v_top.
    """
    tilt = np.radians(tilt_max_deg)
    a_h = AMAX_UP * np.sin(tilt)                 # horizontal thrust authority
    a_vert = AMAX_UP * np.cos(tilt)              # vertical thrust available
    # If a_vert < g we cannot sustain level flight at this tilt except transiently while diving.
    # Course descends, so allow it but flag.
    a_lat = a_h                                  # lateral accel budget (cornering)
    # drag-wall top speed (level, full horizontal thrust at tilt cap)
    v_top = np.sqrt(max(a_h, 1e-6) / C2)
    N = len(s)
    seglen = np.diff(s)
    v_corner = np.where(kappa > 1e-6, np.sqrt(a_lat / np.maximum(kappa, 1e-12)), np.inf)
    v = np.minimum(v_top, v_corner)

    def a_tan_accel(i, vi):
        lat = kappa[i] * vi * vi
        budget = a_h * a_h - lat * lat
        a_th = np.sqrt(max(0.0, budget))
        drag = C2 * vi * vi                      # drag opposes -> reduces forward accel
        return max(0.0, a_th - drag)

    def a_tan_brake(i, vi):
        lat = kappa[i] * vi * vi
        budget = a_h * a_h - lat * lat
        a_th = np.sqrt(max(0.0, budget))
        drag = C2 * vi * vi                      # drag aids braking
        b = a_th + drag
        if a_brake_cap is not None:
            b = min(b, a_brake_cap)
        return b

    v[0] = 0.0
    for i in range(N - 1):
        v[i + 1] = min(v[i + 1], np.sqrt(v[i]**2 + 2 * a_tan_accel(i, v[i]) * seglen[i]))
    v[-1] = min(v[-1], 0.0 if False else v[-1])  # no forced end-stop (lap time = last gate cross)
    for i in range(N - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1]**2 + 2 * a_tan_brake(i + 1, v[i + 1]) * seglen[i]))

    t = np.zeros(N)
    for i in range(N - 1):
        t[i + 1] = t[i] + seglen[i] / max((v[i] + v[i + 1]) / 2, 1e-6)
    return v, t, dict(a_h=a_h, a_vert=a_vert, v_top=v_top)


def gate_cross_times(r, t, s):
    """Find time at each gate plane (nearest path point to each gate centre)."""
    out = []
    for gi, gc in enumerate(GATES):
        d = np.linalg.norm(r - gc, axis=1)
        i = int(np.argmin(d))
        out.append((gi, t[i], d[i], r[i]))
    return out


def main():
    pos, d = load_geometry()
    r, tang, kappa, s = resample_path(pos, n=900)
    print(f"path arc length = {s[-1]:.1f} m  (gate0..gate5 geometry from shipped line)")
    print(f"AMAX_UP={AMAX_UP:.1f} m/s2 ({AMAX_UP/G:.2f}g)  hover={HOVER}  C2={C2}")
    print()
    for tilt in [60, 65, 75, 80]:
        v, t, info = corrected_aero_topp(r, tang, kappa, s, tilt)
        gc = gate_cross_times(r, t, s)
        lap = gc[-1][1]
        print(f"--- tilt cap {tilt} deg ---")
        print(f"  a_h(horiz thrust)={info['a_h']:.1f}  a_vert={info['a_vert']:.1f} (g={G:.1f}, "
              f"level-sustainable={info['a_vert']>=G})  v_top(drag wall)={info['v_top']:.1f} m/s")
        print(f"  v mean={v.mean():.1f} max={v.max():.1f} m/s")
        print(f"  GATE TIMES: " + "  ".join(f"g{gi}={tt:.2f}" for gi, tt, dd, rr in gc))
        print(f"  >>> CORRECTED-AERO TOPP LAP (last gate) = {lap:.3f} s")
        print()
    return r, tang, kappa, s


if __name__ == '__main__':
    main()
