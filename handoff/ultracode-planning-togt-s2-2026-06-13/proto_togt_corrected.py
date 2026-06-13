"""C2 -- Re-evaluate the EXISTING (linear-plant) TOGT trajectories under the CORRECTED-AERO plant.

NO C++ re-run. Pure offline analysis of the shipped TOGT CSVs against the measured convex
collective->accel map + quadratic body-frame drag from racer.rl_plant.

Run:  PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-planning-togt-s2-2026-06-13/proto_togt_corrected.py

What it answers (Fengyou's gap question):
  The shipped 4.55 s reference line + the 4.27 s planning-valid bound were BOTH built on the
  FALSIFIED linear plant (T/W 3.765 LINEAR, linear_drag 0.21 isotropic world-frame, omega_max
  [11,11,7]). The corrected aero is (a) a CONVEX collective map giving ~78.3 m/s^2 (~8 g) at full
  stick -- ~8.5x the old 3.765 g ceiling -- so the line is THRUST-feasible with huge headroom; and
  (b) a QUADRATIC body-frame drag ~0.052/m -- ~2.2x the linear model at 9 m/s, scaling as v^2 ->
  a drag WALL at ~39 m/s. Net effect on the SAME geometry: who wins, thrust headroom or drag wall?

Three independent lenses, all measured from the CSVs:
  LENS 1  Per-node THRUST feasibility: is the TOGT-demanded collective specific accel inside the
          corrected convex map's available accel? (Expect: yes, with massive headroom.)
  LENS 2  DRAG re-accounting: along the TOGT path, recompute the world specific force the corrected
          plant produces given the SAME attitude + the SAME (full-precision) collective the planner
          demanded, and compare the implied tangential accel to what the planned kinematics need.
          Quantifies where the corrected drag eats / frees acceleration.
  LENS 3  PATH-FOLLOWING TOPP re-time: take the TOGT GEOMETRY (x,y,z path) and re-solve the
          time-optimal speed profile under a corrected-aero acceleration envelope (collective
          authority for thrust along body-up, tilt for lateral, MINUS quadratic drag). This is the
          honest "corrected achievable time for the EXISTING geometry" -- reconciled vs the
          exploratory 4.71 s figure (which re-PLANNED the geometry under T/W 8 + flat quad 0.052).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# corrected-aero constants -- the canonical measured plant
from racer.rl_plant import (
    COLL_MAP_THR_MEASURED,
    COLL_MAP_ACCEL_MEASURED,
    QUAD_DRAG_C2_MEASURED,
    QUAD_DRAG_C2_POOLED,
)

G = 9.80665
MASS = 1.0  # TOGT quad mass (peregrine_quad.yaml)
HOVER_THRUST = 0.2656

ROOT = Path("C:/Users/Fengy/Downloads/Projects/Anduril")
CASES_DIR = ROOT / "handoff/laptop-togt-bound-2026-06-10/cases"
OUT_MD = ROOT / "handoff/ultracode-planning-togt-s2-2026-06-13/artifacts/C2_togt_corrected.md"

# TOGT thrust ceiling (linear plant): thrust_max 9.2305 N / mass = 9.2305 m/s^2 = 3.765 g
TOGT_THRUST_MAX_N = 9.2305
TOGT_TW = 3.765
TOGT_LINEAR_DRAG = 0.21  # 1/s isotropic world-frame

# Corrected convex map: full-stick body-up accel and hover knot
COLL_FULL_STICK_ACCEL = float(COLL_MAP_ACCEL_MEASURED[-1])   # ~78.28 m/s^2
COLL_HOVER_ACCEL = float(np.interp(HOVER_THRUST, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED))


# ----------------------------------------------------------------------------- IO + frame helpers
def load_csv(path: Path):
    """Load a TOGT/refined CSV into a dict of float arrays keyed by header column name."""
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = [[float(x) for x in r] for r in rdr if r]
    arr = np.asarray(rows, dtype=np.float64)
    return {h: arr[:, i] for i, h in enumerate(header)}, header


def quat_rotate_wxyz(q, v):
    """Rotate body vector v into world by quaternion q (wxyz, R_world_body). q (N,4), v (3,) -> (N,3)."""
    w = q[:, 0:1]
    u = q[:, 1:4]
    v = np.broadcast_to(np.asarray(v, float), (q.shape[0], 3))
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def quat_rotate_inv_wxyz(q, v):
    """Rotate world vector v (N,3) into body frame by q (N,4)."""
    qc = q.copy()
    qc[:, 1:4] *= -1.0
    w = qc[:, 0:1]
    u = qc[:, 1:4]
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def togt_to_ned(p):
    """TOGT frame (x along-course, y lateral, z UP) -> world NED (z DOWN). Flip z.
    Verified: TOGT init p_z=-0.02 ascends to z<0 as course DESCENDS; NED course descends into +Z."""
    out = p.copy()
    out[..., 2] *= -1.0
    return out


# ----------------------------------------------------------------------------- corrected-aero pieces
def coll_accel_from_norm(coll_norm):
    """Corrected convex map: normalised collective [0,1] -> body-up specific accel (m/s^2)."""
    return np.interp(coll_norm, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)


def togt_coll_norm_from_thrust_N(thrust_N):
    """TOGT per-rotor sum (N) on a 1 kg quad -> the normalised collective stick that the CORRECTED
    convex map would need to MATCH that same body-up specific accel (a = thrust_N / mass).
    i.e. invert the convex map at the demanded accel. Used to place the demand on the stick axis."""
    a_demand = thrust_N / MASS
    return np.interp(a_demand, COLL_MAP_ACCEL_MEASURED, COLL_MAP_THR_MEASURED)


def quad_drag_accel_body(v_body, c2=QUAD_DRAG_C2_MEASURED):
    """Body-frame quadratic drag specific accel (N,3) given body-frame velocity (N,3).
    a_drag_body = -c2[sign] * |v_b| * v_b, per axis, sign-split coefficient."""
    c = np.where(v_body >= 0.0, c2[:, 0], c2[:, 1])  # (3,) broadcast per-axis
    return -(c * np.abs(v_body) * v_body)


# ============================================================================= LENS 1: thrust feasibility
def lens1_thrust_feasibility(d):
    """Per-node: TOGT-demanded collective specific accel vs corrected convex-map available accel."""
    thrust_N = d["u_1"] + d["u_2"] + d["u_3"] + d["u_4"]   # total thrust (N) on 1 kg
    a_demand = thrust_N / MASS                              # body-up specific accel demanded (m/s^2)
    # stick the planner would have ridden under the OLD linear map: thrust_N / thrust_max
    stick_old = thrust_N / TOGT_THRUST_MAX_N
    # stick needed under the CORRECTED convex map to produce the same a_demand
    stick_corr = togt_coll_norm_from_thrust_N(thrust_N)
    headroom_accel = COLL_FULL_STICK_ACCEL - a_demand      # m/s^2 of unused climb authority
    return {
        "a_demand_max": float(a_demand.max()),
        "a_demand_mean": float(a_demand.mean()),
        "a_demand_p95": float(np.percentile(a_demand, 95)),
        "stick_old_max": float(stick_old.max()),
        "stick_old_p84": float(np.percentile(stick_old, 84)),  # ~84% sat reported in analysis.json
        "stick_corr_max": float(stick_corr.max()),
        "stick_corr_mean": float(stick_corr.mean()),
        "stick_corr_p95": float(np.percentile(stick_corr, 95)),
        "frac_demand_above_corr_fullstick": float((a_demand > COLL_FULL_STICK_ACCEL).mean()),
        "headroom_accel_min": float(headroom_accel.min()),
        "coll_full_stick_accel": COLL_FULL_STICK_ACCEL,
        "_a_demand": a_demand,
        "_thrust_N": thrust_N,
    }


# ============================================================================= LENS 2: drag re-accounting
def lens2_drag_reaccount(d, has_vel=True):
    """Along the TOGT path: with the SAME attitude + SAME demanded collective, what world specific
    force does the CORRECTED plant produce, vs the OLD linear plant? Decompose the tangential
    component (the part that sets speed -> lap time).

    Both plants share: thrust magnitude along body-up (we hold the planner's demanded total accel),
    gravity. They DIFFER in the drag model:
      OLD:  a_drag_world = -0.21 * v_world                       (linear, isotropic)
      CORR: a_drag_body  = -c2[sign]*|v_b|*v_b  (per-axis, then rotated to world)
    AND in the available collective accel (LENS 1 covers the climb headroom). Here we hold thrust =
    the planner's demand (so we compare drag at the SAME operating point) and report the tangential
    drag deficit -> the extra braking the corrected plant imposes (slower) or the cruise it allows.
    """
    q = np.column_stack([d["q_w"], d["q_x"], d["q_y"], d["q_z"]])
    # velocities are in the TOGT frame; convert to NED is unnecessary for drag MAGNITUDES along the
    # path tangent, but the body frame must be consistent. The quaternion is R_world_body in the
    # TOGT world frame; v is in the same TOGT world frame -> body velocity is frame-consistent.
    v_world = np.column_stack([d["v_x"], d["v_y"], d["v_z"]])
    speed = np.linalg.norm(v_world, axis=1)
    tang = np.zeros_like(v_world)
    nz = speed > 1e-6
    tang[nz] = v_world[nz] / speed[nz, None]

    # body-frame velocity (for the corrected per-axis quadratic drag)
    v_body = quat_rotate_inv_wxyz(q, v_world)

    # OLD linear drag (world)
    a_drag_old_world = -TOGT_LINEAR_DRAG * v_world
    # CORRECTED quadratic drag: compute in body, rotate to world
    a_drag_corr_body = quad_drag_accel_body(v_body, QUAD_DRAG_C2_MEASURED)
    a_drag_corr_world = quat_rotate_wxyz(q, a_drag_corr_body)

    # tangential components (signed: negative = retards motion)
    tan_old = np.sum(a_drag_old_world * tang, axis=1)
    tan_corr = np.sum(a_drag_corr_world * tang, axis=1)
    deficit = tan_corr - tan_old   # negative -> corrected brakes HARDER than old (extra slowdown)

    mag_old = np.linalg.norm(a_drag_old_world, axis=1)
    mag_corr = np.linalg.norm(a_drag_corr_world, axis=1)

    # speed-bucketed drag ratio (corrected / old) to show the v^2 crossover
    buckets = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 60)]
    ratio_by_speed = {}
    for lo, hi in buckets:
        m = (speed >= lo) & (speed < hi)
        if m.sum() > 0:
            ratio_by_speed[f"{lo}-{hi}"] = {
                "n": int(m.sum()),
                "mag_old_mean": float(mag_old[m].mean()),
                "mag_corr_mean": float(mag_corr[m].mean()),
                "ratio": float(mag_corr[m].mean() / max(mag_old[m].mean(), 1e-9)),
            }
    return {
        "max_speed": float(speed.max()),
        "mean_speed": float(speed.mean()),
        "tan_drag_old_mean": float(tan_old[nz].mean()),
        "tan_drag_corr_mean": float(tan_corr[nz].mean()),
        "tan_deficit_mean": float(deficit[nz].mean()),
        "tan_deficit_min": float(deficit[nz].min()),   # most extra braking at high speed
        "drag_ratio_by_speed": ratio_by_speed,
        "_speed": speed,
    }


# ============================================================================= LENS 3: TOPP re-time
def topp_retime(path_xyz, a_lat_cap, a_long_accel_fn, a_long_brake_fn, kappa_cap=0.5):
    """Time-optimal speed profile along a FIXED geometry under a DECOUPLED accel model.

    Correct physics (vs the broken v1 that charged full v^2-drag against the whole budget):
      * LATERAL (turning) authority comes from TILT of the thrust vector: a_lat_cap (constant ~A_h,
        the horizontal thrust component at full stick). Drag does NOT reduce turning authority.
      * LONGITUDINAL accel/brake are speed-dependent:
          - accel(v): thrust pushes forward, drag opposes -> a_long_accel_fn(v) (drag SUBTRACTED).
          - brake(v): thrust can point backward AND drag adds -> a_long_brake_fn(v) (drag ADDED).
      * Standard friction-circle coupling on the TANGENTIAL budget: with lateral demand
        lat = kappa*v^2 (<= a_lat_cap), the usable tangential magnitude is
        sqrt(cap^2 - lat^2) where cap is the relevant longitudinal cap.

    This is the honest corrected-aero envelope: thrust headroom raises the forward push, the v^2
    drag wall caps top speed (accel(v)->0), and drag HELPS braking into the gates.

    Uses the planner's own dense polyline directly (finite-difference curvature) to avoid the
    spurious curvature of a coarse re-spline. kappa_cap clamps numerical curvature spikes (the
    physical course has gentle turns; spacing >=24 m, lateral offsets <~5 m).
    """
    wp = np.asarray(path_xyz, float)
    # dense, smooth chord-length spline of the planner geometry (it is already smooth + dense)
    from scipy.interpolate import CubicSpline
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    wp = wp[keep]
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    u_wp = np.concatenate([[0.0], np.cumsum(seg)])
    cs = CubicSpline(u_wp, wp, bc_type="natural")
    N = 2000
    u = np.linspace(0.0, u_wp[-1], N)
    d1 = cs(u, 1)
    d2 = cs(u, 2)
    ds_du = np.maximum(np.linalg.norm(d1, axis=1), 1e-9)
    kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / ds_du ** 3
    kappa = np.minimum(kappa, kappa_cap)        # clamp numerical spikes
    s = np.concatenate([[0.0], np.cumsum((ds_du[:-1] + ds_du[1:]) / 2 * np.diff(u))])
    seglen = np.diff(s)
    total_len = float(s[-1])

    # cornering ceiling: kappa * v^2 <= a_lat_cap  -> v_corner = sqrt(a_lat_cap / kappa)
    v_corner = np.where(kappa > 1e-6, np.sqrt(a_lat_cap / np.maximum(kappa, 1e-12)), 1e3)
    v = np.minimum(v_corner, 200.0)

    def a_tan_fwd(i, vi):
        lat = kappa[i] * vi * vi
        cap = a_long_accel_fn(vi)
        return float(np.sqrt(max(0.0, cap * cap - min(lat, cap) ** 2)))

    def a_tan_brk(i, vi):
        lat = kappa[i] * vi * vi
        cap = a_long_brake_fn(vi)
        return float(np.sqrt(max(0.0, cap * cap - min(lat, cap) ** 2)))

    # forward accel pass (start from rest)
    v[0] = 0.0
    for i in range(N - 1):
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2.0 * a_tan_fwd(i, v[i]) * seglen[i]))
    # backward brake pass (finish at rest) -- uses the BRAKE cap (drag helps)
    v[-1] = 0.0
    for i in range(N - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_tan_brk(i + 1, v[i + 1]) * seglen[i]))

    t = 0.0
    for i in range(N - 1):
        vbar = max((v[i] + v[i + 1]) / 2.0, 1e-3)
        t += seglen[i] / vbar
    return {
        "time_s": float(t),
        "path_len_m": total_len,
        "v_max": float(v.max()),
        "v_mean": float(v.mean()),
        "kappa_max": float(kappa.max()),
        "v_corner_min": float(v_corner.min()),
    }


# ============================================================================= driver
def analyze_case(case):
    cdir = CASES_DIR / case
    out = {"case": case}
    # ---- TOGT (planner) trajectory
    togt, _ = load_csv(cdir / "togt_traj.csv")
    out["togt"] = {
        "n": len(togt["t"]),
        "duration_s": float(togt["t"][-1]),
        "lens1": {k: v for k, v in lens1_thrust_feasibility(togt).items() if not k.startswith("_")},
        "lens2": {k: v for k, v in lens2_drag_reaccount(togt).items() if not k.startswith("_")},
    }
    # ---- refined (IPOPT) trajectory, if present
    rpath = cdir / "refined_traj.csv"
    if rpath.exists():
        ref, _ = load_csv(rpath)
        out["refined"] = {
            "n": len(ref["t"]),
            "duration_s": float(ref["t"][-1]),
            "lens1": {k: v for k, v in lens1_thrust_feasibility(ref).items() if not k.startswith("_")},
            "lens2": {k: v for k, v in lens2_drag_reaccount(ref).items() if not k.startswith("_")},
        }
    return out, togt


def main():
    cases = ["bound_nominal", "bound_free", "ref_margin", "bound_nodrag"]
    results = {}
    geom_for_retime = None
    for c in cases:
        res, togt = analyze_case(c)
        results[c] = res
        if c == "bound_nominal":
            geom_for_retime = togt

    # ---- LENS 3: TOPP re-time of the bound_nominal GEOMETRY under corrected aero ----
    # geometry in NED (z flip). Use the planner's dense path positions.
    p_togt = np.column_stack([geom_for_retime["p_x"], geom_for_retime["p_y"], geom_for_retime["p_z"]])
    p_ned_full = togt_to_ned(p_togt)
    # TRIM to the LAP: the planner path overshoots past gate-6 (x=-159.2) to a terminal hover at
    # x=-184; the lap ENDS at the gate-6 crossing (t=5.0657 s, idx 507). Including the post-finish
    # stopping turn injects a spurious tight curve (R~0.5 m) that is NOT part of the race line.
    t_g6 = 5.065659469481976
    i_g6 = int(np.searchsorted(geom_for_retime["t"], t_g6))
    p_ned = p_ned_full[: i_g6 + 1]

    # Corrected-aero caps (decoupled friction-circle model):
    #   thrust magnitude full-stick A = 78.28 m/s^2. Reserve g for altitude hold on the (mostly
    #   level/descending) course -> horizontal tilt authority A_h = sqrt(A^2 - g^2) for turning.
    A = COLL_FULL_STICK_ACCEL
    A_h = float(np.sqrt(A * A - G * G))                 # lateral (turning) cap ~77.7
    # longitudinal accel: forward thrust component minus drag (pooled quad c2); thrust forward
    # available ~A_h too (tilt forward). drag = c2*v^2 opposes -> accel(v) = A_h - c2 v^2.
    a_accel = lambda v: max(0.0, A_h - QUAD_DRAG_C2_POOLED * v * v)
    # longitudinal brake: thrust backward (A_h) PLUS drag helps -> brake(v) = A_h + c2 v^2.
    a_brake = lambda v: A_h + QUAD_DRAG_C2_POOLED * v * v
    retime_corr = topp_retime(p_ned, a_h := A_h, a_accel, a_brake)

    # Reconcile vs the EXPLORATORY 4.71 s case: it re-PLANNED with FLAT T/W 8 (=78.5 m/s^2) + flat
    # quad 0.052 (ball gate). Emulate that envelope on this geometry: A8=8g, same decoupled model.
    A8 = 8.0 * G
    A8_h = float(np.sqrt(A8 * A8 - G * G))
    retime_expl = topp_retime(
        p_ned, A8_h,
        lambda v: max(0.0, A8_h - QUAD_DRAG_C2_POOLED * v * v),
        lambda v: A8_h + QUAD_DRAG_C2_POOLED * v * v,
    )

    # OLD linear plant on the SAME geometry: flat a_max = 3.765 g, linear world drag 0.21/s.
    #   accel(v) = 3.765g*... here the old planner's authority is the thrust ceiling minus linear
    #   drag a_drag = 0.21*v (tiny). lateral cap = sqrt((3.765g)^2 - g^2) (reserve g).
    Aold = TOGT_TW * G
    Aold_h = float(np.sqrt(Aold * Aold - G * G))
    retime_old = topp_retime(
        p_ned, Aold_h,
        lambda v: max(0.0, Aold_h - TOGT_LINEAR_DRAG * v),
        lambda v: Aold_h + TOGT_LINEAR_DRAG * v,
    )

    #  isolate the convex-map headroom: corrected thrust, NO drag at all.
    retime_corr_nodrag = topp_retime(p_ned, A_h, lambda v: A_h, lambda v: A_h)

    results["lens3_topp_retime"] = {
        "geometry": "bound_nominal TOGT path (NED) re-timed; decoupled friction-circle envelope",
        "note": ("TOPP on a fixed geometry is OPTIMISTIC vs TOGT/IPOPT (it ignores jerk/attitude-rate "
                 "transients and treats accel as instantaneous-direction). Use the RATIOS / deltas, "
                 "not the absolute seconds, as the corrected-vs-old signal. Reconcile retime_expl vs "
                 "the shipped expl_corrected_aero refined 4.71 s."),
        "lateral_cap_A_h_mps2": A_h,
        "corrected_aero": retime_corr,
        "exploratory_TW8_flat": retime_expl,
        "old_linear_3p765g": retime_old,
        "corrected_thrust_nodrag": retime_corr_nodrag,
    }

    # ---- margin-cost reconciliation ----
    # refined lap times from the shipped analysis.json (the canonical reported numbers)
    laps = {}
    for c in cases + ["expl_corrected_aero"]:
        aj = CASES_DIR / c / "analysis.json"
        if aj.exists():
            a = json.load(open(aj))
            laps[c] = {
                "togt_init_lap_s": a["togt_init"]["lap_time_s"],
                "refined_lap_s": a["refined"]["lap_time_s"] if "refined" in a else None,
                "refined_max_speed": a["refined"]["max_speed_mps"] if "refined" in a else None,
                "refined_collective_sat": a["refined"]["frac_nodes_collective_sat"] if "refined" in a else None,
            }
    results["shipped_laps"] = laps

    # constants for the report
    results["constants"] = {
        "togt_thrust_ceiling_mps2": TOGT_THRUST_MAX_N / MASS,
        "togt_TW_g": TOGT_TW,
        "togt_linear_drag_per_s": TOGT_LINEAR_DRAG,
        "corrected_full_stick_accel_mps2": COLL_FULL_STICK_ACCEL,
        "corrected_full_stick_g": COLL_FULL_STICK_ACCEL / G,
        "corrected_hover_accel_mps2": COLL_HOVER_ACCEL,
        "quad_drag_pooled_per_m": QUAD_DRAG_C2_POOLED,
        "thrust_headroom_x": COLL_FULL_STICK_ACCEL / (TOGT_THRUST_MAX_N / MASS),
    }

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
