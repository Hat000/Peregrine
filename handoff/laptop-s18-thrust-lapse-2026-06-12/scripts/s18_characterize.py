"""S18 step 1 -- CHARACTERIZE the residual translational gap from 17 inc6 live recordings.

Goal: see the empirical thrust-lapse curve and check lapse-vs-drag identifiability
BEFORE choosing a model form. Reproduces diag_thrust_lapse.py but adds:
  - both pre-fix and post-fix runs, tagged
  - ratio vs |v|, vs v_axial (climb-through-rotor), vs v_edge (in-plane airspeed)
  - ratio vs collective level (multiplicative? additive?)
  - off-axis residual magnitude (drag-consistency / attitude check)
  - frozen-telemetry detection (drop stalled-clock ticks)
  - drag-assumption sensitivity: K_eff curve under {measured drag, zero drag, 2x drag}

Pristine inputs (valid pre- AND post-fix): pos_ned, vel_ned (sim-given),
q_raw_wxyz (true attitude as-is, bcc93f9), collective (wired thrust scalar, sign-mirror
does NOT touch the thrust channel). w_raw used only to gate "smooth" ticks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from racer.rl_plant import (COLL_MAP_ACCEL_MEASURED, COLL_MAP_THR_MEASURED,
                            QUAD_DRAG_C2_MEASURED)

G = 9.80665
D = 2  # transport delay (ticks) command->realized, measured live

DATA = ROOT / "handoff" / "shadowpc-refit-dataset-2026-06-12" / "extracted"
POSTFIX = {"rollfix"}  # name token => post-bcc93f9


def load_run(p):
    rows = []
    with open(p / "debug_obs.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("type") != "header":
                rows.append(r)
    return rows


def K_of(c):
    return np.interp(c, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)


def Rmat(q):
    q = np.asarray(q, float)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


runs = sorted(DATA.glob("20260612_*_inc6_*"))
print(f"{len(runs)} runs")

# columns: speed, v_ax, v_edge, K_model, K_eff(measured drag), K_eff(zero drag),
#          K_eff(2x drag), offaxis_resid, collective, postfix
S = []
n_frozen = 0
for p in runs:
    rows = load_run(p)
    if len(rows) < 30:
        continue
    post = 1.0 if any(t in p.name for t in POSTFIX) else 0.0
    t = np.array([r["t_mono"] for r in rows])
    st = np.array([r["sim_time_ns"] for r in rows], dtype=np.float64)
    v = np.array([r["vel_ned"] for r in rows], dtype=np.float64)
    for k in range(2, len(rows) - 2):
        if np.linalg.norm(rows[k]["w_raw"]) > 1.0:
            continue
        dt = t[k + 1] - t[k - 1]
        if dt <= 0.05 or dt > 0.09:
            continue
        # frozen telemetry: sim clock not advancing across the window
        if st[k + 1] - st[k - 1] < 1e6:  # < 1 ms of sim time over 2 ticks
            n_frozen += 1
            continue
        c = rows[max(k - D, 0)]["collective"]
        Km = K_of(c)
        if Km < 3.0:
            continue
        a_meas = (v[k + 1] - v[k - 1]) / dt
        R = Rmat(rows[k]["q_raw_wxyz"])
        b3 = -R[:, 2]                      # body-up in world
        vv = v[k]
        v_body = R.T @ vv
        c2 = np.asarray(QUAD_DRAG_C2_MEASURED, float)
        c2v = np.where(v_body >= 0, c2[:, 0], c2[:, 1])
        a_drag = R @ (-c2v * np.abs(v_body) * v_body)
        resid = a_meas - np.array([0, 0, G])          # = thrust + drag
        Keff_meas = float((resid - a_drag) @ b3)
        Keff_zero = float(resid @ b3)                 # drag attributed to thrust
        Keff_2x = float((resid - 2 * a_drag) @ b3)
        # off-axis residual after removing best b3 thrust + measured drag
        offax = resid - a_drag - Keff_meas * b3
        v_ax = float(-v_body[2])                       # +z body is down; -z = up = climb inflow
        v_edge = float(np.hypot(v_body[0], v_body[1]))
        S.append((np.linalg.norm(vv), v_ax, v_edge, Km, Keff_meas, Keff_zero,
                  Keff_2x, np.linalg.norm(offax), c, post))

S = np.array(S)
print(f"{len(S)} smooth samples; dropped {n_frozen} frozen-clock ticks")
sp, vax, vedge, Km, Kmeas, Kzero, K2x, offax, coll, post = S.T

print("\n=== ratio K_eff/K_model vs |v| (measured-drag attribution) ===")
print(f"{'|v|':>9} {'n':>5} {'Km_med':>7} {'ratio_meas':>10} {'ratio_zero':>10} {'ratio_2x':>9} {'offax_med':>9}")
bins = [0, 3, 6, 9, 12, 15, 18, 22, 26, 32]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (sp >= lo) & (sp < hi)
    if m.sum() < 10:
        continue
    print(f"{lo:>3}-{hi:<4} {int(m.sum()):>5} {np.median(Km[m]):>7.1f} "
          f"{np.median(Kmeas[m] / Km[m]):>10.2f} {np.median(Kzero[m] / Km[m]):>10.2f} "
          f"{np.median(K2x[m] / Km[m]):>9.2f} {np.median(offax[m]):>9.2f}")

print("\n=== pre-fix vs post-fix (measured-drag ratio, by |v|) ===")
for tag, mask in [("PRE ", post < 0.5), ("POST", post > 0.5)]:
    print(f" {tag}: n={int(mask.sum())}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = mask & (sp >= lo) & (sp < hi)
        if m.sum() < 10:
            continue
        print(f"   {lo:>3}-{hi:<4} n={int(m.sum()):>5} ratio {np.median(Kmeas[m] / Km[m]):.2f}")

print("\n=== ratio vs v_axial (climb inflow) at |v| 3-12 (the lapse band) ===")
mb = (sp >= 3) & (sp < 12)
for lo, hi in [(-12, -6), (-6, -3), (-3, 0), (0, 3), (3, 6), (6, 12)]:
    m = mb & (vax >= lo) & (vax < hi)
    if m.sum() >= 10:
        print(f"  v_ax {lo:>3}..{hi:<3} n={int(m.sum()):>4} ratio {np.median(Kmeas[m] / Km[m]):.2f}")

print("\n=== ratio vs v_edge (in-plane airspeed) at |v| 3-18 ===")
mb = (sp >= 3) & (sp < 18)
for lo, hi in [(0, 3), (3, 6), (6, 9), (9, 12), (12, 16)]:
    m = mb & (vedge >= lo) & (vedge < hi)
    if m.sum() >= 10:
        print(f"  v_edge {lo:>3}..{hi:<3} n={int(m.sum()):>4} ratio {np.median(Kmeas[m] / Km[m]):.2f}  Km_med {np.median(Km[m]):.1f}")

print("\n=== ratio vs collective level at |v| 3-12 (multiplicative check) ===")
mb = (sp >= 3) & (sp < 12)
for lo, hi in [(0.2, 0.32), (0.32, 0.45), (0.45, 0.6), (0.6, 1.01)]:
    m = mb & (coll >= lo) & (coll < hi)
    if m.sum() >= 10:
        print(f"  coll {lo:.2f}-{hi:.2f} n={int(m.sum()):>4} ratio {np.median(Kmeas[m] / Km[m]):.2f}  Km_med {np.median(Km[m]):.1f}")
