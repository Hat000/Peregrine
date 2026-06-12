"""Thrust-lapse curve: K_eff/K_model vs airspeed, smooth ticks only (|w|<1 rad/s),
across all 2026-06-12 inc6 flights (buggy + rollfix -- the physics is the same).

K_eff = (a_meas - g*z - a_drag_model) . b3 ;  ratio = K_eff / K_model(c_applied).
Binned by |v|. Also same with drag REMOVED from the model (attribute everything to
the thrust axis) and a fit of ratio ~ 1/(1 + beta*v_axial^2/K) styles left to the
laptop; here just the empirical curve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.spatial.transform import Rotation

from racer.rl_plant import (COLL_MAP_ACCEL_MEASURED, COLL_MAP_THR_MEASURED,
                            QUAD_DRAG_C2_MEASURED)

G = 9.80665
D = 2

runs = sorted((ROOT / "data" / "runs").glob("20260612_*_inc6_*_f*"))


def load_run(p):
    rows = []
    f = p / "debug_obs.jsonl"
    if not f.exists():
        return []
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("type") != "header":
                rows.append(r)
    return rows


def K_of(c):
    return np.interp(c, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)


samples = []   # (speed, v_axial, K_model, K_eff)
for p in runs:
    rows = load_run(p)
    if len(rows) < 30:
        continue
    t = np.array([r["t_mono"] for r in rows])
    v = np.array([r["vel_ned"] for r in rows])
    for k in range(2, len(rows) - 2):
        w = np.linalg.norm(rows[k]["w_raw"])
        if w > 1.0:
            continue
        dt = t[k + 1] - t[k - 1]
        if dt <= 0.05 or dt > 0.09:
            continue
        c = rows[max(k - D, 0)]["collective"]
        Km = K_of(c)
        if Km < 3.0:
            continue                      # near-idle: ratio ill-conditioned
        a_meas = (v[k + 1] - v[k - 1]) / dt
        q = np.array(rows[k]["q_raw_wxyz"], dtype=np.float64)
        Rm = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        b3 = -Rm[:, 2]
        vv = v[k]
        v_body = Rm.T @ vv
        c2 = np.asarray(QUAD_DRAG_C2_MEASURED, dtype=np.float64)
        if c2.ndim == 2:
            c2 = np.where(v_body >= 0, c2[:, 0], c2[:, 1])
        a_drag = Rm @ (-c2 * np.abs(v_body) * v_body)
        K_eff = float((a_meas - np.array([0, 0, G]) - a_drag) @ b3)
        v_ax = float(-v_body[2])          # axial inflow component (+ = climbing through rotor)
        samples.append((float(np.linalg.norm(vv)), v_ax, Km, K_eff))

samples = np.array(samples)
print(f"{len(samples)} smooth samples across {len(runs)} runs")
bins = [0, 3, 6, 9, 12, 15, 18, 22, 26, 31]
print(f"{'|v| bin':>10} {'n':>5} {'K_model med':>11} {'K_eff med':>10} {'ratio med':>9} {'ratio p10/p90':>14}")
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (samples[:, 0] >= lo) & (samples[:, 0] < hi)
    if m.sum() < 10:
        continue
    r = samples[m, 3] / samples[m, 2]
    print(f"{lo:>4}-{hi:<5} {int(m.sum()):>5} {np.median(samples[m,2]):>11.1f} "
          f"{np.median(samples[m,3]):>10.1f} {np.median(r):>9.2f} "
          f"{np.percentile(r,10):>6.2f}/{np.percentile(r,90):<6.2f}")

# axial-speed dependence at fixed speed band 12-22 (forward flight)
m = (samples[:, 0] >= 12) & (samples[:, 0] < 22)
if m.sum() > 50:
    va = samples[m, 1]
    r = samples[m, 3] / samples[m, 2]
    for lo, hi in [(-8, -4), (-4, -1), (-1, 1), (1, 4), (4, 8)]:
        mm = (va >= lo) & (va < hi)
        if mm.sum() >= 10:
            print(f"  v_axial {lo:>3}..{hi:<3} n={int(mm.sum()):>4} ratio med {np.median(r[mm]):.2f}")
