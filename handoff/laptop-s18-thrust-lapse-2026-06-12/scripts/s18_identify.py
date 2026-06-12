"""S18 step 1b -- (A) the actual standing-start gate-0 approach regime (post-fix rollfix
runs, tick by tick) and (B) a JOINT drag-scale x lapse identifiability fit, to quantify how
much the low-speed thrust deficit trades against drag.

(B) model per smooth tick i (forward flight, exclude fast descent):
    a_meas_i - g = L(|v|) * Km_i * b3_i  -  s_drag * a_drag_meas_i  +  noise
We fit a piecewise-constant L over |v| bins AND a single global drag scale s_drag by
alternating least squares, then report L under s_drag in {0.5,1,1.5,2} to show the trade.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from racer.rl_plant import (COLL_MAP_ACCEL_MEASURED, COLL_MAP_THR_MEASURED,
                            QUAD_DRAG_C2_MEASURED)
G = 9.80665; D = 2
DATA = ROOT / "handoff" / "shadowpc-refit-dataset-2026-06-12" / "extracted"


def load(p):
    return [json.loads(l) for l in open(p / "debug_obs.jsonl", encoding="utf-8")
            if json.loads(l).get("type") != "header"]


def Rm(q):
    q = np.asarray(q, float); return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


K_of = lambda c: np.interp(c, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)

print("=== (A) standing-start gate-0 approach, post-fix rollfix_f1 (tick 0..60) ===")
p = DATA / "20260612_044219_inc6_rollfix_f1"
rows = load(p)
v = np.array([r["vel_ned"] for r in rows], float)
pos = np.array([r["pos_ned"] for r in rows], float)
print(f"{'k':>3} {'sp':>5} {'vN':>6} {'vE':>6} {'vD':>6} {'coll':>5} {'pitch':>6} {'v_ax':>6}")
for k in range(0, 61, 4):
    R = Rm(rows[k]["q_raw_wxyz"])
    vb = R.T @ v[k]
    pitch = np.degrees(np.arcsin(-R[2, 0]))
    print(f"{k:>3} {np.linalg.norm(v[k]):>5.1f} {v[k][0]:>6.1f} {v[k][1]:>6.1f} {v[k][2]:>6.1f} "
          f"{rows[k]['collective']:>5.2f} {pitch:>6.1f} {-vb[2]:>6.1f}")

# ---- build sample table (forward flight, exclude fast descent v_ax<-2) ----
S = []
for p in sorted(DATA.glob("20260612_*_inc6_*")):
    rows = load(p)
    if len(rows) < 30:
        continue
    t = np.array([r["t_mono"] for r in rows]); v = np.array([r["vel_ned"] for r in rows], float)
    for k in range(2, len(rows) - 2):
        if np.linalg.norm(rows[k]["w_raw"]) > 1.0:
            continue
        dt = t[k + 1] - t[k - 1]
        if not (0.05 < dt <= 0.09):
            continue
        c = rows[max(k - D, 0)]["collective"]; Km = K_of(c)
        if Km < 3.0:
            continue
        R = Rm(rows[k]["q_raw_wxyz"]); b3 = -R[:, 2]; vv = v[k]; vb = R.T @ vv
        if -vb[2] < -2.0:        # exclude fast descent (vortex-ring-ish, failure transients)
            continue
        c2 = np.asarray(QUAD_DRAG_C2_MEASURED, float)
        c2v = np.where(vb >= 0, c2[:, 0], c2[:, 1])
        a_drag = R @ (-c2v * np.abs(vb) * vb)
        a = (v[k + 1] - v[k - 1]) / dt - np.array([0, 0, G])
        S.append((np.linalg.norm(vv), Km, float(a @ b3), float(a_drag @ b3), Km, b3, a, a_drag))
S = np.array([(s[0], s[1], s[2], s[3]) for s in S])
sp, Km, a_b3, adrag_b3 = S.T
print(f"\n=== (B) joint identifiability fit, {len(S)} forward/non-descent samples ===")
bins = [3, 6, 9, 12, 15, 18, 24]
print("L(|v|) per bin under assumed global drag-scale s_drag:")
print(f"{'band':>9} {'n':>5} " + " ".join(f"s={s:<4}" for s in (0.5, 1.0, 1.5, 2.0)))
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (sp >= lo) & (sp < hi)
    if m.sum() < 10:
        continue
    # a_b3 = L*Km - s*adrag_b3  => L = (a_b3 + s*adrag_b3)/Km  (per-tick, take median)
    Ls = []
    for s in (0.5, 1.0, 1.5, 2.0):
        L = (a_b3[m] + s * adrag_b3[m]) / Km[m]
        Ls.append(np.median(L))
    print(f"{lo:>3}-{hi:<4} {int(m.sum()):>5} " + " ".join(f"{x:>5.2f}" for x in Ls))

# global drag scale that best flattens L->1 at high speed (12-18, where lapse~recovered)
mh = (sp >= 12) & (sp < 18)
print(f"\nHigh-speed band 12-18 (n={int(mh.sum())}): drag scale s that makes median L=1.0:")
from numpy import median
for s in (0.5, 1.0, 1.5, 2.0, 2.5):
    print(f"  s={s}: median L = {median((a_b3[mh]+s*adrag_b3[mh])/Km[mh]):.2f}")
