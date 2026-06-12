"""S18 step 1d -- lock the lapse knots; 3-fold CV mean-bias (stable metric);
descent-effect quantification (excluded from map, reported for DR)."""
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
K_of = lambda c: np.interp(c, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)

# ---- LOCKED LAPSE MODEL (multiplicative on a_up; np.interp clamps to endpoints) ----
LAPSE_V = np.array([0.0, 4.0, 8.0, 12.0, 15.0])
LAPSE_L = np.array([1.0, 0.78, 0.80, 0.92, 1.0])


def load(p):
    return [r for r in (json.loads(l) for l in open(p / "debug_obs.jsonl", encoding="utf-8"))
            if r.get("type") != "header"]


def Rm(q):
    q = np.asarray(q, float); return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def samples(run_paths, fwd_only=True):
    out = []
    for p in run_paths:
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
            vax = -vb[2]
            if fwd_only and vax < -2.0:
                continue
            c2 = np.asarray(QUAD_DRAG_C2_MEASURED, float)
            c2v = np.where(vb >= 0, c2[:, 0], c2[:, 1])
            a_drag = R @ (-c2v * np.abs(vb) * vb)
            a = (v[k + 1] - v[k - 1]) / dt - np.array([0, 0, G])
            out.append((np.linalg.norm(vv), Km, float(a @ b3), float(a_drag @ b3), vax))
    return np.array(out)


runs = sorted(DATA.glob("20260612_*_inc6_*"))
print("LOCKED lapse:", dict(zip(LAPSE_V, LAPSE_L)))
print(f"  L sampled @ |v|=[2,4,6,8,10,12,14]: {np.round(np.interp([2,4,6,8,10,12,14], LAPSE_V, LAPSE_L),3)}")

# ---- 3-fold CV: per-band mean b3-accel bias, base (S16) vs refit, averaged over folds ----
print("\n=== 3-fold CV mean b3-accel bias (m/s^2), forward flight ===")
rng = np.random.RandomState(1)
idx = rng.permutation(len(runs))
folds = [[runs[i] for i in idx[f::3]] for f in range(3)]
bands = [(3, 6), (6, 9), (9, 12), (12, 18)]
acc_base = {b: [] for b in bands}; acc_refit = {b: [] for b in bands}
for f in range(3):
    test = folds[f]
    St = samples(test)
    sp, Km, a_b3, adrag_b3, vax = St.T
    L = np.interp(sp, LAPSE_V, LAPSE_L)
    rb = a_b3 - (Km + adrag_b3)             # S16 baseline (no lapse)
    rf = a_b3 - (L * Km + adrag_b3)         # refit
    for b in bands:
        m = (sp >= b[0]) & (sp < b[1])
        if m.sum() >= 8:
            acc_base[b].append(np.mean(rb[m])); acc_refit[b].append(np.mean(rf[m]))
print(f"{'band':>9} {'base mean':>10} {'refit mean':>11} {'|bias| reduction':>16}")
for b in bands:
    bb = np.mean(acc_base[b]); rr = np.mean(acc_refit[b])
    print(f"{b[0]:>3}-{b[1]:<4} {bb:>10.2f} {rr:>11.2f} {abs(bb)-abs(rr):>15.2f}")

# ---- descent effect (EXCLUDED from map; reported for DR) ----
print("\n=== descent effect (v_ax < -2, NOT in the fit) ===")
Sall = samples(runs, fwd_only=False)
sp, Km, a_b3, adrag_b3, vax = Sall.T
for lo, hi in [(-16, -8), (-8, -4), (-4, -2), (-2, 2), (2, 6), (6, 14)]:
    m = (vax >= lo) & (vax < hi) & (sp >= 3) & (sp < 14)
    if m.sum() >= 10:
        L = (a_b3[m] - adrag_b3[m]) / Km[m]
        print(f"  v_ax {lo:>4}..{hi:<4} n={int(m.sum()):>4}  L_med {np.median(L):+.2f}")
print("  -> map uses forward/level (v_ax>=-2); descent lapse folded into inc7 DR band, not modeled")
