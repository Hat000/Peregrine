"""S18 step 1c -- FINAL lapse fit + identifiability + held-out validation.

Model (minimal, S16-compatible): a_up_eff = L(|v|) * K(collective), L a piecewise-linear
multiplicative lapse, L(0)=1 (hover map anchored), L(>=v_rec)=1. Defaults L==1 == legacy.

Correct projection: a_meas - g = L*Km*b3 + a_drag  =>  L = (a_b3 - s*adrag_b3)/Km.
Restrict to forward flight (v_ax >= -2) -- excludes fast-descent / failure transients where
the quasi-static rotor model is invalid (reported separately, folded into DR not the map).
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
K_of = lambda c: np.interp(c, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)


def load(p):
    return [r for r in (json.loads(l) for l in open(p / "debug_obs.jsonl", encoding="utf-8"))
            if r.get("type") != "header"]


def Rm(q):
    q = np.asarray(q, float); return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def samples_for(run_paths, fwd_only=True):
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
            if fwd_only and -vb[2] < -2.0:
                continue
            c2 = np.asarray(QUAD_DRAG_C2_MEASURED, float)
            c2v = np.where(vb >= 0, c2[:, 0], c2[:, 1])
            a_drag = R @ (-c2v * np.abs(vb) * vb)
            a = (v[k + 1] - v[k - 1]) / dt - np.array([0, 0, G])
            out.append((np.linalg.norm(vv), Km, float(a @ b3), float(a_drag @ b3)))
    return np.array(out)


runs = sorted(DATA.glob("20260612_*_inc6_*"))
S = samples_for(runs)
sp, Km, a_b3, adrag_b3 = S.T
L1 = (a_b3 - adrag_b3) / Km     # measured-drag (s=1) per-tick lapse

print(f"{len(S)} forward-flight samples")
print("\n=== lapse L(|v|), correct sign, identifiability across drag-scale s ===")
print(f"{'band':>9} {'n':>5} {'s=0.5':>6} {'s=1.0':>6} {'s=1.5':>6} | spread")
fine = [0, 2, 4, 6, 8, 10, 12, 15, 18, 24]
knot_v, knot_L = [], []
for lo, hi in zip(fine[:-1], fine[1:]):
    m = (sp >= lo) & (sp < hi)
    if m.sum() < 10:
        continue
    vals = [np.median((a_b3[m] - s * adrag_b3[m]) / Km[m]) for s in (0.5, 1.0, 1.5)]
    ctr = 0.5 * (lo + hi)
    knot_v.append(ctr); knot_L.append(vals[1])
    print(f"{lo:>3}-{hi:<4} {int(m.sum()):>5} {vals[0]:>6.2f} {vals[1]:>6.2f} {vals[2]:>6.2f} | {max(vals)-min(vals):.2f}")

print("\nKnot candidates (bin center, s=1 median L):")
for vv, LL in zip(knot_v, knot_L):
    print(f"  v={vv:>4.1f}  L={LL:.3f}")

# ---- held-out validation: fit on 12 runs, test predicted vs measured accel on 5 ----
print("\n=== held-out validation (fit on 12 runs, test on 5) ===")
import numpy as _np
rng = _np.random.RandomState(0)
idx = rng.permutation(len(runs))
train = [runs[i] for i in idx[:12]]; test = [runs[i] for i in idx[12:]]


def fit_knots(SS):
    sp, Km, a_b3, adrag_b3 = SS.T
    kv = [0, 4, 7, 10, 13, 16]; kL = []
    for c in kv:
        m = (sp >= max(0, c - 2)) & (sp < c + 2)
        if m.sum() < 8 or c == 0:
            kL.append(1.0 if c == 0 or c >= 16 else None)
        else:
            kL.append(float(_np.clip(_np.median((a_b3[m] - adrag_b3[m]) / Km[m]), 0.5, 1.05)))
    # fill None at v=16 with 1.0; ensure anchors
    kL = [1.0 if x is None else x for x in kL]; kL[0] = 1.0
    return _np.array(kv, float), _np.array(kL)


kv, kL = fit_knots(samples_for(train))
print("train knots L:", dict(zip(kv.astype(int), _np.round(kL, 3))))
St = samples_for(test); spt, Kmt, a_b3t, adrag_b3t = St.T
L_pred = _np.interp(spt, kv, kL)
# predicted thrust-axis accel vs measured (drag held at measured s=1)
a_pred = L_pred * Kmt + adrag_b3t
resid_fit = a_b3t - a_pred
# baseline = S16 plant (no lapse, L=1)
resid_base = a_b3t - (Kmt + adrag_b3t)
print(f"test n={len(St)}")
print(f"  S16 plant (no lapse): b3-accel resid  mean {_np.mean(resid_base):+.2f}  RMS {_np.sqrt(_np.mean(resid_base**2)):.2f} m/s^2")
print(f"  refit (lapse ON)    : b3-accel resid  mean {_np.mean(resid_fit):+.2f}  RMS {_np.sqrt(_np.mean(resid_fit**2)):.2f} m/s^2")
for lo, hi in [(3, 6), (6, 9), (9, 12), (12, 18)]:
    m = (spt >= lo) & (spt < hi)
    if m.sum() < 10:
        continue
    print(f"   |v| {lo}-{hi}: n={int(m.sum()):>4}  base mean {_np.mean(resid_base[m]):+.2f}  refit mean {_np.mean(resid_fit[m]):+.2f}")
