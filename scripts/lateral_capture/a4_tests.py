"""A4: the statistical verdict + the outcome-free replacement measurement.

T1  CONDITIONAL-INDEPENDENCE TEST.  Under pure selection the outcome is a function of the TERMINAL
    geometry alone, so once lat@3m is known, lat@8m must add NO information about the outcome.
    A genuinely worse controller shows up EARLY, so its early state would stay predictive.
    Logistic: died ~ s(|lat@3m|) + |lat@8m|.  Run on observed AND on the null (which has no
    control heterogeneity by construction) so the null supplies the reference coefficient.

T2  THE PARENT'S SECOND CLAIM, against the same null.  "At matched lateral error, dying approaches
    command ~1/3 the corrective roll."  The null has no roll, but it has the QUANTITY THE ROLL
    PRODUCES: the subsequent change in lateral error.  If the null shows the same matched-state
    split, the roll result is the same tautology one derivative up.

T3  OUTCOME-FREE ADEQUACY.  The pooled loop, labels never used: what fraction of an entry lateral
    error survives to the gate plane, and what does that imply for the population?
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a1_reproduce import C, approaches  # noqa: E402
from a2_null import RG  # noqa: E402

P = pickle.load(open(os.path.join(HERE, "a2.pkl"), "rb"))
Y, lab, S, M, Q = P["Y"], P["lab"], P["S"], P["M"], P["Q"]
J = {float(r): i for i, r in enumerate(RG)}
j8, j5, j3 = J[8.0], J[5.0], J[3.0]
term = np.abs(S[:, -1])
slab = term > float(np.nanquantile(term, 1.0 - float(lab.mean())))


def logistic(X, y, iters=60, ridge=1e-4):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p) - ridge * b
        try:
            b = b + np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
    W = np.maximum(p * (1 - p), 1e-9)
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov))


print("=== T1  CONDITIONAL-INDEPENDENCE: does |lat@8m| still predict death given |lat@3m|? ===")
print("    (a real control defect is visible EARLY; pure selection is not)")
for nm, Yx, Lx in (("OBSERVED", Y, lab), ("NULL SIM", S, slab)):
    m = np.isfinite(Yx[:, j8]) & np.isfinite(Yx[:, j3])
    a8, a3 = np.abs(Yx[m, j8]), np.abs(Yx[m, j3])
    y = Lx[m].astype(float)
    # spline-ish basis on lat3 so its shape is not forced linear
    B3 = np.column_stack([a3, a3 ** 2, np.minimum(a3, 0.75), np.maximum(a3 - 0.75, 0)])
    b0, s0 = logistic(B3, y)
    b1, s1 = logistic(np.column_stack([B3, a8]), y)
    z = b1[-1] / max(s1[-1], 1e-9)
    print(f"  {nm:<9} n={m.sum():>6} died={int(y.sum()):>5}   beta(|lat@8m| | lat@3m) = "
          f"{b1[-1]:+.4f} +- {s1[-1]:.4f}   z = {z:+.2f}")

print("\n=== T2  THE ROLL CLAIM, tested as the quantity roll PRODUCES ===")
print("    at matched |lat| in the 8->5 m window, subsequent capture d|lat| by outcome")
print(f"  {'|lat@8m| bin':<14}{'src':<6}{'nP':>6}{'nD':>6}{'P d|lat| 8->5':>15}"
      f"{'D d|lat| 8->5':>15}{'D/P':>8}")
for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5)):
    for nm, Yx, Lx in (("obs", Y, lab), ("null", S, slab)):
        e = np.abs(Yx[:, j8])
        m = np.isfinite(e) & (e >= lo) & (e < hi) & np.isfinite(Yx[:, j5])
        mp, md = m & ~Lx, m & Lx
        if mp.sum() < 8 or md.sum() < 8:
            continue
        dp = float(np.median(np.abs(Yx[mp, j8]) - np.abs(Yx[mp, j5])))
        dd = float(np.median(np.abs(Yx[md, j8]) - np.abs(Yx[md, j5])))
        print(f"  {f'{lo:.1f}-{hi:.1f}':<14}{nm:<6}{int(mp.sum()):>6}{int(md.sum()):>6}"
              f"{dp:>15.3f}{dd:>15.3f}{(dd/dp if abs(dp)>1e-6 else np.nan):>8.2f}")

print("\n--- the actual roll command, same matched-state cut (wire only) ---")
app = approaches(seen_only=True)
print(f"  {'|L_lat| bin':<14}{'nP tick':>9}{'nD tick':>9}{'P corr-roll':>13}{'D corr-roll':>13}"
      f"{'D/P':>8}")
for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5)):
    acc = {0: [], 1: []}
    for a in app:
        A = a["A"]
        m = (A[:, C["rho"]] >= 5.0) & (A[:, C["rho"]] <= 8.0) & (A[:, C["seen"]] > 0.5)
        v = A[m]
        if not len(v):
            continue
        L = v[:, C["L_lat"]]
        sel = (np.abs(L) >= lo) & (np.abs(L) < hi)
        if not sel.any():
            continue
        # corrective roll rate = commanded roll rate signed so that POSITIVE = toward the gate
        # (gate to the LEFT, L>0, needs a LEFT bank = negative roll -> corrective = -sign(L)*cmd)
        cr = -np.sign(L[sel]) * v[sel, C["cmd_roll"]]
        acc[1 if a["died"] else 0].extend(cr[np.isfinite(cr)].tolist())
    p, dd = np.array(acc[0]), np.array(acc[1])
    if len(p) > 20 and len(dd) > 20:
        print(f"  {f'{lo:.1f}-{hi:.1f}':<14}{len(p):>9}{len(dd):>9}{np.mean(p):>13.4f}"
              f"{np.mean(dd):>13.4f}{np.mean(dd)/np.mean(p) if abs(np.mean(p))>1e-9 else np.nan:>8.2f}")

print("\n=== T3  OUTCOME-FREE ADEQUACY of the pooled loop (labels never used) ===")
# deterministic propagation of a unit entry error through the fitted pooled loop
for r0 in (10.0, 8.0, 5.0):
    j = J[r0]
    lat, d = 1.0, 0.0
    for k in range(j, len(RG) - 1):
        if M[k] is None:
            continue
        dn = M[k][0] * lat + M[k][1] * d          # drop the intercept: unit-error response only
        lat += dn
        d = dn
    print(f"  entry error 1.00 m at {r0:>4.1f} m  ->  {lat:.2f} m at the gate plane "
          f"({100*(1-lat):.0f}% nulled)")
print(f"\n  observed |lat| population quantiles (ALL approaches, no labels):")
for r in (8.0, 6.0, 4.0, 2.0):
    v = np.abs(Y[:, J[r]])
    v = v[np.isfinite(v)]
    print(f"    r={r:>4.1f} m  n={len(v):>4}  p50 {np.quantile(v,.5):.2f}  p90 "
          f"{np.quantile(v,.9):.2f}  p99 {np.quantile(v,.99):.2f}   frac>0.75 "
          f"{np.mean(v>0.75):.3f}")
