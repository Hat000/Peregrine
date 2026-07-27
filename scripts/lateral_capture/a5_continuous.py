"""A5: the CONTINUOUS, outcome-free replacement for the died-vs-passed contrast.

The parent asked whether "died at this gate vs passed" is even the right contrast.  It is not:
A2-A4 show the split is a selection signature.  The non-circular object is the terminal lateral
itself, regressed on UPSTREAM state.  Nothing here uses the outcome.

L1  Terminal |lat| distribution and its upstream predictors (all approaches, labels never read).
L2  A NEW outcome-free lead: obs[11:14] is the RAW TILTED-BODY gate vector -- the policy's only
    fine-grained lateral is rel[1], which carries an attitude cross-coupling
        rel[1] = sin(pitch) sin(roll) L_fwd + cos(roll) L_lat + cos(pitch) sin(roll) L_vert
    The middle term is the signal; the first is an artifact proportional to RANGE x sin(pitch).
    Measured here, and tested for a behavioural consequence: at matched TRUE lateral error and
    matched range, does the corrective roll weaken as pitch grows?  (It must, if the loop closes
    on rel[1] uncompensated.)  This is a LEAD, not a verdict.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a1_reproduce import C, approaches  # noqa: E402

app = approaches(seen_only=True)
T = np.vstack([a["A"] for a in app])
T = T[T[:, C["seen"]] > 0.5]
rho = T[:, C["rho"]]
L = T[:, C["L_lat"]]
roll, pitch = T[:, C["roll"]], T[:, C["pitch"]]
Lf, Lv = T[:, C["L_fwd"]], T[:, C["L_vert"]]
rely = T[:, C["rely"]]
cmd = T[:, C["cmd_roll"]]
print(f"ticks: {len(T)}   approaches: {len(app)}")

print("\n=== L1  TERMINAL |lat| -- the continuous target, no outcome anywhere ===")
term, feat = [], []
for a in app:
    A = a["A"][a["A"][:, C["seen"]] > 0.5]
    r = A[:, C["rho"]]
    mt = (r >= 1.5) & (r <= 2.5)
    m8 = np.abs(r - 8.0) <= 0.5
    mw = (r >= 4.0) & (r <= 8.0)
    if not (mt.any() and m8.any() and mw.any()):
        continue
    term.append(np.median(np.abs(A[mt, C["L_lat"]])))
    feat.append([np.median(np.abs(A[m8, C["L_lat"]])),
                 np.median(np.abs(A[mw, C["w_yaw"]])),
                 np.median(np.abs(A[mw, C["roll"]])),
                 np.median(A[mw, C["pitch"]]),
                 np.median(np.abs(A[mw, C["cmd_roll"]])),
                 float(np.mean(A[mw, C["age"]] > 0.05)),
                 (-float(np.median(A[mw, C["drho"]][np.isfinite(A[mw, C["drho"]])]))
                  if np.isfinite(A[mw, C["drho"]]).any() else np.nan)])
term = np.array(term); F = np.array(feat)
names = ["|lat@8m|", "|yaw rate|", "|roll|", "pitch", "|roll cmd|", "stale frac", "closure"]
print(f"  n={len(term)}   terminal |lat| @1.5-2.5 m:  p50 {np.quantile(term,.5):.2f}  "
      f"p90 {np.quantile(term,.9):.2f}  p99 {np.quantile(term,.99):.2f}   "
      f"frac>0.75 {np.mean(term>0.75):.3f}")
ok = np.isfinite(F).all(1)
Fk = F[ok]
keep = [i for i in range(Fk.shape[1]) if Fk[:, i].std() > 1e-9]
names = [names[i] for i in keep]
Fk = Fk[:, keep]
X = np.column_stack([np.ones(len(Fk)), (Fk - Fk.mean(0)) / Fk.std(0)])
b, *_ = np.linalg.lstsq(X, term[ok], rcond=None)
res = term[ok] - X @ b
sig = np.sqrt(np.sum(res ** 2) / (len(res) - X.shape[1]) * np.diag(np.linalg.pinv(X.T @ X)))
print(f"  standardised regression of TERMINAL |lat| on upstream 4-8 m state (n={len(Fk)}):")
for i, nm in enumerate(names):
    print(f"    {nm:<12} {b[i+1]:+.3f} +- {sig[i+1]:.3f}   z={b[i+1]/max(sig[i+1],1e-9):+.1f}")
print(f"    R^2 = {1 - np.var(res)/np.var(term[ok]):.3f}")

print("\n=== L2  LEAD: attitude cross-coupling in the policy's ONLY fine-grained lateral ===")
art = np.sin(pitch) * np.sin(roll) * Lf
sig_ = np.cos(roll) * L
m = np.isfinite(art) & np.isfinite(sig_) & (rho >= 3.0) & (rho <= 9.0)
print(f"  ticks 3-9 m: n={m.sum()}")
print(f"    |signal  cos(roll)*L_lat|        p50 {np.median(np.abs(sig_[m])):.2f}  "
      f"p90 {np.quantile(np.abs(sig_[m]),.9):.2f}")
print(f"    |artifact sin(p)sin(r)*L_fwd|    p50 {np.median(np.abs(art[m])):.2f}  "
      f"p90 {np.quantile(np.abs(art[m]),.9):.2f}")
print(f"    artifact/|signal| ratio           p50 "
      f"{np.median(np.abs(art[m])/np.maximum(np.abs(sig_[m]),1e-3)):.2f}  "
      f"p90 {np.quantile(np.abs(art[m])/np.maximum(np.abs(sig_[m]),1e-3),.9):.2f}")
opp = np.mean(np.sign(art[m]) == -np.sign(sig_[m]))
print(f"    artifact OPPOSES the signal on {100*opp:.0f}% of ticks "
      f"(it cancels the perceived error)")
print(f"    median true pitch {np.degrees(np.median(pitch[m])):.1f} deg   "
      f"p90 |roll| {np.degrees(np.quantile(np.abs(roll[m]),.9)):.1f} deg")

print("\n  behavioural test: corrective roll vs TRUE lateral, stratified by pitch "
      "(matched range 4-8 m)")
print(f"  {'|L_lat| bin':<12}{'pitch bin':<16}{'n':>7}{'corr roll cmd':>15}")
mm = (rho >= 4.0) & (rho <= 8.0) & np.isfinite(cmd)
corr = -np.sign(L) * cmd                      # >0 = banking toward the gate
pq = np.quantile(pitch[mm], [0.25, 0.75])
for lo, hi in ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0)):
    for plab, pm in (("low  (<{:.0f}d)".format(np.degrees(pq[0])), pitch < pq[0]),
                     ("high (>{:.0f}d)".format(np.degrees(pq[1])), pitch > pq[1])):
        s = mm & pm & (np.abs(L) >= lo) & (np.abs(L) < hi)
        if s.sum() < 50:
            continue
        print(f"  {f'{lo}-{hi}':<12}{plab:<16}{int(s.sum()):>7}{np.mean(corr[s]):>15.4f}")
