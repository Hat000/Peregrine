"""A2: THE CIRCULARITY TEST.

The parent's table splits approaches by OUTCOME and finds deaths stop capturing after 8 m.
A flight that is off-centre at 5 m dies BECAUSE it is off-centre, so outcome-conditioned geometry
can be pure tautology.

NULL MODEL (homogeneous control law by construction):
  fit ONE range-indexed linear-Gaussian closed loop  x(r-dr) = M(r) x(r) + eps,  x = (lat, dlat/dr)
  to ALL approaches POOLED (labels never used).  Simulate synthetic approaches from the empirical
  entry distribution, label them by a pure threshold on the terminal lateral tuned to the observed
  death rate, and recompute the parent's table.  Any structure that survives is SELECTION ONLY.

If the null reproduces the observed table, the observed table carries no information about control.
"""
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
from a1_reproduce import C, approaches  # noqa: E402

RG = np.arange(10.0, 1.0 - 1e-9, -0.5)     # range grid, closing


def resample(a, col="L_lat"):
    """Signed lateral on the closing range grid, from the LAST descending crossing of each range."""
    rho = a[:, C["rho"]]
    y = a[:, C[col]]
    ok = np.isfinite(rho) & np.isfinite(y)
    rho, y = rho[ok], y[ok]
    if len(rho) < 4:
        return None
    out = np.full(len(RG), np.nan)
    for j, r in enumerate(RG):
        # last index where rho crosses r downward
        hits = np.where((rho[:-1] >= r) & (rho[1:] < r))[0]
        if len(hits):
            i = hits[-1]
            f = (rho[i] - r) / max(rho[i] - rho[i + 1], 1e-9)
            out[j] = y[i] + f * (y[i + 1] - y[i])
        else:
            m = np.abs(rho - r) <= 0.35
            if m.any():
                out[j] = np.median(y[m])
    return out


def build_matrix(app, col="L_lat"):
    Y, lab, meta = [], [], []
    for a in app:
        v = resample(a["A"], col)
        if v is None:
            continue
        Y.append(v)
        lab.append(a["died"])
        meta.append((a["lineage"], a["gate"]))
    return np.array(Y), np.array(lab, dtype=bool), meta


def fit_pooled(Y):
    """Range-step transition of x=(lat, dlat) fitted on ALL rows, outcome labels never touched."""
    n, m = Y.shape
    dl = np.full_like(Y, np.nan)
    dl[:, 1:] = np.diff(Y, axis=1)              # lat(r_{j}) - lat(r_{j-1}), per 0.5 m closed
    M, Q, keep = [], [], []
    for j in range(m - 1):
        x0 = np.column_stack([Y[:, j], dl[:, j]])
        x1 = np.column_stack([Y[:, j + 1], dl[:, j + 1]])
        ok = np.isfinite(x0).all(1) & np.isfinite(x1).all(1)
        if ok.sum() < 40:
            M.append(None); Q.append(None); keep.append(0); continue
        A0, A1 = x0[ok], x1[ok]
        # lat(j+1) = lat(j) + dl(j+1) identically -> only the dl row needs fitting
        X = np.column_stack([A0, np.ones(len(A0))])
        beta, *_ = np.linalg.lstsq(X, A1[:, 1], rcond=None)
        res = A1[:, 1] - X @ beta
        M.append(beta); Q.append(float(np.std(res))); keep.append(int(ok.sum()))
    return M, Q, keep


def simulate(Y, M, Q, nsim=40000, rng=None):
    rng = rng or np.random.default_rng(0)
    m = Y.shape[1]
    dl = np.full_like(Y, np.nan)
    dl[:, 1:] = np.diff(Y, axis=1)
    j0 = 0
    ok = np.isfinite(Y[:, j0]) & np.isfinite(dl[:, j0 + 1])
    seed_lat = Y[ok, j0]
    seed_dl = dl[ok, j0 + 1]
    pick = rng.integers(0, len(seed_lat), nsim)
    lat = seed_lat[pick].copy()
    d = seed_dl[pick].copy()
    S = np.full((nsim, m), np.nan)
    S[:, 0] = lat
    for j in range(m - 1):
        if M[j] is None:
            S[:, j + 1] = S[:, j]
            continue
        b = M[j]
        dn = b[0] * lat + b[1] * d + b[2] + rng.normal(0.0, Q[j], nsim)
        lat = lat + dn
        d = dn
        S[:, j + 1] = lat
    return S


def tab(Y, lab, rs=(8.0, 5.0, 3.0), name=""):
    idx = [int(np.argmin(np.abs(RG - r))) for r in rs]
    print(f"    {name:<10}{'n':>6}" + "".join(f"{f'@{r:g}m':>9}" for r in rs) + f"{'8->5':>8}{'5->3':>8}")
    for tag, m in (("PASSED", ~lab), ("DIED", lab)):
        v = np.abs(Y[m][:, idx])
        med = [float(np.nanmedian(v[:, k])) for k in range(len(rs))]
        r85 = med[1] / med[0] - 1.0
        r53 = med[2] / med[1] - 1.0
        print(f"    {tag:<10}{int(m.sum()):>6}" + "".join(f"{x:>9.2f}" for x in med)
              + f"{100*r85:>7.0f}%{100*r53:>7.0f}%")


if __name__ == "__main__":
    app = approaches(seen_only=True)
    Y, lab, meta = build_matrix(app)
    print(f"resampled approaches: {Y.shape[0]}  grid {RG[0]}..{RG[-1]} m  died={lab.sum()}")
    print("\n=== OBSERVED (real labels) ===")
    tab(Y, lab, name="observed")

    M, Q, keep = fit_pooled(Y)
    print("\n--- pooled range-step model (labels NEVER used) ---")
    for j in range(0, len(M), 3):
        if M[j] is None:
            continue
        print(f"   r {RG[j]:>5.1f}->{RG[j+1]:>5.1f}  dlat = {M[j][0]:+.4f}*lat {M[j][1]:+.4f}*dlat "
              f"{M[j][2]:+.4f}   sig={Q[j]:.3f}  n={keep[j]}")

    S = simulate(Y, M, Q, nsim=40000)
    # death rule: pure threshold on the terminal lateral, tuned to the observed death rate
    term = np.abs(S[:, -1])
    p_die = float(lab.mean())
    thr = float(np.nanquantile(term, 1.0 - p_die))
    slab = term > thr
    print(f"\n=== NULL SIM (one homogeneous law; death := |lat(1 m)| > {thr:.2f} m, "
          f"rate matched to {100*p_die:.1f}%) ===")
    tab(S, slab, name="null")

    # sensitivity: death threshold at the true half-width instead of rate-matched
    slab2 = term > 0.75
    print(f"\n=== NULL SIM, threshold = the real half-width 0.75 m (death rate "
          f"{100*slab2.mean():.1f}%) ===")
    tab(S, slab2, name="null-0.75")

    # sensitivity: death decided at 1.5 m (before the last blind metre) rather than at 1 m
    j15 = int(np.argmin(np.abs(RG - 1.5)))
    t15 = np.abs(S[:, j15])
    slab3 = t15 > float(np.nanquantile(t15, 1.0 - p_die))
    print(f"\n=== NULL SIM, outcome decided at 1.5 m ===")
    tab(S, slab3, name="null-1.5m")
    pickle.dump(dict(Y=Y, lab=lab, meta=meta, RG=RG, M=M, Q=Q, S=S),
                open(os.path.join(HERE, "a2.pkl"), "wb"))
