"""A1: reproduce the parent's 8/5/3 m lateral-capture table, then stress it.

Primary instrument: per-APPROACH sample of |L_lat| (gravity-levelled lateral offset of the gate
from the drone) in fixed RANGE bands, gates 1-3, pose_seen ticks, aim_off approaches excluded.
"""
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(os.path.join(HERE, "lat_ticks.pkl"), "rb"))
COLS = D["cols"]
C = {c: i for i, c in enumerate(COLS)}


def approaches(seen_only=True, gates=(1, 2, 3)):
    """Split the corpus into per-(session,gate) approaches."""
    out = []
    for s in D["sessions"]:
        A = s["A"]
        for g in gates:
            m = A[:, C["gate"]] == g
            if not m.any():
                continue
            a = A[m]
            if seen_only:
                a = a[a[:, C["seen"]] > 0.5]
            if len(a) < 4:
                continue
            out.append(dict(run=s["run"], lineage=s["lineage"], gate=int(g),
                            died=bool(a[0, C["died"]]), A=a,
                            zbias=s["zbias"], ckpt=s["ckpt"]))
    return out


def band(a, r, half=0.5, col="L_lat"):
    """Median |col| over ticks with |rho - r| <= half on this approach. NaN if unsampled."""
    rho = a[:, C["rho"]]
    m = np.abs(rho - r) <= half
    if m.sum() < 1:
        return np.nan
    return float(np.median(np.abs(a[m, C[col]])))


def band_signed(a, r, half=0.5, col="L_lat"):
    rho = a[:, C["rho"]]
    m = np.abs(rho - r) <= half
    if m.sum() < 1:
        return np.nan
    return float(np.median(a[m, C[col]]))


def turn(a):
    """Integrated |yaw| over the approach (rad) -- the bank-reversal proxy."""
    t = a[:, C["t"]]
    wy = a[:, C["w_yaw"]]
    if len(t) < 3:
        return np.nan
    dt = np.diff(t)
    return float(np.sum(np.abs(0.5 * (wy[1:] + wy[:-1])) * dt))


def summarise(app, tag, cols=("L_lat",)):
    print(f"\n=== {tag} ===")
    for col in cols:
        rows = []
        for a in app:
            rows.append((a["died"], band(a["A"], 8.0, col=col), band(a["A"], 5.0, col=col),
                         band(a["A"], 3.0, col=col), turn(a["A"])))
        R = np.array([[float(r[0])] + list(r[1:]) for r in rows], dtype=float)
        print(f"  [{col}]  n_approach={len(R)}   died={int(R[:,0].sum())}  passed={int((1-R[:,0]).sum())}")
        hdr = f"    {'group':<8}{'n8':>5}{'@8m':>8}{'n5':>5}{'@5m':>8}{'n3':>5}{'@3m':>8}{'turn':>8}"
        print(hdr)
        for lab, mm in (("PASSED", R[:, 0] == 0), ("DIED", R[:, 0] == 1)):
            v = R[mm]
            def md(j):
                x = v[:, j][np.isfinite(v[:, j])]
                return (len(x), float(np.median(x)) if len(x) else np.nan)
            n8, m8 = md(1); n5, m5 = md(2); n3, m3 = md(3); _, mt = md(4)
            print(f"    {lab:<8}{n8:>5}{m8:>8.2f}{n5:>5}{m5:>8.2f}{n3:>5}{m3:>8.2f}{mt:>8.2f}")
    return app


if __name__ == "__main__":
    app = approaches(seen_only=True)
    print(f"total approaches (gates 1-3, aim-free, >=4 seen ticks): {len(app)}")
    from collections import Counter
    print("by gate:", Counter((a["gate"], a["died"]) for a in app))
    print("by lineage:", Counter(a["lineage"] for a in app).most_common())
    summarise(app, "REPRODUCTION (pose_seen, median |L_lat| in +-0.5 m range bands)")
    # sensitivity: body-frame lateral (what the policy is fed) and the raw un-levelled rel[1]
    summarise(app, "SENSITIVITY: body-frame lateral rely (POLICY INPUT, NOT levelled)",
              cols=("rely",))
    summarise(app, "SENSITIVITY: levelled VERTICAL offset", cols=("L_vert",))
