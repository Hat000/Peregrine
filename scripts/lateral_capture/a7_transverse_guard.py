"""A7: the re-lock guard done RIGHT -- gate the TRANSVERSE channel, and drop whole approaches.

A6 gated on TOTAL vision+gyro apparent speed and this is wrong twice over:

 1. At 30 Hz the total is dominated by RADIAL RANGE FLAP, not landmark change.  vg.py measures
    transverse rms 0.07 m vs radial 0.43 m, and the deploy sigma is sig_r = 0.12*rho -- ~1 m at
    8 m, which over a 0.033 s tick is ~30 m/s of apparent speed from NOISE.  A total-speed gate at
    20 m/s therefore rejects range flap, not re-locks (7.2% of ticks, 77% of approaches).
 2. TRUNCATING at the first jump biases the RANGE COVERAGE -- it keeps early/far ticks and discards
    late/near ones, so the surviving @3 m sample is drawn from a different population than the
    @8 m sample.  That is the very confound the range-band design exists to avoid.

A wrong-gate re-lock moves the BEARING, which is the pixel-clean channel.  So:
    gate on TRANSVERSE apparent speed  |dp - (dp.bhat)bhat| / dt
    and DROP the whole approach on any hit, so range coverage stays unbiased.
Still vision + gyro only; the A6 pin test (blind to dead reckoning) applies unchanged.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "failure_profile"))
from a1_reproduce import C, approaches  # noqa: E402
from a2_null import RG, fit_pooled, resample, simulate  # noqa: E402
from a4_tests import logistic  # noqa: E402
from a6_seam_guard import _att  # noqa: E402

J = {float(r): i for i, r in enumerate(RG)}
j8, j5, j3 = J[8.0], J[5.0], J[3.0]


def transverse_speed(A):
    """Vision+gyro TRANSVERSE apparent speed. Radial (range-flap) motion is projected out."""
    t = A[:, C["t"]]
    w = np.column_stack([A[:, C["w_roll"]], A[:, C["w_pitch"]], A[:, C["w_yaw"]]])
    rel = np.column_stack([A[:, C["relx"]], A[:, C["rely"]], A[:, C["relz"]]])
    ok = np.isfinite(rel).all(1) & np.isfinite(t)
    if ok.sum() < 4:
        return None, None
    k = int(ok.sum()) // 2
    R = _att(t[ok], w[ok], A[ok, C["roll"]][k], A[ok, C["pitch"]][k], k)
    r = rel[ok]
    rho = np.maximum(np.linalg.norm(r, axis=1), 1e-9)
    p = -np.einsum("nij,nj->ni", R, r)
    b = np.einsum("nij,nj->ni", R, r / rho[:, None])          # bearing unit vector, levelled
    dp = np.diff(p, axis=0)
    bm = 0.5 * (b[1:] + b[:-1])
    bm /= np.maximum(np.linalg.norm(bm, axis=1), 1e-9)[:, None]
    tr = dp - np.einsum("ni,ni->n", dp, bm)[:, None] * bm
    dt = np.diff(t[ok])
    sp = np.zeros(int(ok.sum()))
    g = dt > 1e-3
    sp[1:][g] = np.linalg.norm(tr[g], axis=1) / dt[g]
    return np.where(ok)[0], sp


def guard_drop(app, thr):
    out, hit_p, hit_d, np_, nd_, tot, nj = [], 0, 0, 0, 0, 0, 0
    for a in app:
        A = a["A"][a["A"][:, C["seen"]] > 0.5]
        if len(A) < 4:
            continue
        idx, sp = transverse_speed(A)
        if idx is None:
            continue
        tot += len(sp); h = int((sp > thr).sum()); nj += h
        if a["died"]:
            nd_ += 1; hit_d += (h > 0)
        else:
            np_ += 1; hit_p += (h > 0)
        if h == 0:
            b = dict(a); b["A"] = A
            out.append(b)
    return out, dict(tick_pct=100 * nj / max(tot, 1), fp=100 * hit_p / max(np_, 1),
                     fd=100 * hit_d / max(nd_, 1))


def matrix(g):
    YY, LL = [], []
    for a in g:
        v = resample(a["A"], "L_lat")
        if v is not None:
            YY.append(v); LL.append(a["died"])
    return np.array(YY), np.array(LL, dtype=bool)


def med3(Y, m):
    v = np.abs(Y[m])
    return [float(np.nanmedian(v[:, k])) for k in (j8, j5, j3)]


if __name__ == "__main__":
    app = approaches(seen_only=True)
    print(f"unguarded approaches: {len(app)}  died={sum(a['died'] for a in app)}\n")
    print("=== TRANSVERSE-CHANNEL contamination, and whether DEATHS re-lock more ===")
    print(f"  {'thr m/s':>8}{'tick %':>9}{'appr hit P':>12}{'appr hit D':>12}{'D-P':>8}{'kept':>7}")
    for thr in (3, 5, 8, 12, 20):
        g, s = guard_drop(app, thr)
        print(f"  {thr:>8}{s['tick_pct']:>8.2f}%{s['fp']:>11.1f}%{s['fd']:>11.1f}%"
              f"{s['fd']-s['fp']:>7.1f}%{len(g):>7}")

    print("\n=== THE PARENT'S TABLE, transverse-guarded, whole approaches dropped ===")
    print(f"  {'cut':<18}{'n':>5}{'D':>5}{'P 8->5->3':>22}{'8->5':>8}{'D 8->5->3':>22}{'8->5':>8}")
    keep = {}
    for thr in (3, 5, 8, 12, 20, 1e9):
        g = app if thr > 1e8 else guard_drop(app, thr)[0]
        Y, lab = matrix(g)
        if len(Y) < 60:
            continue
        keep[thr] = (Y, lab)
        p, d = med3(Y, ~lab), med3(Y, lab)
        lbl = "UNGUARDED" if thr > 1e8 else f"transverse thr={thr}"
        print(f"  {lbl:<18}{len(Y):>5}{int(lab.sum()):>5}"
              f"{f'{p[0]:.2f} -> {p[1]:.2f} -> {p[2]:.2f}':>22}{100*(p[1]/p[0]-1):>7.0f}%"
              f"{f'{d[0]:.2f} -> {d[1]:.2f} -> {d[2]:.2f}':>22}{100*(d[1]/d[0]-1):>7.0f}%")

    print("\n=== THE REFUTATION, re-run at every guard cut ===")
    print("  T1 = logistic beta(|lat@8m| | lat@3m).  A REAL control defect shows up EARLY -> beta>0.")
    print("  gap = median(DIED) - median(PASSED), observed vs the homogeneous-law null refit to the")
    print("  SAME guarded data.  'null >= obs' means selection alone already over-explains it.\n")
    print(f"  {'cut':<18}{'T1 obs':>10}{'T1 null':>10}{'mean gap@5m':>13}{'null gap@5m':>13}"
          f"{'mean gap@3m':>13}{'null gap@3m':>13}")
    for thr, (Y, lab) in keep.items():
        M, Q, _ = fit_pooled(Y)
        S = simulate(Y, M, Q, nsim=40000)
        term = np.abs(S[:, -1])
        slab = term > float(np.nanquantile(term, 1.0 - float(lab.mean())))
        t1 = []
        for Yx, Lx in ((Y, lab), (S, slab)):
            m = np.isfinite(Yx[:, j8]) & np.isfinite(Yx[:, j3])
            a8, a3 = np.abs(Yx[m, j8]), np.abs(Yx[m, j3])
            B3 = np.column_stack([a3, a3 ** 2, np.minimum(a3, .75), np.maximum(a3 - .75, 0)])
            b, s = logistic(np.column_stack([B3, a8]), Lx[m].astype(float))
            t1.append(b[-1])
        gaps = []
        for Yx, Lx in ((Y, lab), (S, slab)):
            g5, g3 = [], []
            for lo, hi in ((0.0, .5), (.5, 1.), (1., 1.5), (1.5, 2.5), (2.5, 6.)):
                e = np.abs(Yx[:, j8])
                m = np.isfinite(e) & (e >= lo) & (e < hi)
                mp, md = m & ~Lx, m & Lx
                if mp.sum() < 8 or md.sum() < 8:
                    continue
                g5.append(np.nanmedian(np.abs(Yx[md, j5])) - np.nanmedian(np.abs(Yx[mp, j5])))
                g3.append(np.nanmedian(np.abs(Yx[md, j3])) - np.nanmedian(np.abs(Yx[mp, j3])))
            gaps.append((np.mean(g5) if g5 else np.nan, np.mean(g3) if g3 else np.nan))
        lbl = "UNGUARDED" if thr > 1e8 else f"transverse thr={thr}"
        print(f"  {lbl:<18}{t1[0]:>+10.3f}{t1[1]:>+10.3f}{gaps[0][0]:>13.2f}{gaps[1][0]:>13.2f}"
              f"{gaps[0][1]:>13.2f}{gaps[1][1]:>13.2f}")

    print("\n=== OUTCOME-FREE quantities under the guard (the numbers I actually report) ===")
    print(f"  {'cut':<18}{'n':>5}{'slope |8m|->|2m|':>18}{'termlat p50':>12}{'p90':>7}{'p99':>7}"
          f"{'>0.75':>8}{'>0.45':>8}")
    for thr, (Y, lab) in keep.items():
        a, b = np.abs(Y[:, j8]), np.abs(Y[:, J[2.0]])
        m = np.isfinite(a) & np.isfinite(b)
        sl = np.polyfit(a[m], b[m], 1)[0] if m.sum() > 30 else np.nan
        tl = np.abs(Y[:, J[2.0]]); tl = tl[np.isfinite(tl)]
        lbl = "UNGUARDED" if thr > 1e8 else f"transverse thr={thr}"
        print(f"  {lbl:<18}{len(Y):>5}{sl:>18.3f}{np.quantile(tl,.5):>12.2f}"
              f"{np.quantile(tl,.9):>7.2f}{np.quantile(tl,.99):>7.2f}"
              f"{np.mean(tl>0.75):>8.3f}{np.mean(tl>0.45):>8.3f}")
